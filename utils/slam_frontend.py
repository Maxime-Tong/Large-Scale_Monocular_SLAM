import time

import os
import cv2
import numpy as np
import torch
import torch.multiprocessing as mp

from gaussian_splatting.gaussian_renderer import render
from gaussian_splatting.utils.graphics_utils import getProjectionMatrix2, getWorld2View2
from gui import gui_utils
from utils.camera_utils import Camera
from utils.eval_utils import eval_ate, save_gaussians
from utils.logging_utils import Log
from utils.multiprocessing_utils import clone_obj
from utils.pose_utils import update_pose
from utils.slam_utils import get_loss_tracking, get_median_depth
from utils.visual import visualize_and_save_images

def eval_rendering(kf_indices, cameras, gaussians, dataset, pipe, background, save_dir):
    import cv2
    for kf in kf_indices:
        viewpoint = cameras[kf]
        gt_image, _, _ = dataset[kf]

        rendering = render(viewpoint, gaussians, pipe, background)["render"]
        image = torch.clamp(rendering, 0.0, 1.0)

        gt = (gt_image.cpu().numpy().transpose((1, 2, 0)) * 255).astype(np.uint8)
        pred = (image.detach().cpu().numpy().transpose((1, 2, 0)) * 255).astype(
            np.uint8
        )
        gt = cv2.cvtColor(gt, cv2.COLOR_BGR2RGB)
        pred = cv2.cvtColor(pred, cv2.COLOR_BGR2RGB)
        visualize_and_save_images(gt, pred, save_dir, filename=f"{kf}.png")
        
class FrontEnd(mp.Process):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.background = None
        self.pipeline_params = None
        self.frontend_queue = None
        self.backend_queue = None
        self.q_main2vis = None
        self.q_vis2main = None

        self.monocular = config["Training"]["monocular"]

        self.requested_submaps = 0
        self.requested_init = True
        self.kf_indices = []

        self.gaussians = None
        self.cameras = dict()
        self.device = "cuda:0"
        self.pause = False
        
        self.submap_size = 16
        self.overlap_size = 1
        self.max_loops = 1
        self.solver = None

    def set_hyperparams(self):
        self.save_dir = self.config["Results"]["save_dir"]
        self.save_results = self.config["Results"]["save_results"]
        self.save_trj = self.config["Results"]["save_trj"]
        self.save_trj_kf_intv = self.config["Results"]["save_trj_kf_intv"]

        self.tracking_itr_num = self.config["Training"]["tracking_itr_num"]
        self.kf_interval = self.config["Training"]["kf_interval"]
        self.window_size = self.config["Training"]["window_size"]
        self.single_thread = self.config["Training"]["single_thread"]
    
    def request_submap_mapping(self, submap_idx, viewpoints, depth_maps):
        msg = ["submap_mapping", submap_idx, viewpoints, depth_maps]
        self.backend_queue.put(msg)
        self.requested_submaps += 1

    def sync_backend(self, data):
        self.gaussians = data[1]
        occ_aware_visibility = data[2]
        keyframes = data[3]
        self.occ_aware_visibility = occ_aware_visibility

        for kf_id, kf_R, kf_T in keyframes:
            self.cameras[kf_id].update_RT(kf_R.clone(), kf_T.clone())
            self.cleanup(kf_id)

    def cleanup(self, cur_frame_idx):
        self.cameras[cur_frame_idx].clean()
        if cur_frame_idx % 10 == 0:
            torch.cuda.empty_cache()
            
    def run(self):
        cur_frame_idx = 0
        cur_submap_idx = 0
        path_window = []
        projection_matrix = getProjectionMatrix2(
            znear=0.01,
            zfar=100.0,
            fx=self.dataset.fx,
            fy=self.dataset.fy,
            cx=self.dataset.cx,
            cy=self.dataset.cy,
            W=self.dataset.width,
            H=self.dataset.height,
        ).transpose(0, 1)
        projection_matrix = projection_matrix.to(device=self.device)
        tic = torch.cuda.Event(enable_timing=True)
        toc = torch.cuda.Event(enable_timing=True)

        while True:
            # if self.q_vis2main.empty():
            #     if self.pause:
            #         continue
            # else:
            #     data_vis2main = self.q_vis2main.get()
            #     self.pause = data_vis2main.flag_pause
            #     if self.pause:
            #         self.backend_queue.put(["pause"])
            #         continue
            #     else:
            #         self.backend_queue.put(["unpause"])
            
            if self.frontend_queue.empty():
                if self.requested_submaps >= 2:
                    time.sleep(0.1)
                    continue
                
                tic.record()
                if cur_frame_idx >= len(self.dataset):
                    if self.requested_submaps > 0: # wait for all submaps sync
                        time.sleep(0.1)
                        continue
                    
                    if self.save_results:
                        eval_ate(
                            self.cameras,
                            self.kf_indices,
                            self.save_dir,
                            0,
                            final=True,
                            monocular=self.monocular,
                        )
                        save_gaussians(
                            self.gaussians, self.save_dir, "final", final=True
                        )
                    break
                
                path_window.append(self.dataset.color_paths[cur_frame_idx])
                if len(path_window) == self.submap_size + self.overlap_size or cur_frame_idx == len(self.dataset) - 1:
                    predictions = self.solver.run_predictions(path_window, self.solver.vggt, self.max_loops)
                    self.solver.add_points(predictions)
                    self.solver.graph.optimize()
                    self.solver.map.update_submap_homographies(self.solver.graph)
                    path_window = path_window[-self.overlap_size:]
                    
                    submap = self.solver.map.get_latest_submap()
                    poses = submap.get_all_poses_world(ignore_loop_closure_frames=True)
                    frame_ids = submap.get_frame_ids()
                    print(poses.shape, frame_ids)
                    print("Total number of submaps in map", self.solver.map.get_num_submaps())
                    print("Total number of loop closures in map", self.solver.graph.get_num_loops())
                    
                    for idx, pose in enumerate(poses[:self.submap_size]):
                        w2c = np.linalg.inv(pose)
                        R = torch.from_numpy(w2c[:3, :3]).float()
                        T = torch.from_numpy(w2c[:3, 3]).float()
                        
                        viewpoint = Camera.init_from_dataset(
                            self.dataset, cur_frame_idx, projection_matrix
                        )
                        viewpoint.update_RT(R, T)
                        frame_idx = cur_submap_idx * self.submap_size + idx
                        self.cameras[frame_idx] = viewpoint
                        if frame_idx % self.kf_interval == 0:
                            self.kf_indices.append(frame_idx)

                    cur_submap_idx += 1                     
                cur_frame_idx += 1
                
                # self.q_main2vis.put(
                #     gui_utils.GaussianPacket(
                #         gaussians=clone_obj(self.gaussians),
                #         current_frame=viewpoint,
                #         keyframes=keyframes,
                #         kf_window=current_window_dict,
                #     )
                # )

                if (
                    self.save_results
                    and self.save_trj
                    and len(self.kf_indices) > self.save_trj_kf_intv
                    # and not self.requested_init
                ):
                    self.save_trj_kf_intv += self.save_trj_kf_intv
                    Log("Evaluating ATE at frame: ", cur_frame_idx)
                    eval_ate(
                        self.cameras,
                        self.kf_indices,
                        self.save_dir,
                        cur_frame_idx,
                        monocular=self.monocular,
                    )
                    
                    # eval_rendering(
                    #     self.kf_indices,
                    #     self.cameras,
                    #     self.gaussians,
                    #     self.dataset,
                    #     self.pipeline_params,
                    #     self.background,
                    #     'test_output/rendering'
                    # )
                toc.record()
                torch.cuda.synchronize()
            else:
                data = self.frontend_queue.get()
                if data[0] == "sync_backend":
                    self.sync_backend(data)
                    self.requested_submaps -= 1
                    
                elif data[0] == 'submap_mapping':
                    self.sync_backend(data)
                    self.requested_submaps -= 1
                    self.requested_init = False
                
                elif data[0] == 'init':
                    self.sync_backend(data)
                    self.requested_init = False

                elif data[0] == "stop":
                    Log("Frontend Stopped.")
                    break
