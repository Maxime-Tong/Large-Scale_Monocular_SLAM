import time

import os
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
        self.kf_indices = []

        self.gaussians = None
        self.cameras = dict()
        self.device = "cuda:0"
        self.pause = False
        
        self.vggtl = None
        self.chunk_data = None
        self.use_vggtl_depth = False

    def set_hyperparams(self):
        self.save_dir = self.config["Results"]["save_dir"]
        self.save_results = self.config["Results"]["save_results"]
        self.save_trj = self.config["Results"]["save_trj"]
        self.save_trj_kf_intv = self.config["Results"]["save_trj_kf_intv"]

        self.tracking_itr_num = self.config["Training"]["tracking_itr_num"]
        self.kf_interval = self.config["Training"]["kf_interval"]
        self.window_size = self.config["Training"]["window_size"]
        self.single_thread = self.config["Training"]["single_thread"]
    
    def request_submap_mapping(self, submap_idx, viewpoints, pcd_path):
        msg = ["submap_mapping", submap_idx, viewpoints, pcd_path]
        self.backend_queue.put(msg)
        self.requested_submaps += 1

    def sync_backend(self, data):
        self.gaussians = data[1]
        occ_aware_visibility = data[2]
        keyframes = data[3]
        self.occ_aware_visibility = occ_aware_visibility

        for kf_id, kf_R, kf_T in keyframes:
            self.cameras[kf_id].update_RT(kf_R.clone(), kf_T.clone())

    def cleanup(self, cur_frame_idx):
        self.cameras[cur_frame_idx].clean()
        if cur_frame_idx % 10 == 0:
            torch.cuda.empty_cache()

    def run(self):
        cur_frame_idx = 0
        cur_submap_idx = 0
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
                
                start_idx = cur_frame_idx
                end_idx = min(start_idx + self.vggtl.chunk_size, len(self.dataset))
                
                color_paths = self.dataset.color_paths[start_idx:end_idx]
                self.vggtl.update_submap(color_paths)
                
                submap_viewpoints = {}
                for frame_idx in range(start_idx, end_idx):
                    frame_idx_in_submap = frame_idx - start_idx
                    Rt = self.vggtl.get_frame_RT(frame_idx_in_submap)
                    
                    viewpoint = Camera.init_from_dataset(
                        self.dataset, frame_idx, projection_matrix
                    )
                    viewpoint.update_RT(*Rt)
                    self.cameras[frame_idx] = viewpoint
                    
                    if frame_idx % self.kf_interval == 0:
                        self.kf_indices.append(frame_idx)
                        
                    submap_viewpoints[frame_idx] = viewpoint             

                    # self.cleanup(frame_idx)
                
                pcd_path = os.path.join(self.vggtl.aligned_point_cloud_dir, f'chunk_{cur_submap_idx}.ply')
                self.request_submap_mapping(cur_submap_idx, submap_viewpoints, pcd_path)
                cur_frame_idx = end_idx
                cur_submap_idx += 1

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
                    and len(self.kf_indices) % self.save_trj_kf_intv == 0
                ):
                    Log("Evaluating ATE at frame: ", cur_frame_idx)
                    eval_ate(
                        self.cameras,
                        self.kf_indices,
                        self.save_dir,
                        cur_frame_idx,
                        monocular=self.monocular,
                    )
                toc.record()
                torch.cuda.synchronize()
            else:
                data = self.frontend_queue.get()
                if data[0] == "sync_backend":
                    self.sync_backend(data)
                    self.requested_submaps -= 1
                    
                elif data[0] == 'submap':
                    self.sync_backend(data)
                    self.requested_submaps -= 1

                elif data[0] == "stop":
                    Log("Frontend Stopped.")
                    break
