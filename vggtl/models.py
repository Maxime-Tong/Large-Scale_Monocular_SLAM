import numpy as np
import argparse

import os
import glob
import threading
import torch
from tqdm.auto import tqdm
import cv2

import gc

try:
    import onnxruntime
except ImportError:
    print("onnxruntime not found. Sky segmentation may not work.")

from LoopModels.LoopModel import LoopDetector
from LoopModelDBoW.retrieval.retrieval_dbow import RetrievalDBOW
# from loop_utils.visual_util import segment_sky, download_file_from_url
from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images
from vggt.utils.geometry import closed_form_inverse_se3, unproject_depth_map_to_point_map
from vggt.utils.pose_enc import pose_encoding_to_extri_intri

import numpy as np

from loop_utils.sim3loop import Sim3LoopOptimizer
from loop_utils.sim3utils import *
from datetime import datetime

from PIL import Image

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sys

import pypose as pp
from scipy.spatial.transform import Rotation as R

def sim3_transform(prev_sim3, curr_sim3):
    prev_s, prev_R, prev_t = prev_sim3
    curr_s, curr_R, curr_t = curr_sim3
    R_new = prev_R @ curr_R
    s_new = prev_s * curr_s
    t_new = prev_s * (prev_R @ curr_t) + prev_t
    return s_new, R_new, t_new
    
def joint_bilateral_upsampling_batch(depths, images, radius=8, eps=1e-3):
    B, H, W, C = images.shape
    
    depths_upsampled = []
    for depth, image in zip(depths, images):
        # d_refined = cv2.ximgproc.guidedFilter(guide=image, src=d_init, radius=radius, eps=eps)
        d_refined = cv2.resize(depth, (W, H), interpolation=cv2.INTER_CUBIC).astype(np.float32)
        depths_upsampled.append(d_refined)
    return np.stack(depths_upsampled, axis=0)

def remove_duplicates(data_list):
    """
        data_list: [(67, (3386, 3406), 48, (2435, 2455)), ...]
    """
    seen = {} 
    result = []
    
    for item in data_list:
        if item[0] == item[2]:
            continue

        key = (item[0], item[2])
        
        if key not in seen.keys():
            seen[key] = True
            result.append(item)
    
    return result

class VGGT_Long:
    def __init__(self, save_dir, config):
        self.config = config

        self.chunk_size = self.config['Model']['chunk_size']
        self.overlap = self.config['Model']['overlap']
        self.step = self.chunk_size - self.overlap
        self.conf_threshold = 1.5
        self.seed = 42
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
        self.sky_mask = False
        self.useDBoW = self.config['Model']['useDBoW']

        self.output_dir = save_dir
        self.save_result = True
        
        self.result_unaligned_dir = os.path.join(save_dir, 'results_unaligned')
        self.result_aligned_dir = os.path.join(save_dir, 'results_aligned')
        self.result_loop_dir = os.path.join(save_dir, 'results_loop')
        self.pcd_dir = os.path.join(save_dir, 'pcd')
        if self.save_result:
            os.makedirs(self.result_unaligned_dir, exist_ok=True)
            os.makedirs(self.result_aligned_dir, exist_ok=True)
            os.makedirs(self.result_loop_dir, exist_ok=True)
            os.makedirs(self.pcd_dir, exist_ok=True)
        
        print('Loading model...')

        self.model = VGGT()
        self.model = self.model.to(self.device)
        # _URL = "https://huggingface.co/facebook/VGGT-1B/resolve/main/model.pt"
        # model.load_state_dict(torch.hub.load_state_dict_from_url(_URL))
        _URL = self.config['Weights']['VGGT']
        state_dict = torch.load(_URL, map_location=self.device)
        self.model.load_state_dict(state_dict, strict=False)
        self.model.eval()

        self.skyseg_session = None
        
        self.img_list = []
        self.current_chunk_idx = -1
        self.current_chunk_data = None
        self.all_camera_poses = []
        self.sim3_list = []
        self.cumulative_sim3_list = []
        self.chunk_indices = []
        # if self.sky_mask:
        #     print('Loading skyseg.onnx...')
        #     # Download skyseg.onnx if it doesn't exist
        #     if not os.path.exists("skyseg.onnx"):
        #         print("Downloading skyseg.onnx...")
        #         download_file_from_url("https://huggingface.co/JianyuanWang/skyseg/resolve/main/skyseg.onnx", "skyseg.onnx")

        #     self.skyseg_session = onnxruntime.InferenceSession("skyseg.onnx")
        
        self.loop_enable = self.config['Model']['loop_enable']
        self.loop_optimizer = Sim3LoopOptimizer(self.config)
        self.half_window = int(self.config['Model']['loop_chunk_size'] / 2)
        self.loop_list = [] # e.g. [(1584, 139), ...]
        self.loop_seen = set()
        self.loop_sim3_list = []
        self.loop_ptr = 0
        
        self.loop_info_save_path = os.path.join(save_dir, 'loop_closures.txt')
        
        print('init done.')
    
    def init_loop_pairs(self):
        loop_detector = LoopDetector(
            image_dir=None,
            output=self.loop_info_save_path,
            config=self.config
        )
        loop_detector.image_paths = self.img_list
        loop_detector.load_model()
        loop_detector.extract_descriptors()
        loop_detector.find_loop_closures()
        
        if self.save_result:
            loop_detector.save_results()
            
        self.loop_list = sorted(loop_detector.get_loop_list(), key=lambda x: x[0])
        
        del loop_detector
        torch.cuda.empty_cache()
        # print(self.loop_list)
                
    def process_single_chunk(self, range_1, chunk_idx=None, range_2=None, is_loop=False):
        start_idx, end_idx = range_1
        chunk_image_paths = self.img_list[start_idx:end_idx]
        if range_2 is not None:
            start_idx, end_idx = range_2
            chunk_image_paths += self.img_list[start_idx:end_idx]

        images = load_and_preprocess_images(chunk_image_paths).to(self.device)
        print(f"Loaded {len(images)} images")
        
        # images: [B, 3, H, W]
        assert len(images.shape) == 4
        assert images.shape[1] == 3

        torch.cuda.empty_cache()
        with torch.no_grad():
            with torch.cuda.amp.autocast(dtype=self.dtype):
                predictions = self.model(images)
        torch.cuda.empty_cache()

        print("Converting pose encoding to extrinsic and intrinsic matrices...")
        extrinsic, intrinsic = pose_encoding_to_extri_intri(predictions["pose_enc"], images.shape[-2:])
        predictions["extrinsic"] = extrinsic
        predictions["intrinsic"] = intrinsic
        print("Processing model outputs...")
        for key in predictions.keys():
            if isinstance(predictions[key], torch.Tensor):
                predictions[key] = predictions[key].cpu().numpy().squeeze(0)
        
        if not is_loop and range_2 is None:
            extrinsics = predictions['extrinsic']
            chunk_range = self.chunk_indices[chunk_idx]
            self.all_camera_poses.append((chunk_range, extrinsics))
        
        predictions['depth'] = np.squeeze(predictions['depth'])
        
        if self.save_result:
            # Save predictions to disk instead of keeping in memory
            if is_loop:
                save_dir = self.result_loop_dir
                filename = f"loop_{range_1[0]}_{range_1[1]}_{range_2[0]}_{range_2[1]}.npy"
            else:
                if chunk_idx is None:
                    raise ValueError("chunk_idx must be provided when is_loop is False")
                save_dir = self.result_unaligned_dir
                filename = f"chunk_{chunk_idx}.npy"
            
            save_path = os.path.join(save_dir, filename)
            np.save(save_path, predictions)
                    
        return predictions if is_loop or range_2 is not None else None
    
    def get_frame_RT(self, frame_idx):
        chunk_idx = frame_idx // self.step
        frame_idx_in_chunk = frame_idx % self.step
        chunk_range, chunk_extrinsics = self.all_camera_poses[chunk_idx]
        
        cumulative_sim3_list = accumulate_sim3_transforms(self.sim3_list)
        if chunk_idx > 0:
            s, R, t = cumulative_sim3_list[chunk_idx - 1]
            S = np.eye(4)
            S[:3, :3] = s * R
            S[:3, 3] = t
            
            # TODO: compute transformed_w2c directly
            w2c = np.eye(4)
            w2c[:3, :] = chunk_extrinsics[frame_idx_in_chunk]
            c2w = np.linalg.inv(w2c)
            transformed_c2w = S @ c2w  # Be aware of the left multiplication!
            transformed_w2c = np.linalg.inv(transformed_c2w)
            R = transformed_w2c[:3, :3]
            t = transformed_w2c[:3, 3]
        return R, t
    
    def get_frame_depth(self, frame_idx):
        return self.current_chunk_data['depth'][frame_idx]
    
    def update(self, chunk_range):
        previous_chunk_idx = self.current_chunk_idx
        current_chunk_idx = self.current_chunk_idx + 1
        current_data = self.process_single_chunk(chunk_range, chunk_idx=current_chunk_idx)
        
        if current_chunk_idx > 0:
            print(f"Aligning chunk {current_chunk_idx} to chunk {previous_chunk_idx}")
            previous_data = self.current_chunk_data
            
            overlap = min(self.overlap, chunk_range[1] - chunk_range[0])
            prev_points = previous_data['world_points'][-overlap:]
            curr_points = current_data['world_points'][:overlap]
            prev_conf = previous_data['world_points_conf'][-overlap:]
            curr_conf = current_data['world_points_conf'][:overlap]

            conf_threshold = min(np.median(prev_conf), np.median(curr_conf)) * 0.1
            
            current_sim3 = weighted_align_point_maps(
                prev_points, prev_conf, curr_points, curr_conf, 
                conf_threshold=conf_threshold, config=self.config
            )
            
            self.sim3_list.append(current_sim3)

        # if self.save_result:        
        #     points = aligned_points.reshape(-1, 3)
        #     colors = (current_data['images'].transpose(0, 2, 3, 1).reshape(-1, 3) * 255).astype(np.uint8)
        #     confs = current_data['world_points_conf'].reshape(-1)
        #     save_path = os.path.join(self.pcd_dir, f'chunk_{current_chunk_idx}.ply')
            
        #     save_confident_pointcloud_batch(
        #         points=points, colors=colors, confs=confs, output_path=save_path,
        #         conf_threshold=np.mean(confs) * self.config['Model']['Pointcloud_Save']['conf_threshold_coef'],
        #         sample_ratio=self.config['Model']['Pointcloud_Save']['sample_ratio']
        #     )

        self.current_chunk_idx = current_chunk_idx
        self.current_chunk_data = current_data
        self.chunk_indices.append(chunk_range)

        return current_data
    
    def loop_optimization(self, cur_frame_idx):
        while self.loop_ptr < len(self.loop_list):
            idx1, idx2 = self.loop_list[self.loop_ptr]
            if idx1 <= cur_frame_idx:
                print('Loop SIM(3) estimating...')
                chunk_idx1_0based = find_chunk_index(self.chunk_indices, idx1)
                chunk1 = self.chunk_indices[chunk_idx1_0based]
                range1 = get_frame_range(chunk1, idx1, self.half_window)
                
                chunk_idx2_0based = find_chunk_index(self.chunk_indices, idx2)
                chunk2 = self.chunk_indices[chunk_idx2_0based]
                range2 = get_frame_range(chunk2, idx2, self.half_window)

                key = (chunk_idx1_0based, chunk_idx2_0based)
                if key not in self.loop_seen:
                    self.loop_seen.add(key)

                    loop_predictions = self.process_single_chunk(range1, range_2=range2, is_loop=True)
                    print('chunk_a align')
                    point_map_loop = loop_predictions['world_points'][:range1[1] - range1[0]]
                    conf_loop = loop_predictions['world_points_conf'][:range1[1] - range1[0]]
                    chunk_a_rela_begin = range1[0] - self.chunk_indices[chunk_idx1_0based][0]
                    chunk_a_rela_end = chunk_a_rela_begin + range1[1] - range2[0]

                    chunk_data_a = np.load(os.path.join(self.result_unaligned_dir, f"chunk_{chunk_idx1_0based}.npy"), allow_pickle=True).item()
                    
                    point_map_a = chunk_data_a['world_points'][chunk_a_rela_begin:chunk_a_rela_end]
                    conf_a = chunk_data_a['world_points_conf'][chunk_a_rela_begin:chunk_a_rela_end]
                
                    conf_threshold = min(np.median(conf_a), np.median(conf_loop)) * 0.1
                    s_a, R_a, t_a = weighted_align_point_maps(point_map_a, 
                                                            conf_a, 
                                                            point_map_loop, 
                                                            conf_loop, 
                                                            conf_threshold=conf_threshold,
                                                            config=self.config)
                    print("Estimated Scale:", s_a)
                    print("Estimated Rotation:\n", R_a)
                    print("Estimated Translation:", t_a)

                    print('chunk_a align')
                    point_map_loop = loop_predictions['world_points'][-range2[1] + range2[0]:]
                    conf_loop = loop_predictions['world_points_conf'][-range2[1] + range2[0]:]
                    chunk_b_rela_begin = range2[0] - self.chunk_indices[chunk_idx2_0based][0]
                    chunk_b_rela_end = chunk_b_rela_begin + range2[1] - range2[0]

                    chunk_data_b = np.load(os.path.join(self.result_unaligned_dir, f"chunk_{chunk_idx2_0based}.npy"), allow_pickle=True).item()
                    
                    point_map_b = chunk_data_b['world_points'][chunk_b_rela_begin:chunk_b_rela_end]
                    conf_b = chunk_data_b['world_points_conf'][chunk_b_rela_begin:chunk_b_rela_end]
                
                    conf_threshold = min(np.median(conf_b), np.median(conf_loop)) * 0.1
                    s_b, R_b, t_b = weighted_align_point_maps(point_map_b, 
                                                            conf_b, 
                                                            point_map_loop, 
                                                            conf_loop, 
                                                            conf_threshold=conf_threshold,
                                                            config=self.config)
                    print("Estimated Scale:", s_b)
                    print("Estimated Rotation:\n", R_b)
                    print("Estimated Translation:", t_b)

                    print('a -> b SIM 3')
                    s_ab, R_ab, t_ab = compute_sim3_ab((s_a, R_a, t_a), (s_b, R_b, t_b))
                    print("Estimated Scale:", s_ab)
                    print("Estimated Rotation:\n", R_ab)
                    print("Estimated Translation:", t_ab)
                    
                    self.loop_sim3_list.append((chunk_idx1_0based, chunk_idx2_0based, (s_ab, R_ab, t_ab)))
            else:
                break
            self.loop_ptr += 1
        print(self.loop_sim3_list)
        self.sim3_list = self.loop_optimizer.optimize(self.sim3_list, self.loop_sim3_list)
