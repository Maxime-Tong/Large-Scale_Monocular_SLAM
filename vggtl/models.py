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

        self.result_chunk_dir = os.path.join(save_dir, 'result_chunks')
        self.aligned_point_cloud_dir = os.path.join(save_dir, 'aligned_point_clouds')
        os.makedirs(self.result_chunk_dir, exist_ok=True)
        os.makedirs(self.aligned_point_cloud_dir, exist_ok=True)
        
        print('Loading model...')

        self.model = VGGT()
        # _URL = "https://huggingface.co/facebook/VGGT-1B/resolve/main/model.pt"
        # model.load_state_dict(torch.hub.load_state_dict_from_url(_URL))
        _URL = self.config['Weights']['VGGT']
        state_dict = torch.load(_URL, map_location='cuda')
        self.model.load_state_dict(state_dict, strict=False)

        self.model.eval()
        self.model = self.model.to(self.device)

        self.skyseg_session = None
        
        self.current_chunk_idx = -1
        self.current_chunk_data = None
        self.current_sim3 = (1.0, np.eye(3), np.zeros(3))
        
        self.use_point_map = False
        # if self.sky_mask:
        #     print('Loading skyseg.onnx...')
        #     # Download skyseg.onnx if it doesn't exist
        #     if not os.path.exists("skyseg.onnx"):
        #         print("Downloading skyseg.onnx...")
        #         download_file_from_url("https://huggingface.co/JianyuanWang/skyseg/resolve/main/skyseg.onnx", "skyseg.onnx")

        #     self.skyseg_session = onnxruntime.InferenceSession("skyseg.onnx")
        
        print('init done.')
    
    def update_current_sim3(self, s, R, t): # R = c2w.R, t = c2w.t
        prev_s, prev_R, prev_t = self.current_sim3
        curr_s, curr_R, curr_t = s, R, t
        
        R_new = prev_R @ curr_R
        s_new = prev_s * curr_s
        t_new = prev_s * (prev_R @ curr_t) + prev_t 
        
        self.current_sim3 = (s_new, R_new, t_new)
        
    def get_frame_RT(self, frame_idx):
        pose = self.current_chunk_data['aligned_poses'][frame_idx]
        w2c = np.linalg.inv(pose)
        return (
            torch.from_numpy(w2c[:3, :3]).float(),
            torch.from_numpy(w2c[:3, 3]).float()
        )
        
    def process_single_chunk(self, images, chunk_idx=None):
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
        
        # Save predictions to disk instead of keeping in memory
        save_dir = self.result_chunk_dir
        filename = f"chunk_{chunk_idx}.npy"
        
        save_path = os.path.join(save_dir, filename)
                    
        # predictions['depth'] = np.squeeze(predictions['depth'])

        # np.save(save_path, predictions)
        return predictions
    
    def numpy_to_pypose_sim3(self, s: float, R_mat: np.ndarray, t_vec: np.ndarray):
        """Convert numpy s,R,t to pypose Sim3"""
        q = R.from_matrix(R_mat).as_quat()  # [x,y,z,w]
        # pypose requires [t, q, s] format
        data = np.concatenate([t_vec, q, np.array([s])])
        return pp.Sim3(torch.from_numpy(data).float().to(self.device))
    
    def align_chunk_pair(self, source_chunk_data, target_chunk_data):
        source_points = source_chunk_data['world_points']
        source_confs = source_chunk_data['world_points_conf']
        target_points = target_chunk_data['world_points']
        target_confs = target_chunk_data['world_points_conf']
        
        s_rel, R_rel, t_rel = weighted_align_point_maps(
            target_points, target_confs, source_points, source_confs,
            conf_threshold=self.conf_threshold, config=self.config
        )
        aligned_points = apply_sim3_direct(source_points, s_rel, R_rel, t_rel)

        return aligned_points, (s_rel, R_rel, t_rel) 
    
    def update_submap(self, image_paths):
        images_crop = load_and_preprocess_images(image_paths, mode='crop', return_originals=False)
        images_crop = images_crop.to(self.device)
        
        previous_idx = self.current_chunk_idx
        current_idx = self.current_chunk_idx + 1
        current_data = self.process_single_chunk(images_crop, chunk_idx=current_idx)
        
        if not self.use_point_map:
            depth_map = current_data["depth"]  # (S, H, W, 1)
            conf = current_data["depth_conf"]  # (S, H, W)
            # print(depth_map.shape, conf.shape, current_data['extrinsic'].shape, current_data['intrinsic'].shape)
            world_points = unproject_depth_map_to_point_map(depth_map, current_data['extrinsic'], current_data['intrinsic'])
            
            current_data['world_points'] = world_points
            current_data['world_points_conf'] = conf
        
        if current_idx > 0:
            print(f"Aligning chunk {current_idx} to chunk {previous_idx}")
            previous_data = self.current_chunk_data
            
            prev_points = previous_data['world_points'][-self.overlap:]
            curr_points = current_data['world_points'][:self.overlap]
            prev_conf = previous_data['world_points_conf'][-self.overlap:]
            curr_conf = current_data['world_points_conf'][:self.overlap]

            conf_threshold = min(np.median(prev_conf), np.median(curr_conf)) * 0.1
            
            s_rel, R_rel, t_rel = weighted_align_point_maps(
                prev_points, prev_conf, curr_points, curr_conf, 
                conf_threshold=conf_threshold, config=self.config
            )
            
            self.update_current_sim3(s_rel, R_rel, t_rel)
            aligned_points = apply_sim3_direct(current_data['world_points'], *self.current_sim3)
        else:
            aligned_points = current_data['world_points']
        
            
        # only save the last self.overlap points
        points = aligned_points[:self.step].reshape(-1, 3)
        colors = (current_data['images'][:self.step].transpose(0, 2, 3, 1).reshape(-1, 3) * 255).astype(np.uint8)
        confs = current_data['world_points_conf'][:self.step].reshape(-1)
        save_path = os.path.join(self.aligned_point_cloud_dir, f'chunk_{current_idx}.ply')
        
        save_confident_pointcloud_batch(
            points=points, colors=colors, confs=confs, output_path=save_path,
            conf_threshold=np.mean(confs) * self.config['Model']['Pointcloud_Save']['conf_threshold_coef'],
            sample_ratio=self.config['Model']['Pointcloud_Save']['sample_ratio']
        )
        
        S = np.eye(4)
        S[:3, :3] = self.current_sim3[0] * self.current_sim3[1] # s * R
        S[:3, 3] = self.current_sim3[2] # t
        current_chunk_poses = []
        
        for extrinsic in current_data['extrinsic']:
            w2c = np.eye(4)
            w2c[:3, :] = extrinsic
            c2w = np.linalg.inv(w2c)
            aligned_c2w = S @ c2w
            current_chunk_poses.append(aligned_c2w)
        current_data['aligned_poses'] = np.stack(current_chunk_poses, axis=0)
        # current_data['depth_upsampled'] = joint_bilateral_upsampling_batch(depths = current_data['depth'], images = images)
        
        self.current_chunk_idx = current_idx
        self.current_chunk_data = current_data
        return current_data
        