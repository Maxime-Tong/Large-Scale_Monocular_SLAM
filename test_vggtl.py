import sys
import os
import glob

module_path = os.path.join(os.path.dirname(__file__), 'vggtl')
sys.path.insert(0, module_path)

from models import VGGT_Long
from loop_utils.config_utils import load_config
from vggt.utils.load_fn import load_and_preprocess_images

# img_dir = '/data/xthuang/SLAM/large_scale/data/video'
img_dir = '/data/xthuang/dataset/slam/Replica/room0/results'
img_list = sorted(glob.glob(os.path.join(img_dir, "frame*.jpg")) + 
                                glob.glob(os.path.join(img_dir, "frame*.png")))
config_path = 'vggtl/configs/base_config.yaml'
config = load_config(config_path)

vggtl = VGGT_Long(save_dir='test_output', config=config)

step = vggtl.chunk_size - vggtl.overlap
for start_idx in range(0, len(img_list), step):
    end_idx = min(start_idx + vggtl.chunk_size, len(img_list))
    images = load_and_preprocess_images(img_list[start_idx:end_idx], mode='crop').to(vggtl.device)
    predictions = vggtl.process_single_chunk(images, chunk_idx=start_idx // step)
    # print(predictions.keys()) 
    # dict_keys(['pose_enc', 'depth', 'depth_conf', 'world_points', 'world_points_conf', 'images', 'extrinsic', 'intrinsic'])
    
    # print(predictions['world_points'].shape)
    # (60, 294, 518, 3)
    
    # print(predictions['images'].shape)
    # (60, 3, 294, 518)
    
    # print(predictions['extrinsic'].shape)
    # (60, 3, 4)