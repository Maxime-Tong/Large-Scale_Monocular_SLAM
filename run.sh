# dataset=office0
# python slam.py --config configs/rgbd/replica/$dataset.yaml --data /data/xthuang/dataset/slam/Replica/$dataset

# Monocular
python slam.py --config configs/mono/tum/fr2_xyz.yaml --data /data/zh/slam/TUM_RGBD/rgbd_dataset_freiburg2_xyz
# python slam.py --config configs/mono/tum/fr1_desk.yaml --data /data/zh/slam/TUM_RGBD/rgbd_dataset_freiburg1_desk