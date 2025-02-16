# Copyright (C) 2024-present Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).
#
# --------------------------------------------------------
# Dataloader for preprocessed BlendedMVS
# dataset at https://github.com/YoYo000/BlendedMVS
# See datasets_preprocess/preprocess_blendedmvs.py
# --------------------------------------------------------
import os.path as osp
import numpy as np
import glob
from dust3r.datasets.base.base_stereo_view_dataset import BaseStereoViewDataset
from dust3r.utils.image import imread_cv2
import json
from collections import deque
import random
import json

def read_split_file(file_path):
        # 打开并读取JSON文件
    with open(file_path, 'r') as f:
        folder_dict = json.load(f)

    # 打印字典内容
    return folder_dict
    

class BlendedMVStest(BaseStereoViewDataset):
    """ Dataset of outdoor street scenes, 5 images each time
    """

    def __init__(self, *args, ROOT, split=None, **kwargs):
        self.ROOT = ROOT
        super().__init__(*args, **kwargs)
        self._load_data(split)
        self.i = 0

    def _load_data(self, split):
        pair_paths = read_split_file(self.ROOT)
        self.scenes = []
        for pair_path in pair_paths:
            # pairs = glob.glob(osp.join(pair_path, '*.jpg'))
            self.scenes += [pair_path]

    def __len__(self):
        return len(self.scenes)

    def get_stats(self):
        return f'{len(self)} pairs from {len(self.scenes)} scenes'

    def _get_views(self, pair_idx, resolution, rng):
        impaths = self.scenes[self.i]
        self.i += 1
        views = []
        while len(impaths) > 0:  # some images (few) have zero depth
            impath = impaths.pop()
            impath = impath[0]
            # impath = self.scenes[scene][im_idx]
            # load camera params
            input_metadata = np.load(impath.replace('jpg', 'npz'))
            camera_pose = np.eye(4, dtype=np.float32)
            camera_pose[:3, :3] = input_metadata['R_cam2world']
            camera_pose[:3, 3] = input_metadata['t_cam2world']
            intrinsics = input_metadata['intrinsics'].astype(np.float32)
            # load image and depth
            rgb_image = imread_cv2(impath)
            depthmap = imread_cv2(impath.replace('jpg', 'exr'))
            rgb_image, depthmap, intrinsics = self._crop_resize_if_necessary(
                rgb_image, depthmap, intrinsics, resolution, rng=rng, info=impath)
            rgb_image_orig = rgb_image.copy()
            H, W = depthmap.shape[:2]
            fxfycxcy = np.array([intrinsics[0, 0]/W, intrinsics[1, 1]/H, intrinsics[0,2]/W, intrinsics[1,2]/H]).astype(np.float32)
            views.append(dict(
                img_org=rgb_image_orig,
                img=rgb_image,
                depthmap=depthmap,
                camera_pose=camera_pose,
                camera_intrinsics=intrinsics,
                fxfycxcy=fxfycxcy,
                dataset='blendedmvs',
                label=impath,
                instance=osp.split(impath)[1],
            ))

        return views



if __name__ == '__main__':
    from dust3r.datasets.base.base_stereo_view_dataset import view_name
    from dust3r.viz import SceneViz, auto_cam_size
    from dust3r.utils.image import rgb

    dataset = BlendedMVS(split='train', ROOT="data/blendedmvs_processed", resolution=224, aug_crop=16)

    for idx in np.random.permutation(len(dataset)):
        views = dataset[idx]
        assert len(views) == 2
        print(idx, view_name(views[0]), view_name(views[1]))
        viz = SceneViz()
        poses = [views[view_idx]['camera_pose'] for view_idx in [0, 1]]
        cam_size = max(auto_cam_size(poses), 0.001)
        for view_idx in [0, 1]:
            pts3d = views[view_idx]['pts3d']
            valid_mask = views[view_idx]['valid_mask']
            colors = rgb(views[view_idx]['img'])
            viz.add_pointcloud(pts3d, colors, valid_mask)
            viz.add_camera(pose_c2w=views[view_idx]['camera_pose'],
                           focal=views[view_idx]['camera_intrinsics'][0, 0],
                           color=(idx * 255, (1 - idx) * 255, 0),
                           image=colors,
                           cam_size=cam_size)
        viz.show()
