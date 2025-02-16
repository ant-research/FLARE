# Copyright (C) 2024-present Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).
#
# --------------------------------------------------------
# Dataloader for preprocessed arkitscenes
# dataset at https://github.com/apple/ARKitScenes - Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International Public License https://github.com/apple/ARKitScenes/tree/main?tab=readme-ov-file#license
# See datasets_preprocess/preprocess_arkitscenes.py
# --------------------------------------------------------
import os.path as osp
import cv2
import numpy as np
import random
import mast3r.utils.path_to_dust3r  # noqa
# check the presence of models directory in repo to be sure its cloned
from dust3r.datasets.base.base_stereo_view_dataset import BaseStereoViewDataset
from dust3r.utils.image import imread_cv2, imread_cv2_orig
from collections import deque
import os
import json
import time
import glob
import tqdm
import torch
from io import BytesIO

try:
    from pcache_fileio import fileio
    flag_pcache = True
except:
    flag_pcache = False
from enum import Enum, auto
from pathlib import Path
from PIL import Image
oss_folder_path = 'oss://antsys-vilab/datasets/pcache_datasets/'
# oss_file_path = 'oss://antsys-vilab/datasets/pcache_datasets/SAM_1B/sa_000000/images/sa_1011.jpg'
pcache_folder_path = 'pcache://vilabpcacheproxyi-pool.cz50c.alipay.com:39999/mnt/antsys-vilab_datasets_pcache_datasets/'
# pcache_file_path = oss_file_path.replace(oss_folder_path,pcache_folder_path)
from decord import VideoReader


def read_camera_parameters(filename):
    with open(filename) as f:
        lines = f.readlines()
        lines = [line.rstrip() for line in lines]
    # extrinsics: line [1,5), 4x4 matrix
    extrinsics = np.fromstring(' '.join(lines[1:5]), dtype=np.float32, sep=' ').reshape((4, 4))
    # intrinsics: line [7-10), 3x3 matrix
    intrinsics = np.fromstring(' '.join(lines[7:10]), dtype=np.float32, sep=' ').reshape((3, 3))
    # TODO: assume the feature is 1/4 of the original image size
    return intrinsics, extrinsics


def load_from_json(filename):
    """Load a dictionary from a JSON filename.

    Args:
        filename: The filename to load from.
    """
    with open(filename, encoding="UTF-8") as file:
        return json.load(file)

class Re10K_ps_demo(BaseStereoViewDataset):
    def __init__(self, *args, split, ROOT, only_pose=False, min_gap=10, max_gap=22, interal=32, **kwargs):
        self.ROOT = ROOT
        # Collect chunks.
        ROOT = Path(ROOT)
        self.chunks = []
        self.split = split
        images_path = osp.join(ROOT, 'images')
        images_list = sorted(os.listdir(images_path))
        images_list = [os.path.join(images_path, image) for image in images_list]
        self.images_list = images_list[::interal]
        self.global_idx = 0
        # for root in ROOT:
        super().__init__(*args, **kwargs)
        self.rendering = True

    def __len__(self):
        return len(self.images_list)-1
    
    @staticmethod
    def image_read(image_file):
        img = cv2.imread(image_file)
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    @staticmethod
    def depth_read(depth_file):
        depth = np.load(depth_file)
        depth_min = np.percentile(depth, 5)
        depth[depth==np.nan] = 0
        depth[depth==np.inf] = 0
        depth_max =  np.percentile(depth, 95)  
        depth[depth>=depth_max] = 0
        depth[depth>=depth_min+200] = 0
        return depth

    def _get_views(self, idx, resolution, rng):
        # a, b = camera_pose_right[:3, 3], camera_pose_left[:3, 3]
        # scale = np.linalg.norm(a - b)
        images = [self.images_list[self.global_idx], self.images_list[self.global_idx+1]]
        self.global_idx += 1
        views = []
        poses = []
        for image in images:
            frame = imread_cv2(image)
            frame_num = int((image.split('/')[-1].split('.')[0]).replace('frame_', ''))
            pose_path = osp.join(self.ROOT,  'cams', '{:08d}_cam.txt'.format(frame_num))
            intrinsics, extrinsics = read_camera_parameters(pose_path)
            poses.append(extrinsics)
            
        a, b = poses[0][:3, 3], poses[1][:3, 3]
        scale = np.linalg.norm(a - b)
        for image in images:
            frame = imread_cv2(image)
            frame_num = int((image.split('/')[-1].split('.')[0]).replace('frame_', ''))
            pose_path = osp.join(self.ROOT, 'cams', '{:08d}_cam.txt'.format(frame_num))
            intrinsics, extrinsics = read_camera_parameters(pose_path)
            extrinsics =  np.linalg.inv(extrinsics)
            H, W = frame.shape[:2]
            depthmap = np.zeros((H, W), dtype=np.float32)
            rgb_image, depthmap, intrinsics = self._crop_resize_if_necessary(
                frame, depthmap, intrinsics, resolution, rng=rng, info=f'')
            rgb_image_orig = rgb_image.copy()
            H, W = depthmap.shape[:2]
            extrinsics[:3, 3] = extrinsics[:3, 3]/scale
            camera_pose = extrinsics
            fxfycxcy = np.array([intrinsics[0, 0]/W, intrinsics[1, 1]/H, intrinsics[0,2]/W, intrinsics[1,2]/H]).astype(np.float32)
            views.append(dict(
                img_org=rgb_image_orig,
                img=rgb_image,
                depthmap=depthmap.astype(np.float32),
                camera_pose=camera_pose.astype(np.float32),
                camera_intrinsics=intrinsics.astype(np.float32),
                fxfycxcy=fxfycxcy,
                dataset='re10k',
                label=image,
                instance=image,
            ))
        return views


if __name__ == "__main__":
    from dust3r.datasets.base.base_stereo_view_dataset import view_name
    from dust3r.viz import SceneViz, auto_cam_size
    from dust3r.utils.image import rgb
    import nerfvis.scene as scene_vis 
    dataset = Re10K_ps(split='train', ROOT="/nas3/zsz/re10k/re10k",resolution=[(256, 256)], aug_crop=16)
    for idx in np.random.permutation(len(dataset)):
        views = dataset[idx]
        # assert len(views) == 2
        # print(view_name(views[0]), view_name(views[1]))
        # view_idxs = list(range(len(views)))
        # poses = [views[view_idx]['camera_pose'] for view_idx in view_idxs]
        # cam_size = max(auto_cam_size(poses), 0.001)
        # pts3ds = []
        # colors = []
        # valid_masks = []
        # c2ws = []
        # intrinsics = []
        # for view_idx in view_idxs:
        #     pts3d = views[view_idx]['pts3d']
        #     pts3ds.append(pts3d)
        #     valid_mask = views[view_idx]['valid_mask']
        #     valid_masks.append(valid_mask)
        #     color = rgb(views[view_idx]['img'])
        #     colors.append(color)
            # viz.add_pointcloud(pts3d, colors, valid_mask)
        #     c2ws.append(views[view_idx]['camera_pose'])

        
        # pts3ds = np.stack(pts3ds, axis=0)
        # colors = np.stack(colors, axis=0)
        # valid_masks = np.stack(valid_masks, axis=0)
        # c2ws = np.stack(c2ws)
        # scene_vis.set_title("My Scene")
        # scene_vis.set_opencv() 
        # # colors = torch.zeros_like(structure).to(structure)
        # # scene_vis.add_points("points", pts3ds.reshape(-1,3)[valid_masks.reshape(-1)], vert_color=colors.reshape(-1,3)[valid_masks.reshape(-1)], point_size=1)
        # # for i in range(len(c2ws)):
        # f = 1111.0 / 2.5
        # z = 10.
        # scene_vis.add_camera_frustum("cameras", r=c2ws[:, :3, :3], t=c2ws[:, :3, 3], focal_length=f,
        #                 image_width=colors.shape[2], image_height=colors.shape[1],
        #                 z=z, connect=False, color=[1.0, 0.0, 0.0])
        # for i in range(len(c2ws)):
        #     scene_vis.add_image(
        #                     f"images/{i}",
        #                     colors[i], # Can be a list of paths too (requires joblib for that) 
        #                     r=c2ws[i, :3, :3],
        #                     t=c2ws[i, :3, 3],
        #                     # Alternatively: from nerfvis.utils import split_mat4; **split_mat4(c2ws)
        #                     focal_length=f,
        #                     z=z,
        #                 )
        # scene_vis.display(port=8081)

