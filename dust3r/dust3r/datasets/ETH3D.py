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
from dust3r.datasets.base.base_stereo_view_dataset import BaseStereoViewDataset, BaseStereoViewDataset_test
from dust3r.utils.image import imread_cv2, imread_cv2_orig
from collections import deque
import os
import json
import time
import glob
import tqdm
try:
    from pcache_fileio import fileio
    flag_pcache = True
except:
    flag_pcache = False
from enum import Enum, auto
from pathlib import Path

oss_folder_path = 'oss://antsys-vilab/datasets/pcache_datasets/'
# oss_file_path = 'oss://antsys-vilab/datasets/pcache_datasets/SAM_1B/sa_000000/images/sa_1011.jpg'
pcache_folder_path = 'pcache://vilabpcacheproxyi-pool.cz50c.alipay.com:39999/mnt/antsys-vilab_datasets_pcache_datasets/'
# pcache_file_path = oss_file_path.replace(oss_folder_path,pcache_folder_path)

class ETH3D(BaseStereoViewDataset_test):
    def __init__(self, *args, split, ROOT, meta, only_pose=False, **kwargs):
        self.ROOT = ROOT
        self.index = 0
        with open(meta, 'rb') as f:
            self.meta = json.load(f)    
        self.scenes = list(self.meta.keys())
        super().__init__(*args, **kwargs)

    def __len__(self):
        return len(self.scenes)
    
    @staticmethod
    def image_read(image_file):
        img = cv2.imread(image_file)
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

  
    def _get_views(self, idx, resolution, rng):
        scene = self.scenes[self.index]
        images_list = self.meta[scene]['images']
        intrinsics_org = np.array(self.meta[scene]['intrinsics'])
        depth_maps = self.meta[scene]['depths']
        camera_poses = np.array(self.meta[scene]['poses'])
        views = []
        for image, depthmap, camera_pose in zip(images_list, depth_maps, camera_poses):
            rgb_image = self.image_read(image)
            H, W = rgb_image.shape[:2]
            # intrinsics = np.array([[W, 0, W/2], [0, H, H/2], [0, 0, 1]])
            depthmap_mask = np.load(depthmap.split('.npy')[0] + '_valid.npy')
            depthmap = 1/np.load(depthmap) 
            depthmap[~depthmap_mask] = 0
            camera_pose = np.linalg.inv(camera_pose)
            rgb_image, depthmap, intrinsics = self._crop_resize_if_necessary(
                rgb_image, depthmap, intrinsics_org, resolution, rng=rng, info=None)
            rgb_image_orig = rgb_image.copy()
            H, W = depthmap.shape[:2]
            fxfycxcy = np.array([intrinsics[0, 0]/W, intrinsics[1, 1]/H, intrinsics[0,2]/W, intrinsics[1,2]/H]).astype(np.float32)
            views.append(dict(
                img_org=rgb_image_orig,
                img=rgb_image,
                depthmap=depthmap.astype(np.float32),
                camera_pose=camera_pose.astype(np.float32),
                camera_intrinsics=intrinsics.astype(np.float32),
                fxfycxcy=fxfycxcy,
                dataset='own',
                label=scene,
                instance=scene,
            ))
        self.index += 1
        return views


if __name__ == "__main__":
    from dust3r.datasets.base.base_stereo_view_dataset import view_name
    from dust3r.viz import SceneViz, auto_cam_size
    from dust3r.utils.image import rgb
    import nerfvis.scene as scene_vis 
    dataset = ETH3D(split='train', ROOT="/nas3/zsz/ETH3D_results/",meta='/nas3/zsz/DPSNet/metas.json',resolution=[(512, 384)], aug_crop=16)

    for idx in np.random.permutation(len(dataset)):
        views = dataset[idx]
        # assert len(views) == 2
        # print(view_name(views[0]), view_name(views[1]))
        view_idxs = list(range(len(views)))
        poses = [views[view_idx]['camera_pose'] for view_idx in view_idxs]
        cam_size = max(auto_cam_size(poses), 0.001)
        pts3ds = []
        colors = []
        valid_masks = []
        c2ws = []
        intrinsics = []
        for view_idx in view_idxs:
            pts3d = views[view_idx]['pts3d']
            pts3ds.append(pts3d)
            valid_mask = views[view_idx]['valid_mask']
            valid_masks.append(valid_mask)
            color = rgb(views[view_idx]['img'])
            colors.append(color)
            # viz.add_pointcloud(pts3d, colors, valid_mask)
            c2ws.append(views[view_idx]['camera_pose'])

        
        pts3ds = np.stack(pts3ds, axis=0)
        colors = np.stack(colors, axis=0)
        valid_masks = np.stack(valid_masks, axis=0)
        c2ws = np.stack(c2ws)
        scene_vis.set_title("My Scene")
        scene_vis.set_opencv() 
        # colors = torch.zeros_like(structure).to(structure)
        scene_vis.add_points("points", pts3ds.reshape(-1,3)[valid_masks.reshape(-1)], vert_color=colors.reshape(-1,3)[valid_masks.reshape(-1)], point_size=1)
        # for i in range(len(c2ws)):
        f = 1111.0 / 2.5
        z = 10.
        scene_vis.add_camera_frustum("cameras", r=c2ws[:, :3, :3], t=c2ws[:, :3, 3], focal_length=f,
                        image_width=colors.shape[2], image_height=colors.shape[1],
                        z=z, connect=False, color=[1.0, 0.0, 0.0])
        for i in range(len(c2ws)):
            scene_vis.add_image(
                            f"images/{i}",
                            colors[i], # Can be a list of paths too (requires joblib for that) 
                            r=c2ws[i, :3, :3],
                            t=c2ws[i, :3, 3],
                            # Alternatively: from nerfvis.utils import split_mat4; **split_mat4(c2ws)
                            focal_length=f,
                            z=z,
                        )
        scene_vis.display(port=8081)

