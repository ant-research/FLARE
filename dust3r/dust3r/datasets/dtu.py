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

class DTU(BaseStereoViewDataset_test):
    def __init__(self, *args, split, ROOT, num_views, only_pose=False, **kwargs):
        self.datapath = ROOT
        self.listfile = os.listdir(self.datapath)
        self.nviews = num_views
        self.meta = self.build_list()
        self.index = 0
        super().__init__(*args, **kwargs)

    def build_list(self):
        metas = []
        scans = self.listfile

        interval_scale_dict = {}
        # scans
        for scan in scans:
            pair_file = "{}/pair.txt".format(scan)
            # read the pair file
            with open(os.path.join(self.datapath, pair_file)) as f:
                num_viewpoint = int(f.readline())-1
                # viewpoints
                # for view_idx in range(num_viewpoint):
                #     ref_view = int(f.readline().rstrip())
                #     src_views = [int(x) for x in f.readline().rstrip().split()[1::2]]
                #     # filter by no src view and fill to nviews
                #     if len(src_views) > 0:
                #         if len(src_views) < self.nviews:
                #             print("{}< num_views:{}".format(len(src_views), self.nviews))
                #             src_views += [src_views[0]] * (self.nviews - len(src_views))
                interal = num_viewpoint // self.nviews
                ref_view = num_viewpoint
                # step = num_viewpoint // 7
                src_views = list(range(num_viewpoint))[::interal]
                metas.append((scan, ref_view, src_views, scan))
        return metas


    def __len__(self):
        return len(self.meta)
    
    @staticmethod
    def image_read(image_file):
        img = cv2.imread(image_file)
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


    def read_cam_file(self, filename):
        with open(filename) as f:
            lines = [line.rstrip() for line in f.readlines()]
        # extrinsics: line [1,5), 4x4 matrix
        extrinsics = np.fromstring(' '.join(lines[1:5]), dtype=np.float32, sep=' ')
        extrinsics = extrinsics.reshape((4, 4))
        # intrinsics: line [7-10), 3x3 matrix
        intrinsics = np.fromstring(' '.join(lines[7:10]), dtype=np.float32, sep=' ')
        intrinsics = intrinsics.reshape((3, 3))
        # depth_min & depth_interval: line 11
        depth_min = float(lines[11].split()[0])
        depth_interval = float(lines[11].split()[1])
        return intrinsics, extrinsics, depth_min, depth_interval

  
    def _get_views(self, idx, resolution, rng):
        meta = self.meta[self.index]
        self.index += 1
        scan, ref_view, src_views, scene_name = meta
        view_ids = [ref_view] + src_views[:self.nviews - 1]
        imgs = []
        depth_values = None
        proj_matrices = []
        views = []
        for i, vid in enumerate(view_ids):
            img_filename = os.path.join(self.datapath, '{}/images_post/{:0>8}.jpg'.format(scan, vid))
            if not os.path.exists(img_filename):
                img_filename = os.path.join(self.datapath, '{}/images/{:0>8}.jpg'.format(scan, vid))

            proj_mat_filename = os.path.join(self.datapath, '{}/cams/{:0>8}_cam.txt'.format(scan, vid))
            rgb_image = self.image_read(img_filename)
            depthmap = np.zeros_like(rgb_image)[..., 0]
            intrinsics, extrinsics, depth_min, depth_interval = self.read_cam_file(proj_mat_filename)
            # scale input
            camera_pose = np.linalg.inv(extrinsics)
            rgb_image, depthmap, intrinsics = self._crop_resize_if_necessary(
                rgb_image, depthmap, intrinsics, resolution, rng=rng, info=None)
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
                label=scene_name,
                instance=scene_name,
                vid=str(self.nviews)
            ))
            
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

