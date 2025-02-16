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
from dust3r.datasets.base.base_stereo_view_dataset import BaseStereoViewDataset_test
from dust3r.utils.image import imread_cv2, imread_cv2_orig
from dust3r.utils.geometry import colmap_to_opencv_intrinsics#, opencv_to_colmap_intrinsics  # noqa

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

def read_camera_parameters(filename):
    with open(filename) as f:
        lines = f.readlines()
        lines = [line.rstrip() for line in lines]
    # extrinsics: line [1,5), 4x4 matrix
    extrinsics = np.fromstring(' '.join(lines[1:5]), dtype=np.float32, sep=' ').reshape((4, 4))
    # intrinsics: line [7-10), 3x3 matrix
    intrinsics = np.fromstring(' '.join(lines[7:10]), dtype=np.float32, sep=' ').reshape((3, 3))
    # TODO: assume the feature is 1/4 of the original image size

    # intrinsics[:2, :] *= scale

    # if (flag==0):
    #     intrinsics[0,2]-=index
    # else:
    #     intrinsics[1,2]-=index
  
    return intrinsics, extrinsics

class DL3DV_test_eval(BaseStereoViewDataset_test):
    def __init__(self, *args, split,meta, ROOT, only_pose=False, **kwargs):
        self.ROOT = ROOT 
        self.meta = meta
        self.npy_list = os.listdir(os.path.join(meta))
        # self.json_paths = os.listdir(ROOT)
        super().__init__(*args, **kwargs)
        self.global_idx = 0
    def __len__(self):
        return len(self.npy_list) 
    
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
    
    def sequential_sample(self, im_start, last, interal):
        im_list = [
            im_start + i * interal
            for i in range(self.num_image)
        ]
        im_list += [
            random.choice(im_list) + random.choice(list(range(-interal//2, interal//2)))
            for _ in range(self.gt_num_image)
        ]
        return im_list
        
    def _get_views(self, idx, resolution, rng):
        # image_idx1, image_idx2 = self.pairs[idx]
        idx = self.global_idx
        npy = self.npy_list[idx] #random.choice(self.npy_list)
        self.global_idx += 1
        # import ipdb; ipdb.set_trace()
        npy_path = os.path.join(self.meta, npy)
        # set(imgs_idxs.tolist())
        imgs_idxs = np.load(npy_path)        
        image_list = [image_idx[0] for image_idx in imgs_idxs]

        self.num_image_input = self.num_image
        # scene_name = os.path.basename(scene).split('.')[0]
        scene_name = image_list[0].split('/')[-3]
        image_path = osp.join(self.ROOT, scene_name,  'images')

        imgs_idxs = deque(image_list)
        views = []
        i = 1
        while len(imgs_idxs) > 0:  # some images (few) have zero depth
            frame = imgs_idxs.pop()
            frame_num = int((frame.split('/')[-1].split('.')[0]).replace('frame_', ''))
            rgb_image = imread_cv2(frame)
            pose_path = osp.join(self.ROOT, scene_name, 'cams', '{:08d}_cam.txt'.format(frame_num))
            intrinsics, camera_pose = read_camera_parameters(pose_path)
            camera_pose = np.linalg.inv(camera_pose)
            # try:
            depth_path = osp.join(self.ROOT, scene_name, 'filter', 'depth', '{:05d}.npy'.format(frame_num))
            depthmap = np.load(depth_path)
            mask_path = osp.join(self.ROOT, scene_name, 'filter', 'mask', '{:05d}_final.png'.format(frame_num))
            mask = imread_cv2(mask_path).sum(-1)
            # depthmap = np.zeros_like(rgb_image[..., 0])
            # mask = np.zeros_like(rgb_image[..., 0])
                
            mask = mask.astype(np.uint8)
            mask = cv2.erode(mask, np.ones((3, 3), np.uint8)) > 0

            depthmap = depthmap * mask
            intrinsics = colmap_to_opencv_intrinsics(intrinsics.astype(np.float32))
            # if os.path.exists(frame.replace('images', 'depth_anything_aligned').replace('.png', '.npy')):
            #     depth_anything = np.load(frame.replace('images', 'depth_anything_aligned').replace('.png', '.npy'))
            # else:
            depth_anything = np.zeros_like(depthmap)
            if i > self.num_image_input:
                resolution = (256, 256)
            i += 1
            rgb_image, depthmap, intrinsics, depth_anything = self._crop_resize_if_necessary(
                rgb_image, depthmap, intrinsics, resolution, rng=rng, info=None, depth_anything=depth_anything)
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
                dataset='DL3DV',
                label=self.meta,
                instance=scene_name+'/'+str(frame_num),
                depth_anything=depth_anything,
            ))
        return views


if __name__ == "__main__":
    from dust3r.datasets.base.base_stereo_view_dataset import view_name
    from dust3r.viz import SceneViz, auto_cam_size
    from dust3r.utils.image import rgb
    import nerfvis.scene as scene_vis 
    dataset = DL3DV(split='train', ROOT="/input1/datasets/DL3DV_dust3r", meta='/input1/zsz/DL3DV/json', resolution=[(512, 384)], aug_crop=16)

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
        mean = pts3ds.reshape(-1,3)[valid_masks.reshape(-1)].mean(0)
        scene_vis.add_points("points", pts3ds.reshape(-1,3)[valid_masks.reshape(-1)] - mean, vert_color=colors.reshape(-1,3)[valid_masks.reshape(-1)], point_size=1)
        # for i in range(len(c2ws)):
        f = 1111.0 
        z = 1.
        scene_vis.add_camera_frustum("cameras", r=c2ws[:, :3, :3], t=c2ws[:, :3, 3] - mean, focal_length=f,
                        image_width=colors.shape[2], image_height=colors.shape[1],
                        z=z, connect=False, color=[1.0, 0.0, 0.0])
        for i in range(len(c2ws)):
            scene_vis.add_image(
                            f"images/{i}",
                            colors[i], # Can be a list of paths too (requires joblib for that) 
                            r=c2ws[i, :3, :3],
                            t=c2ws[i, :3, 3] - mean,
                            # Alternatively: from nerfvis.utils import split_mat4; **split_mat4(c2ws)
                            focal_length=f,
                            z=z,
                        )
        scene_vis.export('vis', embed_output=True)
        break
