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
from decord import VideoReader

def load_from_json(filename):
    """Load a dictionary from a JSON filename.

    Args:
        filename: The filename to load from.
    """
    with open(filename, encoding="UTF-8") as file:
        return json.load(file)

class ACID(BaseStereoViewDataset):
    def __init__(self, *args, split, ROOT, meta, only_pose=False, **kwargs):
        self.ROOT = ROOT.replace(oss_folder_path, pcache_folder_path)
        self.meta_path = meta
        self.meta = np.load(meta, allow_pickle=True).item()['train']
        self.file_paths = list(self.meta.keys())
        super().__init__(*args, **kwargs)
        self.rendering = True
        # self._load_data()


    # def _load_data(self):
    #     # if os.path.exists(self.meta_path):
    #     # import ipdb; ipdb.set_trace()
    #     for i in len(self.meta):
    #         video_reader = VideoReader(os.path.join(self.ROOT, self.meta[0]['clip_path']))
    #         video_length = len(video_reader)
    #         pose_path = osp.join(self.ROOT, 'pose_files')
    #         with open(pose_path, 'r') as f:
    #             poses = f.readlines()
    #         poses = [pose.strip().split(' ') for pose in poses[1:]]
    #         camera_params = [[float(x) for x in pose] for pose in poses]
    #         cameras = []
    #         for cam_param in camera_params:
    #             fx, fy, cx, cy = cam_param[1:5]
    #             w2c_mat = np.array(cam_param[7:]).reshape(3, 4).astype(np.float32)
    #             K = np.asarray([[fx, 0., cx], [0., fy, cy], [0., 0., 1.]], dtype=np.float32)
    #             cameras.append((w2c_mat, K))

    def __len__(self):
        return 684000
    
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

    def _random_index(self, rng):
        scene = random.choice(list(self.meta.keys()))
        self.num_image_input = self.num_image
        meta =  self.images_path[scene]
        frames = self.images_path[scene]['frames']
        image_num = len(frames)
        end = image_num-self.num_image_input*32
        end = max(1, end)
        im_start = random.choice(range(end))
        # # add a bit of randomness
        last = image_num - 1
        im_list = [im_start + i * 32 + int(random.choice(list(np.linspace(-16,16,33)))) for i in range(self.num_image_input)]
        im_end = min(im_start + self.num_image_input * 32, last)
        im_list += [random.choice(im_list) + int(random.choice(list(np.linspace(-16,16,33)))) for _ in range(self.gt_num_image)]
        imgs_idxs = [max(0, min(im_idx, im_end)) for im_idx in im_list]
        if len(imgs_idxs) < self.num_image_input+self.gt_num_image:
            imgs_idxs = random.choices(imgs_idxs, k=self.num_image_input+self.gt_num_image)
        else:
            imgs_idxs = imgs_idxs[:self.num_image_input+self.gt_num_image]
        random.shuffle(imgs_idxs)

        # imgs_idxs = range(0, (self.gt_num_image + self.num_image_input) * 8, 8)
        imgs_idxs = deque(imgs_idxs)
        return imgs_idxs, meta, frames, scene
    
    def _sequential_index(self, rng):
        if self.overfit == True:
            scene = random.choice(list(self.file_paths)[:30])
        else:
            scene = random.choice(list(self.file_paths))
        self.num_image_input = self.num_image
        meta =  self.meta[scene]
        frames = meta['image']
        image_num = len(frames)
        end = image_num-self.num_image_input*32
        end = max(1, end)
        im_start = random.choice(range(end))
        # # add a bit of randomness
        last = image_num - 1
        im_list = [im_start + i * 8 for i in range(self.num_image_input)]
        im_max = max(im_list)
        im_end = min(im_start + self.num_image_input * 16, last)
        im_max = min(im_max, im_end)
        im_list += [random.choice(im_list) + int(random.choice(list(np.linspace(-8,8,16)))) for _ in range(self.gt_num_image)]
        imgs_idxs = [max(0, min(im_idx, im_max)) for im_idx in im_list]
        # imgs_idxs = range(0, (self.gt_num_image + self.num_image_input) * 8, 8)
        imgs_idxs = deque(imgs_idxs)
        return imgs_idxs, meta, frames, scene

    def _get_views(self, idx, resolution, rng):
        # image_idx1, image_idx2 = self.pairs[idx]
        # imgs_idxs = self._random_index(rng)
        imgs_idxs, meta, frames, scene = self._sequential_index(rng)
        # The naming is somewhat confusing, but:
        # - transform_matrix contains the transformation to dataparser output coordinates from saved coordinates.
        # - dataparser_transform_matrix contains the transformation to dataparser output coordinates from original data coordinates.
        # - applied_transform contains the transformation to saved coordinates from original data coordinates.
        views = []
        while len(imgs_idxs) > 0:  # some images (few) have zero depth
            view_idx = imgs_idxs.pop()
            frame = frames[view_idx]
            image_file = frame
            image_file = osp.join(self.ROOT,  'train', scene, image_file)
            rgb_image = imread_cv2(image_file)
            camera_pose = np.array(meta['extrinsics'])[view_idx]
            H, W = rgb_image.shape[:2]
            intrinsics = meta['intrinsics'][view_idx]
            intrinsics[0, 0] = intrinsics[0, 0] * W
            intrinsics[1, 1] = intrinsics[1, 1] * H
            intrinsics[0, 2] = intrinsics[0, 2] * W
            intrinsics[1, 2] = intrinsics[1, 2] * H
            depthmap = np.zeros_like(rgb_image)[...,0]
            rgb_image, depthmap, intrinsics = self._crop_resize_if_necessary(
                rgb_image, depthmap, intrinsics, resolution, rng=rng, info=f'{str(idx)}_{str(view_idx)}')
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
                dataset='tartantair',
                label=image_file,
                instance=image_file.split('/')[-3],
            ))
        
        return views


if __name__ == "__main__":
    from dust3r.datasets.base.base_stereo_view_dataset import view_name
    from dust3r.viz import SceneViz, auto_cam_size
    from dust3r.utils.image import rgb
    import nerfvis.scene as scene_vis 
    dataset = ACID(split='train', ROOT="/data1/zsz/acid/acid",meta='/data0/zsz/mast3recon/data/acid_meta.npy',resolution=[(512, 384)], aug_crop=16)

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
        # scene_vis.add_points("points", pts3ds.reshape(-1,3)[valid_masks.reshape(-1)], vert_color=colors.reshape(-1,3)[valid_masks.reshape(-1)], point_size=1)
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

