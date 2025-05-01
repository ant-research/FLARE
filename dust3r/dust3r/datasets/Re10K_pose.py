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
txt_path = '/data0/zsz/mast3recon/data/re10k_test_1800.txt'
train_path = '/nas3/zsz/RealEstate10k_v2/train_captions.json'
test_path = '/nas3/zsz/RealEstate10k_v2/test_captions.json'

def load_from_json(filename):
    """Load a dictionary from a JSON filename.

    Args:
        filename: The filename to load from.
    """
    with open(filename, encoding="UTF-8") as file:
        return json.load(file)

class Re10K_pose(BaseStereoViewDataset_test):
    def __init__(self, *args, split, ROOT, only_pose=False, **kwargs):
        self.ROOT = ROOT.replace(oss_folder_path, pcache_folder_path)
        # self.meta_path = meta
        self.test_json = load_from_json(test_path)
        self.train_json = load_from_json(train_path)
        clips = {}
        for key in self.train_json.keys():
            path = os.path.basename(key).split('.')[0]
            sub_file = key.split('/')[-2]
            clips[path] = os.path.join(sub_file, path)
        for key in self.test_json.keys():
            path = os.path.basename(key).split('.')[0]
            sub_file = key.split('/')[-2]
            clips[path] = os.path.join(sub_file, path)
        self.list_path = list(np.loadtxt(txt_path, dtype=str))
        self.list_path = [path for path in self.list_path if path in clips.keys()]
        self.clips = clips
        super().__init__(*args, **kwargs)
        self.rendering = True
        self.global_index = 0
        
    def __len__(self):
        return len(self.list_path)
    
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

    
    
    def _sequential_index(self, scene, video_path, rng):
        # for i in range(len(self.meta)):
        interal = 0
        num_view = self.num_image
        while interal == 0:
            video_reader = VideoReader(os.path.join(self.ROOT, 'video_clips', video_path))
            H, W = video_reader[0].shape[:2]
            video_length = len(video_reader)
            interal = video_length//num_view
        pose_path = osp.join(self.ROOT, 'pose_files', scene + '.txt')
        im_list = rng.choice(range(0, video_length), self.num_image + self.gt_num_image, replace=False)
        # im_list = [i * interal for i in range(num_view)]
        # im_list += [random.choice(im_list) + int(random.choice(list(np.linspace(-interal//2,interal//2,interal)))) for _ in range(self.num_image-5)]
        im_list = [max(0, min(im_idx, video_length-1)) for im_idx in im_list]
        with open(pose_path, 'r') as f:
            poses = f.readlines()
        poses = [pose.strip().split(' ') for pose in poses[1:]]
        camera_params = [[float(x) for x in pose] for pose in poses]
        w2c_mats = []
        Ks = []
        frames = video_reader.get_batch(im_list).asnumpy()
        for cam_param in camera_params:
            fx, fy, cx, cy = cam_param[1:5]
            w2c_mat = np.array(cam_param[7:]).reshape(3, 4).astype(np.float32)
            K = np.asarray([[fx, 0., cx], [0., fy, cy], [0., 0., 1.]], dtype=np.float32)
            K[0, 0] = K[0, 0] * W
            K[1, 1] = K[1, 1] * H
            K[0, 2] = K[0, 2] * W
            K[1, 2] = K[1, 2] * H
            w2c_mats.append(w2c_mat)
            Ks.append(K)
        w2c_mats = np.stack(w2c_mats)
        Ks = np.stack(Ks)
        w2c_mats = w2c_mats[im_list]
        Ks = Ks[im_list]
        return frames, w2c_mats, Ks, os.path.join(self.ROOT, 'video_clips', video_path)

    def _get_views(self, idx, resolution, rng):
        idx = self.global_index
        flag = False
        while flag==False:
            scene = self.list_path[self.global_index % len(self.list_path)]
            self.global_index += 1
            video_path = self.clips[scene] + '.mp4'
            try:
                frames, w2c_mats, Ks, clip_path = self._sequential_index(scene, video_path, rng)
                flag=True
            except:
                flag = False
            
        views = []
        num = len(frames)
        for frame, w2c_mat, K in zip(frames, w2c_mats, Ks):
            depthmap = np.zeros_like(frame)[..., 0]
            bottom_ = np.array([[0,0,0,1]])
            w2c_mat = np.concatenate([w2c_mat, bottom_], axis=0)
            camera_pose = np.linalg.inv(w2c_mat)
            rgb_image, depthmap, intrinsics = self._crop_resize_if_necessary(
                frame, depthmap, K, resolution, rng=rng, info=f'')
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
                dataset='re10k',
                label=clip_path,
                instance=clip_path,
                num = num
            ))
        
        return views


if __name__ == "__main__":
    from dust3r.datasets.base.base_stereo_view_dataset import view_name
    from dust3r.viz import SceneViz, auto_cam_size
    from dust3r.utils.image import rgb
    import nerfvis.scene as scene_vis 
    dataset = Re10K_pose(split='train', ROOT="/nas3/zsz/RealEstate10k_v2/",resolution=[(512, 384)], aug_crop=16)
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

