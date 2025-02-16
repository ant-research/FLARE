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

def load_from_json(filename):
    """Load a dictionary from a JSON filename.

    Args:
        filename: The filename to load from.
    """
    with open(filename, encoding="UTF-8") as file:
        return json.load(file)

class Re10K_ps(BaseStereoViewDataset):
    def __init__(self, *args, split, ROOT, only_pose=False, min_gap=15, max_gap=192, **kwargs):
        self.ROOT = ROOT.replace(oss_folder_path, pcache_folder_path)
        # Collect chunks.
        ROOT = Path(ROOT)
        self.chunks = []
        self.split = split
        self.min_gap = min_gap
        self.max_gap = max_gap
        # for root in ROOT:
        root =  ROOT / split
        self.root = root
        if os.path.exists(os.path.join(root, 'index.json')) and 're10k' not in str(root):
            # import ipdb; ipdb.set_trace()
            with open(os.path.join(root, 'index.json')) as f:
                self.chunks = json.load(f)
            self.json = True
        else:
            root_chunks = sorted(
                [path for path in root.iterdir() if path.suffix == ".torch"]
            )
            self.chunks = root_chunks
            self.json = False
        super().__init__(*args, **kwargs)
        self.rendering = True

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
        scene = random.choice(list(self.file_paths))
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
        # for i in range(len(self.meta)):
        interal = 0
        while interal == 0:
            meta = rng.choice(self.meta)
            video_reader = VideoReader(os.path.join(self.ROOT, meta['clip_path']))
            H, W = video_reader[0].shape[:2]
            video_length = len(video_reader)
            interal = video_length//(self.num_image)
        pose_path = osp.join(self.ROOT, meta['pose_file'])
        im_list = [i * interal + int(random.choice(list(np.linspace(-interal//2,interal//2,interal)))) for i in range(self.num_image)]
        im_list += [random.choice(im_list) + int(random.choice(list(np.linspace(-interal//2,interal//2,interal)))) for _ in range(self.gt_num_image)]
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
        return frames, w2c_mats, Ks, meta['clip_path']

    def _get_views(self, idx, resolution, rng):
        # image_idx1, image_idx2 = self.pairs[idx]
        # imgs_idxs = self._random_index(rng)
        scale = -1
        while scale < 1e-4 or scale > 1e10:
            if self.json ==False:
                chunk_path = rng.choice(self.chunks)
                chunk = torch.load(chunk_path, weights_only=False)
            else:
                chunk_path = rng.choice(list(self.chunks.values()))
                chunk_path = os.path.join(self.root, chunk_path)
                chunk = torch.load(chunk_path, weights_only=False)[0]
            image_num = len(chunk["cameras"])
            min_gap = self.min_gap
            max_gap = self.max_gap
            device = 'cpu'
            max_gap = min(max_gap, image_num - 1)
            min_gap = min(min_gap, max_gap)
            context_gap = torch.randint(
                min_gap,
                max_gap + 1,
                size=tuple(),
                device=device,
            ).item()
            index_context_left = torch.randint(
                image_num - context_gap,
                size=tuple(),
                device=device,
            ).item()
            index_context_right = index_context_left + context_gap
            index_target = torch.randint(
                    index_context_left,
                    index_context_right + 1,
                    size=(self.gt_num_image,),
                    device=device,
                )
            index_list = [index_context_right, index_context_left]
            poses_right = chunk["cameras"][index_context_right]
            w2c_right = np.eye(4)
            w2c_right[:3] = poses_right[6:].reshape(3, 4)
            camera_pose_right =  np.linalg.inv(w2c_right)
            poses_left = chunk["cameras"][index_context_left]
            w2c_left = np.eye(4)
            w2c_left[:3] = poses_left[6:].reshape(3, 4)
            camera_pose_left =  np.linalg.inv(w2c_left)
            random.shuffle(index_list)
            a, b = camera_pose_right[:3, 3], camera_pose_left[:3, 3]
            scale = np.linalg.norm(a - b)

        index_list.extend(index_target.tolist())
        views = []
        for index in index_list:
            poses = chunk["cameras"][index]
            intrinsics = np.eye(3)
            fx, fy, cx, cy = poses[:4]
            intrinsics[0, 0] = fx
            intrinsics[1, 1] = fy
            intrinsics[0, 2] = cx
            intrinsics[1, 2] = cy
            # Convert the extrinsics to a 4x4 OpenCV-style W2C matrix.
            w2c = np.eye(4)
            w2c[:3] = poses[6:].reshape(3, 4)
            camera_pose =  np.linalg.inv(w2c)
            camera_pose[:3, 3] = camera_pose[:3, 3] / scale
            scene = chunk["key"]
            frame = chunk["images"][index] 
            frame = Image.open(BytesIO(frame.numpy().tobytes())).convert('RGB')
            frame = np.asarray(frame)
            depthmap = np.zeros_like(frame)[..., 0]
            H, W = frame.shape[:2]
            intrinsics[0, 0] = intrinsics[0, 0] * W
            intrinsics[1, 1] = intrinsics[1, 1] * H
            intrinsics[0, 2] = intrinsics[0, 2] * W
            intrinsics[1, 2] = intrinsics[1, 2] * H
            rgb_image, depthmap, intrinsics = self._crop_resize_if_necessary(
                frame, depthmap, intrinsics, resolution, rng=rng, info=f'')
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
                label=scene,
                instance=scene,
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

