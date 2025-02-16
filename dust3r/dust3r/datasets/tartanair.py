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
try:
    # from pcache_fileio import fileio
    import fsspec
    PCACHE_HOST = "vilabpcacheproxyi-pool.cz50c.alipay.com"
    PCACHE_PORT = 39999
    pcache_kwargs = {"host": PCACHE_HOST, "port": PCACHE_PORT}
    pcache_fs = fsspec.filesystem("pcache", pcache_kwargs=pcache_kwargs)
    oss_folder_path = 'oss://antsys-vilab/datasets/pcache_datasets/'
    pcache_folder_path = '/mnt/antsys-vilab_datasets_pcache_datasets/'
    flag_pcache = True
except:
    flag_pcache = False
    
class TartanAir(BaseStereoViewDataset):
    def __init__(self, *args, split, ROOT, meta, only_pose=False, **kwargs):
        self.ROOT = ROOT.replace(oss_folder_path, pcache_folder_path)
        super().__init__(*args, **kwargs)
        if split == "train":
            self.split = "Training"
        elif split == "test":
            self.split = "Test"
        else:
            raise ValueError("")
        self.only_pose = only_pose
        self.meta = np.load(meta, allow_pickle=True)
        self.loaded_data = self._load_data(self.split)

    def _load_data(self, split):
        self.images_left = self.meta['images_left'].item()
        self.depths_left = self.meta['depths_left'].item()
        self.images_right = self.meta['images_right'].item()
        self.depths_right = self.meta['depths_right'].item()
        self.camera_left = self.meta['camera_left'].item()
        self.camera_right = self.meta['camera_right'].item()
    def __len__(self):
        return 684000
    
    @staticmethod
    def image_read(image_file):
        img = cv2.imread(image_file)
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    @staticmethod
    def depth_read(depth_file):
        with pcache_fs.open(depth_file, 'rb') as f:
            depth = np.load(f)
        depth_min = np.percentile(depth, 5)
        depth[depth==np.nan] = 0
        depth[depth==np.inf] = 0
        depth_max =  np.percentile(depth, 95)  
        depth[depth>=depth_max] = 0
        depth[depth>=depth_min+150] = 0
        return depth
    
    def _get_views(self, idx, resolution, rng):
        # image_idx1, image_idx2 = self.pairs[idx]
        scene = random.choice(list(self.images_left.keys()))
        # imgs_idxs = random.choice(subgraphs)
        image_num = len(self.images_left[scene])

        if self.sequential_input == False:
            imgs_idxs = rng.choice(range(image_num), self.num_image + self.gt_num_image)
        else:
            last = image_num - 1
            interal = 8
            end = last - self.num_image * interal//2
            end = max(1, end)
            im_start = rng.choice(range(end))
            im_list = self.sequential_sample(im_start, last, interal)
            imgs_idxs = [max(0, min(im_idx, last)) for im_idx in im_list]

        if len(imgs_idxs) < self.num_image+self.gt_num_image:
            imgs_idxs = random.choices(imgs_idxs, k=self.num_image+self.gt_num_image)
        else:
            imgs_idxs = imgs_idxs[:self.num_image+self.gt_num_image]
        imgs_idxs = deque(imgs_idxs)
        views = []
        view_flags = random.choices([0,1], k=self.num_image+self.gt_num_image)
        view_flags = deque(view_flags)
        while len(imgs_idxs) > 0:  # some images (few) have zero depth
            view_idx = imgs_idxs.pop()
            view_flag = view_flags.pop()
            K = np.array([[320., 0, 320.], [0, 320., 240.], [0, 0, 1]])
            intrinsics = K
            # if self.only_pose:
            #     if view_flag > 0.5:
            #         basename = self.images_left[scene][view_idx]
            #         camera_pose = np.load(osp.join(self.ROOT, scene, 'image_left', basename).replace('png', 'npz'))['camera_pose']
            #     else:
            #         basename = self.images_right[scene][view_idx]
            #         camera_pose = np.load(osp.join(self.ROOT, scene, 'image_right', basename).replace('png', 'npz'))['camera_pose']
            #     views.append(dict(
            #         camera_pose=camera_pose.astype(np.float32),
            #         camera_intrinsics=intrinsics.astype(np.float32),
            #         dataset='tartantair',
            #         label=osp.join(self.ROOT, scene, 'image_left', basename),
            #         instance=f'{str(idx)}_{str(view_idx)}',
            #     ))
            # else:
            if view_flag > 0.5:
                basename = self.images_left[scene][view_idx]
                rgb_image = imread_cv2(osp.join(self.ROOT, *scene.split('/')[-3:], 'image_left', basename))
                camera_pose = self.camera_left[scene][basename] #p.load(osp.join(self.ROOT, scene.split('/')[-3:], 'image_left', basename).replace('png', 'npz'))['camera_pose']
                basename = self.depths_left[scene][view_idx]
                depth = osp.join(self.ROOT, *scene.split('/')[-3:], 'depth_left', basename)
                depthmap = self.depth_read(depth)
            else:
                basename = self.images_right[scene][view_idx]
                rgb_image = imread_cv2(osp.join(self.ROOT, *scene.split('/')[-3:], 'image_right', basename))
                camera_pose = self.camera_right[scene][basename] #p.load(osp.join(self.ROOT, scene.split('/')[-3:], 'image_left', basename).replace('png', 'npz'))['camera_pose']
                basename = self.depths_right[scene][view_idx]
                depth = osp.join(self.ROOT, *scene.split('/')[-3:], 'depth_right', basename)
                depthmap = self.depth_read(depth)

            if rng.random() > 0.5:
                random_number = rng.integers(40, 99)
                depthmap[depthmap > np.percentile(depthmap, random_number)] = 0
            # camera_pose = np.linalg.inv(camera_pose)
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
                label=osp.join(self.ROOT, scene, 'image_left', basename),
                instance=f'{str(idx)}_{str(view_idx)}',
            ))
        
        return views


if __name__ == "__main__":
    from dust3r.datasets.base.base_stereo_view_dataset import view_name
    from dust3r.viz import SceneViz, auto_cam_size
    from dust3r.utils.image import rgb
    import nerfvis.scene as scene_vis 
    dataset = TartanAir(split='train', ROOT="/nas7/datasets/tartanair/data_unzip",meta='/nas7/datasets/tartanair/tartanair_metadata.npz',resolution=[(512, 384)], aug_crop=16)

    for idx in np.random.permutation(len(dataset)):
        views = dataset[idx]
        # assert len(views) == 2
        # print(view_name(views[0]), view_name(views[1]))
        viz = SceneViz()
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
            # viz.add_camera(pose_c2w=views[view_idx]['camera_pose'],
            #                focal=views[view_idx]['camera_intrinsics'][0, 0],
            #                color=(idx * 255, (1 - idx) * 255, 0),
            #                image=colors,
            #                cam_size=cam_size)
        
        pts3ds = np.stack(pts3ds, axis=0)
        colors = np.stack(colors, axis=0)
        valid_masks = np.stack(valid_masks, axis=0)
        c2ws = np.stack(c2ws)
        scene_vis.set_title("My Scene")
        scene_vis.set_opencv() 
        # colors = torch.zeros_like(structure).to(structure)
        scene_vis.add_points("points", pts3ds.reshape(-1,3)[valid_masks.reshape(-1)], vert_color=colors.reshape(-1,3)[valid_masks.reshape(-1)], point_size=1)
        # for i in range(len(c2ws)):
        scene_vis.add_camera_frustum(
            "gt_cameras",
            r=c2ws[:, :3, :3],
            t=c2ws[:, :3, 3],
            focal_length=320,
            z=1,
            connect=False,
            image_width=640,
            image_height=480,
            color=[0.0, 1.0, 0.0],
        )
        scene_vis.display()

