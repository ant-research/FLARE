# Copyright (C) 2024-present Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).
#
# --------------------------------------------------------
# Dataloader for preprocessed Co3d_v2
# dataset at https://github.com/facebookresearch/co3d - Creative Commons Attribution-NonCommercial 4.0 International
# See datasets_preprocess/preprocess_co3d.py
# --------------------------------------------------------
import os.path as osp
import json
import itertools
from collections import deque
import random
import cv2
import os
import numpy as np
# import mast3r.utils.path_to_dust3r  # noqa
from dust3r.datasets.base.base_stereo_view_dataset import BaseStereoViewDataset
from dust3r.utils.image import imread_cv2
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
class Co3d(BaseStereoViewDataset):
    def __init__(self, mask_bg=True, *args, ROOT, sequential_input=False, **kwargs):
        self.ROOT = ROOT
        self.sequential_input = sequential_input
        super().__init__(*args, **kwargs)
        assert mask_bg in (True, False, 'rand')
        self.mask_bg = mask_bg
        self.dataset_label = 'Co3d_v2'
        # load all scenes
        if 'pcache' in self.ROOT:
            self.ROOT = self.ROOT.replace(oss_folder_path, pcache_folder_path)
            with pcache_fs.open(osp.join(self.ROOT, f'selected_seqs_{self.split}.json'), 'rb') as f:
                self.scenes = json.load(f)
                self.scenes = {k: v for k, v in self.scenes.items() if len(v) > 0}
                self.scenes = {(k, k2): v2 for k, v in self.scenes.items()
                            for k2, v2 in v.items()}
        else:
            with open(osp.join(self.ROOT, f'selected_seqs_{self.split}.json'), 'r') as f:
                self.scenes = json.load(f)
                self.scenes = {k: v for k, v in self.scenes.items() if len(v) > 0}
                self.scenes = {(k, k2): v2 for k, v in self.scenes.items()
                            for k2, v2 in v.items()}
        self.scene_list = list(self.scenes.keys())
        self.combinations = [(i, j)
                             for i, j in itertools.combinations(range(100), 2)
                             if 0 < abs(i - j) <= 30 and abs(i - j) % 5 == 0]

        self.invalidate = {scene: {} for scene in self.scene_list}

    def __len__(self):
        return 684000

    def _get_metadatapath(self, obj, instance, view_idx):
        return osp.join(self.ROOT, obj, instance, 'images', f'frame{view_idx:06n}.npz')

    def _get_impath(self, obj, instance, view_idx):
        return osp.join(self.ROOT, obj, instance, 'images', f'frame{view_idx:06n}.jpg')

    def _get_depthpath(self, obj, instance, view_idx):
        return osp.join(self.ROOT, obj, instance, 'depths', f'frame{view_idx:06n}.jpg.geometric.png')

    def _get_maskpath(self, obj, instance, view_idx):
        return osp.join(self.ROOT, obj, instance, 'masks', f'frame{view_idx:06n}.png')

    def _read_depthmap(self, depthpath, input_metadata):
        depthmap = imread_cv2(depthpath, cv2.IMREAD_UNCHANGED)
        depthmap = (depthmap.astype(np.float32) / 65535) * np.nan_to_num(input_metadata['maximum_depth'])
        return depthmap


    def _get_views(self, idx, resolution, rng):
        # choose a scene
        obj, instance = rng.choice(self.scene_list)
        image_pool = self.scenes[obj, instance]
        # im1_idx, im2_idx = self.combinations[idx % len(self.combinations)]
        last = len(image_pool)-1
        interal = 16
        end = last - self.num_image*interal//2
        end = max(1, end)
        im_start = random.choice(range(end))
        if self.sequential_input:
            im_list = self.sequential_sample(im_start, last, interal)
        else:
            im_list = rng.choice(range(last + 1), self.num_image + self.gt_num_image)
        # add a bit of randomness
        if resolution not in self.invalidate[obj, instance]:  # flag invalid images
            self.invalidate[obj, instance][resolution] = [False for _ in range(len(image_pool))]

        # decide now if we mask the bg  
        mask_bg = (self.mask_bg == True) or (self.mask_bg == 'rand' and rng.choice(2))

        views = []
        imgs_idxs = [max(0, min(im_idx, last)) for im_idx in im_list]
        imgs_idxs = deque(imgs_idxs)
        while len(imgs_idxs) > 0:  # some images (few) have zero depth
            im_idx = imgs_idxs.pop()
            if self.invalidate[obj, instance][resolution][im_idx]:
                # search for a valid image
                random_direction = 2 * rng.choice(2) - 1
                for offset in range(1, len(image_pool)):
                    tentative_im_idx = (im_idx + (random_direction * offset)) % len(image_pool)
                    if not self.invalidate[obj, instance][resolution][tentative_im_idx]:
                        im_idx = tentative_im_idx
                        break

            view_idx = image_pool[im_idx]

            impath = self._get_impath(obj, instance, view_idx)
            dapath = impath.replace('images', 'depth_anything_aligned')+ '.npy'

            depthpath = self._get_depthpath(obj, instance, view_idx)

            # load camera params
            metadata_path = self._get_metadatapath(obj, instance, view_idx)
            input_metadata = np.load(metadata_path)
            camera_pose = input_metadata['camera_pose'].astype(np.float32)
            intrinsics = input_metadata['camera_intrinsics'].astype(np.float32)

            # load image and depth
            rgb_image = imread_cv2(impath)
            depthmap = self._read_depthmap(depthpath, input_metadata)
            # if os.path.exists(dapath):
            #     depth_anything = np.load(dapath)
            # else:
            depth_anything = np.zeros_like(depthmap)

            if mask_bg:
                # load object mask
                maskpath = self._get_maskpath(obj, instance, view_idx)
                maskmap = imread_cv2(maskpath, cv2.IMREAD_UNCHANGED).astype(np.float32)
                maskmap = (maskmap / 255.0) > 0.1

                # update the depthmap with mask
                depthmap *= maskmap

            rgb_image, depthmap, intrinsics, depth_anything= self._crop_resize_if_necessary(
                rgb_image, depthmap, intrinsics, resolution, rng=rng, info=impath, depth_anything=depth_anything)
            img_org = rgb_image.copy()
            num_valid = (depthmap > 0.0).sum()
            if num_valid == 0:
                # problem, invalidate image and retry
                self.invalidate[obj, instance][resolution][im_idx] = True
                imgs_idxs.append(im_idx)
                continue
            H, W = depthmap.shape[:2]
            fxfycxcy = np.array([intrinsics[0, 0]/W, intrinsics[1, 1]/H, intrinsics[0,2]/W, intrinsics[1,2]/H]).astype(np.float32)
            views.append(dict(
                fxfycxcy=fxfycxcy,
                img_org=img_org,
                img=rgb_image,
                depthmap=depthmap,
                camera_pose=camera_pose,
                camera_intrinsics=intrinsics,
                dataset=self.dataset_label,
                label=impath,
                instance=osp.split(impath)[1],
                depth_anything=depth_anything
            ))
        return views


if __name__ == "__main__":
    from dust3r.datasets.base.base_stereo_view_dataset import view_name
    from dust3r.viz import SceneViz, auto_cam_size
    from dust3r.utils.image import rgb
    import nerfvis.scene as scene_vis 

    dataset = Co3d(split='train', ROOT="/nas7/vilab/zsz/dust3r/data/co3d_subset_processed_full", resolution=[(512, 384)], aug_crop=16)

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