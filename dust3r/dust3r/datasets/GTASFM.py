# Copyright (C) 2024-present Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).
#
# --------------------------------------------------------
# Dataloader for preprocessed MegaDepth
# dataset at https://www.cs.cornell.edu/projects/megadepth/
# See datasets_preprocess/preprocess_megadepth.py
# --------------------------------------------------------
import os.path as osp
import numpy as np
import mast3r.utils.path_to_dust3r  # noqa
from dust3r.datasets.base.base_stereo_view_dataset import BaseStereoViewDataset
from dust3r.utils.image import imread_cv2
import os
import glob
import json
import imageio
import pickle
import random
from PIL import Image
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

# def read_pkl_array(filename):
#         with open(filename, 'rb') as f:
#             array = pickle.load(f)
#         return array 

class GTASFM(BaseStereoViewDataset):
    def __init__(self, *args, split, ROOT,  **kwargs):
        self.ROOT = ROOT.replace(oss_folder_path, pcache_folder_path)
        super().__init__(*args, **kwargs)
        json_path = os.path.join(self.ROOT, 'scenes.json')
        # with open(json_path, 'r') as f:
        with pcache_fs.open(json_path, 'rb') as f:
            self.scenes = json.load(f)

    def __len__(self):
        return 684000

    def get_stats(self):
        return f'{len(self)} pairs from {len(list(self.scenes.keys()))} scenes'

    def read_cam(self, pose_filename, K_filename):
        extrinsics = self.read_pkl_array(pose_filename)
        intrinsics = self.read_pkl_array(K_filename)
        return intrinsics, extrinsics


    def read_depth(self, filename):
        return self.read_pkl_array(filename)


    def read_pkl_array(self, filename):
        # with open(filename, 'rb') as f:
        with pcache_fs.open(filename, 'rb') as f:
            array = pickle.load(f)
        return array 

    def _get_views(self, pair_idx, resolution, rng):
        scene_id = rng.choice(list(self.scenes.keys()))
        img_indx = self.scenes[scene_id]['images']
        if self.sequential_input == False:
            img_indx = rng.choice(img_indx, self.num_image + self.gt_num_image)
        else:
            last = len(img_indx)-1
            interal = 12
            end = last - self.num_image * interal//2
            end = max(1, end)
            im_start = rng.choice(range(end))
            im_list = self.sequential_sample(im_start, last, interal)
            im_list = [max(0, min(im_idx, last)) for im_idx in im_list]
            img_indx = [img_indx[im_idx] for im_idx in im_list]

        views = []
        seq_path = osp.join(self.ROOT, scene_id)
        for im_id in img_indx:
            im_id = osp.join(*(im_id.split('/')[-8:]))
            img = osp.join(self.ROOT, im_id)
            image = imread_cv2(img)

            depthmap = self.read_pkl_array(img.replace('.png', '.pkl').replace('images', 'depth'))
            depthmap[np.isinf(depthmap)] = 0

            pose_filename = img.replace('.png', '.pkl').replace('images', 'pose')
            K_filename = img.replace('.png', '.pkl').replace('images', 'K')
            extrinsics = self.read_pkl_array(pose_filename).astype(np.float32)
            intrinsics = self.read_pkl_array(K_filename).astype(np.float32)
            camera_pose = extrinsics
            image, depthmap, intrinsics = self._crop_resize_if_necessary(
                image, depthmap, intrinsics, resolution, rng, info=(seq_path, img))
            sky_mask = depthmap > 10000-1
            depthmap[depthmap > 10000-1] = 0
            if rng.integers(low=0, high=100) > 50:
                sky_mask = sky_mask | (depthmap > depthmap.mean())
                depthmap[sky_mask] = 0
            else:
                random_number = rng.integers(low=60, high=90)
                sky_mask = sky_mask | (depthmap > np.percentile(depthmap, random_number))
                depthmap[sky_mask] = 0

            H, W = depthmap.shape[:2]
            fxfycxcy = np.array([intrinsics[0, 0]/W, intrinsics[1, 1]/H, intrinsics[0,2]/W, intrinsics[1,2]/H]).astype(np.float32)
            img_org = image.copy()

            views.append(dict(
                sky_mask=sky_mask,
                img_org=img_org,
                fxfycxcy=fxfycxcy,
                img=image,
                depthmap=depthmap,
                camera_pose=camera_pose,  # cam2world
                camera_intrinsics=intrinsics,
                dataset='GTAV',
                label=osp.relpath(seq_path, self.ROOT),
                instance=img))

        return views


if __name__ == "__main__":
    from dust3r.datasets.base.base_stereo_view_dataset import view_name
    from dust3r.viz import SceneViz, auto_cam_size
    from dust3r.utils.image import rgb
    import nerfvis.scene as scene_vis 
    dataset = GTASFM(split='train', ROOT='oss://antsys-vilab/datasets/pcache_datasets/gta_sfm_clean/gta_sfm_clean/', resolution=[(512, 384)], aug_crop='auto',  aug_monocular=0.005,  num_views=8, gt_num_image=0)
    for idx in np.random.permutation(len(dataset)):
        views = dataset[idx]
        print('+1')