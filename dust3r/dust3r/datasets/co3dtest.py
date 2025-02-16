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
import numpy as np
# import mast3r.utils.path_to_dust3r  # noqa
from dust3r.datasets.base.base_stereo_view_dataset import BaseStereoViewDataset
from dust3r.utils.image import imread_cv2
try:
    from pcache_fileio import fileio
    flag_pcache = True
except:
    flag_pcache = False
    
oss_folder_path = 'oss://antsys-vilab/datasets/pcache_datasets/'
pcache_folder_path = 'pcache://vilabpcacheproxyi-pool.cz50c.alipay.com:39999/mnt/antsys-vilab_datasets_pcache_datasets/'

def read_split_file(file_path):
        # 打开并读取JSON文件
    with open(file_path, 'r') as f:
        folder_dict = json.load(f)

    # 打印字典内容
    return folder_dict
    

class Co3dtest(BaseStereoViewDataset):
    def __init__(self, mask_bg=True, *args, ROOT, **kwargs):
        self.ROOT = ROOT
        super().__init__(*args, **kwargs)
        assert mask_bg in (True, False, 'rand')
        self.mask_bg = mask_bg
        self.dataset_label = 'Co3d_v2'
        self._load_data()
        self.i = 0

    def _load_data(self, ):
        pair_paths = read_split_file(self.ROOT)
        self.scenes = []
        for pair_path in pair_paths:
            # pairs = glob.glob(osp.join(pair_path, '*.jpg'))
            self.scenes += [pair_path]

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

    def _get_views(self, pair_idx, resolution, rng):
        impaths = self.scenes[self.i]
        print(self.i)
        self.i += 1
        views = []
        while len(impaths) > 0:  # some images (few) have zero depth
            impath = impaths.pop()
            impath = impath[0]
            # impath = self.scenes[scene][im_idx]
            # load camera params
            input_metadata = np.load(impath.replace('jpg', 'npz'))
            camera_pose = input_metadata['camera_pose']
            intrinsics = input_metadata['camera_intrinsics'].astype(np.float32)
            # load image and depth
            rgb_image = imread_cv2(impath)
            depthmap = np.zeros_like(rgb_image)[..., 0]
            rgb_image, depthmap, intrinsics = self._crop_resize_if_necessary(
                rgb_image, depthmap, intrinsics, resolution, rng=rng, info=impath)
            rgb_image_orig = rgb_image.copy()
            H, W = depthmap.shape[:2]
            fxfycxcy = np.array([intrinsics[0, 0]/W, intrinsics[1, 1]/H, intrinsics[0,2]/W, intrinsics[1,2]/H]).astype(np.float32)
            views.append(dict(
                img_org=rgb_image_orig,
                img=rgb_image,
                depthmap=depthmap,
                camera_pose=camera_pose,
                camera_intrinsics=intrinsics,
                fxfycxcy=fxfycxcy,
                dataset='co3d',
                label=impath,
                instance=osp.split(impath)[1],
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