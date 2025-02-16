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
from dust3r.utils.image import imread_cv2, imread_cv2_orig
import os
import glob
import json
import imageio
oss_folder_path = 'oss://antsys-vilab/datasets/pcache_datasets/'
# oss_file_path = 'oss://antsys-vilab/datasets/pcache_datasets/SAM_1B/sa_000000/images/sa_1011.jpg'
pcache_folder_path = 'pcache://vilabpcacheproxyi-pool.cz50c.alipay.com:39999/mnt/antsys-vilab_datasets_pcache_datasets/'
# pcache_file_path = oss_file_path.replace(oss_folder_path,pcache_folder_path)

class MatrixCity(BaseStereoViewDataset):
    def __init__(self, *args, split, ROOT, meta,**kwargs):
        self.ROOT = ROOT.replace(oss_folder_path, pcache_folder_path)
        super().__init__(*args, **kwargs)
        self.meta = np.load(meta, allow_pickle=True).item()

    def __len__(self):
        return 684000

    def get_stats(self):
        return f'{len(self)} pairs from {len(self.all_scenes)} scenes'

    def _get_views(self, pair_idx, resolution, rng):
        scene_id = rng.choice(list(self.scenes.keys()))
        img_indx = self.scenes[scene_id]['images']
        self.num_image_input = self.num_image
        img_indx = rng.choice(img_indx, self.num_image_input + self.gt_num_image)
        views = []
        seq_path = osp.join(self.ROOT, scene_id)
        for im_id in img_indx:
            im_id = osp.join(*(im_id.split('/')[-3:]))
            img = osp.join(self.ROOT, im_id)
            image = imread_cv2(img)
            depthmap = imread_cv2(img.replace('.png', '.exr').replace('images', 'depths'))
            depthmap[np.isinf(depthmap)] = 0
            random_number = rng.integers(low=50, high=99)
            depthmap[depthmap > np.percentile(depthmap, random_number)] = 0
            camera_params = self.scenes[scene_id]['poses'][os.path.basename(img).split('.')[0]]
            c_x, c_y, f_x, f_y = np.float32(camera_params['c_x']), np.float32(camera_params['c_y']), np.float32(camera_params['f_x']), np.float32(camera_params['f_y'])
            intrinsics = np.float32([[f_x * 2 * 810 / 1920, 0, c_x], [0, f_y, c_y], [0, 0, 1]])
            camera_pose = np.linalg.inv(np.float32(camera_params['extrinsic']))
            image, depthmap, intrinsics = self._crop_resize_if_necessary(
                image, depthmap, intrinsics, resolution, rng, info=(seq_path, img))
            H, W = depthmap.shape[:2]
            fxfycxcy = np.array([intrinsics[0, 0]/W, intrinsics[1, 1]/H, intrinsics[0,2]/W, intrinsics[1,2]/H]).astype(np.float32)
            img_org = image.copy()

            views.append(dict(
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

    dataset = GTAV(split='train', ROOT="/nas7/vilab/zsz/mast3recon/data/matrix_city', resolution=[(512, 380)], aug_crop=16)
    for idx in np.random.permutation(len(dataset)):
        views = dataset[idx]
        print('+1')