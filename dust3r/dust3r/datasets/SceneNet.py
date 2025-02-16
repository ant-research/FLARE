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
from PIL import Image
from dust3r.utils.geometry import geotrf, colmap_to_opencv_intrinsics, opencv_to_colmap_intrinsics
import math
# try:
#     # from pcache_fileio import fileio
#     import fsspec
#     PCACHE_HOST = "vilabpcacheproxyi-pool.cz50c.alipay.com"
#     PCACHE_PORT = 39999
#     pcache_kwargs = {"host": PCACHE_HOST, "port": PCACHE_PORT}
#     pcache_fs = fsspec.filesystem("pcache", pcache_kwargs=pcache_kwargs)
#     oss_folder_path = 'oss://antsys-vilab/datasets/pcache_datasets/'
#     pcache_folder_path = '/mnt/antsys-vilab_datasets_pcache_datasets/'
#     flag_pcache = True
# except:
#     flag_pcache = False
oss_folder_path = 'oss://antsys-vilab/datasets/pcache_datasets/'
pcache_folder_path = '/mnt/antsys-vilab_datasets_pcache_datasets/'

def load_depth_map_in_m(file_name):
    image = Image.open(file_name)
    pixel = np.array(image)
    return (pixel * 0.001)

def pixel_to_ray(pixel,vfov=45,hfov=60,pixel_width=320,pixel_height=240):
    x, y = pixel
    x_vect = math.tan(math.radians(hfov/2.0)) * ((2.0 * ((x+0.5)/pixel_width)) - 1.0)
    y_vect = math.tan(math.radians(vfov/2.0)) * ((2.0 * ((y+0.5)/pixel_height)) - 1.0)
    return (x_vect,y_vect,1.0)

def normalize(v):
    return v/np.linalg.norm(v)

def normalised_pixel_to_ray_array(width=320,height=240):
    pixel_to_ray_array = np.zeros((height,width,3))
    for y in range(height):
        for x in range(width):
            pixel_to_ray_array[y,x] = normalize(np.array(pixel_to_ray((x,y),pixel_height=height,pixel_width=width)))
    return pixel_to_ray_array

class SceneNet(BaseStereoViewDataset):
    def __init__(self, *args, split, meta, ROOT, **kwargs):
        self.ROOT = ROOT.replace(oss_folder_path, pcache_folder_path)
        super().__init__(*args, **kwargs)
        meta_path = os.path.join(meta)
        self.meta = np.load(meta_path, allow_pickle=True)
        self.meta = self.meta['arr_0']
        self.scenes =  self.meta.item()[split]
        self.split = split
        self.cached_pixel_to_ray_array = normalised_pixel_to_ray_array().astype(np.float32)


    def __len__(self):
        return 684000

    def get_stats(self):
        return f'{len(self)} pairs from {len(self.all_scenes)} scenes'
        
    def photo_path_from_view(self, render_path,view):
        photo_path = os.path.join(render_path,'photo')
        image_path = os.path.join(photo_path,'{0}.jpg'.format(view.frame_num))
        return os.path.join(self.ROOT,image_path)
    
    def instance_path_from_view(self, render_path,view):
        photo_path = os.path.join(render_path,'instance')
        image_path = os.path.join(photo_path,'{0}.png'.format(view.frame_num))
        return os.path.join(self.ROOT,image_path)
    
    def _get_views(self, pair_idx, resolution, rng):
        scene_id = rng.choice(list(self.scenes.keys()))
        img_indx = self.scenes[scene_id].keys()
        img_indx = list(img_indx)
        img_indx.remove('intrinsic_matrix')
        self.num_image_input = self.num_image
        img_indx = rng.choice(img_indx, self.num_image_input + self.gt_num_image)
        views = []
        seq_path = osp.join(self.ROOT, self.split, scene_id)
        for im_id in img_indx:
            img = osp.join(seq_path, 'photo', '{0}.jpg'.format(im_id))
            image = imread_cv2(img)
            depth = osp.join(seq_path, 'depth', '{0}.png'.format(im_id))
            depthmap = load_depth_map_in_m(depth).astype(np.float32)
            # import ipdb; ipdb.set_trace()
            depthmap = depthmap * self.cached_pixel_to_ray_array[:,:,2]
            depthmap[np.isinf(depthmap)] = 0 
            extrinsics = self.scenes[scene_id][im_id].astype(np.float32)
            intrinsics = self.scenes[scene_id]['intrinsic_matrix'].astype(np.float32)
            intrinsics = intrinsics[:,:3]
            # intrinsics = opencv_to_colmap_intrinsics(intrinsics)
            camera_pose = extrinsics
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
                depth_anything=depthmap,
                label=osp.relpath(seq_path, self.ROOT),
                instance=img))

        return views


if __name__ == "__main__":
    from dust3r.datasets.base.base_stereo_view_dataset import view_name
    from dust3r.viz import SceneViz, auto_cam_size
    from dust3r.utils.image import rgb
    import nerfvis.scene as scene_vis 

    dataset = SceneNet(split='train', ROOT="oss://antsys-vilab/datasets/pcache_datasets/scenergbd_net/scenergbd_net/", meta='/input_ssd/zsz/dust3r_dataset/SceneNet_meta.npz', resolution=[(512, 380)], aug_crop=16)
    for idx in np.random.permutation(len(dataset)):
        views = dataset[idx]
        view_idxs = list(range(len(views)))
        print('+1')
