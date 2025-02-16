# Copyright (C) 2024-present Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).
#
# --------------------------------------------------------
# Dataloader for preprocessed scannet++
# dataset at https://github.com/scannetpp/scannetpp - non-commercial research and educational purposes
# https://kaldir.vc.in.tum.de/scannetpp/static/scannetpp-terms-of-use.pdf
# See datasets_preprocess/preprocess_scannetpp.py
# --------------------------------------------------------
import os.path as osp
import cv2
import numpy as np
import mast3r.utils.path_to_dust3r  # noqa
from dust3r.datasets.base.base_stereo_view_dataset import BaseStereoViewDataset
from dust3r.utils.image import imread_cv2
import glob
import random
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
    
class ScanNetpp(BaseStereoViewDataset):
    def __init__(self, *args, ROOT, meta,**kwargs):
        if flag_pcache:
            self.ROOT = ROOT.replace(oss_folder_path, pcache_folder_path)
        else:
            self.ROOT = ROOT
        self.meta = meta
        super().__init__(*args, **kwargs)
        assert self.split == 'train'
        self.loaded_data = self._load_data()

    def _load_data(self):
        self.meta = np.load(self.meta, allow_pickle=True)
        self.images = self.meta['images'].item()
        self.trajectories = self.meta['trajectories'].item()
        self.intrinsics = self.meta['intrinsics'].item()
        # file_paths = glob.glob(self.ROOT + '/*')
        # self.images = {}
        # self.trajectories = {}
        # self.intrinsics = {}
        # for file_path in file_paths:
        #     scene = osp.basename(file_path)
        #     self.images[scene] = {}
        #     self.trajectories[scene] = {}
        #     self.intrinsics[scene] = {}
        #     scene_metadata_dslr = osp.join(file_path, 'scene_metadata_dslr.npz')
        #     if osp.exists(scene_metadata_dslr) == False:
        #         continue
        #     else:
        #         scene_metadata_dslr = np.load(scene_metadata_dslr)
        #         self.images[scene]['dslr'] = scene_metadata_dslr['images']
        #         self.trajectories[scene]['dslr'] = scene_metadata_dslr['trajectories']
        #         self.intrinsics[scene]['dslr'] = scene_metadata_dslr['intrinsics']

        #     scene_metadata_iphone = osp.join(file_path, 'scene_metadata_iphone.npz')
        #     if osp.exists(scene_metadata_iphone) == False:
        #         continue
        #     else:
        #         scene_metadata_iphone = np.load(scene_metadata_iphone)
        #         self.images[scene]['iphone'] = scene_metadata_iphone['images']
        #         self.trajectories[scene]['iphone'] = scene_metadata_iphone['trajectories']
        #         self.intrinsics[scene]['iphone'] = scene_metadata_iphone['intrinsics']
        # np.savez(osp.join('/nas7/vilab/zsz/mast3recon/data', 'scannetpp_meta.npz'), images=self.images, trajectories=self.trajectories, intrinsics=self.intrinsics)
            
    def __len__(self):
        return 684000

    def _get_views(self, idx, resolution, rng):
        key = 'dslr'
        while key == 'dslr':
            scene_id = rng.choice(list(self.images.keys()))
            keys = self.images[scene_id].keys()
            key = rng.choice(list(keys))
        img_names = self.images[scene_id][key]
        if self.sequential_input == False:
            img_indx = rng.choice(range(len(img_names)), self.num_image + self.gt_num_image)
        else:
            last = len(img_names)-1
            interal = 6
            end = last - self.num_image * interal//2
            end = max(1, end)
            im_start = rng.choice(range(end))
            im_list = self.sequential_sample(im_start, last, interal)
            img_indx = [max(0, min(im_idx, last)) for im_idx in im_list]
        views = []

        views = []
        for im_id in img_indx:
            basename = img_names[im_id]
            intrinsics = self.intrinsics[scene_id][key][im_id]
            camera_pose = self.trajectories[scene_id][key][im_id]
            # Load RGB image
            filename =  basename.split('.')[0]
            rgb_image = imread_cv2(osp.join(self.ROOT, scene_id, 'images', filename + '.jpg'))
            # Load depthmap
            depthmap = imread_cv2(osp.join(self.ROOT, scene_id, 'depth', filename + '.png'), cv2.IMREAD_UNCHANGED)
            depthmap = depthmap.astype(np.float32) / 1000
            depthmap[~np.isfinite(depthmap)] = 0  # invalid
            rgb_image, depthmap, intrinsics = self._crop_resize_if_necessary(
                rgb_image, depthmap, intrinsics, resolution, rng=rng, info=im_id)
            H, W = depthmap.shape[:2]
            fxfycxcy = np.array([intrinsics[0, 0]/W, intrinsics[1, 1]/H, intrinsics[0,2]/W, intrinsics[1,2]/H]).astype(np.float32)
            img_org = rgb_image.copy()
            views.append(dict(
                fxfycxcy=fxfycxcy,
                img_org=img_org,
                img=rgb_image,
                depthmap=depthmap.astype(np.float32),
                depth_anything=depthmap.astype(np.float32),
                camera_pose=camera_pose.astype(np.float32),
                camera_intrinsics=intrinsics.astype(np.float32),
                dataset='ScanNet++',
                label=osp.join(self.ROOT, scene_id, 'images', filename + '.jpg'),
                instance=f'{str(idx)}_{str(im_id)}',
            ))
        return views


if __name__ == "__main__":
    from dust3r.datasets.base.base_stereo_view_dataset import view_name
    from dust3r.viz import SceneViz, auto_cam_size
    from dust3r.utils.image import rgb
    import nerfvis.scene as scene_vis 

    dataset = ScanNetpp(split='train', ROOT="/nas8/datasets/scannetpp/scannetpp_processed_v3/", meta=osp.join('/nas7/vilab/zsz/mast3recon/data', 'scannetpp_meta.npz'), resolution=[(512, 380)], aug_crop=16)

    for idx in np.random.permutation(len(dataset)):
        views = dataset[idx]
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
        pts_mean = pts3ds.reshape(-1,3)[valid_masks.reshape(-1)].mean(0)
        scene_vis.add_points("points", pts3ds.reshape(-1,3)[valid_masks.reshape(-1)]-pts_mean, vert_color=colors.reshape(-1,3)[valid_masks.reshape(-1)], point_size=1)
        # for i in range(len(c2ws)):
        scene_vis.add_camera_frustum(
            "gt_cameras",
            r=c2ws[:, :3, :3],
            t=c2ws[:, :3, 3]-pts_mean,
            focal_length=320,
            z=1,
            connect=False,
            image_width=640,
            image_height=480,
            color=[0.0, 1.0, 0.0],
        )
        scene_vis.display()