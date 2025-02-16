# Copyright (C) 2024-present Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).
#
# --------------------------------------------------------
# Dataloader for preprocessed WayMo
# dataset at https://github.com/waymo-research/waymo-open-dataset
# See datasets_preprocess/preprocess_waymo.py
# --------------------------------------------------------
import os.path as osp
import numpy as np
import mast3r.utils.path_to_dust3r  # noqa

from dust3r.datasets.base.base_stereo_view_dataset import BaseStereoViewDataset
from dust3r.utils.image import imread_cv2
import glob
import tqdm 
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
    oss_folder_path = ''
    pcache_folder_path = ''
    flag_pcache = False
    
class Hypersim (BaseStereoViewDataset):
    """ Dataset of outdoor street scenes, 5 images each time
    """

    def __init__(self, *args, ROOT, meta, **kwargs):
        self.ROOT = ROOT.replace(oss_folder_path, pcache_folder_path)
        self.meta = meta
        super().__init__(*args, **kwargs)
        self._load_data()

    def _load_data(self):
        self.meta = np.load(self.meta, allow_pickle=True)['arr_0'].item()
        keys = list(self.meta.keys())
        for scene, cam in keys:
            if len(self.meta[scene, cam]['Ks'].keys()) == 0:
                self.meta.pop((scene, cam))

    def __len__(self):
        return 684000

    def get_stats(self):
        return f'{len(self)} pairs from { len(self.meta.keys())} scenes'
    
    def _get_views(self, pair_idx, resolution, rng):
        # seq, img1, img2 = self.pairs[pair_idx]
        # seq_path = osp.join(self.ROOT, self.scenes[seq])
        if self.overfit == True:
            scene, cam = random.choice(list(self.meta.keys())[:30])
        else:
            scene, cam = random.choice(list(self.meta.keys()))
        image_idexs = list(self.meta[scene, cam]['Ks'].keys())
        if self.sequential_input == False:
            imgs_idx = rng.choice(image_idexs, self.num_image + self.gt_num_image)
        else:
            last = len(image_idexs)-1
            interal = 4
            end = last - self.num_image * interal//2
            end = max(1, end)
            im_start = rng.choice(range(end))
            im_list = self.sequential_sample(im_start, last, interal)
            im_list = [max(0, min(im_idx, last)) for im_idx in im_list]
            imgs_idx = [image_idexs[im_idx] for im_idx in im_list]
        views = []
        seq_path = osp.join(self.ROOT, scene)
        for view_index in imgs_idx:
            image = imread_cv2(osp.join(seq_path, cam, view_index))
            image = image[..., ::-1]
            if flag_pcache:
                with pcache_fs.open(osp.join(seq_path, cam,  view_index.replace('.jpg', '.npy')), 'rb') as f:
                    depthmap = np.load(f)#)
            else:
                depthmap = np.load(osp.join(seq_path, cam,  view_index.replace('.jpg', '.npy')))
            # camera_params = np.load(osp.join(seq_path, impath + ".npz"))
            intrinsics = np.float32(self.meta[scene, cam]['Ks'][view_index])
            camera_pose = np.float32(self.meta[scene, cam]['poses'][view_index])
            image, depthmap, intrinsics = self._crop_resize_if_necessary(
                image, depthmap, intrinsics, resolution, rng, info=(seq_path))
            # random_number = rng.integers(low=90, high=95)
            # sky_mask = depthmap > np.percentile(depthmap, random_number)
            # depthmap[sky_mask] = 0
            img_org = image.copy()
            H, W = depthmap.shape[:2]
            fxfycxcy = np.array([intrinsics[0, 0]/W, intrinsics[1, 1]/H, intrinsics[0,2]/W, intrinsics[1,2]/H]).astype(np.float32)
            views.append(dict(
                # sky_mask=sky_mask,
                fxfycxcy=fxfycxcy,
                img_org=img_org,
                img=image,
                depthmap=depthmap,
                camera_pose=camera_pose,  # cam2world
                camera_intrinsics=intrinsics,
                dataset='Waymo',
                label=osp.join(seq_path, cam, view_index),
                depth_anything=depthmap,
                instance=view_index))

        return views



if __name__ == '__main__':
    from dust3r.datasets.base.base_stereo_view_dataset import view_name
    from dust3r.viz import SceneViz, auto_cam_size
    from dust3r.utils.image import rgb
    import nerfvis.scene as scene_vis 

    dataset =  Hypersim(split='train', ROOT='/nas7/vilab/zsz/mast3recon/data/hypersim/', meta='/input_ssd/zsz/dust3r_dataset/hypersim_meta.npz', resolution=[(512, 384)], aug_crop='auto', aug_monocular=0.005,  num_views=8, gt_num_image=0) 
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

