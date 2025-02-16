# Copyright (C) 2024-present Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).
#
# --------------------------------------------------------
# Dataloader for preprocessed StaticThings3D
# dataset at https://github.com/lmb-freiburg/robustmvd/
# See datasets_preprocess/preprocess_staticthings3d.py
# --------------------------------------------------------
import os.path as osp
import numpy as np
import mast3r.utils.path_to_dust3r  # noqa
import glob
import os
from dust3r.datasets.base.base_stereo_view_dataset import BaseStereoViewDataset
from dust3r.utils.image import imread_cv2
try:
    from pcache_fileio import fileio
    flag_pcache = True
except:
    flag_pcache = False
oss_folder_path = 'oss://antsys-vilab/datasets/pcache_datasets/'
# oss_file_path = 'oss://antsys-vilab/datasets/pcache_datasets/SAM_1B/sa_000000/images/sa_1011.jpg'
pcache_folder_path = 'pcache://vilabpcacheproxyi-pool.cz50c.alipay.com:39999/mnt/antsys-vilab_datasets_pcache_datasets/'
# pcache_file_path = oss_file_path.replace(oss_folder_path,pcache_folder_path)


class StaticThings3D (BaseStereoViewDataset):
    """ Dataset of indoor scenes, 5 images each time
    """
    def __init__(self, ROOT, meta, *args, mask_bg='rand', **kwargs):
        self.ROOT = ROOT.replace(oss_folder_path, pcache_folder_path)
        super().__init__(*args, **kwargs)

        assert mask_bg in (True, False, 'rand')
        self.mask_bg = mask_bg

        # loading all pairs
        assert self.split is None
        self.meta = np.load(meta, allow_pickle=True)
        # data_path = list(self.scenes.keys())[0]
        # data_path = 
        self.scenes_orig = self.meta['scenes'].item()
        self.scenes = {}
        for key in self.scenes_orig.keys():
            self.scenes[self.ROOT+ '/' + '/'.join(key.split('/')[-3:])] = self.scenes_orig[key]
        # self.scenes_new = {}
        # for scene in self.scenes:
        self.folders = list(self.scenes.keys())
        # folders = glob.glob(osp.join(data_path, '*/*/*'))
        # self.folders = folders
        # self.scenes = {}
        # for folder in folders:
        #     assert osp.exists(osp.join(folder, 'left'))
        #     assert osp.exists(osp.join(folder, 'right'))
        #     folder_left = folder + '/left/*.jpg'
        #     folder_right = folder + '/right/*.jpg'
        #     self.scenes[folder] = glob.glob(folder_left) + glob.glob(folder_right)
        # np.savez('/nas7/vilab/zsz/mast3recon/dust3r/datasets_preprocess/data/staticthings3d_metadata.npz', scenes=self.scenes)

    def __len__(self):
        return len(self.folders) * 2 * 87 * 88

    def get_stats(self):
        return f'{len(self)} pairs'

    def _get_views(self, pair_idx, resolution, rng):
        scene = rng.choice(self.folders)
        # seq_path = osp.join('TRAIN', scene.decode('ascii'), f'{seq:04d}')
        self.num_image_input = self.num_image
        image_idx = rng.choice(self.scenes[scene], self.num_image_input + self.gt_num_image)
        views = []

        mask_bg = (self.mask_bg == True) or (self.mask_bg == 'rand' and rng.choice(2))

        for idx in image_idx:
            idx = '/'.join(idx.split('/')[-5:])
            idx = osp.join(self.ROOT, idx)
            image = imread_cv2(idx)
            depth_path = idx.replace('_final.jpg', '').replace('_clean.jpg', '')
            depthmap = imread_cv2(osp.join(depth_path+".exr"))
            cam_path = idx.replace('_final.jpg', '').replace('_clean.jpg', '')
            camera_params = np.load(osp.join(cam_path+".npz"))

            intrinsics = camera_params['intrinsics']
            camera_pose = camera_params['cam2world']
            
            if mask_bg:
                depthmap[depthmap > 200] = 0

            if rng.random() > 0.5:
                depthmap[np.isinf(depthmap)] = 0
                random_number = rng.integers(50, 99)
                depthmap[depthmap > np.percentile(depthmap, random_number)] = 0

            image, depthmap, intrinsics = self._crop_resize_if_necessary(image, depthmap, intrinsics, resolution, rng, info=(depth_path))
            img_org = image.copy()
            H, W = depthmap.shape[:2]
            fxfycxcy = np.array([intrinsics[0, 0]/W, intrinsics[1, 1]/H, intrinsics[0,2]/W, intrinsics[1,2]/H]).astype(np.float32)
            views.append(dict(
                fxfycxcy = fxfycxcy,
                img_org = img_org,
                img = image, 
                depthmap = depthmap,
                camera_pose = camera_pose, # cam2world
                camera_intrinsics = intrinsics,
                dataset = 'StaticThings3D',
                label = depth_path,
                instance = os.path.basename(depth_path)))

        return views

if __name__ == '__main__':
    from dust3r.datasets.base.base_stereo_view_dataset import view_name
    from dust3r.viz import SceneViz, auto_cam_size
    from dust3r.utils.image import rgb
    import nerfvis.scene as scene_vis 

    dataset = StaticThings3D(ROOT="oss://antsys-vilab/datasets/pcache_datasets/staticthings3d_processed/", meta='/nas7/vilab/zsz/mast3recon/dust3r/datasets_preprocess/data/staticthings3d_metadata.npz', resolution=[(512, 384)], aug_crop=16)

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