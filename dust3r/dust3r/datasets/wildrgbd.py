# Copyright (C) 2024-present Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).
#
# --------------------------------------------------------
# Dataloader for preprocessed WildRGB-D
# dataset at https://github.com/wildrgbd/wildrgbd/
# See datasets_preprocess/preprocess_wildrgbd.py
# --------------------------------------------------------
import os.path as osp

import cv2
import numpy as np
import mast3r.utils.path_to_dust3r  # noqa

from dust3r.datasets.co3d import Co3d
from dust3r.utils.image import imread_cv2
import tqdm
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
# pcache_file_path = oss_file_path.replace(oss_folder_path,pcache_folder_path)
from collections import deque
import random

class WildRGBD(Co3d):
    def __init__(self, mask_bg='True', *args, ROOT, meta, **kwargs):
        super().__init__(mask_bg, *args, ROOT=ROOT, **kwargs)
        self.ROOT = ROOT.replace(oss_folder_path, pcache_folder_path)
        self.dataset_label = 'WildRGBD'
        self.camera_pose = {}
        self.camera_intrinsics = {}
        # # for obj, instance in tqdm.tqdm(self.scene_list):
        # #     image_pool = self.scenes[obj, instance]
        # #     self.camera_intrinsics[obj, instance] = {}
        # #     self.camera_pose[obj, instance] = {}
        # #     for view_idx in image_pool:
        # #         metadata_path = self._get_metadatapath(obj, instance, view_idx)
        # #         input_metadata = np.load(metadata_path)
        # #         camera_pose = input_metadata['camera_pose'].astype(np.float32)
        # #         intrinsics = input_metadata['camera_intrinsics'].astype(np.float32)
        # #         self.camera_pose[obj, instance][view_idx] = camera_pose
        # #         self.camera_intrinsics[obj, instance][view_idx] = intrinsics
        # for obj, instance in tqdm.tqdm(self.scene_list):
        #     self.process_scene(obj, instance, self.scenes, self.camera_pose, self.camera_intrinsics)
        # np.savez('/nas7/vilab/zsz/mast3recon/checkpoints/wildrgbd_metadata.npz', camera_pose=self.camera_pose, camera_intrinsics=self.camera_intrinsics)
        self.meta = np.load(meta, allow_pickle=True)
        self.camera_pose = self.meta['camera_pose'].item()
        self.camera_intrinsics = self.meta['camera_intrinsics'].item()

    def process_scene(self, obj, instance, scenes, camera_pose, camera_intrinsics):
        image_pool = scenes[obj, instance]
        camera_pose[obj, instance] = {}
        camera_intrinsics[obj, instance] = {}
        with concurrent.futures.ThreadPoolExecutor(max_workers=32) as executor:
            futures = [
                executor.submit(self.process_view, view_idx, obj, instance, scenes, camera_pose[obj, instance], camera_intrinsics[obj, instance])
                for view_idx in image_pool
            ]
            for future in concurrent.futures.as_completed(futures):
                future.result()

    def process_view(self, view_idx, obj, instance, scenes, camera_pose, camera_intrinsics):
        metadata_path = self._get_metadatapath(obj, instance, view_idx)
        input_metadata = np.load(metadata_path)
        camera_pose_data = input_metadata['camera_pose'].astype(np.float32)
        intrinsics = input_metadata['camera_intrinsics'].astype(np.float32)
        camera_pose[view_idx] = camera_pose_data
        camera_intrinsics[view_idx] = intrinsics
        
    def _get_metadatapath(self, obj, instance, view_idx):
        return osp.join(self.ROOT, obj, instance, 'metadata', f'{view_idx:0>5d}.npz')

    def _get_impath(self, obj, instance, view_idx):
        return osp.join(self.ROOT, obj, instance, 'rgb', f'{view_idx:0>5d}.jpg')

    def _get_depthpath(self, obj, instance, view_idx):
        return osp.join(self.ROOT, obj, instance, 'depth', f'{view_idx:0>5d}.png')

    def _get_maskpath(self, obj, instance, view_idx):
        return osp.join(self.ROOT, obj, instance, 'masks', f'{view_idx:0>5d}.png')

    def _read_depthmap(self, depthpath, input_metadata):
        # We store depths in the depth scale of 1000.
        # That is, when we load depth image and divide by 1000, we could get depth in meters.
        depthmap = imread_cv2(depthpath, cv2.IMREAD_UNCHANGED)
        depthmap = depthmap.astype(np.float32) / 1000.0
        return depthmap

    def _get_views(self, idx, resolution, rng):
        # choose a scene
        # obj, instance = self.scene_list[idx // len(self.combinations)]
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
        # decide now if we mask the bg
        mask_bg = (self.mask_bg == True) or (self.mask_bg == 'rand' and rng.choice(2))

        views = []
        imgs_idxs = [max(0, min(im_idx, last)) for im_idx in im_list]
        imgs_idxs = deque(imgs_idxs)
        while len(imgs_idxs) > 0:  # some images (few) have zero depth
            im_idx = imgs_idxs.pop()

            # if self.invalidate[obj, instance][resolution][im_idx]:
            #     # search for a valid image
            #     random_direction = 2 * rng.choice(2) - 1
            #     for offset in range(1, len(image_pool)):
            #         tentative_im_idx = (im_idx + (random_direction * offset)) % len(image_pool)
            #         if not self.invalidate[obj, instance][resolution][tentative_im_idx]:
            #             im_idx = tentative_im_idx
            #             break

            view_idx = image_pool[im_idx]

            impath = self._get_impath(obj, instance, view_idx)
            depthpath = self._get_depthpath(obj, instance, view_idx)

            # load camera params
            camera_pose = self.camera_pose[obj, instance][view_idx].astype(np.float32)
            intrinsics = self.camera_intrinsics[obj, instance][view_idx].astype(np.float32)

            # load image and depth
            rgb_image = imread_cv2(impath)
            depthmap = self._read_depthmap(depthpath, None)

            if mask_bg:
                # load object mask
                maskpath = self._get_maskpath(obj, instance, view_idx)
                maskmap = imread_cv2(maskpath, cv2.IMREAD_UNCHANGED).astype(np.float32)
                maskmap = (maskmap / 255.0) > 0.1

                # update the depthmap with mask
                depthmap *= maskmap

            rgb_image, depthmap, intrinsics = self._crop_resize_if_necessary(
                rgb_image, depthmap, intrinsics, resolution, rng=rng, info=impath)
            img_org = rgb_image.copy()
            num_valid = (depthmap > 0.0).sum()
            # if num_valid == 0:
            #     # problem, invalidate image and retry
            #     self.invalidate[obj, instance][resolution][im_idx] = True
            #     imgs_idxs.append(im_idx)
            #     continue
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
                label=osp.join(obj, instance),
                instance=osp.split(impath)[1],
            ))
        return views


if __name__ == "__main__":
    from dust3r.datasets.base.base_stereo_view_dataset import view_name
    from dust3r.viz import SceneViz, auto_cam_size
    from dust3r.utils.image import rgb
    import nerfvis.scene as scene_vis 

    dataset = WildRGBD(mask_bg='rand', split='train', ROOT="/nas8/datasets/wildrgbd_processed", meta='/nas7/vilab/zsz/mast3recon/checkpoints/wildrgbd_metadata.npz', resolution=[(512, 384)], aug_crop=16)

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