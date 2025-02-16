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
from dust3r.datasets.base.base_stereo_view_dataset import BaseStereoViewDataset_test
from dust3r.utils.image import imread_cv2, imread_cv2_orig
from dust3r.utils.geometry import colmap_to_opencv_intrinsics#, opencv_to_colmap_intrinsics  # noqa
from torchvision import transforms
from collections import deque
import os
import json
import time
import glob
import tqdm
try:
    from pcache_fileio import fileio
    flag_pcache = True
except:
    flag_pcache = False
from enum import Enum, auto
from pathlib import Path
from PIL import Image
import torch
import re
from copy import deepcopy
from dust3r.datasets.crop_shim import apply_crop_shim
oss_folder_path = 'oss://antsys-vilab/datasets/pcache_datasets/'
pcache_folder_path = 'pcache://vilabpcacheproxyi-pool.cz50c.alipay.com:39999/mnt/antsys-vilab_datasets_pcache_datasets/'

def read_camera_parameters(filename):
    with open(filename) as f:
        lines = f.readlines()
        lines = [line.rstrip() for line in lines]
    # extrinsics: line [1,5), 4x4 matrix
    extrinsics = np.fromstring(' '.join(lines[1:5]), dtype=np.float32, sep=' ').reshape((4, 4))
    # intrinsics: line [7-10), 3x3 matrix
    intrinsics = np.fromstring(' '.join(lines[7:10]), dtype=np.float32, sep=' ').reshape((3, 3))
    # TODO: assume the feature is 1/4 of the original image size
    # intrinsics[:2, :] *= scale
    # if (flag==0):
    #     intrinsics[0,2]-=index
    # else:
    #     intrinsics[1,2]-=index
    return intrinsics, extrinsics

def get_context_target_data(cams_target, cams_src, intrinsic, intrinsic_src, target_index, context_index):
    transform = transforms.ToTensor()
    context_imgs = [transform(Image.open(img)) for img in context_index]
    context_imgs = torch.stack(context_imgs)
    b, c, h, w = context_imgs.shape
    # import pdb; pdb.set_trace()
    context_extrinsics = [torch.tensor(np.load(cam)) for cam in cams_src]
    context_intrinsics = [torch.tensor(np.load(intrinsic)) for intrinsic in intrinsic_src]
    context_extrinsics = torch.cat([extrinsic for extrinsic in context_extrinsics],0)  # [batch, 4, 4]
    context_intrinsics = torch.cat(context_intrinsics,0)
    context_intrinsics[:, :1] = context_intrinsics[:, :1] / w
    context_intrinsics[:, 1:2] = context_intrinsics[:, 1:2] / h
    # if '84' in str(target_index):
    #     import ipdb; ipdb.set_trace()
    target_img = Image.open(target_index)
    target_img = transform(target_img)
    target_img = torch.stack([target_img])  
    
    b, c, h, w = target_img.shape
    target_extrinsics = [torch.tensor(np.load(cams_target))]
    target_extrinsics = torch.cat([extrinsic for extrinsic in target_extrinsics], 0)  # [1, 4, 4]
    target_intrinsics = torch.cat([torch.tensor(np.load(intrinsic))], 0)  # [1,
    target_intrinsics[:, :1] = target_intrinsics[:, :1] / w
    target_intrinsics[:, 1:2] = target_intrinsics[:, 1:2] / h
    context_data = {
        "image": context_imgs,
        "extrinsics": context_extrinsics,
        "intrinsics": context_intrinsics,
        "index": [str(context_index_per) for context_index_per in context_index]
    }
    target_data = {
        "image": target_img,
        "extrinsics": target_extrinsics,
        "intrinsics": target_intrinsics,
        "index": [str(target_index)]
    }

    return context_data, target_data

class DL3DV_test_eval_2views(BaseStereoViewDataset_test):
    def __init__(self, *args, split,meta, ROOT, only_pose=False, **kwargs):
        self.ROOT = ROOT 
        self.meta = meta
        self.npy_list = os.listdir(os.path.join(meta))
        self.target_image_list = []
        self.target_cam_list = []
        self.target_intrinsic_list = []
        self.src_images_list = []
        self.src_cams_list = []
        self.src_intrinsic_list = []
        self.index_list = []
        for npy in self.npy_list:
            # if  'a8bb155bab889d8b0a2f62977dfbe3388ea2226849f42e809522f97d9893688b' not in npy: #'9cd1493527b5a9eae415fe1649828c014c6f40ab1f480315ea0239a3d0b765e9' not in npy: # and '9937f80f08a339ebf05fdb14704fdb6c59abf70ba6457b46c63afac6ae21f783' not in npy
            #     continue
            npy_path = Path(self.meta) / npy
            image_list = np.load(npy_path)
            scene_name = npy.split('.')[0]
            self.img_root = Path(self.meta + '_results')/ scene_name / 'images'
            self.img_list = sorted(glob.glob(str(self.img_root / 'input_*')))
            self.img_list = sorted([file  for file in self.img_list if 'input_.' not in file])
            self.cam_root = Path(self.meta + '_results')/ scene_name / 'cams'
            self.cam_list = sorted(glob.glob(str(self.cam_root / 'input_*')))
            self.cam_list = sorted([file  for file in self.cam_list if 'input_.' not in file])

            self.intrinsic_root =  Path(self.meta + '_results')/ scene_name / 'K'
            self.intrinsic_list = sorted(glob.glob(str(self.intrinsic_root / 'input_*')))
            self.intrinsic_list = sorted([file  for file in self.intrinsic_list if 'input_.' not in file])

            
            self.imgs_list_gt = sorted(self.img_root.glob('gt_*'))
            pattern = re.compile(r'^gt_\d+(\.\w+)?$') 
            self.imgs_list_gt = sorted([file for file in self.imgs_list_gt if pattern.match(file.name)])
            
            self.cam_root_gt = Path(self.meta + 'results')  / 'cams'
            self.cams_list_gt = sorted(self.cam_root.glob('gt_*'))
            self.cams_list_gt = sorted([file for file in self.cams_list_gt if pattern.match(file.name)])
            
            self.intrinsic_root_gt = Path(self.meta+ 'results') / 'K'
            self.intrinsics_list_gt = sorted(self.intrinsic_root.glob('gt_*'))
            self.intrinsics_list_gt = sorted([file for file in self.intrinsics_list_gt if pattern.match(file.name)])
            
            
            src_indexs = [int(img_list.split('/')[-1].split('.')[0].split('_')[-1]) for img_list in self.img_list if 'input_.png' not in img_list and 'gt_.png' not in img_list]
 

            src_indexs = np.array(src_indexs)
            for idx, (img_list_gt, cam_list_gt, intrinsic_list_gt) in enumerate(zip(self.imgs_list_gt, self.cams_list_gt, self.intrinsics_list_gt)):
                target_index = int(str(img_list_gt).split('/')[-1].split('.')[0].split('_')[-1])
                # if target_index != 84:
                #     continue
                self.target_image_list.append(img_list_gt)
                self.target_cam_list.append(cam_list_gt)
                self.target_intrinsic_list.append(intrinsic_list_gt)
                diffs = np.abs(src_indexs - target_index)
                closest_indices = diffs.argsort()[:2]
                # if len(closest_indices) < 1:
                img_per = [self.img_list[i] for i in closest_indices]
                # img_per = [self.img_list[0], self.img_list[1]]
                cam_per = [self.cam_list[i] for i in closest_indices]
                intrinsic_per = [self.intrinsic_list[i] for i in closest_indices]
                self.src_images_list.append(img_per)
                self.src_cams_list.append(cam_per) 
                self.src_intrinsic_list.append(intrinsic_per)
                self.index_list.append(target_index)
        super().__init__(*args, **kwargs)
        self.global_idx = 0
    def __len__(self):
        return len(self.src_images_list) 
    
    @staticmethod
    def image_read(image_file):
        img = cv2.imread(image_file)
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    @staticmethod
    def depth_read(depth_file):
        depth = np.load(depth_file)
        depth_min = np.percentile(depth, 5)
        depth[depth==np.nan] = 0
        depth[depth==np.inf] = 0
        depth_max =  np.percentile(depth, 95)  
        depth[depth>=depth_max] = 0
        depth[depth>=depth_min+200] = 0
        return depth
    
    def sequential_sample(self, im_start, last, interal):
        im_list = [
            im_start + i * interal
            for i in range(self.num_image)
        ]
        im_list += [
            random.choice(im_list) + random.choice(list(range(-interal//2, interal//2)))
            for _ in range(self.gt_num_image)
        ]
        return im_list
        
    def _get_views(self, idx, resolution, rng):
        # image_idx1, image_idx2 = self.pairs[idx]
        idx = self.global_idx
        self.num_image_input = self.num_image
        views = []
        target_index = self.target_image_list[idx]
        context_index = self.src_images_list[idx]
        intrinsic = self.target_intrinsic_list[idx]
        intrinsic_src = self.src_intrinsic_list[idx]
        cams_src = self.src_cams_list[idx]
        cams_target = self.target_cam_list[idx]
        idex = self.index_list[idx]
        context_data, target_data = get_context_target_data(cams_target, cams_src, intrinsic, intrinsic_src, target_index, context_index)  
        processed_data = {
            'context': context_data,
            'target': target_data,
            'scene': self.ROOT,
        }
        # import ipdb; ipdb.set_trace()
        # import matplotlib.pyplot as plt
        # plt.imsave('test.png', ret['target']['image'][0].permute(1,2,0).numpy())
        ret = apply_crop_shim(processed_data, (256,256))
        self.global_idx += 1
        scene_name = context_index[0].split('/')[-3]
        camera_pose_right = ret['context']['extrinsics'][0]
        camera_pose_left = ret['context']['extrinsics'][1]
        a, b = camera_pose_right[:3, 3], camera_pose_left[:3, 3]
        scale = np.linalg.norm(a - b)
        for i in range(len(ret['context']['image'])):
            image = ret['context']['image'][i]
            H, W = image.shape[1:]
            extrinsics = ret['context']['extrinsics'][i].cpu().numpy()
            extrinsics[:3, 3] = extrinsics[:3, 3] / scale

            intrinsics = ret['context']['intrinsics'][i].cpu().numpy()
            intrinsics[:1] = intrinsics[:1] * H 
            intrinsics[1:2] = intrinsics[1:2] * W
            image = image.permute(1,2,0).numpy()
            depth_anything = np.zeros_like(image[..., 0])

            image = Image.fromarray((image * 255).astype(np.uint8))
            fxfycxcy = np.array([intrinsics[0, 0]/W, intrinsics[1, 1]/H, intrinsics[0,2]/W, intrinsics[1,2]/H]).astype(np.float32)
            views.append(dict(
                img_org=image,
                img=image,
                depthmap=depth_anything.astype(np.float32),
                camera_pose=extrinsics.astype(np.float32),
                camera_intrinsics=intrinsics.astype(np.float32),
                fxfycxcy=fxfycxcy,
                dataset='DL3DV',
                label=self.meta,
                instance=scene_name+'/'+str(self.global_idx),
                depth_anything=depth_anything,
            ))

        for i in range(len(ret['target']['image'])):
            image = ret['target']['image'][i]
            H, W = image.shape[1:]
            extrinsics = ret['target']['extrinsics'][i].cpu().numpy()
            extrinsics[:3, 3] = extrinsics[:3, 3] / scale
            intrinsics = ret['target']['intrinsics'][i].cpu().numpy()
            intrinsics[:1] = intrinsics[:1] * H 
            intrinsics[1:2] = intrinsics[1:2] * W
            image = image.permute(1,2,0).numpy()
            depth_anything = np.zeros_like(image[..., 0])
            image = Image.fromarray((image * 255).astype(np.uint8))
            fxfycxcy = np.array([intrinsics[0, 0]/W, intrinsics[1, 1]/H, intrinsics[0,2]/W, intrinsics[1,2]/H]).astype(np.float32)
            views.append(dict(
                img_org=image,
                img=image,
                depthmap=depth_anything.astype(np.float32),
                camera_pose=extrinsics.astype(np.float32),
                camera_intrinsics=intrinsics.astype(np.float32),
                fxfycxcy=fxfycxcy,
                dataset='DL3DV',
                label=self.meta,
                instance=scene_name+'/' + str(idex),
                depth_anything=depth_anything,
            ))
        views = [views[1], views[0], views[2]]
        # image.save('test.png')
        # import ipdb; ipdb.set_trace()
        return views


if __name__ == "__main__":
    from dust3r.datasets.base.base_stereo_view_dataset import view_name
    from dust3r.viz import SceneViz, auto_cam_size
    from dust3r.utils.image import rgb
    import nerfvis.scene as scene_vis 
    dataset = DL3DV(split='train', ROOT="/input1/datasets/DL3DV_dust3r", meta='/input1/zsz/DL3DV/json', resolution=[(512, 384)], aug_crop=16)

    for idx in np.random.permutation(len(dataset)):
        views = dataset[idx]
        # assert len(views) == 2
        # print(view_name(views[0]), view_name(views[1]))
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

        
        pts3ds = np.stack(pts3ds, axis=0)
        colors = np.stack(colors, axis=0)
        valid_masks = np.stack(valid_masks, axis=0)
        c2ws = np.stack(c2ws)
        scene_vis.set_title("My Scene")
        scene_vis.set_opencv() 
        # colors = torch.zeros_like(structure).to(structure)
        mean = pts3ds.reshape(-1,3)[valid_masks.reshape(-1)].mean(0)
        scene_vis.add_points("points", pts3ds.reshape(-1,3)[valid_masks.reshape(-1)] - mean, vert_color=colors.reshape(-1,3)[valid_masks.reshape(-1)], point_size=1)
        # for i in range(len(c2ws)):
        f = 1111.0 
        z = 1.
        scene_vis.add_camera_frustum("cameras", r=c2ws[:, :3, :3], t=c2ws[:, :3, 3] - mean, focal_length=f,
                        image_width=colors.shape[2], image_height=colors.shape[1],
                        z=z, connect=False, color=[1.0, 0.0, 0.0])
        for i in range(len(c2ws)):
            scene_vis.add_image(
                            f"images/{i}",
                            colors[i], # Can be a list of paths too (requires joblib for that) 
                            r=c2ws[i, :3, :3],
                            t=c2ws[i, :3, 3] - mean,
                            # Alternatively: from nerfvis.utils import split_mat4; **split_mat4(c2ws)
                            focal_length=f,
                            z=z,
                        )
        scene_vis.export('vis', embed_output=True)
        break
