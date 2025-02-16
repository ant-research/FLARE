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
from collections import deque
import os
import json
import time
import glob
import tqdm
import torch
from io import BytesIO
import random

try:
    from pcache_fileio import fileio
    flag_pcache = True
except:
    flag_pcache = False
from enum import Enum, auto
from pathlib import Path
from PIL import Image
import PIL
index_json = '/nas3/zsz/NoPoSplat/assets/evaluation_index_re10k.json'
testset_json = '/nas3/zsz/re10k/test/index.json'
oss_folder_path = 'oss://antsys-vilab/datasets/pcache_datasets/'
# oss_file_path = 'oss://antsys-vilab/datasets/pcache_datasets/SAM_1B/sa_000000/images/sa_1011.jpg'
pcache_folder_path = 'pcache://vilabpcacheproxyi-pool.cz50c.alipay.com:39999/mnt/antsys-vilab_datasets_pcache_datasets/'

# pcache_file_path = oss_file_path.replace(oss_folder_path,pcache_folder_path)
from decord import VideoReader
from collections import OrderedDict
import numpy as np
import torch
from einops import rearrange
from jaxtyping import Float
from PIL import Image
from torch import Tensor



def rescale(
    image: Float[Tensor, "3 h_in w_in"],
    shape: tuple[int, int],
) -> Float[Tensor, "3 h_out w_out"]:
    h, w = shape
    image_new = (image * 255).clip(min=0, max=255).type(torch.uint8)
    image_new = rearrange(image_new, "c h w -> h w c").detach().cpu().numpy()
    image_new = Image.fromarray(image_new)
    image_new = image_new.resize((w, h), Image.LANCZOS)
    image_new = np.array(image_new) / 255
    image_new = torch.tensor(image_new, dtype=image.dtype, device=image.device)
    return rearrange(image_new, "h w c -> c h w")


def center_crop(
    images: Float[Tensor, "*#batch c h w"],
    intrinsics: Float[Tensor, "*#batch 3 3"],
    shape: tuple[int, int],
) -> tuple[
    Float[Tensor, "*#batch c h_out w_out"],  # updated images
    Float[Tensor, "*#batch 3 3"],  # updated intrinsics
]:
    *_, h_in, w_in = images.shape
    h_out, w_out = shape

    # Note that odd input dimensions induce half-pixel misalignments.
    row = (h_in - h_out) // 2
    col = (w_in - w_out) // 2

    # Center-crop the image.
    images = images[..., :, row : row + h_out, col : col + w_out]

    # Adjust the intrinsics to account for the cropping.
    intrinsics = intrinsics.clone()
    intrinsics[..., 0, 0] *= w_in / w_out  # fx
    intrinsics[..., 1, 1] *= h_in / h_out  # fy

    return images, intrinsics


def rescale_and_crop(
    images: Float[Tensor, "*#batch c h w"],
    intrinsics: Float[Tensor, "*#batch 3 3"],
    shape: tuple[int, int],
) -> tuple[
    Float[Tensor, "*#batch c h_out w_out"],  # updated images
    Float[Tensor, "*#batch 3 3"],  # updated intrinsics
]:
    *_, h_in, w_in = images.shape
    h_out, w_out = shape
    assert h_out <= h_in and w_out <= w_in

    scale_factor = max(h_out / h_in, w_out / w_in)
    h_scaled = round(h_in * scale_factor)
    w_scaled = round(w_in * scale_factor)
    assert h_scaled == h_out or w_scaled == w_out

    # Reshape the images to the correct size. Assume we don't have to worry about
    # changing the intrinsics based on how the images are rounded.
    *batch, c, h, w = images.shape
    images = images.reshape(-1, c, h, w)
    images = torch.stack([rescale(image, (h_scaled, w_scaled)) for image in images])
    images = images.reshape(*batch, c, h_scaled, w_scaled)

    return center_crop(images, intrinsics, shape)



import numpy as np
import torch
from einops import rearrange
from jaxtyping import Float
from PIL import Image
from torch import Tensor


def rescale(
    image: Float[Tensor, "3 h_in w_in"],
    shape: tuple[int, int],
) -> Float[Tensor, "3 h_out w_out"]:
    h, w = shape
    image_new = (image * 255).clip(min=0, max=255).type(torch.uint8)
    image_new = rearrange(image_new, "c h w -> h w c").detach().cpu().numpy()
    image_new = Image.fromarray(image_new)
    image_new = image_new.resize((w, h), Image.LANCZOS)
    image_new = np.array(image_new) / 255
    image_new = torch.tensor(image_new, dtype=image.dtype, device=image.device)
    return rearrange(image_new, "h w c -> c h w")


def center_crop(
    images: Float[Tensor, "*#batch c h w"],
    intrinsics: Float[Tensor, "*#batch 3 3"],
    shape: tuple[int, int],
) -> tuple[
    Float[Tensor, "*#batch c h_out w_out"],  # updated images
    Float[Tensor, "*#batch 3 3"],  # updated intrinsics
]:
    *_, h_in, w_in = images.shape
    h_out, w_out = shape

    # Note that odd input dimensions induce half-pixel misalignments.
    row = (h_in - h_out) // 2
    col = (w_in - w_out) // 2

    # Center-crop the image.
    images = images[..., :, row : row + h_out, col : col + w_out]

    # Adjust the intrinsics to account for the cropping.
    intrinsics = intrinsics.clone()
    intrinsics[..., 0, 0] *= w_in / w_out  # fx
    intrinsics[..., 1, 1] *= h_in / h_out  # fy

    return images, intrinsics


def rescale_and_crop(
    images: Float[Tensor, "*#batch c h w"],
    intrinsics: Float[Tensor, "*#batch 3 3"],
    shape: tuple[int, int],
) -> tuple[
    Float[Tensor, "*#batch c h_out w_out"],  # updated images
    Float[Tensor, "*#batch 3 3"],  # updated intrinsics
]:
    *_, h_in, w_in = images.shape
    h_out, w_out = shape
    assert h_out <= h_in and w_out <= w_in

    scale_factor = max(h_out / h_in, w_out / w_in)
    h_scaled = round(h_in * scale_factor)
    w_scaled = round(w_in * scale_factor)
    assert h_scaled == h_out or w_scaled == w_out

    # Reshape the images to the correct size. Assume we don't have to worry about
    # changing the intrinsics based on how the images are rounded.
    *batch, c, h, w = images.shape
    images = images.reshape(-1, c, h, w)
    images = torch.stack([rescale(image, (h_scaled, w_scaled)) for image in images])
    images = images.reshape(*batch, c, h_scaled, w_scaled)

    return center_crop(images, intrinsics, shape)



def load_from_json(filename):
    """Load a dictionary from a JSON filename.

    Args:
        filename: The filename to load from.
    """
    with open(filename, encoding="UTF-8") as file:
        return json.load(file)



def get_overlap_tag(overlap):
    if 0.05 <= overlap <= 0.3:
        overlap_tag = "small"
    elif 0 < overlap <= 0.55:
        overlap_tag = "medium"
    elif overlap <= 0.8:
        overlap_tag = "large"
    else:
        overlap_tag = "ignore"

    return overlap_tag

class Re10K_nopo(BaseStereoViewDataset_test):
    def __init__(self, *args, split, ROOT, meta='/nas3/zsz/NoPoSplat/assets/evaluation_index_re10k.json', only_pose=False, **kwargs):
        self.ROOT = ROOT.replace(oss_folder_path, pcache_folder_path)
        # import pdb; pdb.set_trace()
        # Collect chunks.
        ROOT = Path(ROOT)
        self.chunks = []
        self.index_json = load_from_json(meta)
        # self.index_iter = iter(self.index_json)
        # for root in ROOT:
        self.root =  ROOT / split
        
        self.test_path = load_from_json(testset_json)
        # import pdb; pdb.set_trace()
        self.available_scenes = []
        # output_image = glob.glob('/data0/zsz/mast3recon/data/output/*')
        # output_image = [osp.basename(x).split('.')[0] for x in output_image]
        # output_image = ['2ad09b7837010330', '17d841670d2da942', '18ce480be0ececbd', '28f5ebf3c3e2fe54', '64a0d6a31e6484ee', '398c4688209874c9', 'd38139cf5c8c1d40', 'e40ca395753837ce', 'ffe67ac537febe41']
        for current_scene, chunk_gt in self.index_json.items():
            if 'low_score_data' in meta:
                if current_scene not in output_image:
                    continue
            if chunk_gt is None:
                continue
            if 'overlap_tag' not in chunk_gt:
                if current_scene in self.test_path.keys() and chunk_gt is not None: #  and get_overlap_tag(chunk_gt['overlap']) == "small"
                    # chunk_path, indx = self.scene_dict[current_scene]
                    # chunk = torch.load(chunk_path)
                    self.available_scenes.append(current_scene)
            else:
                if current_scene in self.test_path.keys() and chunk_gt is not None and chunk_gt['overlap_tag'] == "small":
                    self.available_scenes.append(current_scene)
            
        # chunk_gt = self.index_json[current_scene]
        # index_list = chunk_gt['context']
        # index_target = chunk_gt['target']
        super().__init__(*args, **kwargs)
        self.rendering = True
        self.global_idx = 0
        
    def __len__(self):
        return len(self.available_scenes)
    
    @staticmethod
    def image_read(image_file):
        img = cv2.imread(image_file)
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
   
    def _get_views(self, idx, resolution, rng):
        # image_idx1, image_idx2 = self.pairs[idx]
        # imgs_idxs = self._random_index(rng)
        # scale = -1
        
        # current_scene = next(self.index_iter)
        # # i = 0

        # while current_scene not in self.scene_dict or self.index_json[current_scene] is None:
        #     current_scene = next(self.index_iter)
        #     # i += 1
        # #     current_scene = self.scene_dict.keys()[0]
        # try:
        #     chunk_gt = self.index_json[current_scene]
        #     index_list = chunk_gt['context']
        #     index_target = chunk_gt['target']
        #     # print('get a index list', index_list)
        # except:
        #     import pdb; pdb.set_trace() 

        # chunk_path, indx = self.scene_dict[current_scene]
        # chunk = torch.load(chunk_path)
        # chunk = chunk[indx]
        idx = self.global_idx
        current_scene = self.available_scenes[idx]
        self.global_idx += 1
        chunk_gt = self.index_json[current_scene]
        file_path = self.test_path[current_scene]
        chunk_path = self.root/ file_path
        if 'overlap_tag' not in chunk_gt:
            index_list = list(chunk_gt['context'])
            index_target = list(chunk_gt['target'])
        else:
            index_list = list(chunk_gt['context_index'])
            index_target = list(chunk_gt['target_index'])
        # print("index_list", index_list)
        # print("index_target", index_target)
        # chunk_path, indx = self.scene_dict[current_scene]
        chunk = torch.load(chunk_path)
        name_dict = {}
        final_i = None
        for i in range(len(chunk)):
            if chunk[i]['key']==current_scene:
                final_i = i
        chunk = chunk[final_i]
        if 'overlap_tag' in chunk_gt:
            overlap = chunk_gt['overlap_tag']
            psnr = chunk_gt['psnr']
        else:
            overlap = 0#chunk_gt['overlap']
            psnr = 0
        poses_right = chunk["cameras"][index_list[0]]
        w2c_right = np.eye(4)
        w2c_right[:3] = poses_right[6:].reshape(3, 4)
        camera_pose_right =  np.linalg.inv(w2c_right)
        poses_left = chunk["cameras"][index_list[1]]
        w2c_left = np.eye(4)
        w2c_left[:3] = poses_left[6:].reshape(3, 4)
        camera_pose_left =  np.linalg.inv(w2c_left)
        a, b = camera_pose_right[:3, 3], camera_pose_left[:3, 3]
        scale = np.linalg.norm(a - b)
        
        index_list.extend(index_target)
        views = []
        # print('extended index_list, total num', index_list, len(self.available_scenes))
        for index in index_list:
            poses = chunk["cameras"][index]
            intrinsics = np.eye(3)
            fx, fy, cx, cy = poses[:4]
            intrinsics[0, 0] = fx
            intrinsics[1, 1] = fy
            intrinsics[0, 2] = cx
            intrinsics[1, 2] = cy
            # Convert the extrinsics to a 4x4 OpenCV-style W2C matrix.
            w2c = np.eye(4)
            w2c[:3] = poses[6:].reshape(3, 4)
            camera_pose =  np.linalg.inv(w2c)
            camera_pose[:3, 3] = camera_pose[:3, 3] / scale
            
            scene = chunk["key"]
            frame = chunk["images"][index] 
            frame = Image.open(BytesIO(frame.numpy().tobytes())).convert('RGB')
            frame = np.asarray(frame)
            depthmap = np.zeros_like(frame)[..., 0]
            frame = torch.tensor(frame).permute(2, 0, 1).float() / 255.0
            intrinsics = torch.tensor(intrinsics)
            images, intrinsics = rescale_and_crop(frame, intrinsics, resolution)
            images = images.permute(1, 2, 0).numpy() * 255
            H, W = images.shape[:2]
            images = PIL.Image.fromarray(images.astype(np.uint8))
            intrinsics[0, 0] = intrinsics[0, 0] * W
            intrinsics[1, 1] = intrinsics[1, 1] * H
            intrinsics[0, 2] = intrinsics[0, 2] * W
            intrinsics[1, 2] = intrinsics[1, 2] * H
            rgb_image_orig = images.copy()
            depthmap = np.zeros_like(images)[..., 0]
            fxfycxcy = np.array([intrinsics[0, 0]/W, intrinsics[1, 1]/H, intrinsics[0,2]/W, intrinsics[1,2]/H]).astype(np.float32)
            intrinsics = intrinsics.numpy()
            views.append(dict(
                img_org=rgb_image_orig,
                img=images,
                depthmap=depthmap.astype(np.float32),
                camera_pose=camera_pose.astype(np.float32),
                camera_intrinsics=intrinsics.astype(np.float32),
                fxfycxcy=fxfycxcy,
                dataset='re10k',
                label=scene,
                instance=scene,
                overlap=get_overlap_tag(overlap) if 'overlap_tag' not in chunk_gt else chunk_gt['overlap_tag'],
                psnr = np.array([psnr]).astype(np.float32)
            ))

        return views


# if __name__ == "__main__":
#     from dust3r.datasets.base.base_stereo_view_dataset import view_name
#     from dust3r.viz import SceneViz, auto_cam_size
#     from dust3r.utils.image import rgb
#     import nerfvis.scene as scene_vis 
#     dataset = Re10K_ps(split='train', ROOT="/nas3/zsz/re10k/re10k",resolution=[(256, 256)], aug_crop=16)
#     for idx in np.random.permutation(len(dataset)):
#         views = dataset[idx]
        # assert len(views) == 2
        # print(view_name(views[0]), view_name(views[1]))
        # view_idxs = list(range(len(views)))
        # poses = [views[view_idx]['camera_pose'] for view_idx in view_idxs]
        # cam_size = max(auto_cam_size(poses), 0.001)
        # pts3ds = []
        # colors = []
        # valid_masks = []
        # c2ws = []
        # intrinsics = []
        # for view_idx in view_idxs:
        #     pts3d = views[view_idx]['pts3d']
        #     pts3ds.append(pts3d)
        #     valid_mask = views[view_idx]['valid_mask']
        #     valid_masks.append(valid_mask)
        #     color = rgb(views[view_idx]['img'])
        #     colors.append(color)
            # viz.add_pointcloud(pts3d, colors, valid_mask)
        #     c2ws.append(views[view_idx]['camera_pose'])

        
        # pts3ds = np.stack(pts3ds, axis=0)
        # colors = np.stack(colors, axis=0)
        # valid_masks = np.stack(valid_masks, axis=0)
        # c2ws = np.stack(c2ws)
        # scene_vis.set_title("My Scene")
        # scene_vis.set_opencv() 
        # # colors = torch.zeros_like(structure).to(structure)
        # # scene_vis.add_points("points", pts3ds.reshape(-1,3)[valid_masks.reshape(-1)], vert_color=colors.reshape(-1,3)[valid_masks.reshape(-1)], point_size=1)
        # # for i in range(len(c2ws)):
        # f = 1111.0 / 2.5
        # z = 10.
        # scene_vis.add_camera_frustum("cameras", r=c2ws[:, :3, :3], t=c2ws[:, :3, 3], focal_length=f,
        #                 image_width=colors.shape[2], image_height=colors.shape[1],
        #                 z=z, connect=False, color=[1.0, 0.0, 0.0])
        # for i in range(len(c2ws)):
        #     scene_vis.add_image(
        #                     f"images/{i}",
        #                     colors[i], # Can be a list of paths too (requires joblib for that) 
        #                     r=c2ws[i, :3, :3],
        #                     t=c2ws[i, :3, 3],
        #                     # Alternatively: from nerfvis.utils import split_mat4; **split_mat4(c2ws)
        #                     focal_length=f,
        #                     z=z,
        #                 )
        # scene_vis.display(port=8081)

