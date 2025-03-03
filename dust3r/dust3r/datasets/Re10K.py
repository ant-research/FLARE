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
from dust3r.datasets.base.base_stereo_view_dataset import BaseStereoViewDataset_test
from collections import deque
import os
import json
import time
import glob
import tqdm
import torch
from io import BytesIO
import random
from enum import Enum, auto
from pathlib import Path
from PIL import Image
import PIL
from decord import VideoReader
from collections import OrderedDict
import numpy as np
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

class Re10K(BaseStereoViewDataset_test):
    def __init__(self, *args, split, ROOT, meta='assets/evaluation_index_re10k.json', testset_json='assets/index.json', only_pose=False, **kwargs):
        self.ROOT = ROOT
        ROOT = Path(ROOT)
        self.chunks = []
        self.index_json = load_from_json(meta)
        self.root =  ROOT / split
        self.test_path = load_from_json(testset_json)
        self.available_scenes = []
        for current_scene, chunk_gt in self.index_json.items():
            if chunk_gt is None:
                continue
            if 'overlap_tag' not in chunk_gt:
                if current_scene in self.test_path.keys() and chunk_gt is not None: 
                    self.available_scenes.append(current_scene)
            else:
                if current_scene in self.test_path.keys() and chunk_gt is not None and get_overlap_tag(chunk_gt['overlap']) != 'large':
                    self.available_scenes.append(current_scene)
            
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
            overlap = 0
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
