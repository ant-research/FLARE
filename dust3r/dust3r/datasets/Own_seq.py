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
try:
    from pcache_fileio import fileio
    flag_pcache = True
except:
    flag_pcache = False
from enum import Enum, auto
from pathlib import Path

oss_folder_path = 'oss://antsys-vilab/datasets/pcache_datasets/'
# oss_file_path = 'oss://antsys-vilab/datasets/pcache_datasets/SAM_1B/sa_000000/images/sa_1011.jpg'
pcache_folder_path = 'pcache://vilabpcacheproxyi-pool.cz50c.alipay.com:39999/mnt/antsys-vilab_datasets_pcache_datasets/'
# pcache_file_path = oss_file_path.replace(oss_folder_path,pcache_folder_path)

def load_from_json(filename):
    """Load a dictionary from a JSON filename.

    Args:
        filename: The filename to load from.
    """
    with open(filename, encoding="UTF-8") as file:
        return json.load(file)


class CameraType(Enum):
    """Supported camera types."""

    PERSPECTIVE = auto()
    FISHEYE = auto()
    EQUIRECTANGULAR = auto()
    OMNIDIRECTIONALSTEREO_L = auto()
    OMNIDIRECTIONALSTEREO_R = auto()
    VR180_L = auto()
    VR180_R = auto()
    ORTHOPHOTO = auto()
    FISHEYE624 = auto()


CAMERA_MODEL_TO_TYPE = {
    "SIMPLE_PINHOLE": CameraType.PERSPECTIVE,
    "PINHOLE": CameraType.PERSPECTIVE,
    "SIMPLE_RADIAL": CameraType.PERSPECTIVE,
    "RADIAL": CameraType.PERSPECTIVE,
    "OPENCV": CameraType.PERSPECTIVE,
    "OPENCV_FISHEYE": CameraType.FISHEYE,
    "EQUIRECTANGULAR": CameraType.EQUIRECTANGULAR,
    "OMNIDIRECTIONALSTEREO_L": CameraType.OMNIDIRECTIONALSTEREO_L,
    "OMNIDIRECTIONALSTEREO_R": CameraType.OMNIDIRECTIONALSTEREO_R,
    "VR180_L": CameraType.VR180_L,
    "VR180_R": CameraType.VR180_R,
    "ORTHOPHOTO": CameraType.ORTHOPHOTO,
    "FISHEYE624": CameraType.FISHEYE624,
}


def _get_fname(filepath, data_dir, downsample_folder_prefix="images_"):
    """Get the filename of the image file.
    downsample_folder_prefix can be used to point to auxiliary image data, e.g. masks

    filepath: the base file name of the transformations.
    data_dir: the directory of the data that contains the transform file
    downsample_folder_prefix: prefix of the newly generated downsampled images
    """

    downscale_factor = 4
    return data_dir / f"{downsample_folder_prefix}{downscale_factor}" / filepath.name



def get_distortion_params(
    k1: float = 0.0,
    k2: float = 0.0,
    k3: float = 0.0,
    k4: float = 0.0,
    p1: float = 0.0,
    p2: float = 0.0,
):
    """Returns a distortion parameters matrix.

    Args:
        k1: The first radial distortion parameter.
        k2: The second radial distortion parameter.
        k3: The third radial distortion parameter.
        k4: The fourth radial distortion parameter.
        p1: The first tangential distortion parameter.
        p2: The second tangential distortion parameter.
    Returns:
        torch.Tensor: A distortion parameters matrix.
    """
    return [k1, k2, k3, k4, p1, p2]


# def parser(meta, data_dir):
#     data_dir = Path(data_dir)
#     image_filenames = []
#     mask_filenames = []
#     depth_filenames = []
#     poses = []

#     fx_fixed = "fl_x" in meta
#     fy_fixed = "fl_y" in meta
#     cx_fixed = "cx" in meta
#     cy_fixed = "cy" in meta
#     height_fixed = "h" in meta
#     width_fixed = "w" in meta
#     distort_fixed = False
#     for distort_key in ["k1", "k2", "k3", "p1", "p2", "distortion_params"]:
#         if distort_key in meta:
#             distort_fixed = True
#             break
#     fisheye_crop_radius = meta.get("fisheye_crop_radius", None)
#     fx = []
#     fy = []
#     cx = []
#     cy = []
#     height = []
#     width = []
#     distort = []

#     # sort the frames by fname
#     fnames = []
#     for frame in meta["frames"]:
#         filepath = Path(frame["file_path"])
#         fname = _get_fname(filepath, data_dir)
#         fnames.append(fname)
#     inds = np.argsort(fnames)
#     frames = [meta["frames"][ind] for ind in inds]

#     for frame in frames:
#         filepath = Path(frame["file_path"])
#         fname = _get_fname(filepath, data_dir)

#         if not fx_fixed:
#             assert "fl_x" in frame, "fx not specified in frame"
#             fx.append(float(frame["fl_x"]))
#         if not fy_fixed:
#             assert "fl_y" in frame, "fy not specified in frame"
#             fy.append(float(frame["fl_y"]))
#         if not cx_fixed:
#             assert "cx" in frame, "cx not specified in frame"
#             cx.append(float(frame["cx"]))
#         if not cy_fixed:
#             assert "cy" in frame, "cy not specified in frame"
#             cy.append(float(frame["cy"]))
#         if not height_fixed:
#             assert "h" in frame, "height not specified in frame"
#             height.append(int(frame["h"]))
#         if not width_fixed:
#             assert "w" in frame, "width not specified in frame"
#             width.append(int(frame["w"]))
#         if not distort_fixed:
#             distort.append(
#                 torch.tensor(frame["distortion_params"], dtype=torch.float32)
#                 if "distortion_params" in frame
#                 else get_distortion_params(
#                     k1=float(frame["k1"]) if "k1" in frame else 0.0,
#                     k2=float(frame["k2"]) if "k2" in frame else 0.0,
#                     k3=float(frame["k3"]) if "k3" in frame else 0.0,
#                     k4=float(frame["k4"]) if "k4" in frame else 0.0,
#                     p1=float(frame["p1"]) if "p1" in frame else 0.0,
#                     p2=float(frame["p2"]) if "p2" in frame else 0.0,
#                 )
#             )

#         image_filenames.append(fname)
#         poses.append(np.array(frame["transform_matrix"]))

#     poses = np.array(poses).astype(np.float32)
#     # Scale poses
#     scale_factor = 1.0
#     poses[:, :3, 3] *= scale_factor
#     indices = range(len(image_filenames))
#     # Choose image_filenames and poses based on split, but after auto orient and scaling the poses.
#     image_filenames = [image_filenames[i] for i in indices]
#     mask_filenames =  []
#     depth_filenames = []

#     idx_tensor = np.array(indices)
#     poses = poses[idx_tensor]

#     # in x,y,z order
#     # assumes that the scene is centered at the origin

#     if "camera_model" in meta:
#         camera_type = CAMERA_MODEL_TO_TYPE[meta["camera_model"]]
#     else:
#         camera_type = CameraType.PERSPECTIVE

#     fx = float(meta["fl_x"] * 1/4) if fx_fixed else torch.tensor(fx, dtype=torch.float32)[idx_tensor]
#     fy = float(meta["fl_y"] * 1/4) if fy_fixed else torch.tensor(fy, dtype=torch.float32)[idx_tensor]
#     cx = float(meta["cx"] * 1/4) if cx_fixed else torch.tensor(cx, dtype=torch.float32)[idx_tensor]
#     cy = float(meta["cy"] * 1/4) if cy_fixed else torch.tensor(cy, dtype=torch.float32)[idx_tensor]
#     height = int(meta["h"] * 1/4) if height_fixed else torch.tensor(height, dtype=torch.int32)[idx_tensor]
#     width = int(meta["w"] * 1/4) if width_fixed else torch.tensor(width, dtype=torch.int32)[idx_tensor]
#     # The naming is somewhat confusing, but:
#     # - transform_matrix contains the transformation to dataparser output coordinates from saved coordinates.
#     # - dataparser_transform_matrix contains the transformation to dataparser output coordinates from original data coordinates.
#     # - applied_transform contains the transformation to saved coordinates from original data coordinates.
#     applied_transform = None
#     if "applied_transform" in meta:
#         applied_transform = np.array(meta["applied_transform"])

#     if applied_transform is not None:
#         dataparser_transform_matrix = poses @ np.concatenate(
#             [applied_transform, np.array([[0, 0, 0, 1]])], 0
#         )
#     else:
#         dataparser_transform_matrix = transform_matrix
#     dataparser_transform_matrix[0:3, 1:3] *= -1
#     c2w_opencv = dataparser_transform_matrix
#     metadata = {}
#     # K = geometry.colmap_to_opencv_intrinsics(K)
#     # map1, map2 = cv2.initUndistortRectifyMap(K, distortion, np.eye(3), new_K, (width, height), cv2.CV_32FC1)
#     distortion_params = np.array([meta['k1'], meta['k2'], meta['p1'], meta['p2']])
#     #k2, k3, k4, p1, p2]

#     # because OpenCV expects the pixel coord to be top-left, we need to shift the principal point by 0.5
#     # see https://github.com/nerfstudio-project/nerfstudio/issues/3048
#     K = np.zeros((3, 3))
#     K[0, 0] = fx
#     K[1, 1] = fy
#     K[0, 2] = cx
#     K[1, 2] = cy
#     K[2, 2] = 1.0
#     K[0, 2] = K[0, 2] - 0.5
#     K[1, 2] = K[1, 2] - 0.5
#     newK, roi = cv2.getOptimalNewCameraMatrix(K, distortion_params, np.array((meta['w']//4, meta["h"]//4)), 0)
#     K[0, 0] = fx * 4
#     K[1, 1] = fy * 4
#     K[0, 2] = cx * 4
#     K[1, 2] = cy * 4
#     K[2, 2] = 1.0
#     K[0, 2] = K[0, 2] - 0.5
#     K[1, 2] = K[1, 2] - 0.5
#     newK1, roi = cv2.getOptimalNewCameraMatrix(K, distortion_params, np.array((meta['w'], meta["h"])), 0)
#     newK1[:2] = newK1[:2]/4
#     import ipdb; ipdb.set_trace()

#     image = cv2.undistort(image, K, distortion_params, None, newK)  # type: ignore
#     x, y, w, h = roi
#     newK[0, 2] -= x
#     newK[1, 2] -= y
#     newK[0, 2] = newK[0, 2] + 0.5
#     newK[1, 2] = newK[1, 2] + 0.5
#     K = newK
#     import ipdb; ipdb.set_trace()
#     return K, new_K, c2w_opencv, distortion

def find_closest_anchors(global_idx, anchor_idx, top_n=7):
    # 将anchor_idx转换为numpy数组
    anchor_array = np.array(anchor_idx)
    # 计算每个anchor与global_idx的距离
    distances = np.abs(anchor_array - global_idx)
    # 按距离从小到大排序，获取最小的top_n个索引
    closest_indices = np.argsort(distances)[:top_n]
    # 提取对应的anchor值
    closest_anchors = anchor_array[closest_indices]
    return closest_anchors.tolist()

class Own_seq(BaseStereoViewDataset_test):
    def __init__(self, *args, split, ROOT, only_pose=False, pose_est=False, **kwargs):
        self.ROOT = ROOT
        self.global_idx = 0
        self.images_list = glob.glob(osp.join(self.ROOT, '*.png')) + glob.glob(osp.join(self.ROOT, '*.jpg')) + glob.glob(osp.join(self.ROOT, '*.JPG'))
        self.images_list = sorted(self.images_list)
        self.anchor_idx = list(range(len(self.images_list)))[::8]
        # self.anchor_idx = list(range(len(self.images_list)))
        total_images = len(self.images_list)
        self.len = len(self.images_list)

        if pose_est == True:
            interval = total_images // 7  # 计算间隔
            selected_indices = [i * interval for i in range(7)]
            self.anchor_idx = selected_indices#[self.images_list[i] for i in selected_indices]
            middle_index = len(selected_indices) // 2
            middle_value = selected_indices[middle_index]
            selected_indices.remove(middle_value)  # 移除中间的数
            selected_indices.insert(0, middle_value)  # 将中间的数插入到第一个位置
            self.anchor_idx = selected_indices
            
            i = 0
            cont = 1
            while i < self.len:
                i = i + 8
                cont += 1
            self.len = cont
        self.pose_est = pose_est
        
        super().__init__(*args, **kwargs)

    def __len__(self):
        return self.len
    
    @staticmethod
    def image_read(image_file):
        img = cv2.imread(image_file)
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

  
    def _get_views(self, idx, resolution, rng):
        idx = self.global_idx
        min_idx = max(self.global_idx - 8 * 8, 0)
        max_idx = min(self.global_idx + 8 * 8, len(self.images_list))
        # images_list = [self.images_list[self.global_idx]] + self.anchor_idx
        if self.pose_est == False:
            anchor = [self.images_list[i] for i in find_closest_anchors(self.global_idx, self.anchor_idx)]
            anchor = sorted(anchor)
            images_list = [self.images_list[self.global_idx]] + anchor 
            self.global_idx = self.global_idx + 1
        else:
            index_list =  [self.global_idx] + self.anchor_idx
            index_list = [min(len(self.images_list) - 1, i) for i in index_list]
            images_list = [self.images_list[i] for i in index_list]
            self.global_idx = self.global_idx + 8
        views = []
        for image in images_list:
            rgb_image = self.image_read(image)
            H, W = rgb_image.shape[:2]
            intrinsics = np.array([[W, 0, W/2], [0, H, H/2], [0, 0, 1]])
            camera_pose = np.eye(4)
            depthmap = np.zeros((H, W))
            rgb_image, depthmap, intrinsics = self._crop_resize_if_necessary(
                rgb_image, depthmap, intrinsics, resolution, rng=rng, info=None)
            rgb_image_orig = rgb_image.copy()
            H, W = depthmap.shape[:2]
            fxfycxcy = np.array([intrinsics[0, 0]/W, intrinsics[1, 1]/H, intrinsics[0,2]/W, intrinsics[1,2]/H]).astype(np.float32)
            views.append(dict(
                img_org=rgb_image_orig,
                img=rgb_image,
                depthmap=depthmap.astype(np.float32),
                camera_pose=camera_pose.astype(np.float32),
                camera_intrinsics=intrinsics.astype(np.float32),
                fxfycxcy=fxfycxcy,
                dataset='own',
                label=image,
                instance=image,
                pose_est=self.pose_est
            ))
        
        return views


if __name__ == "__main__":
    from dust3r.datasets.base.base_stereo_view_dataset import view_name
    from dust3r.viz import SceneViz, auto_cam_size
    from dust3r.utils.image import rgb
    import nerfvis.scene as scene_vis 
    dataset = DL3DV(split='train', ROOT="/data0/zsz/mast3recon/data/DL3DV",meta='data/DL3DV_metadata.npz',resolution=[(512, 384)], aug_crop=16)

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
        # scene_vis.add_points("points", pts3ds.reshape(-1,3)[valid_masks.reshape(-1)], vert_color=colors.reshape(-1,3)[valid_masks.reshape(-1)], point_size=1)
        # for i in range(len(c2ws)):
        f = 1111.0 / 2.5
        z = 10.
        scene_vis.add_camera_frustum("cameras", r=c2ws[:, :3, :3], t=c2ws[:, :3, 3], focal_length=f,
                        image_width=colors.shape[2], image_height=colors.shape[1],
                        z=z, connect=False, color=[1.0, 0.0, 0.0])
        for i in range(len(c2ws)):
            scene_vis.add_image(
                            f"images/{i}",
                            colors[i], # Can be a list of paths too (requires joblib for that) 
                            r=c2ws[i, :3, :3],
                            t=c2ws[i, :3, 3],
                            # Alternatively: from nerfvis.utils import split_mat4; **split_mat4(c2ws)
                            focal_length=f,
                            z=z,
                        )
        scene_vis.display(port=8081)

