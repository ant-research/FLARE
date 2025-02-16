# Copyright (C) 2024-present Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).
#
# --------------------------------------------------------
# Implementation of DUSt3R training losses
# --------------------------------------------------------
from copy import copy, deepcopy
import torch
import torch.nn as nn
import random
import torch.nn.functional as F
import mast3r.utils.path_to_dust3r  # noqa
from dust3r.inference import get_pred_pts3d, find_opt_scaling
from dust3r.utils.geometry import inv, geotrf, normalize_pointcloud#, matrix_to_quaternion, quaternion_to_matrix
from dust3r.utils.geometry import get_joint_pointcloud_depth, get_joint_pointcloud_center_scale
from dust3r.renderers.gaussian_renderer import GaussianRenderer
from dust3r.renderers.gaussian_utils import GRMGaussianModel, render_image
from dust3r.utils.triangulation import triangulate_tracks
import numpy as np
import bisect
from dust3r.utils.misc import transpose_to_landscape_render, transpose_to_landscape  # noqa
from pytorch3d.transforms.rotation_conversions import matrix_to_quaternion, quaternion_to_matrix
import pytorch3d.transforms
# from torch_batch_svd import svd
import roma
from torch.cuda.amp import autocast
import nerfvis.scene as scene_vis
import math
from mast3r.fast_nn import extract_correspondences_nonsym, bruteforce_reciprocal_nns
import poselib
import pycolmap
# from dust3r_visloc.localization import run_pnp
from mast3r.cloud_opt.triangulation import batched_triangulate
import viser
import viser.transforms as tf
from tqdm import tqdm
from dust3r.depth_loss import ScaleAndShiftInvariantLoss
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp

def interpolate_pose(pose1, pose2, t):
    """
    Interpolate between two camera poses (4x4 matrices).

    :param pose1: First pose (4x4 matrix)
    :param pose2: Second pose (4x4 matrix)
    :param t: Interpolation factor, t in [0, 1]
    :return: Interpolated pose (4x4 matrix)
    """

    # Extract translation and rotation from both poses
    translation1 = pose1[:3, 3].detach().cpu().numpy()
    translation2 = pose2[:3, 3].detach().cpu().numpy()
    rotation1 = pose1[:3, :3].detach().cpu().numpy()
    rotation2 = pose2[:3, :3].detach().cpu().numpy()

    # Interpolate the translation (linear interpolation)
    interpolated_translation = (1 - t) * translation1 + t * translation2
    
    # Convert rotation matrices to quaternions
    quat1 = R.from_matrix(rotation1).as_quat()
    quat2 = R.from_matrix(rotation2).as_quat()

    # Slerp for rotation interpolation
    slerp = Slerp([0, 1], R.from_quat([quat1, quat2]))

    interpolated_rotation = slerp(t).as_matrix()
    # Combine the interpolated rotation and translation
    interpolated_pose = np.eye(4)
    interpolated_pose[:3, :3] = interpolated_rotation
    interpolated_pose[:3, 3] = interpolated_translation
    return interpolated_pose

def rotation_6d_to_matrix(d6):
    """
    Converts 6D rotation representation by Zhou et al. [1] to rotation matrix
    using Gram--Schmidt orthogonalization per Section B of [1]. Adapted from pytorch3d.
    Args:
        d6: 6D rotation representation, of size (*, 6)

    Returns:
        batch of rotation matrices of size (*, 3, 3)

    [1] Zhou, Y., Barnes, C., Lu, J., Yang, J., & Li, H.
    On the Continuity of Rotation Representations in Neural Networks.
    IEEE Conference on Computer Vision and Pattern Recognition, 2019.
    Retrieved from http://arxiv.org/abs/1812.07035
    """

    a1, a2 = d6[..., :3], d6[..., 3:]
    b1 = F.normalize(a1, dim=-1)
    b2 = a2 - (b1 * a2).sum(-1, keepdim=True) * b1
    b2 = F.normalize(b2, dim=-1)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack((b1, b2, b3), dim=-2)

def vis(poses, images, points3d, colors, fxfycxcy, H, W):
    server = viser.ViserServer()
    server.gui.configure_theme(titlebar_content=None, control_layout="collapsible")
    # Load the colmap info.
    gui_reset_up = server.gui.add_button(
        "Reset up direction",
        hint="Set the camera control 'up' direction to the current camera's 'up'.",
    )

    @gui_reset_up.on_click
    def _(event: viser.GuiEvent) -> None:
        client = event.client
        assert client is not None
        client.camera.up_direction = tf.SO3(client.camera.wxyz) @ np.array([0.0, -1.0, 0.0])

    gui_points = server.gui.add_slider(
        "Max points",
        min=1,
        max=len(points3d),
        step=1,
        initial_value=min(len(points3d), 50_000),
    )
    gui_frames = server.gui.add_slider(
        "Max frames",
        min=1,
        max=len(images),
        step=1,
        initial_value=min(len(images), 100),
    )
    gui_point_size = server.gui.add_slider(
        "Point size", min=0.0001, max=0.05, step=0.0001, initial_value=0.001
    )

    # points = points3d.detach().cpu().numpy()
    # colors = colors.detach().cpu().numpy()
    poses, images, points3d, colors, fxfycxcy = poses.detach().cpu().numpy(), images.detach().cpu().numpy(), points3d.detach().cpu().numpy(), colors, fxfycxcy.detach().cpu().numpy()
    points = points3d.reshape(-1,3)
    colors = colors.reshape(-1,3)
    point_mask = np.random.choice(points.shape[0], gui_points.value, replace=False)
    point_cloud = server.scene.add_point_cloud(
        name="/colmap/pcd",
        points=points[point_mask],
        colors=colors[point_mask] ,
        point_shape='circle',
        point_size=gui_point_size.value,
    )
    frames: List[viser.FrameHandle] = []

    def visualize_frames() -> None:
        """Send all COLMAP elements to viser for visualization. This could be optimized
        a ton!"""

        # Remove existing image frames.
        for frame in frames:
            frame.remove()
        frames.clear()

        # Interpret the images and cameras.
        # img_ids = [im.id for im in images.values()]
        # random.shuffle(img_ids)
        # img_ids = sorted(img_ids[: gui_frames.value])
        img_ids = range(len(images))
        def attach_callback(
            frustum: viser.CameraFrustumHandle, frame: viser.FrameHandle
        ) -> None:
            @frustum.on_click
            def _(_) -> None:
                for client in server.get_clients().values():
                    client.camera.wxyz = frame.wxyz
                    client.camera.position = frame.position

        for img_id in tqdm(img_ids):
            cam = poses[img_id]
            T_world_camera = tf.SE3.from_rotation_and_translation(
                tf.SO3.from_matrix(cam[:3,:3]), cam[:3,3]
            )
            frame = server.scene.add_frame(
                f"/colmap/frame_{img_id}",
                wxyz=T_world_camera.rotation().wxyz,
                position=T_world_camera.translation(),
                axes_length=0,
                axes_radius=0,
            )
            frames.append(frame)

            # For pinhole cameras, cam.params will be (fx, fy, cx, cy).
            fy = fxfycxcy[img_id, 1]
            # fy = cam.params[1]
            image = images[img_id]
            # image = image[::downsample_factor, ::downsample_factor]
            frustum = server.scene.add_camera_frustum(
                f"/colmap/frame_{img_id}/frustum",
                fov=2 * np.arctan2(1 / 2, fy),
                aspect=W / H,
                scale=0.1,
                image=image,
                color=[255, 0., 0.],
            )
            attach_callback(frustum, frame)

    need_update = True

    @gui_points.on_update
    def _(_) -> None:
        point_mask = np.random.choice(points.shape[0], gui_points.value, replace=False)
        point_cloud.points = points[point_mask]
        point_cloud.colors = colors[point_mask]

    @gui_frames.on_update
    def _(_) -> None:
        nonlocal need_update
        need_update = True

    @gui_point_size.on_update
    def _(_) -> None:
        point_cloud.point_size = gui_point_size.value
    # try:
    import time
    start = time.time()
    while True:
        if need_update:
            need_update = False
            visualize_frames()
        time.sleep(1e-3)


        


def plucker_embedder(h, w, camera, intrinsics):        
    b, v, _, _ = camera.size()
    c2w = camera.reshape(b,v,-1)[:, :, :16]
    fxfycxcy = intrinsics
    c2w = c2w.reshape(b * v, 4, 4)
    fxfycxcy = fxfycxcy.reshape(b * v, 4)
    y, x = torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij")
    y, x = y.to(camera.device), x.to(camera.device)
    x = x[None, :, :].expand(b * v, -1, -1).reshape(b * v, -1) / w
    y = y[None, :, :].expand(b * v, -1, -1).reshape(b * v, -1) / h
    x = (x  - fxfycxcy[:, 2:3]) / fxfycxcy[:, 0:1]
    y = (y  - fxfycxcy[:, 3:4]) / fxfycxcy[:, 1:2]
    z = torch.ones_like(x)
    ray_d = torch.stack([x, y, z], dim=2)  # [b*v, h*w, 3]
    ray_d = torch.bmm(ray_d, c2w[:, :3, :3].transpose(1, 2))  # [b*v, h*w, 3]
    ray_d = ray_d / torch.norm(ray_d, dim=2, keepdim=True)  # [b*v, h*w, 3]
    ray_o = c2w[:, :3, 3][:, None, :].expand_as(ray_d)  # [b*v, h*w, 3]
    ray_o = ray_o.reshape(b, v, h, w, 3).permute(0, 1, 4, 2, 3)
    ray_d = ray_d.reshape(b, v, h, w, 3).permute(0, 1, 4, 2, 3)
    return ray_d 

def reproj2d(fxfy, c2w, pts3d, true_shape):
    # undebug
    dtype = fxfy.dtype
    device = fxfy.device
    H, W = true_shape
    assert pts3d.shape[-2] == true_shape[-1]
    # K = torch.eye(3, dtype=dtype, device=device)[None].expand(len(fxfy), 3, 3).clone()
    # K[:, 0, 0] = fxfy[:,0] * W
    # K[:, 1, 1] = fxfy[:,1] * H
    # K[:, 0:2, 2] = fxfy[:,2:] * torch.tensor([W, H], dtype=dtype, device=device)[None].repeat(len(fxfy), 1)
    # proj_matrix = K @ w2cam[:, :3]
    # res = (pts3d.reshape(len(fxfy),-1,3) @ proj_matrix[:,:3, :3].transpose(-1, -2)) + proj_matrix[:,:3, 3][:,None]
    # clipped_z = res[..., 2:3].clip(min=1e-3)  # make sure we don't have nans!
    # uv = res[..., 0:2] / clipped_z
    # x, y = torch.meshgrid(torch.arange(W), torch.arange(H), indexing='xy')
    # xy = torch.stack((x, y), dim=-1).to(dtype=dtype, device=device)
    # R_cam2world = c2w[:, :3, :3]
    t_cam2world = c2w[:, :3, 3]
    pred_ray = pts3d - t_cam2world[:, None, None]
    pred_ray = pred_ray / (torch.norm(pred_ray, dim=-1, keepdim=True) + 1e-6)
    ray_d = plucker_embedder(H, W, c2w[None], fxfy)
    ray_d = ray_d.squeeze(0).permute(0, 2, 3, 1)
    # H, W = X_cam_v2.shape[1:3]
    # u_gt, v_gt = torch.meshgrid(torch.arange(W).to(R_cam2world), torch.arange(H).to(R_cam2world), indexing='xy')
    # z_cam = X_cam_v2[..., 2]
    # fu, fv, cu, cv = fxfy[:, 0] * W, fxfy[:, 1] * H, fxfy[:, 2] * W, fxfy[:, 3] * H
    # u1 = X_cam_v2[...,0] * fu[:,None,None] / z_cam + cu[:,None,None] 
    # v1 = X_cam_v2[...,1] * fv[:,None,None] / z_cam + cv[:,None,None] 
    # uv = torch.stack((u1, v1), dim=-1)
    # uv_gt = torch.stack((u_gt, v_gt), dim=-1)
    return torch.abs(pred_ray - ray_d) * 5

def interpolate_poses(pose_timestamps, abs_poses, requested_timestamps, origin_timestamp):
    """Interpolate between absolute poses.
    Args:
        pose_timestamps (list[int]): Timestamps of supplied poses. Must be in ascending order.
        abs_poses (list[numpy.matrixlib.defmatrix.matrix]): SE3 matrices representing poses at the timestamps specified.
        requested_timestamps (list[int]): Timestamps for which interpolated timestamps are required.
        origin_timestamp (int): UNIX timestamp of origin frame. Poses will be reported relative to this frame.
    Returns:
        list[numpy.matrixlib.defmatrix.matrix]: SE3 matrix representing interpolated pose for each requested timestamp.
    Raises:
        ValueError: if pose_timestamps and abs_poses are not the same length
        ValueError: if pose_timestamps is not in ascending order
    """
    requested_timestamps.insert(0, origin_timestamp)
    requested_timestamps = np.array(requested_timestamps)
    pose_timestamps = np.array(pose_timestamps)
    if len(pose_timestamps) != len(abs_poses):
        raise ValueError('Must supply same number of timestamps as poses')

    abs_quaternions = np.zeros((4, len(abs_poses)))
    abs_positions = np.zeros((3, len(abs_poses)))
    for i, pose in enumerate(abs_poses):
        if i > 0 and pose_timestamps[i - 1] >= pose_timestamps[i]:
            raise ValueError('Pose timestamps must be in ascending order')

        abs_quaternions[:, i] = pose[
                                3:]  # np.roll(pose[3:], -1) uncomment this if the quaternion is saved as [w, x, y, z]
        abs_positions[:, i] = pose[:3]

    upper_indices = [bisect.bisect(pose_timestamps, pt) for pt in requested_timestamps]
    lower_indices = [u - 1 for u in upper_indices]

    if max(upper_indices) >= len(pose_timestamps):
        upper_indices = [min(i, len(pose_timestamps) - 1) for i in upper_indices]
    fractions = (requested_timestamps - pose_timestamps[lower_indices]) / (pose_timestamps[upper_indices] - pose_timestamps[lower_indices])

    quaternions_lower = abs_quaternions[:, lower_indices]
    quaternions_upper = abs_quaternions[:, upper_indices]

    d_array = (quaternions_lower * quaternions_upper).sum(0)

    linear_interp_indices = np.nonzero(d_array >= 1)
    sin_interp_indices = np.nonzero(d_array < 1)

    scale0_array = np.zeros(d_array.shape)
    scale1_array = np.zeros(d_array.shape)

    scale0_array[linear_interp_indices] = 1 - fractions[linear_interp_indices]
    scale1_array[linear_interp_indices] = fractions[linear_interp_indices]

    theta_array = np.arccos(np.abs(d_array[sin_interp_indices]))

    scale0_array[sin_interp_indices] = \
        np.sin((1 - fractions[sin_interp_indices]) * theta_array) / np.sin(theta_array)
    scale1_array[sin_interp_indices] = \
        np.sin(fractions[sin_interp_indices] * theta_array) / np.sin(theta_array)

    negative_d_indices = np.nonzero(d_array < 0)
    scale1_array[negative_d_indices] = -scale1_array[negative_d_indices]

    quaternions_interp = np.tile(scale0_array, (4, 1)) * quaternions_lower \
                         + np.tile(scale1_array, (4, 1)) * quaternions_upper

    positions_lower = abs_positions[:, lower_indices]
    positions_upper = abs_positions[:, upper_indices]

    positions_interp = np.multiply(np.tile((1 - fractions), (3, 1)), positions_lower) \
                       + np.multiply(np.tile(fractions, (3, 1)), positions_upper)

    poses_mat = np.zeros((4, 4 * len(requested_timestamps)))

    poses_mat[0, 0::4] = 1 - 2 * np.square(quaternions_interp[2, :]) - \
                         2 * np.square(quaternions_interp[3, :])
    poses_mat[0, 1::4] = 2 * np.multiply(quaternions_interp[1, :], quaternions_interp[2, :]) - \
                         2 * np.multiply(quaternions_interp[3, :], quaternions_interp[0, :])
    poses_mat[0, 2::4] = 2 * np.multiply(quaternions_interp[1, :], quaternions_interp[3, :]) + \
                         2 * np.multiply(quaternions_interp[2, :], quaternions_interp[0, :])

    poses_mat[1, 0::4] = 2 * np.multiply(quaternions_interp[1, :], quaternions_interp[2, :]) \
                         + 2 * np.multiply(quaternions_interp[3, :], quaternions_interp[0, :])
    poses_mat[1, 1::4] = 1 - 2 * np.square(quaternions_interp[1, :]) \
                         - 2 * np.square(quaternions_interp[3, :])
    poses_mat[1, 2::4] = 2 * np.multiply(quaternions_interp[2, :], quaternions_interp[3, :]) - \
                         2 * np.multiply(quaternions_interp[1, :], quaternions_interp[0, :])

    poses_mat[2, 0::4] = 2 * np.multiply(quaternions_interp[1, :], quaternions_interp[3, :]) - \
                         2 * np.multiply(quaternions_interp[2, :], quaternions_interp[0, :])
    poses_mat[2, 1::4] = 2 * np.multiply(quaternions_interp[2, :], quaternions_interp[3, :]) + \
                         2 * np.multiply(quaternions_interp[1, :], quaternions_interp[0, :])
    poses_mat[2, 2::4] = 1 - 2 * np.square(quaternions_interp[1, :]) - \
                         2 * np.square(quaternions_interp[2, :])

    poses_mat[0:3, 3::4] = positions_interp
    poses_mat[3, 3::4] = 1

    poses_mat = np.linalg.solve(poses_mat[0:4, 0:4], poses_mat)

    poses_out = []
    for i in range(1, len(requested_timestamps)):
        pose_mat = poses_mat[0:4, i * 4:(i + 1) * 4]
        pose_rot = pose_mat.copy()
        pose_rot[:3, -1] = 0
        pose_rot[-1, :3] = 0
        pose_position = pose_mat[:3, -1]
        pose_rot[:3, -1] = pose_position
        # pose_quaternion = matrix_to_quaternion(torch.tensor(pose_rot)-)  # [w x y z]
        poses_out.append(pose_rot)
        # poses_out[i - 1] = poses_mat[0:4, i * 4:(i + 1) * 4]

    return poses_out


def Sum(*losses_and_masks):
    loss, mask = losses_and_masks[0]
    if loss.ndim > 0:
        # we are actually returning the loss for every pixels
        return losses_and_masks
    else:
        # we are returning the global loss
        for loss2, mask2 in losses_and_masks[1:]:
            loss = loss + loss2
        return loss


def rotation_angle(rot_gt, rot_pred, batch_size=None, eps=1e-15):
    #########
    q_pred = matrix_to_quaternion(rot_pred)
    q_gt = matrix_to_quaternion(rot_gt)

    loss_q = (1 - (q_pred * q_gt).sum(dim=1) ** 2).clamp(min=eps)
    err_q = torch.arccos(1 - 2 * loss_q)

    rel_rangle_deg = err_q * 180 / np.pi

    if batch_size is not None:
        rel_rangle_deg = rel_rangle_deg.reshape(batch_size, -1)

    return rel_rangle_deg

class BaseCriterion(nn.Module):
    def __init__(self, reduction='mean'):
        super().__init__()
        self.reduction = reduction


class LLoss (BaseCriterion):
    """ L-norm loss
    """

    def forward(self, a, b):
        assert a.shape == b.shape and a.ndim >= 2 and 1 <= a.shape[-1] <= 3, f'Bad shape = {a.shape}'
        dist = self.distance(a, b)
        assert dist.ndim == a.ndim - 1  # one dimension less
        if self.reduction == 'none':
            return dist
        if self.reduction == 'sum':
            return dist.sum()
        if self.reduction == 'mean':
            return dist.mean() if dist.numel() > 0 else dist.new_zeros(())
        raise ValueError(f'bad {self.reduction=} mode')

    def distance(self, a, b):
        raise NotImplementedError()


class L21Loss (LLoss):
    """ Euclidean distance between 3d points  """

    def distance(self, a, b):
        return torch.norm(a - b, dim=-1)  # normalized L2 distance


L21 = L21Loss()


class Criterion (nn.Module):
    def __init__(self, criterion=None):
        super().__init__()
        assert isinstance(criterion, BaseCriterion), f'{criterion} is not a proper criterion!'
        self.criterion = copy(criterion)

    def get_name(self):
        return f'{type(self).__name__}({self.criterion})'

    def with_reduction(self, mode='none'):
        res = loss = deepcopy(self)
        while loss is not None:
            assert isinstance(loss, Criterion)
            loss.criterion.reduction = mode  # make it return the loss for each sample
            loss = loss._loss2  # we assume loss is a Multiloss
        return res


class MultiLoss (nn.Module):
    """ Easily combinable losses (also keep track of individual loss values):
        loss = MyLoss1() + 0.1*MyLoss2()
    Usage:
        Inherit from this class and override get_name() and compute_loss()
    """

    def __init__(self):
        super().__init__()
        self._alpha = 1
        self._loss2 = None

    def compute_loss(self, *args, **kwargs):
        raise NotImplementedError()

    def get_name(self):
        raise NotImplementedError()

    def __mul__(self, alpha):
        assert isinstance(alpha, (int, float))
        res = copy(self)
        res._alpha = alpha
        return res
    __rmul__ = __mul__  # same

    def __add__(self, loss2):
        assert isinstance(loss2, MultiLoss)
        res = cur = copy(self)
        # find the end of the chain
        while cur._loss2 is not None:
            cur = cur._loss2
        cur._loss2 = loss2
        return res

    def __repr__(self):
        name = self.get_name()
        if self._alpha != 1:
            name = f'{self._alpha:g}*{name}'
        if self._loss2:
            name = f'{name} + {self._loss2}'
        return name

    def forward(self, *args, **kwargs):
        loss = self.compute_loss(*args, **kwargs)
        if isinstance(loss, tuple) and len(loss) == 2:
            loss, details = loss
            monitoring = None
        elif isinstance(loss, tuple) and len(loss) == 3:
            loss, details, monitoring = loss
        elif loss.ndim == 0:
            details = {self.get_name(): float(loss)}
        else:
            details = {}
        loss = loss * self._alpha

        if self._loss2:
            if isinstance(loss, tuple) and len(loss) == 2:
                loss2, details2 = self._loss2(*args, **kwargs)
                loss = loss + loss2
                details |= details2
                monitoring = None
            elif isinstance(loss, tuple) and len(loss) == 3:
                loss2, details2, monitoring2 = self._loss2(*args, **kwargs)
                loss = loss + loss2
                details |= details2
                monitoring |= monitoring2

        return loss, details, monitoring


def rotation_distance(R1,R2,eps=1e-7):
    # http://www.boris-belousov.net/2016/12/01/quat-dist/
    R_diff = R1@R2.transpose(-2,-1)
    trace = R_diff[...,0,0]+R_diff[...,1,1]+R_diff[...,2,2]
    angle = ((trace-1)/2).clamp(-1+eps,1-eps).acos_() # numerical stability near -1/+1
    return angle

class RandomCrop(object):
    """Crop the given PIL Image at a random location.

    Args:
        size (sequence or int): Desired output size of the crop. If size is an
            int instead of sequence like (h, w), a square crop (size, size) is
            made.
        padding (int or sequence, optional): Optional padding on each border
            of the image. Default is 0, i.e no padding. If a sequence of length
            4 is provided, it is used to pad left, top, right, bottom borders
            respectively.
        pad_if_needed (boolean): It will pad the image if smaller than the
            desired size to avoid raising an exception.
    """

    def __init__(self, size, padding=0, pad_if_needed=False):
        self.size = (int(size), int(size))
        self.padding = padding
        self.pad_if_needed = pad_if_needed

    @staticmethod
    def get_params(img, output_size):
        """Get parameters for ``crop`` for a random crop.

        Args:
            img (PIL Image): Image to be cropped.
            output_size (tuple): Expected output size of the crop.

        Returns:
            tuple: params (i, j, h, w) to be passed to ``crop`` for random crop.
        """
        h,w = img.shape[-2], img.shape[-1]
        th, tw = output_size
        if w == tw and h == th:
            return 0, 0, h, w

        i = random.randint(0, h - th)
        j = random.randint(0, w - tw)
        return i, j, th, tw

    def __call__(self, img, img_gt):
        """
        Args:
            img (PIL Image): Image to be cropped.

        Returns:
            PIL Image: Cropped image.
        """
        if self.padding > 0:
            img = F.pad(img, self.padding)

        # pad the width if needed
        if self.pad_if_needed and img.size[0] < self.size[1]:
            img = F.pad(img, (int((1 + self.size[1] - img.size[0]) / 2), 0))
        # pad the height if needed
        if self.pad_if_needed and img.size[1] < self.size[0]:
            img = F.pad(img, (0, int((1 + self.size[0] - img.size[1]) / 2)))

        i, j, h, w = self.get_params(img, self.size)
        img, img_gt = img[..., i:i+self.size[0],j:j+self.size[1]],img_gt[..., i:i+self.size[0],j:j+self.size[1]]
        return img, img_gt
        # return F.crop(img, i, j, h, w),F.crop(img_gt, i, j, h, w)

    def __repr__(self):
        return self.__class__.__name__ + '(size={0}, padding={1})'.format(self.size, self.padding)


def apply_log_to_norm(xyz):
    d = xyz.norm(dim=-1, keepdim=True)
    xyz = xyz / d.clip(min=1e-8)
    xyz = xyz * torch.log1p(d)
    return xyz


def compare_translation_by_angle(t_gt, t, eps=1e-15, default_err=1e6):
    """Normalize the translation vectors and compute the angle between them."""
    t_norm = torch.norm(t, dim=1, keepdim=True)
    t = t / (t_norm + eps)

    t_gt_norm = torch.norm(t_gt, dim=1, keepdim=True)
    t_gt = t_gt / (t_gt_norm + eps)

    loss_t = torch.clamp_min(1.0 - torch.sum(t * t_gt, dim=1) ** 2, eps)
    err_t = torch.acos(torch.sqrt(1 - loss_t))

    err_t[torch.isnan(err_t) | torch.isinf(err_t)] = default_err
    return err_t

def translation_angle(tvec_gt, tvec_pred, batch_size=None, ambiguity=True):
    # tvec_gt, tvec_pred (B, 3,)
    rel_tangle_deg = compare_translation_by_angle(tvec_gt, tvec_pred)
    rel_tangle_deg = rel_tangle_deg * 180.0 / np.pi
    if ambiguity:
        rel_tangle_deg = torch.min(rel_tangle_deg, (180 - rel_tangle_deg).abs())

    if batch_size is not None:
        rel_tangle_deg = rel_tangle_deg.reshape(batch_size, -1)

    return rel_tangle_deg



def batched_all_pairs(B, N):
    # B, N = se3.shape[:2]
    i1_, i2_ = torch.combinations(torch.arange(N), 2, with_replacement=False).unbind(-1)
    i1, i2 = [(i[None] + torch.arange(B)[:, None] * N).reshape(-1) for i in [i1_, i2_]]

    return i1, i2


# def closed_form_inverse(se3, R=None, T=None):
#     """
#     Computes the inverse of each 4x4 SE3 matrix in the batch.

#     Args:
#     - se3 (Tensor): Nx4x4 tensor of SE3 matrices.

#     Returns:
#     - Tensor: Nx4x4 tensor of inverted SE3 matrices.


#     | R t |
#     | 0 1 |
#     -->
#     | R^T  -R^T t|
#     | 0       1  |
#     """
#     if R is None:
#         R = se3[:, :3, :3]

#     if T is None:
#         T = se3[:, :3, 3:]

#     # Compute the transpose of the rotation
#     R_transposed = R.transpose(1, 2)

#     # -R^T t
#     top_right = -R_transposed.bmm(T)

#     inverted_matrix = torch.eye(4, 4)[None].repeat(len(se3), 1, 1)
#     inverted_matrix = inverted_matrix.to(R.dtype).to(R.device)

#     inverted_matrix[:, :3, :3] = R_transposed
#     inverted_matrix[:, :3, 3:] = top_right

#     return inverted_matrix

def closed_form_inverse(se3, R=None, T=None):
    """
    Computes the inverse of each 4x4 SE3 matrix in the batch.

    Args:
    - se3 (Tensor): Nx4x4 tensor of SE3 matrices.

    Returns:
    - Tensor: Nx4x4 tensor of inverted SE3 matrices.


    | R t |
    | 0 1 |
    -->
    | R^T  -R^T t|
    | 0       1  |
    """
    if R is None:
        R = se3[:, :3, :3]

    if T is None:
        T = se3[:, :3, 3:]

    # Compute the transpose of the rotation
    R_transposed = R.transpose(1, 2)

    # -R^T t
    top_right = -R_transposed.bmm(T)

    inverted_matrix = torch.eye(4, 4)[None].repeat(len(se3), 1, 1)
    inverted_matrix = inverted_matrix.to(R.dtype).to(R.device)

    inverted_matrix[:, :3, :3] = R_transposed
    inverted_matrix[:, :3, 3:] = top_right

    return inverted_matrix





class RVQ(MultiLoss):
    """ Ensure that all 3D points are correct.
        Asymmetric loss: view1 is supposed to be the anchor.

        P1 = RT1 @ D1
        P2 = RT2 @ D2
        loss1 = (I @ pred_D1) - (RT1^-1 @ RT1 @ D1)
        loss2 = (RT21 @ pred_D2) - (RT1^-1 @ P2)
              = (RT21 @ pred_D2) - (RT1^-1 @ RT2 @ D2)
    """

    def __init__(self,):
        super().__init__()
        self.epoch = 0

    def compute_loss(self, outputs, trajectory_pred, trajectory_gt, **kw):
        loss_rec_lat = outputs['loss_total']
        loss_recon = outputs['loss_recon']
        with torch.no_grad():
            batch_size = trajectory_pred.shape[0]
            R = quaternion_to_matrix(trajectory_pred[..., 3:].reshape(-1, 4)).reshape(batch_size, -1, 3, 3)
            R_gt = quaternion_to_matrix(trajectory_gt[..., 3:].reshape(-1, 4)).reshape(batch_size, -1, 3, 3)
            se3_gt = torch.cat((R_gt, trajectory_gt[..., :3][..., None]), dim=-1).reshape(batch_size, -1 , 3, 4)
            se3_pred = torch.cat((R, trajectory_pred[..., :3][..., None]), dim=-1).reshape(batch_size, -1 , 3, 4)
            rot_err = torch.rad2deg(rotation_distance(R.reshape(*R_gt.shape).float(), R_gt.float())).mean()
            se3_gt = se3_gt[0]
            se3_pred = se3_pred[0]
            pair_idx_i1, pair_idx_i2 = batched_all_pairs(1, 7)
            bottom_ = torch.tensor([[[0,0,0,1]]]).to(se3_gt.device)
            se3_gt = torch.cat((se3_gt, bottom_.repeat(se3_gt.shape[0],1,1)), dim=1)
            se3_pred = torch.cat((se3_pred, bottom_.repeat(se3_pred.shape[0],1,1)), dim=1)
            relative_pose_gt = closed_form_inverse(se3_gt[pair_idx_i1]).bmm(se3_gt[pair_idx_i2])
            relative_pose_pred = closed_form_inverse(se3_pred[pair_idx_i1]).bmm(se3_pred[pair_idx_i2])
            rel_tangle_deg = translation_angle(relative_pose_gt[:, :3, 3], relative_pose_pred[:, :3, 3]).mean()
        return loss_rec_lat, dict(loss_recon=float(loss_recon), loss_rec_lat=float(loss_rec_lat), rot_err=float(rot_err), rel_tangle_deg=float(rel_tangle_deg))



class Regr3D_clean_SA(Criterion, MultiLoss):
    """ Ensure that all 3D points are correct.
        Asymmetric loss: view1 is supposed to be the anchor.

        P1 = RT1 @ D1
        P2 = RT2 @ D2
        loss1 = (I @ pred_D1) - (RT1^-1 @ RT1 @ D1)
        loss2 = (RT21 @ pred_D2) - (RT1^-1 @ P2)
              = (RT21 @ pred_D2) - (RT1^-1 @ RT2 @ D2)
    """

    def __init__(self, criterion, norm_mode='avg_dis', disable_rigid=True, gt_scale=False, test=False):
        super().__init__(criterion)
        self.norm_mode = norm_mode
        self.gt_scale = gt_scale
        self.epoch = 0
        self.test = test
        self.gt_num_image = 0
        self.disable_rigid = disable_rigid

    def get_all_pts3d(self, gt1, gt2, pred1, pred2, trajectory_pred, dist_clip=None, render_gt=None):
        pr_ref = pred1['pts3d']
        pr_stage2 = pred2['pts3d']
        B, num_views, H, W, _ = pr_ref.shape
        num_views_src = num_views - 1
        # pr_pts1 = pr_pts1.view(B, num_views, H, W, 3)
        # GT pts3d
        gt_pts3d = torch.stack([gt1_per['pts3d'] for gt1_per in gt1 + gt2], dim=1)
        B, _, H, W, _ = gt_pts3d.shape
        trajectory = torch.stack([gt1_per['camera_pose'] for gt1_per in gt1 + gt2], dim=1)
        in_camera1 = inv(trajectory[:, :1])
        # import ipdb; ipdb.set_trace()
        # trajectory_1 = closed_form_inverse(camera_pose.repeat(1, trajectory.shape[1],1,1).reshape(-1,4,4)).bmm(trajectory.reshape(-1,4,4))
        trajectory = torch.einsum('bnjk,bnkl->bnjl', in_camera1.repeat(1, trajectory.shape[1],1,1), trajectory)
        # import ipdb; ipdb.set_trace()
        trajectory_t_gt = trajectory[..., :3, 3].clone()
        trajectory_t_gt = trajectory_t_gt / (trajectory_t_gt.norm(dim=-1, keepdim=True).mean(dim=1, keepdim=True) + 1e-5)
        trajectory_normalize = trajectory.clone().detach()
        trajectory_normalize[..., :3, 3] = trajectory_t_gt
        quaternion_R = matrix_to_quaternion(trajectory[...,:3,:3])
        trajectory_gt = torch.cat([trajectory_t_gt, quaternion_R], dim=-1)[None]
        R = trajectory[...,:3,:3]
        fxfycxcy1 = torch.stack([view['fxfycxcy'] for view in gt1], dim=1).float()
        fxfycxcy2 = torch.stack([view['fxfycxcy'] for view in gt2], dim=1).float()
        fxfycxcy = torch.cat((fxfycxcy1, fxfycxcy2), dim=1).to(fxfycxcy1)
        focal_length_gt = fxfycxcy[...,:2]
        with torch.no_grad():
            trajectory_R_prior = trajectory_pred[:4][-1]['R'].reshape(B, -1, 3, 3)
            trajectory_R_post = trajectory_pred[-1]['R'].reshape(B, -1, 3, 3)
        trajectory_t_pred = torch.stack([view["T"].reshape(B, -1, 3) for view in trajectory_pred], dim=1)
        trajectory_r_pred = torch.stack([view["quaternion_R"].reshape(B, -1, 4) for view in trajectory_pred], dim=1)
        focal_length_pred = torch.stack([view["focal_length"].reshape(B, -1, 2) for view in trajectory_pred], dim=1)
        trajectory_t_pred = trajectory_t_pred / (trajectory_t_pred.norm(dim=-1, keepdim=True).mean(dim=-2, keepdim=True) + 1e-5)
        trajectory_pred = torch.cat([trajectory_t_pred, trajectory_r_pred], dim=-1)
        trajectory_pred = trajectory_pred.permute(1,0,2,3)
        focal_length_pred = focal_length_pred.permute(1,0,2,3)
        se3_gt = torch.cat((R, trajectory_t_gt[..., None]), dim=-1)
        with torch.no_grad():
            pred_R = quaternion_to_matrix(trajectory_r_pred)
            se3_pred = torch.cat((pred_R, trajectory_t_pred[..., None]), dim=-1)
        # gt_pts3d_orig = gt_pts3d.clone()
        gt_pts3d = geotrf(in_camera1.repeat(1,num_views,1,1).view(-1,4,4), gt_pts3d.view(-1,H,W,3))  # B,H,W,3
        gt_pts3d = gt_pts3d.view(B,-1,H,W,3)

        # trajectory_orig = torch.stack([gt1_per['camera_pose'] for gt1_per in gt1 + gt2], dim=1)
        # pr_ref = geotrf(trajectory_orig.view(-1,4,4).inverse(), gt_pts3d_orig.view(-1,H,W,3)).view(B,-1,H,W,3)

        pr_ref = geotrf(trajectory_normalize.view(-1,4,4), pr_ref.view(-1,H,W,3)).view(B,-1,H,W,3)
        
        # valid mask
        valid1 = torch.stack([gt1_per['valid_mask'] for gt1_per in gt1], dim=1).view(B,-1,H,W).clone()
        valid2 = torch.stack([gt2_per['valid_mask'] for gt2_per in gt2], dim=1).view(B,-1,H,W).clone()
        dist_clip = 500
        if dist_clip is not None:
            # points that are too far-away == invalid
            dis1 = pr_stage2[:,:1].reshape(B, -1,W,3).norm(dim=-1)  # (B, H, W)
            dis2 = pr_stage2[:,1:].reshape(B, -1,W,3).norm(dim=-1)  # (B, H, W)
            valid1 = valid1 & (dis1 <= dist_clip).reshape(*valid1.shape)
            valid2 = valid2 & (dis2 <= dist_clip).reshape(*valid2.shape)
        gt_pts1, gt_pts2, norm_factor_gt = normalize_pointcloud(gt_pts3d[:,:1].reshape(B, -1,W,3), gt_pts3d[:,1:].reshape(B, -1,W,3), self.norm_mode, valid1.reshape(B, -1,W), valid2.reshape(B, -1,W), ret_factor=True)
        pr_pts1_ref, pr_pts2_ref, norm_factor_pr = normalize_pointcloud(pr_ref[:,:1].reshape(B, -1,W,3), pr_ref[:,1:].reshape(B, -1,W,3), self.norm_mode, valid1.reshape(B, -1,W), valid2.reshape(B, -1,W), ret_factor=True)
        
        # import nerfvis.scene as scene_vis
        # pr_pts = torch.cat((pr_pts1_ref, pr_pts2_ref), dim=1)
        # gt_pts = torch.cat((gt_pts1, gt_pts2), dim=1)
        # images_gt = torch.stack([gt['img_org'] for gt in gt1+gt2], dim=1)
        # images_gt = images_gt.permute(0,1,3,4,2).reshape(B, -1, H, W, 3)
        # scene_vis.add_points("pred_points_align", pr_pts[0].reshape(-1,3), vert_color=images_gt[0].reshape(-1,3) / 2 + 0.5, point_size=1)
        # scene_vis.add_points("gt_points", gt_pts[0].reshape(-1,3), vert_color=images_gt[0].reshape(-1,3) / 2 + 0.5, point_size=1)
        # scene_vis.display(port=8000)
        # reshape point map

        pr_stage2_pts1, pr_stage2_pts2, norm_factor_pr_stage2 = normalize_pointcloud(pr_stage2[:,:1].reshape(B, -1,W,3), pr_stage2[:,1:].reshape(B, -1,W,3), self.norm_mode, valid1.reshape(B, -1,W), valid2.reshape(B, -1,W), ret_factor=True)

        conf_ref = pred1['conf']
        conf_stage2 = pred2['conf']
        # conf2 = conf2.view(B, num_views, H, W)
        # 统计每个 (B, num_views) 中 True 的数量
        true_counts = valid2.view(B, num_views_src, -1).sum(dim=2)
        # 计算每个 B 中各视图的最小 True 数量
        min_true_counts_per_B = true_counts.min().item()
        if self.disable_rigid:
            min_true_counts_per_B = 0

        if min_true_counts_per_B > 10:
            #target = src @ R_gt.T # target vectors
            #R_predicted = roma.rigid_vectors_registration(src, target)
            mask = valid2.view(B, num_views_src, H, W)
            mask = mask.view(B*num_views_src, H, W)
            true_coords = []
            for i in range(mask.shape[0]):
                true_indices = torch.nonzero(mask[i])  # 获取所有 True 的坐标
                if true_indices.size(0) > 0:  # 确保有 True 值
                    sampled_indices = true_indices[torch.randint(0, true_indices.size(0), (min_true_counts_per_B,))]
                    true_coords.append(sampled_indices)
            true_coords = torch.stack(true_coords, dim=0)
            sampled_pts = []
            pr_pts2_reshaped = pr_stage2_pts2.reshape(B*num_views_src, H, W, 3)
            for i in range(len(true_coords)):
                coords = true_coords[i]
                # 直接用坐标从 pr_pts2 中采样
                sampled_points = pr_pts2_reshaped[i][coords[:, 0], coords[:, 1], :]
                sampled_pts.append(sampled_points)
            sampled_pts = torch.stack(sampled_pts, dim=0)
            gt_pts2_reshaped = gt_pts2.reshape(B*num_views_src, H, W, 3)
            sampled_gt_pts = []
            for i in range(len(true_coords)):
                coords = true_coords[i]
                # 直接用坐标从 pr_pts2 中采样
                sampled_points = gt_pts2_reshaped[i][coords[:, 0], coords[:, 1], :]
                sampled_gt_pts.append(sampled_points)
            sampled_gt_pts = torch.stack(sampled_gt_pts, dim=0)
           
            with torch.no_grad():
                R_pts2, T, s = roma.rigid_points_registration(sampled_pts, sampled_gt_pts, compute_scaling=True)
                # assert not torch.isnan(s).any()
                # assert not torch.isnan(R_pts2).any()
                # assert not torch.isnan(T).any()
            if torch.isnan(s).any() or torch.isnan(R_pts2).any() or torch.isnan(T).any():
                pr_pts2_transform = gt_pts2.reshape(B, num_views_src * H, W, 3)
            else:
                pr_pts2_transform = s[:,None, None, None] * torch.einsum('bik,bhwk->bhwi', R_pts2, pr_stage2_pts2.reshape(B * num_views_src, H, W, 3)) + T[:,None,None, :]
                # (gt_pts2.view(B*num_views, H, W,3) - pr_pts2_transform)[valid2.reshape(B * num_views, H, W)].abs().mean()
                pr_pts2_transform = pr_pts2_transform.reshape(B, num_views_src * H, W, 3)
        else:
            pr_pts2_transform = gt_pts2.reshape(B, num_views_src * H, W, 3)
        return gt_pts1, gt_pts2, pr_pts1_ref, pr_pts2_ref, pr_stage2_pts1, pr_stage2_pts2, valid1, valid2, conf_ref, conf_stage2, trajectory_gt, trajectory_pred, R, trajectory_R_prior, trajectory_R_post, focal_length_gt, focal_length_pred, se3_gt, se3_pred, pr_pts2_transform, [norm_factor_gt, norm_factor_pr_stage2]

    def compute_loss(self, gt1, gt2, pred1, pred2, trajectory_pred, render_gt=None, **kw):
        gt_pts1, gt_pts2, pr_pts1_ref, pr_pts2_ref, pr_stage2_pts1, pr_stage2_pts2, valid1, valid2, conf_ref, conf_stage2, trajectory_gt, trajectory_pred, R, trajectory_R_prior, trajectory_R_post, focal_length_gt, focal_length_pred, se3_gt, se3_pred, pr_pts2_transform, monitoring = self.get_all_pts3d(gt1, gt2, pred1, pred2, trajectory_pred, render_gt=render_gt, **kw)
        valid1 = valid1.flatten(1,2)    
        valid2 = valid2.flatten(1,2)   
        l1_ref = self.criterion(pr_pts1_ref[valid1], gt_pts1[valid1])
        l2_ref = self.criterion(pr_pts2_ref[valid2], gt_pts2[valid2])
        l1_stage2 = self.criterion(pr_stage2_pts1[valid1], gt_pts1[valid1])
        l2_stage2 = self.criterion(pr_stage2_pts2[valid2], gt_pts2[valid2])
        norm_factor_gt, norm_factor_pr_stage2 = monitoring
        Reg_1_ref = l1_ref.mean() if l1_ref.numel() > 0 else 0
        Reg_2_ref = l2_ref.mean() if l2_ref.numel() > 0 else 0
        Reg_1_stage2 = l1_stage2.mean() if l1_stage2.numel() > 0 else 0
        Reg_2_stage2 = l2_stage2.mean() if l2_stage2.numel() > 0 else 0

        n_predictions = len(trajectory_pred)
        pose_r_loss = 0.0
        pose_t_loss = 0.0
        pose_f_loss = 0.0
        gamma = 0.8
        l2_rigid = self.criterion(pr_pts2_transform[valid2], gt_pts2[valid2])
        Reg_2_rigid = l2_rigid.mean() if l2_rigid.numel() > 0 else 0
        if Reg_2_rigid < 1e-5:
            Reg_2_rigid = 0
            l2_rigid = 0
        for i in range(n_predictions):
            i_weight = gamma ** (n_predictions - i  - 1)
            trajectory_pred_iter = trajectory_pred[i]
            trajectory_gt_iter = trajectory_gt[0]
            focal_length_gt_iter = focal_length_pred[i]
            fxfy_gt = focal_length_gt
            trajectory_pred_r = trajectory_pred_iter[..., 3:]
            trajectory_pred_t = trajectory_pred_iter[..., :3]
            trajectory_gt_r = trajectory_gt_iter[..., 3:]
            trajectory_gt_t = trajectory_gt_iter[..., :3]
            pose_f_loss += 0. #i_weight * (focal_length_gt_iter - fxfy_gt).abs().mean()
            pose_r_loss += i_weight * (trajectory_gt_r - trajectory_pred_r).abs().mean() * 5
            pose_t_loss += i_weight * (trajectory_gt_t - trajectory_pred_t).abs().mean() * 5

        pose_t_loss = pose_t_loss 
        pose_r_loss = pose_r_loss
        with torch.no_grad():
            # rot_err_prior = torch.rad2deg(rotation_distance(R.reshape(*trajectory_R_prior.shape).float(), trajectory_R_prior.float())).mean()
            # rot_err_post = torch.rad2deg(rotation_distance(R.reshape(*trajectory_R_post.shape).float(), trajectory_R_post.float())).mean()
            # batch_size, nviews= se3_gt.shape[:2]
            # se3_gt = se3_gt.reshape(-1, 3, 4)
            # se3_pred_post = se3_pred[:,-1].reshape(-1, 3, 4)
            # pair_idx_i1, pair_idx_i2 = batched_all_pairs(batch_size, nviews)
            # bottom_ = torch.tensor([[[0,0,0,1]]]).to(se3_gt.device)
            # se3_gt = torch.cat((se3_gt, bottom_.repeat(se3_gt.shape[0],1,1)), dim=1)
            # se3_pred_post = torch.cat((se3_pred_post, bottom_.repeat(se3_pred_post.shape[0],1,1)), dim=1)
            # relative_pose_gt = closed_form_inverse(se3_gt[pair_idx_i1]).bmm(se3_gt[pair_idx_i2])
            # relative_pose_pred = closed_form_inverse(se3_pred_post[pair_idx_i1]).bmm(se3_pred_post[pair_idx_i2])
            # rel_tangle_deg = translation_angle(relative_pose_gt[:, :3, 3], relative_pose_pred[:, :3, 3]).mean()
            # se3_pred_prior = se3_pred[:, :4][-1].reshape(-1, 3, 4)
            # se3_pred_prior = torch.cat((se3_pred_prior, bottom_.repeat(se3_pred_prior.shape[0],1,1)), dim=1)
            # relative_pose_pred_prior = closed_form_inverse(se3_pred_prior[pair_idx_i1]).bmm(se3_pred_prior[pair_idx_i2])
            # rel_tangle_deg_prior = translation_angle(relative_pose_gt[:, :3, 3], relative_pose_pred_prior[:, :3, 3]).mean()
            batch_size, num_views = trajectory_pred.shape[1:3]
            rot_err_prior = torch.rad2deg(rotation_distance(R.reshape(*trajectory_R_prior.shape).float(), trajectory_R_prior.float())).mean()
            rot_err_post = torch.rad2deg(rotation_distance(R.reshape(*trajectory_R_post.shape).float(), trajectory_R_post.float())).mean()
            pair_idx_i1, pair_idx_i2 = batched_all_pairs(batch_size, num_views)
            se3_gt = torch.cat((R, trajectory_gt[0,..., :3, None]), dim=-1).reshape(-1 , 3, 4)
            se3_pred_prior = torch.cat((trajectory_R_prior, trajectory_pred[3][..., :3][..., None]), dim=-1).reshape(-1 , 3, 4)
            se3_pred_post = torch.cat((trajectory_R_post, trajectory_pred[-1][..., :3][..., None]), dim=-1).reshape(-1 , 3, 4)
            bottom_ = torch.tensor([[[0,0,0,1]]]).to(se3_gt.device)
            se3_gt = torch.cat((se3_gt, bottom_.repeat(se3_gt.shape[0],1,1)), dim=1)
            se3_pred_prior = torch.cat((se3_pred_prior, bottom_.repeat(se3_pred_prior.shape[0],1,1)), dim=1)
            se3_pred_post = torch.cat((se3_pred_post, bottom_.repeat(se3_pred_post.shape[0],1,1)), dim=1)
            relative_pose_gt = closed_form_inverse(se3_gt[pair_idx_i1]).bmm(se3_gt[pair_idx_i2])
            relative_pose_pred_prior = closed_form_inverse(se3_pred_prior[pair_idx_i1]).bmm(se3_pred_prior[pair_idx_i2])
            relative_pose_pred_post = closed_form_inverse(se3_pred_post[pair_idx_i1]).bmm(se3_pred_post[pair_idx_i2])
            rel_tangle_deg_prior = translation_angle(relative_pose_gt[:, :3, 3], relative_pose_pred_prior[:, :3, 3]).mean()
            rel_tangle_deg_post = translation_angle(relative_pose_gt[:, :3, 3], relative_pose_pred_post[:, :3, 3]).mean()

        self_name = type(self).__name__
        details = {'pr_stage2_pts1': float(pr_stage2_pts1.mean()), 'pr_stage2_pts2': float(pr_stage2_pts2.mean()), 'norm_factor_pr_stage2': float(norm_factor_pr_stage2.mean()), 'translation_error_prior': float(rel_tangle_deg_prior), 'translation_error': float(rel_tangle_deg_post), 'rot_err_post': float(rot_err_post), 'rot_err_prior': float(rot_err_prior), self_name+'_2_rigid': float(Reg_2_rigid),
        self_name+'_f_pose': float(pose_f_loss), self_name+'_t_pose': float(pose_t_loss), self_name+'_r_pose': float(pose_r_loss), 'trajectory_gt_t_first': float(trajectory_gt_t[:,0].abs().mean()), 'trajectory_pred_t_first': float(trajectory_pred_t[:,0].abs().mean()), self_name+'_1_ref': float(Reg_1_ref), self_name+'_2_ref': float(Reg_2_ref), self_name+'_1_stage2': float(Reg_1_stage2), self_name+'_2_stage2': float(Reg_2_stage2)}
        return Sum((l1_ref, valid1), (l2_ref, valid2), (l1_stage2, valid1), (l2_stage2, valid2), (pose_r_loss, None), (pose_t_loss, None), (pose_f_loss, None),  (0, None), (0, None), (0, None), (0, None), (l2_rigid, valid2)), details, monitoring



class ConfLoss_SA_ab (MultiLoss):
    """ Weighted regression by learned confidence.
        Assuming the input pixel_loss is a pixel-level regression loss.

    Principle:
        high-confidence means high conf = 0.1 ==> conf_loss = x / 10 + alpha*log(10)
        low  confidence means low  conf = 10  ==> conf_loss = x * 10 - alpha*log(10) 

        alpha: hyperparameter
    """

    def __init__(self, pixel_loss, alpha=1):
        super().__init__()
        assert alpha > 0
        self.alpha = alpha
        self.pixel_loss = pixel_loss.with_reduction('none')

    def get_name(self):
        return f'ConfLoss({self.pixel_loss})'

    def get_conf_log(self, x):
        return x, torch.log(x)

    def compute_loss(self, gt1, gt2, pred1, pred2, trajectory_pred, **kw):
        # compute per-pixel loss
        ((l1_ref, msk1), (l2_ref, msk2), (l1_stage2, msk1), (l2_stage2, msk2), (pose_r_loss, _), (pose_t_loss, _), (pose_f_loss, _), (loss_image, _), (loss_2d, _), (loss1_coarse, msk1_coarse), (loss2_coarse, msk2_coarse), (l2_rigid, _)), details, monitoring = self.pixel_loss(gt1, gt2, pred1, pred2, trajectory_pred, **kw)

        # if loss1.numel() == 0:
        #     print('NO VALID POINTS in img1', force=True)
        # if loss2.numel() == 0:
        #     print('NO VALID POINTS in img2', force=True)

        # weight by confidence
        conf_ref = pred1['conf']
        conf_ref1 = conf_ref[:,0]
        conf_ref2 = conf_ref[:,1:]
        conf_stage2 = pred2['conf']
        conf_stage2_1 = conf_stage2[:,0]
        conf_stage2_2 = conf_stage2[:,1:]
        
        conf1_ref, log_conf1_ref = self.get_conf_log(conf_ref1[msk1])
        conf2_ref, log_conf2_ref = self.get_conf_log(conf_ref2[msk2.view(*conf_ref2.shape)])
        conf1_stage2, log_conf1_stage2 = self.get_conf_log(conf_stage2_1[msk1])
        conf2_stage2, log_conf2_stage2 = self.get_conf_log(conf_stage2_2[msk2.view(*conf_stage2_2.shape)])
        # conf2, log_conf2 = self.get_conf_log(pred2['conf'][msk2.view(*pred2['conf'].shape)])
        # if msk1_coarse is not None:
        #     conf1_coarse, log_conf1_coarse = self.get_conf_log(pred1['conf_coarse'][msk1_coarse])
        #     conf2_coarse, log_conf2_coarse = self.get_conf_log(pred2['conf_coarse'][msk2_coarse.view(*pred2['conf_coarse'].shape)])
        #     conf_loss1_coarse = loss1_coarse * conf1_coarse - self.alpha * log_conf1_coarse
        #     conf_loss2_coarse = loss2_coarse * conf2_coarse - self.alpha * log_conf2_coarse
        # else:
        # conf_loss1_ref = l1_ref * conf1_ref - self.alpha * log_conf1_ref
        # conf_loss2_ref = l2_ref * conf2_ref - self.alpha * log_conf2_ref
        conf_loss1_ref = 0
        conf_loss2_ref = 0
        conf_loss1_stage2 = l1_stage2 * conf1_stage2 - self.alpha * log_conf1_stage2
        conf_loss2_stage2 = l2_stage2 * conf2_stage2 - self.alpha * log_conf2_stage2

        if type(l2_rigid) != int:
            conf_loss2_rigid = l2_rigid * conf2_stage2 - self.alpha * log_conf2_stage2
            conf_loss2_rigid = conf_loss2_rigid.mean() if conf_loss2_rigid.numel() > 0 else 0
        else:
            conf_loss2_rigid = 0 
        # average + nan protection (in case of no valid pixels at all)
        # conf_loss1_ref = conf_loss1_ref.mean() if conf_loss1_ref.numel() > 0 else 0
        # conf_loss2_ref = conf_loss2_ref.mean() if conf_loss2_ref.numel() > 0 else 0
        conf_loss1_stage2 = conf_loss1_stage2.mean() if conf_loss1_stage2.numel() > 0 else 0
        conf_loss2_stage2 = conf_loss2_stage2.mean() if conf_loss2_stage2.numel() > 0 else 0
        if loss_image is None:
            loss_image = 0
        # if pose_f_loss is None:
        pose_f_loss = 0
        return  conf_loss1_ref * 0.5 + conf_loss2_ref * 0.5 + conf_loss1_stage2 + conf_loss2_stage2 + pose_f_loss + pose_r_loss + pose_t_loss  + loss_image + loss_2d + conf_loss2_rigid * 0.1, dict(conf_loss2_rigid = float(conf_loss2_rigid), conf_loss_1=float(conf_loss1_ref), conf_loss_2=float(conf_loss2_ref), conf_loss_1_stage2=float(conf_loss1_stage2), conf_loss_2_stage2=float(conf_loss2_stage2), pose_r_loss=float(pose_r_loss), pose_t_loss=float(pose_t_loss), pose_f_loss=float(pose_f_loss), image_loss=float(loss_image), **details)



class ConfLoss_SA (MultiLoss):
    """ Weighted regression by learned confidence.
        Assuming the input pixel_loss is a pixel-level regression loss.

    Principle:
        high-confidence means high conf = 0.1 ==> conf_loss = x / 10 + alpha*log(10)
        low  confidence means low  conf = 10  ==> conf_loss = x * 10 - alpha*log(10) 

        alpha: hyperparameter
    """

    def __init__(self, pixel_loss, alpha=1):
        super().__init__()
        assert alpha > 0
        self.alpha = alpha
        self.pixel_loss = pixel_loss.with_reduction('none')

    def get_name(self):
        return f'ConfLoss({self.pixel_loss})'

    def get_conf_log(self, x):
        return x, torch.log(x)

    def compute_loss(self, gt1, gt2, pred1, pred2, trajectory_pred, **kw):
        # compute per-pixel loss
        ((l1_ref, msk1), (l2_ref, msk2), (l1_stage2, msk1), (l2_stage2, msk2), (pose_r_loss, _), (pose_t_loss, _), (pose_f_loss, _), (loss_image, _), (loss_2d, _), (loss1_coarse, msk1_coarse), (loss2_coarse, msk2_coarse), (l2_rigid, _)), details, monitoring = self.pixel_loss(gt1, gt2, pred1, pred2, trajectory_pred, **kw)

        # if loss1.numel() == 0:
        #     print('NO VALID POINTS in img1', force=True)
        # if loss2.numel() == 0:
        #     print('NO VALID POINTS in img2', force=True)

        # weight by confidence
        conf_ref = pred1['conf']
        conf_ref1 = conf_ref[:,0]
        conf_ref2 = conf_ref[:,1:]
        conf_stage2 = pred2['conf']
        conf_stage2_1 = conf_stage2[:,0]
        conf_stage2_2 = conf_stage2[:,1:]
        
        conf1_ref, log_conf1_ref = self.get_conf_log(conf_ref1[msk1])
        conf2_ref, log_conf2_ref = self.get_conf_log(conf_ref2[msk2.view(*conf_ref2.shape)])
        conf1_stage2, log_conf1_stage2 = self.get_conf_log(conf_stage2_1[msk1])
        conf2_stage2, log_conf2_stage2 = self.get_conf_log(conf_stage2_2[msk2.view(*conf_stage2_2.shape)])
        # conf2, log_conf2 = self.get_conf_log(pred2['conf'][msk2.view(*pred2['conf'].shape)])
        # if msk1_coarse is not None:
        #     conf1_coarse, log_conf1_coarse = self.get_conf_log(pred1['conf_coarse'][msk1_coarse])
        #     conf2_coarse, log_conf2_coarse = self.get_conf_log(pred2['conf_coarse'][msk2_coarse.view(*pred2['conf_coarse'].shape)])
        #     conf_loss1_coarse = loss1_coarse * conf1_coarse - self.alpha * log_conf1_coarse
        #     conf_loss2_coarse = loss2_coarse * conf2_coarse - self.alpha * log_conf2_coarse
        # else:
        conf_loss1_ref = l1_ref * conf1_ref - self.alpha * log_conf1_ref
        conf_loss2_ref = l2_ref * conf2_ref - self.alpha * log_conf2_ref
        conf_loss1_stage2 = l1_stage2 * conf1_stage2 - self.alpha * log_conf1_stage2
        conf_loss2_stage2 = l2_stage2 * conf2_stage2 - self.alpha * log_conf2_stage2

        if type(l2_rigid) != int:
            conf_loss2_rigid = l2_rigid * conf2_stage2 - self.alpha * log_conf2_stage2
            conf_loss2_rigid = conf_loss2_rigid.mean() if conf_loss2_rigid.numel() > 0 else 0
        else:
            conf_loss2_rigid = 0 
        # average + nan protection (in case of no valid pixels at all)
        conf_loss1_ref = conf_loss1_ref.mean() if conf_loss1_ref.numel() > 0 else 0
        conf_loss2_ref = conf_loss2_ref.mean() if conf_loss2_ref.numel() > 0 else 0
        conf_loss1_stage2 = conf_loss1_stage2.mean() if conf_loss1_stage2.numel() > 0 else 0
        conf_loss2_stage2 = conf_loss2_stage2.mean() if conf_loss2_stage2.numel() > 0 else 0
        if loss_image is None:
            loss_image = 0
        # if pose_f_loss is None:
        pose_f_loss = 0
        return  conf_loss1_ref * 0.5 + conf_loss2_ref * 0.5 + conf_loss1_stage2 + conf_loss2_stage2 + pose_f_loss + pose_r_loss + pose_t_loss  + loss_image + loss_2d + conf_loss2_rigid * 0.1, dict(conf_loss2_rigid = float(conf_loss2_rigid), conf_loss_1=float(conf_loss1_ref), conf_loss_2=float(conf_loss2_ref), conf_loss_1_stage2=float(conf_loss1_stage2), conf_loss_2_stage2=float(conf_loss2_stage2), pose_r_loss=float(pose_r_loss), pose_t_loss=float(pose_t_loss), pose_f_loss=float(pose_f_loss), image_loss=float(loss_image), **details)



class Regr3D_clean(Criterion, MultiLoss):
    """ Ensure that all 3D points are correct.
        Asymmetric loss: view1 is supposed to be the anchor.

        P1 = RT1 @ D1
        P2 = RT2 @ D2
        loss1 = (I @ pred_D1) - (RT1^-1 @ RT1 @ D1)
        loss2 = (RT21 @ pred_D2) - (RT1^-1 @ P2)
              = (RT21 @ pred_D2) - (RT1^-1 @ RT2 @ D2)
    """

    def __init__(self, criterion, norm_mode='avg_dis', gt_scale=False, test=False):
        super().__init__(criterion)
        self.norm_mode = norm_mode
        self.gt_scale = gt_scale
        self.epoch = 0
        self.test = test
        self.gt_num_image = 0

    def get_all_pts3d(self, gt1, gt2, pred1, pred2, trajectory_pred, dist_clip=None, render_gt=None, wop=False):
        # predicted pts3d
        pr_pts1 = pred1['pts3d']
        pr_pts2 = pred2['pts3d']
        # GT pts3d
        gt1_pts3d = torch.stack([gt1_per['pts3d'] for gt1_per in gt1], dim=1)
        B, _, H, W, _ = gt1_pts3d.shape
        # caculate the number of views and read the gt2
        num_views = pr_pts2.shape[0] // B
        gt2_pts3d = torch.stack([gt2_per['pts3d'] for gt2_per in gt2[:num_views]], dim=1)
        # camera trajectory
        camera_pose = torch.stack([gt1_per['camera_pose'] for gt1_per in gt1], dim=1)
        in_camera1 = inv(camera_pose)
        trajectory = torch.stack([view['camera_pose'] for view in gt2], dim=1)
        trajectory = torch.cat((camera_pose, trajectory), dim=1)
        # import ipdb; ipdb.set_trace()

        # trajectory_1 = closed_form_inverse(camera_pose.repeat(1, trajectory.shape[1],1,1).reshape(-1,4,4)).bmm(trajectory.reshape(-1,4,4))
        trajectory = torch.einsum('bnjk,bnkl->bnjl', in_camera1.repeat(1, trajectory.shape[1],1,1), trajectory)
        # trajectory_1 - trajectory.reshape(-1,4,4)
        # transform the gt points to the coordinate of the first view
        gt_pts1 = geotrf(in_camera1.view(-1,4,4), gt1_pts3d.view(-1,H,W,3))  # B,H,W,3
        gt_pts2 = geotrf(in_camera1.repeat(1,num_views,1,1).view(-1,4,4), gt2_pts3d.view(-1,H,W,3))  # B,H,W,3
        gt_pts1 = gt_pts1.view(B,-1,H,W,3)
        gt_pts2 = gt_pts2.view(B,-1,H,W,3)
        Rs = []
        Ts = []


        # valid mask
        valid1 = torch.stack([gt1_per['valid_mask'] for gt1_per in gt1], dim=1).view(B,-1,H,W).clone()
        valid2 = torch.stack([gt2_per['valid_mask'] for gt2_per in gt2[:num_views]], dim=1).view(B,-1,H,W).clone()


        if dist_clip is not None:
            # points that are too far-away == invalid
            dis1 = gt_pts1.norm(dim=-1)  # (B, H, W)
            dis2 = gt_pts2.norm(dim=-1)  # (B, H, W)
            valid1 = valid1 & (dis1 <= dist_clip)
            valid2 = valid2 & (dis2 <= dist_clip)
        
        # reshape point map
        gt_pts1 = gt_pts1.reshape(B,H,W,3)
        gt_pts2 = gt_pts2.reshape(B,H*num_views,W,3)
        pr_pts1 = pr_pts1.reshape(B,H,W,3)
        pr_pts2 = pr_pts2.reshape(B,H*num_views,W,3)
        valid1 = valid1.view(B,H,W)
        valid2 = valid2.view(B,H*num_views,W)
        
        if valid1.sum() == 0 and valid2.sum() == 0:
            valid1 = torch.ones_like(valid1).to(valid1) > 0
            valid2 = torch.ones_like(valid2).to(valid2) > 0
            # import ipdb; ipdb.set_trace()

        
        # normalize point map
        if self.norm_mode and not self.gt_scale:
            gt_pts1, gt_pts2, norm_factor_gt = normalize_pointcloud(gt_pts1, gt_pts2, self.norm_mode, valid1, valid2, ret_factor=True)
            # import ipdb; ipdb.set_trace()
            # output_c2ws = trajectory.clone() #torch.einsum('bnjk,bnkl->bnjl', in_camera1.repeat(1,output_c2ws.shape[1], 1, 1), output_c2ws)
            # output_c2ws[..., :3, 3:] = output_c2ws[..., :3, 3:] / norm_factor_gt.detach()
        
        # generate the gt camera trajectory
        trajectory_t_gt = trajectory[..., :3, 3]
        trajectory_t_gt = trajectory_t_gt / (trajectory_t_gt.norm(dim=-1, keepdim=True).mean(dim=1, keepdim=True) + 1e-5)
        quaternion_R = matrix_to_quaternion(trajectory[...,:3,:3])
        trajectory_gt = torch.cat([trajectory_t_gt, quaternion_R], dim=-1)[None]
        R = trajectory[...,:3,:3]
        fxfycxcy1 = torch.stack([view['fxfycxcy'] for view in gt1], dim=1).float()
        fxfycxcy2 = torch.stack([view['fxfycxcy'] for view in gt2], dim=1).float()
        fxfycxcy = torch.cat((fxfycxcy1, fxfycxcy2), dim=1).to(fxfycxcy1)
        focal_length_gt = fxfycxcy[...,:2]

        # generate the predicted camera trajectory
        if wop == False:
            with torch.no_grad():
                trajectory_R_prior = trajectory_pred[:4][-1]['R'].reshape(B, -1, 3, 3)
                trajectory_R_post = trajectory_pred[-1]['R'].reshape(B, -1, 3, 3)
        else:
            trajectory_R_prior = None
            trajectory_R_post = None
        if wop == False:
            trajectory_t_pred = torch.stack([view["T"].reshape(B, -1, 3) for view in trajectory_pred], dim=1)
            trajectory_r_pred = torch.stack([view["quaternion_R"].reshape(B, -1, 4) for view in trajectory_pred], dim=1)
            focal_length_pred = torch.stack([view["focal_length"].reshape(B, -1, 2) for view in trajectory_pred], dim=1)
            trajectory_t_pred = trajectory_t_pred / (trajectory_t_pred.norm(dim=-1, keepdim=True).mean(dim=-2, keepdim=True) + 1e-5)
            trajectory_pred = torch.cat([trajectory_t_pred, trajectory_r_pred], dim=-1)
            trajectory_pred = trajectory_pred.permute(1,0,2,3)
            focal_length_pred = focal_length_pred.permute(1,0,2,3)
        else:
            trajectory_pred = None
            focal_length_pred = None
            trajectory_t_pred = None
            trajectory_r_pred = None

        if self.norm_mode:
            pr_pts1, pr_pts2, norm_factor_pr = normalize_pointcloud(pr_pts1, pr_pts2, self.norm_mode, valid1, valid2, ret_factor=True)



        conf1 = pred1['conf']
        conf2 = pred2['conf']
        conf2 = conf2.view(B, num_views, H, W)
        # 统计每个 (B, num_views) 中 True 的数量
        true_counts = valid2.view(B, num_views, -1).sum(dim=2)
        # 计算每个 B 中各视图的最小 True 数量
        min_true_counts_per_B = true_counts.min().item()
        min_true_counts_per_B = 0   
        if min_true_counts_per_B != 0:
            #target = src @ R_gt.T # target vectors
            #R_predicted = roma.rigid_vectors_registration(src, target)
            mask = valid2.view(B, num_views, H, W)
            mask = mask.view(B*num_views, H, W)
            true_coords = []
            for i in range(mask.shape[0]):
                true_indices = torch.nonzero(mask[i])  # 获取所有 True 的坐标
                if true_indices.size(0) > 0:  # 确保有 True 值
                    sampled_indices = true_indices[torch.randint(0, true_indices.size(0), (min_true_counts_per_B,))]
                    true_coords.append(sampled_indices)
            true_coords = torch.stack(true_coords, dim=0)
            sampled_pts = []
            pr_pts2_reshaped = pr_pts2.reshape(B*num_views, H, W, 3)
            for i in range(len(true_coords)):
                coords = true_coords[i]
                # 直接用坐标从 pr_pts2 中采样
                sampled_points = pr_pts2_reshaped[i][coords[:, 0], coords[:, 1], :]
                sampled_pts.append(sampled_points)
            sampled_pts = torch.stack(sampled_pts, dim=0)
            gt_pts2_reshaped = gt_pts2.reshape(B*num_views, H, W, 3)
            sampled_gt_pts = []
            for i in range(len(true_coords)):
                coords = true_coords[i]
                # 直接用坐标从 pr_pts2 中采样
                sampled_points = gt_pts2_reshaped[i][coords[:, 0], coords[:, 1], :]
                sampled_gt_pts.append(sampled_points)
            sampled_gt_pts = torch.stack(sampled_gt_pts, dim=0)
            with torch.no_grad():
                R_pts2, T = roma.rigid_points_registration(sampled_pts, sampled_gt_pts, compute_scaling=False)
                # assert not torch.isnan(s).any()
                assert not torch.isnan(R_pts2).any()
                assert not torch.isnan(T).any()
                #verfiy not nan 
                
            pr_pts2_transform = torch.einsum('bik,bhwk->bhwi', R_pts2, pr_pts2.reshape(B * num_views, H, W, 3)) + T[:,None,None, :]
            # (gt_pts2.view(B*num_views, H, W,3) - pr_pts2_transform)[valid2.reshape(B * num_views, H, W)].abs().mean()
            pr_pts2_transform = pr_pts2_transform.reshape(B, num_views * H, W, 3).float()
            # import nerfvis.scene as scene_vis 
            # scene_vis.set_title("My Scene")
            # scene_vis.set_opencv() 
            # pts = pr_pts2_transform[0].reshape(-1,3)
            # pts_mean = torch.mean(pts, axis=0)
            # img_org = torch.stack([render['img'] for render in gt2], dim=1)[0].permute(0,2,3,1).reshape(-1,3).float()
            # scene_vis.add_points("pred_points_align", (pts-pts_mean), vert_color=img_org.reshape(-1,3) / 2 + 0.5, point_size=1)
            # scene_vis.add_points("gt_points", (gt_pts2[0].reshape(-1,3) - pts_mean), vert_color=img_org.reshape(-1,3) / 2 + 0.5, point_size=1)
            # scene_vis.add_points("pred_points", (pr_pts2[0].reshape(-1,3) - pts_mean), vert_color=img_org.reshape(-1,3) / 2 + 0.5, point_size=1)
            # scene_vis.display()
        else:
            pr_pts2_transform = pr_pts2.reshape(B * num_views, H, W, 3)#gt_pts2.reshape(B, num_views * H, W, 3)
        if wop == False:
            se3_gt = torch.cat((R, trajectory_t_gt[..., None]), dim=-1)
            with torch.no_grad():
                pred_R = quaternion_to_matrix(trajectory_r_pred)
                se3_pred = torch.cat((pred_R, trajectory_t_pred[..., None]), dim=-1)
        else:
            se3_gt = None
            se3_pred = None
            pred_R = None
        return gt_pts1, gt_pts2, pr_pts1, pr_pts2, valid1, valid2, conf1, conf2, trajectory_gt, trajectory_pred, R, trajectory_R_prior, trajectory_R_post, focal_length_gt, focal_length_pred, se3_gt, se3_pred, pr_pts2_transform, [norm_factor_gt, norm_factor_pr]

    def compute_loss(self, gt1, gt2, pred1, pred2, trajectory_pred, render_gt=None, **kw):
        gt_pts1, gt_pts2, pr_pts1, pr_pts2,  mask1, mask2, conf1, conf2, trajectory, trajectory_pred, R, trajectory_R_prior, trajectory_R_post, focal_length_gt, focal_length_pred, se3_gt, se3_pred, pr_pts2_transform, monitoring = self.get_all_pts3d(gt1, gt2, pred1, pred2, trajectory_pred, render_gt=render_gt, **kw)
        l1 = self.criterion(pr_pts1[mask1], gt_pts1[mask1])
        l2 = self.criterion(pr_pts2[mask2], gt_pts2[mask2])
        Reg_1 = l1.mean() if l1.numel() > 0 else 0
        Reg_2 = l2.mean() if l2.numel() > 0 else 0
        n_predictions = len(trajectory_pred)
        pose_r_loss = 0.0
        pose_t_loss = 0.0
        pose_f_loss = 0.0
        gamma = 0.8
        l2_rigid = self.criterion(pr_pts2_transform[mask2], gt_pts2[mask2])
        Reg_2_rigid = l2_rigid.mean() if l2_rigid.numel() > 0 else 0
        for i in range(n_predictions):
            i_weight = gamma ** (n_predictions - i % 4 - 1)
            trajectory_pred_iter = trajectory_pred[i]
            trajectory_gt_iter = trajectory[0]
            focal_length_gt_iter = focal_length_pred[i]
            fxfy_gt = focal_length_gt
            trajectory_pred_r = trajectory_pred_iter[..., 3:]
            trajectory_pred_t = trajectory_pred_iter[..., :3]
            trajectory_gt_r = trajectory_gt_iter[..., 3:]
            trajectory_gt_t = trajectory_gt_iter[..., :3]
            pose_f_loss += 0. #i_weight * (focal_length_gt_iter - fxfy_gt).abs().mean()
            pose_r_loss += i_weight * (trajectory_gt_r - trajectory_pred_r).abs().mean() * 2.5
            pose_t_loss += i_weight * (trajectory_gt_t - trajectory_pred_t).abs().mean() * 2.5

        pose_t_loss = pose_t_loss 
        pose_r_loss = pose_r_loss
        with torch.no_grad():
            # rot_err_prior = torch.rad2deg(rotation_distance(R.reshape(*trajectory_R_prior.shape).float(), trajectory_R_prior.float())).mean()
            # rot_err_post = torch.rad2deg(rotation_distance(R.reshape(*trajectory_R_post.shape).float(), trajectory_R_post.float())).mean()
            # batch_size, nviews= se3_gt.shape[:2]
            # se3_gt = se3_gt.reshape(-1, 3, 4)
            # se3_pred_post = se3_pred[:,-1].reshape(-1, 3, 4)
            # pair_idx_i1, pair_idx_i2 = batched_all_pairs(batch_size, nviews)
            # bottom_ = torch.tensor([[[0,0,0,1]]]).to(se3_gt.device)
            # se3_gt = torch.cat((se3_gt, bottom_.repeat(se3_gt.shape[0],1,1)), dim=1)
            # se3_pred_post = torch.cat((se3_pred_post, bottom_.repeat(se3_pred_post.shape[0],1,1)), dim=1)
            # relative_pose_gt = closed_form_inverse(se3_gt[pair_idx_i1]).bmm(se3_gt[pair_idx_i2])
            # relative_pose_pred = closed_form_inverse(se3_pred_post[pair_idx_i1]).bmm(se3_pred_post[pair_idx_i2])
            # rel_tangle_deg = translation_angle(relative_pose_gt[:, :3, 3], relative_pose_pred[:, :3, 3]).mean()
            # se3_pred_prior = se3_pred[:, :4][-1].reshape(-1, 3, 4)
            # se3_pred_prior = torch.cat((se3_pred_prior, bottom_.repeat(se3_pred_prior.shape[0],1,1)), dim=1)
            # relative_pose_pred_prior = closed_form_inverse(se3_pred_prior[pair_idx_i1]).bmm(se3_pred_prior[pair_idx_i2])
            # rel_tangle_deg_prior = translation_angle(relative_pose_gt[:, :3, 3], relative_pose_pred_prior[:, :3, 3]).mean()
            batch_size, num_views = trajectory_pred.shape[1:3]
            rot_err_prior = torch.rad2deg(rotation_distance(R.reshape(*trajectory_R_prior.shape).float(), trajectory_R_prior.float())).mean()
            rot_err_post = torch.rad2deg(rotation_distance(R.reshape(*trajectory_R_post.shape).float(), trajectory_R_post.float())).mean()
            pair_idx_i1, pair_idx_i2 = batched_all_pairs(batch_size, num_views)
            se3_gt = torch.cat((R, trajectory[0,..., :3, None]), dim=-1).reshape(-1 , 3, 4)
            se3_pred_prior = torch.cat((trajectory_R_prior, trajectory_pred[3][..., :3][..., None]), dim=-1).reshape(-1 , 3, 4)
            se3_pred_post = torch.cat((trajectory_R_post, trajectory_pred[-1][..., :3][..., None]), dim=-1).reshape(-1 , 3, 4)
            bottom_ = torch.tensor([[[0,0,0,1]]]).to(se3_gt.device)
            se3_gt = torch.cat((se3_gt, bottom_.repeat(se3_gt.shape[0],1,1)), dim=1)
            se3_pred_prior = torch.cat((se3_pred_prior, bottom_.repeat(se3_pred_prior.shape[0],1,1)), dim=1)
            se3_pred_post = torch.cat((se3_pred_post, bottom_.repeat(se3_pred_post.shape[0],1,1)), dim=1)
            relative_pose_gt = closed_form_inverse(se3_gt[pair_idx_i1]).bmm(se3_gt[pair_idx_i2])
            relative_pose_pred_prior = closed_form_inverse(se3_pred_prior[pair_idx_i1]).bmm(se3_pred_prior[pair_idx_i2])
            relative_pose_pred_post = closed_form_inverse(se3_pred_post[pair_idx_i1]).bmm(se3_pred_post[pair_idx_i2])
            rel_tangle_deg_prior = translation_angle(relative_pose_gt[:, :3, 3], relative_pose_pred_prior[:, :3, 3]).mean()
            rel_tangle_deg_post = translation_angle(relative_pose_gt[:, :3, 3], relative_pose_pred_post[:, :3, 3]).mean()

        self_name = type(self).__name__
        details = {'translation_error_prior': float(rel_tangle_deg_prior), 'translation_error': float(rel_tangle_deg_post), 'rot_err_post': float(rot_err_post), 'rot_err_prior': float(rot_err_prior),self_name+'_1': float(Reg_1), self_name+'_2': float(Reg_2), self_name+'_2_rigid': float(Reg_2_rigid),
        self_name+'_f_pose': float(pose_f_loss), self_name+'_t_pose': float(pose_t_loss), self_name+'_r_pose': float(pose_r_loss), 'trajectory_gt_t_first': float(trajectory_gt_t[:,0].abs().mean()), 'trajectory_pred_t_first': float(trajectory_pred_t[:,0].abs().mean())}
        return Sum((l1, mask1), (l2, mask2), (pose_r_loss, None), (pose_t_loss, None), (pose_f_loss, None),  (0, None), (0, None), (0, None), (0, None), (l2_rigid, mask2)), details, monitoring
        #  ((loss1, msk1), (loss2, msk2), (pose_r_loss, _), (pose_t_loss, _), (pose_f_loss, _), (loss_image, _), (loss_2d, _), (loss1_coarse, msk1_coarse), (loss2_coarse, msk2_coarse))
    def get_all_pose(self, gt1, gt2, pred1, pred2, trajectory_pred, dist_clip=None, render_gt=None):
        # predicted pts3d
        camera_pose = torch.stack([gt1_per['camera_pose'] for gt1_per in gt1], dim=1)
        in_camera1 = inv(camera_pose)
        trajectory = torch.stack([view['camera_pose'] for view in gt2], dim=1)
        B = trajectory.shape[0]
        trajectory = torch.cat((camera_pose, trajectory), dim=1)
        # trajectory_1 = closed_form_inverse(camera_pose.repeat(1, trajectory.shape[1],1,1).reshape(-1,4,4)).bmm(trajectory.reshape(-1,4,4))
        trajectory = torch.einsum('bnjk,bnkl->bnjl', in_camera1.repeat(1, trajectory.shape[1],1,1), trajectory)
        # transform the gt points to the coordinate of the first view
        Rs = []
        Ts = []
        # generate the gt camera trajectory
        trajectory_t_gt = trajectory[..., :3, 3].clone()
        trajectory_t_gt = trajectory_t_gt / (trajectory_t_gt.norm(dim=-1, keepdim=True).mean(dim=1, keepdim=True) + 1e-5)
        quaternion_R = matrix_to_quaternion(trajectory[...,:3,:3])
        trajectory_gt = torch.cat([trajectory_t_gt, quaternion_R], dim=-1)[None]
        R = trajectory[...,:3,:3]
        fxfycxcy1 = torch.stack([view['fxfycxcy'] for view in gt1], dim=1).float()
        fxfycxcy2 = torch.stack([view['fxfycxcy'] for view in gt2], dim=1).float()
        fxfycxcy = torch.cat((fxfycxcy1, fxfycxcy2), dim=1).to(fxfycxcy1)
        focal_length_gt = fxfycxcy[...,:2]
        # generate the predicted camera trajectory
        with torch.no_grad():
            trajectory_R_prior = trajectory_pred[:4][-1]['R'].reshape(B, -1, 3, 3)
            trajectory_R_post = trajectory_pred[-1]['R'].reshape(B, -1, 3, 3)
        trajectory_t_pred = torch.stack([view["T"].reshape(B, -1, 3) for view in trajectory_pred], dim=1)
        trajectory_r_pred = torch.stack([view["quaternion_R"].reshape(B, -1, 4) for view in trajectory_pred], dim=1)
        focal_length_pred = torch.stack([view["focal_length"].reshape(B, -1, 2) for view in trajectory_pred], dim=1)
        trajectory_t_pred = trajectory_t_pred / (trajectory_t_pred.norm(dim=-1, keepdim=True).mean(dim=-2, keepdim=True) + 1e-5)
        trajectory_pred = torch.cat([trajectory_t_pred, trajectory_r_pred], dim=-1)
        trajectory_pred = trajectory_pred.permute(1,0,2,3)
        focal_length_pred = focal_length_pred.permute(1,0,2,3)
        se3_gt = torch.cat((R, trajectory_t_gt[..., None]), dim=-1)
        with torch.no_grad():
            pred_R = quaternion_to_matrix(trajectory_r_pred)
            se3_pred = torch.cat((pred_R, trajectory_t_pred[..., None]), dim=-1)
        return trajectory_gt, trajectory_pred, R, trajectory_R_prior, trajectory_R_post, focal_length_gt, focal_length_pred, se3_gt, se3_pred

    def compute_pose_loss(self, gt1, gt2, pred1, pred2, trajectory_pred, render_gt=None, **kw):
        trajectory, trajectory_pred, R, trajectory_R_prior, trajectory_R_post, focal_length_gt, focal_length_pred, se3_gt, se3_pred = self.get_all_pose(gt1, gt2, pred1, pred2, trajectory_pred, render_gt=render_gt, **kw)
        n_predictions = len(trajectory_pred)
        pose_r_loss = 0.0
        pose_t_loss = 0.0
        pose_f_loss = 0.0
        gamma = 0.8
        for i in range(n_predictions):
            i_weight = gamma ** (n_predictions - i % 4 - 1)
            trajectory_pred_iter = trajectory_pred[i]
            trajectory_gt_iter = trajectory[0]
            focal_length_gt_iter = focal_length_pred[i]
            fxfy_gt = focal_length_gt
            trajectory_pred_r = trajectory_pred_iter[..., 3:]
            trajectory_pred_t = trajectory_pred_iter[..., :3]
            trajectory_gt_r = trajectory_gt_iter[..., 3:]
            trajectory_gt_t = trajectory_gt_iter[..., :3]
            pose_f_loss += 0. #i_weight * (focal_length_gt_iter - fxfy_gt).abs().mean()
            pose_r_loss += i_weight * (trajectory_gt_r - trajectory_pred_r).abs().mean() * 2.5
            pose_t_loss += i_weight * (trajectory_gt_t - trajectory_pred_t).abs().mean() * 2.5

        pose_t_loss = pose_t_loss 
        pose_r_loss = pose_r_loss
        with torch.no_grad():
            batch_size, num_views = trajectory_pred.shape[1:3]
            rot_err_prior = torch.rad2deg(rotation_distance(R.reshape(*trajectory_R_prior.shape).float(), trajectory_R_prior.float())).mean()
            rot_err_post = torch.rad2deg(rotation_distance(R.reshape(*trajectory_R_post.shape).float(), trajectory_R_post.float())).mean()
            pair_idx_i1, pair_idx_i2 = batched_all_pairs(batch_size, num_views)
            se3_gt = torch.cat((R, trajectory[0,..., :3, None]), dim=-1).reshape(-1 , 3, 4)
            se3_pred_prior = torch.cat((trajectory_R_prior, trajectory_pred[3][..., :3][..., None]), dim=-1).reshape(-1 , 3, 4)
            se3_pred_post = torch.cat((trajectory_R_post, trajectory_pred[-1][..., :3][..., None]), dim=-1).reshape(-1 , 3, 4)
            bottom_ = torch.tensor([[[0,0,0,1]]]).to(se3_gt.device)
            se3_gt = torch.cat((se3_gt, bottom_.repeat(se3_gt.shape[0],1,1)), dim=1)
            se3_pred_prior = torch.cat((se3_pred_prior, bottom_.repeat(se3_pred_prior.shape[0],1,1)), dim=1)
            se3_pred_post = torch.cat((se3_pred_post, bottom_.repeat(se3_pred_post.shape[0],1,1)), dim=1)
            relative_pose_gt = closed_form_inverse(se3_gt[pair_idx_i1]).bmm(se3_gt[pair_idx_i2])
            relative_pose_pred_prior = closed_form_inverse(se3_pred_prior[pair_idx_i1]).bmm(se3_pred_prior[pair_idx_i2])
            relative_pose_pred_post = closed_form_inverse(se3_pred_post[pair_idx_i1]).bmm(se3_pred_post[pair_idx_i2])
            rel_tangle_deg_prior = translation_angle(relative_pose_gt[:, :3, 3], relative_pose_pred_prior[:, :3, 3]).mean()
            rel_tangle_deg_post = translation_angle(relative_pose_gt[:, :3, 3], relative_pose_pred_post[:, :3, 3]).mean()

        self_name = type(self).__name__
        details = {'translation_error_prior': float(rel_tangle_deg_prior), 'translation_error': float(rel_tangle_deg_post), 'rot_err_post': float(rot_err_post), 'rot_err_prior': float(rot_err_prior), self_name+'_f_pose': float(pose_f_loss), self_name+'_t_pose': float(pose_t_loss), self_name+'_r_pose': float(pose_r_loss), 'trajectory_gt_t_first': float(trajectory_gt_t[:,0].abs().mean()), 'trajectory_pred_t_first': float(trajectory_pred_t[:,0].abs().mean())}
        return (pose_r_loss, None), (pose_t_loss, None), (pose_f_loss, None), details


def reproject(pts3d, camera_intrinsics, world2cam):
    R_cam2world = world2cam[..., :3, :3]
    t_cam2world = world2cam[..., :3, 3]
    B,N = world2cam.shape[:2]
    X_cam = torch.einsum("bnik, bnpk -> bnpi", R_cam2world, pts3d.reshape(B,1,-1,3).repeat(1,N,1,1)) + t_cam2world[..., None, :]
    X_cam = X_cam/X_cam[...,2:3]
    X_cam = torch.einsum("bnik, bnpk -> bnpi", camera_intrinsics, X_cam)
    X_cam = X_cam[...,:2]
    return X_cam


def homogeneous_to_euclidean(points):
    """Converts homogeneous points to euclidean

    Args:
        points numpy array or torch tensor of shape (N, M + 1): N homogeneous points of dimension M

    Returns:
        numpy array or torch tensor of shape (N, M): euclidean points
    """
    if isinstance(points, np.ndarray):
        return (points.T[:-1] / points.T[-1]).T
    elif torch.is_tensor(points):
        return (points.transpose(1, 0)[:-1] / points.transpose(1, 0)[-1]).transpose(1, 0)
    else:
        raise TypeError("Works only with numpy arrays and PyTorch tensors.")


def triangulate_point_from_multiple_views_linear_torch(proj_matricies, points, confidences=None):
    """Similar as triangulate_point_from_multiple_views_linear() but for PyTorch.
    For more information see its documentation.
    Args:
        proj_matricies torch tensor of shape (N, 3, 4): sequence of projection matricies (3x4)
        points torch tensor of of shape (N, 2): sequence of points' coordinates
        confidences None or torch tensor of shape (N,): confidences of points [0.0, 1.0].
                                                        If None, all confidences are supposed to be 1.0
    Returns:
        point_3d numpy torch tensor of shape (3,): triangulated point
    """
    assert len(proj_matricies) == len(points)
    n_views = proj_matricies.shape[0]
    batch_size = points.shape[1]
    points = points.permute(1,0,2)

    A = proj_matricies[:, 2:3].expand(n_views, 2, 4)[None].repeat(batch_size, 1,1,1) * points.view(batch_size, n_views, 2, 1) # n_views, 2, 4  x n_views,2,1
    A -= proj_matricies[:, :2][None].repeat(batch_size, 1,1,1) # n_views, 2, 4 
    if confidences is not None:
        # confidences = torch.ones(batch_size, n_views, dtype=torch.float32, device=points.device)
        A *= confidences.permute(1,0,2).view(batch_size, -1, 1, 1)
    u, s, vh = torch.svd(A.view(batch_size, -1, 4)) # 16, 4
    point_3d_homo = -vh[..., 3]
    point_3d = homogeneous_to_euclidean(point_3d_homo)
    return point_3d




def triangulate_batch_of_points(proj_matricies_batch, points_batch, confidences_batch=None):
    batch_size, n_views, n_joints = points_batch.shape[:3]
    point_3d_batch = torch.zeros(batch_size, n_joints, 3, dtype=torch.float32, device=points_batch.device)
    for batch_i in range(batch_size):
        points = points_batch[batch_i, :, :, :]
        confidences = confidences_batch[batch_i] if confidences_batch is not None else None
        point_3d = triangulate_point_from_multiple_views_linear_torch(proj_matricies_batch[batch_i], points, confidences=confidences)
        point_3d_batch[batch_i] = point_3d
    return point_3d_batch




class Regr3D_gs (Regr3D_clean):
    """ Ensure that all 3D points are correct.
        Asymmetric loss: view1 is supposed to be the anchor.

        P1 = RT1 @ D1
        P2 = RT2 @ D2
        loss1 = (I @ pred_D1) - (RT1^-1 @ RT1 @ D1)
        loss2 = (RT21 @ pred_D2) - (RT1^-1 @ P2)
                = (RT21 @ pred_D2) - (RT1^-1 @ RT2 @ D2)
    """
    def __init__(self, criterion, disable_rayloss = False, scaling_mode='interp_5e-4_0.1', norm_mode='avg_dis', gt_scale=False, test=False):
        super().__init__(criterion)
        self.norm_mode = norm_mode
        self.disable_rayloss = disable_rayloss
        self.scaling_mode = {'type':scaling_mode.split('_')[0], 'min_scaling': eval(scaling_mode.split('_')[1]), 'max_scaling': eval(scaling_mode.split('_')[2])}
        self.gt_scale = gt_scale
        from .perceptual import PerceptualLoss
        self.perceptual_loss = PerceptualLoss().cuda().eval()
        for param in self.perceptual_loss.parameters():
            param.requires_grad = False
        self.random_crop = RandomCrop(224)
        self.epoch = 0
        self.test = test
        self.gt_num_image = 0
        self.render_landscape = transpose_to_landscape_render(self.render)
        self.loss2d_landscape = transpose_to_landscape(self.loss2d)

    def render(self, render_params, image_size):
        latent, output_fxfycxcy, output_c2ws = render_params
        (H_org, W_org) = image_size
        H = H_org
        W = W_org
        scaling_factor = latent.pop('scaling_factor')
        gs_render = GaussianRenderer(H_org, W_org, patch_size=128, gs_kwargs={'type':self.scaling_mode['type'], 'min_scaling': self.scaling_mode['min_scaling'], 'max_scaling': self.scaling_mode['max_scaling'], 'scaling_factor': scaling_factor}) # scaling_factor
        results = gs_render(latent, output_fxfycxcy.reshape(1,-1,4), output_c2ws.reshape(1,-1,4,4))
        images = results['image']
        image_mask = results['alpha']
        image_mask = image_mask.reshape(-1,1,H,W).permute(0,2,3,1)
        images = images.reshape(-1,3,H,W).permute(0,2,3,1)
        return {'images': images, 'image_mask': image_mask}

    def loss2d(self, input, image_size):
        pr_pts, fxfycxcy, c2ws = input
        H, W = image_size
        loss_2d = reproj2d(fxfycxcy, c2ws, pr_pts, image_size)
        return {'loss_2d': loss_2d}

    def get_all_pts3d(self, gt1, gt2, pred1, pred2, trajectory_pred, dist_clip=None, render_gt=None):
        # everything is normalized w.r.t. camera of view1
        gt_pts1, gt_pts2, pr_pts1, pr_pts2, valid1, valid2, conf1, conf2, trajectory_gt, trajectory_pred, R, trajectory_R_prior, trajectory_R_post, focal_length_gt, focal_length_pred, se3_gt, se3_pred, pr_pts2_transform, monitoring = super().get_all_pts3d(gt1, gt2, pred1, pred2, trajectory_pred, dist_clip=None, render_gt=render_gt)
        norm_factor_gt, norm_factor_pr = monitoring
        B, H, W, _ = gt_pts1.shape
        pr_pts1_pre, pr_pts2_pre = pred1['pts3d_pre'].reshape(*pr_pts1.shape), pred2['pts3d_pre'].reshape(*pr_pts2.shape)
        pr_pts1_pre, pr_pts2_pre, _ = normalize_pointcloud(pr_pts1_pre, pr_pts2_pre, self.norm_mode, valid1, valid2, ret_factor=True)
        feature1 = pred1['feature'].reshape(B,H,W,3)
        feature2 = pred2['feature'].reshape(B,-1,W,3)
        feature = torch.cat((feature1, feature2), dim=1).float()
        image_2d = torch.cat((feature1.reshape(B,-1, H,W,3), feature2.reshape(B,-1, H,W,3)), dim=1)
        image_2d = torch.sigmoid(image_2d)
        opacity1 = pred1['opacity'].reshape(B,H,W,1)
        opacity2 = pred2['opacity'].reshape(B,-1,W,1)
        opacity = torch.cat((opacity1, opacity2), dim=1).float()
        scaling1 = pred1['scaling'].reshape(B,H,W,3)
        scaling2 = pred2['scaling'].reshape(B,-1,W,3)
        scaling = torch.cat((scaling1, scaling2), dim=1).float()
        rotation1 = pred1['rotation'].reshape(B,H,W,4)
        rotation2 = pred2['rotation'].reshape(B,-1,W,4)
        rotation = torch.cat((rotation1, rotation2), dim=1).float()
        output_fxfycxcy = torch.stack([gt['fxfycxcy'] for gt in render_gt], dim=1)
        render_mask = torch.stack([gt['render_mask'] for gt in render_gt], dim=1)
        shape = torch.stack([gt['true_shape'] for gt in render_gt], dim=1)
        xyz = torch.cat((pr_pts1, pr_pts2), dim=1)
        pr_pts = torch.cat((pr_pts1.reshape(B,-1,H,W,3), pr_pts2.reshape(B,-1,H,W,3)), dim=1)
        xyz_gt = torch.cat((gt_pts1.reshape(B,-1,H,W,3), gt_pts2.reshape(B,-1,H,W,3)), dim=1).detach()
        # TODO rewrite transform cameras

        images_list = []
        image_mask_list = []
        images_gt = torch.stack([gt['img_org'] for gt in render_gt], dim=1)
        images_gt = images_gt / 2 + 0.5
        feature = feature.reshape(B, -1, 3)
        norm_factor_gt = monitoring[0]
        output_c2ws = torch.stack([gt['camera_pose'] for gt in render_gt], dim=1)
        camera_pose = torch.stack([gt1_per['camera_pose'] for gt1_per in gt1], dim=1)
        in_camera1 = inv(camera_pose)
        output_c2ws = torch.einsum('bnjk,bnkl->bnjl', in_camera1.repeat(1,output_c2ws.shape[1],1,1), output_c2ws)
        output_c2ws[..., :3, 3:] = output_c2ws[..., :3, 3:] / (norm_factor_gt.detach() + 1e-5)
        for i in range(B):
            # try:
            latent = {'xyz': xyz.reshape(B, -1, 3)[i:i+1], 'feature': feature.reshape(B, -1, 3)[i:i+1], 'opacity': opacity.reshape(B, -1, 1)[i:i+1], 'scaling': scaling.reshape(B, -1, 3)[i:i+1], 'rotation': rotation.reshape(B, -1, 4)[i:i+1], 'scaling_factor': norm_factor_pr.reshape(B, -1)[i:i+1]}
            ret = self.render_landscape([latent, output_fxfycxcy[i:i+1].reshape(-1,4), output_c2ws[i:i+1].reshape(-1,4,4)], shape[i:i+1].reshape(-1,2))
            # except:
            #     import ipdb; ipdb.set_trace()
            images = ret['images']
            image_mask = ret['image_mask']
            images_list.append(images)
            image_mask_list.append(image_mask)
        images = torch.stack(images_list, dim=0).permute(0,1,4,2,3)
        image_mask = torch.stack(image_mask_list, dim=0).permute(0,1,4,2,3)
        # with torch.no_grad():
        #     images_gt_geo, image_mask_gt_geo = [], []
        #     for i in range(B):
        #         latent = {'xyz': xyz_gt.reshape(B, -1, 3)[i:i+1], 'feature': feature.reshape(B, -1, 3)[i:i+1], 'opacity': opacity.reshape(B, -1, 1)[i:i+1], 'scaling': scaling.reshape(B, -1, 3)[i:i+1], 'rotation': rotation.reshape(B, -1, 4)[i:i+1]}
        #         ret = self.render_landscape([latent, output_fxfycxcy[i:i+1].reshape(-1,4), output_c2ws[i:i+1].reshape(-1,4,4)], shape[i:i+1].reshape(-1,2))
        #         images_gt_render = ret['images']
        #         image_mask_gt = ret['image_mask']
        #         images_gt_geo.append(images_gt_render)
        #         image_mask_gt_geo.append(image_mask_gt)
        #     images_gt_geo = torch.stack(images_gt_geo, dim=0).permute(0,1,4,2,3)
        #     image_mask_gt_geo = torch.stack(image_mask_gt_geo, dim=0).permute(0,1,4,2,3)
        image_mask_gt_geo = image_mask.clone()
        images_gt_geo = images.clone()
        shape_input = torch.stack([gt['true_shape'] for gt in gt1+gt2], dim=1)
        input_fxfycxcy = output_fxfycxcy[:,:4]
        input_c2ws = output_c2ws[:,:4]
        shape_input = shape_input[:,:4]
        pr_pts = pr_pts[:,:4]
        loss_2d = self.loss2d_landscape([pr_pts.reshape(-1, H, W, 3), input_fxfycxcy.reshape(-1,4), input_c2ws.reshape(-1,4,4)], shape_input.reshape(-1,2))
        # import ipdb; ipdb.set_trace()
        # masks = torch.cat((valid1.reshape(B, -1, H, W), valid2.reshape(B, -1, H, W)), 1)

        loss_2d = loss_2d['loss_2d'].mean()
        if self.disable_rayloss:
            loss_2d = torch.zeros_like(loss_2d).to(loss_2d.device)

        if self.test:
            with torch.no_grad():
                render_pose = interpolate_poses([0,1,2,3,4,5,6,7], trajectory_pred.detach().cpu().numpy()[0,0], list(np.linspace(0,1,280)), 0)
                render_pose = np.stack(render_pose)
                render_pose = torch.tensor(render_pose).cuda()[:200]
                i=0
                B = 1 
                clip = 8
                # pr_pts1_pre, pr_pts2_pre, norm_factor_pr = normalize_pointcloud(pr_pts1_pre, pr_pts2_pre, self.norm_mode, valid1, valid2, ret_factor=True)
                latent = {'xyz': xyz.reshape(B, -1, 3)[i:i+1], 'feature': feature.reshape(B, -1, 3)[i:i+1], 'opacity': opacity.reshape(B, -1, 1)[i:i+1], 'scaling': scaling.reshape(B, -1, 3)[i:i+1], 'rotation': rotation.reshape(B, -1, 4)[i:i+1], 'scaling_factor': norm_factor_pr.reshape(B, -1)[i:i+1]}
                ret = self.render_landscape([latent, output_fxfycxcy[i:i+1][:1].reshape(-1,4).repeat(render_pose.reshape(-1,4,4).shape[0],1), render_pose.reshape(-1,4,4)], shape[i:i+1].reshape(-1,2))
                pc = GRMGaussianModel(sh_degree=0,
                        scaling_kwargs={'type':'interp', 'min_scaling': 5e-4, 'max_scaling': 0.1, 'scaling_factor': norm_factor_pr[i:i+1].reshape(B, -1)}, **latent)
                
                # results = gs_render.render(latent, output_fxfycxcy[:, :1].repeat(1,len(render_pose),1).to(xyz), render_pose.reshape(B,-1,4,4).to(xyz))
                video = ret['images'].detach().cpu().numpy()
                import imageio
                import os
                i = 0
                output_path = f'test/output_video_{i}.mp4'
                os.makedirs('test', exist_ok=True)
                while os.path.exists(output_path):
                    i = i  + 1
                    output_path = f'test/output_video_{i}.mp4'
                
                pc.save_ply_v2(f'test/{i}.ply')
                import matplotlib.pyplot as plt
                plt.imshow(images[0,0].detach().cpu().numpy().transpose(1,2,0))
                plt.savefig('test/output.png')
                writer = imageio.get_writer(output_path, fps=30)  # You can specify the filename and frames per second
                for frame in video:  # Example for 100 frames
                    # Add the frame to the video
                    frame = (frame * 255).astype(np.uint8)
                    writer.append_data(frame)
                # Close the writer to finalize the video
                writer.close()

        metric_conf1 = pred1['conf']
        metric_conf2 = pred2['conf']
        return gt_pts1, gt_pts2, pr_pts1, pr_pts2, valid1, valid2, metric_conf1, metric_conf2, trajectory_gt, trajectory_pred, images, images_gt, image_mask, images_gt_geo, image_mask_gt_geo, render_mask, R, trajectory_R_prior, trajectory_R_post, pr_pts1_pre, pr_pts2_pre, loss_2d, image_2d, {}

    def compute_loss(self, gt1, gt2, pred1, pred2, trajectory_pred, render_gt=None, **kw):
        gt_pts1, gt_pts2, pr_pts1, pr_pts2,  mask1, mask2, metric_conf1, metric_conf2, trajectory, trajectory_pred, images, images_gt, image_mask, images_gt_geo, image_mask_gt_geo, render_mask,  R, trajectory_R_prior, trajectory_R_post, pr_pts1_pre, pr_pts2_pre, loss_2d, image_2d, monitoring = self.get_all_pts3d(gt1, gt2, pred1, pred2, trajectory_pred, render_gt=render_gt, **kw)
        # loss on img1 side
        # loss on gt2 side
        l1 = self.criterion(pr_pts1[mask1], gt_pts1[mask1])
        l2 = self.criterion(pr_pts2[mask2], gt_pts2[mask2])
        l1_pre = self.criterion(pr_pts1_pre[mask1], gt_pts1[mask1])
        l2_pre = self.criterion(pr_pts2_pre[mask2], gt_pts2[mask2])
        n_predictions = len(trajectory_pred)
        pose_r_loss = 0.0
        pose_t_loss = 0.0
        render_mask = render_mask[:,:,None].repeat(1,1,3,1,1)
        render_mask = render_mask.float()
        image_2d = image_2d.permute(0,1,4,2,3)
        images_org = images.clone() * render_mask.float()
        
        loss_image_pred_geo = ((images.reshape(*render_mask.shape)[:,:4][render_mask[:,:4] > 0.] - images_gt.reshape(*render_mask.shape)[:,:4][render_mask[:,:4] > 0.]) ** 2)
        loss_image_pred_geo = loss_image_pred_geo.mean() if loss_image_pred_geo.numel() > 0 else 0
        loss_image_pred_geo_novel = ((images.reshape(*render_mask.shape)[:,4:][render_mask[:,4:] > 0.] - images_gt.reshape(*render_mask.shape)[:,4:][render_mask[:,4:] > 0.]) ** 2)
        loss_image_pred_geo_novel = loss_image_pred_geo_novel.mean() if loss_image_pred_geo_novel.numel() > 0 else 0
        # loss_image_pred_self = ((image_2d[:,:4] - images_gt[:,:4]) ** 2).mean()
        loss_image_pred_geo = loss_image_pred_geo  + loss_image_pred_geo_novel * 2#+ loss_image_pred_self
        images_masked = images 
        # images_masked[:,:4] = (images_masked[:,:4] * 0.5 + image_2d[:,:4] * 0.5)

        images_masked = images_masked * render_mask + images_gt * (1 - render_mask)
        images_masked, images_gt_1 = self.random_crop(images_masked, images_gt)
        images_masked = images_masked.reshape(-1,*images_gt_1.shape[-3:])
        images_gt_1 = images_gt_1.reshape(-1,*images_gt_1.shape[-3:])
        loss_vgg = self.perceptual_loss(images_masked, images_gt_1) * 0.1
        loss_image = loss_vgg + loss_image_pred_geo
        # image_2d
        gamma = 0.8
        for i in range(n_predictions):
            i_weight = gamma ** (n_predictions - 1 - i%4) 
            trajectory_pred_iter = trajectory_pred[i]
            trajectory_gt_iter = trajectory[0]
            trajectory_pred_r = trajectory_pred_iter[..., 3:]
            trajectory_pred_t = trajectory_pred_iter[..., :3]
            trajectory_gt_r = trajectory_gt_iter[..., 3:]
            trajectory_gt_t = trajectory_gt_iter[..., :3]
            pose_r_loss += i_weight * (trajectory_gt_r - trajectory_pred_r).abs().mean() * 2.5
            pose_t_loss += i_weight * (trajectory_gt_t - trajectory_pred_t).abs().mean() * 2.5

        pose_t_loss = pose_t_loss
        pose_r_loss = pose_r_loss
        rot_err_prior = torch.rad2deg(rotation_distance(R.reshape(*trajectory_R_prior.shape).float(), trajectory_R_prior.float())).mean()
        rot_err_post = torch.rad2deg(rotation_distance(R.reshape(*trajectory_R_post.shape).float(), trajectory_R_post.float())).mean()
        mse = torch.mean(((images_org.reshape(*render_mask.shape)[render_mask > 0.] - images_gt.reshape(*render_mask.shape)[render_mask > 0.]) * 255) ** 2)
        PSNR = 20 * torch.log10(255/torch.sqrt(mse))
        Reg_1 = l1.mean() if l1.numel() > 0 else 0
        Reg_2 = l2.mean() if l2.numel() > 0 else 0
        Reg_1_pre = l1_pre.mean() if l1_pre.numel() > 0 else 0
        Reg_2_pre = l2_pre.mean() if l2_pre.numel() > 0 else 0
        with torch.no_grad():
            batch_size, num_views = trajectory_pred.shape[1:3]
            rot_err_prior = torch.rad2deg(rotation_distance(R.reshape(*trajectory_R_prior.shape).float(), trajectory_R_prior.float())).mean()
            rot_err_post = torch.rad2deg(rotation_distance(R.reshape(*trajectory_R_post.shape).float(), trajectory_R_post.float())).mean()
            pair_idx_i1, pair_idx_i2 = batched_all_pairs(batch_size, num_views)
            se3_gt = torch.cat((R, trajectory[0,..., :3, None]), dim=-1).reshape(-1 , 3, 4)
            se3_pred_prior = torch.cat((trajectory_R_prior, trajectory_pred[3][..., :3][..., None]), dim=-1).reshape(-1 , 3, 4)
            se3_pred_post = torch.cat((trajectory_R_post, trajectory_pred[-1][..., :3][..., None]), dim=-1).reshape(-1 , 3, 4)
            bottom_ = torch.tensor([[[0,0,0,1]]]).to(se3_gt.device)
            se3_gt = torch.cat((se3_gt, bottom_.repeat(se3_gt.shape[0],1,1)), dim=1)
            se3_pred_prior = torch.cat((se3_pred_prior, bottom_.repeat(se3_pred_prior.shape[0],1,1)), dim=1)
            se3_pred_post = torch.cat((se3_pred_post, bottom_.repeat(se3_pred_post.shape[0],1,1)), dim=1)
            relative_pose_gt = closed_form_inverse(se3_gt[pair_idx_i1]).bmm(se3_gt[pair_idx_i2])
            relative_pose_pred_prior = closed_form_inverse(se3_pred_prior[pair_idx_i1]).bmm(se3_pred_prior[pair_idx_i2])
            relative_pose_pred_post = closed_form_inverse(se3_pred_post[pair_idx_i1]).bmm(se3_pred_post[pair_idx_i2])
            rel_tangle_deg_prior = translation_angle(relative_pose_gt[:, :3, 3], relative_pose_pred_prior[:, :3, 3]).mean()
            rel_tangle_deg_post = translation_angle(relative_pose_gt[:, :3, 3], relative_pose_pred_post[:, :3, 3]).mean()
        self_name = type(self).__name__
        details = {'PSNR': float(PSNR), 'loss_2d': float(loss_2d),'post_tangle_deg': float(rel_tangle_deg_post), 'prior_tangle_deg': float(rel_tangle_deg_prior), 'post_roterr': float(rot_err_post), 'prior_roterr': float(rot_err_prior), 'loss_pred_geo': float(loss_image_pred_geo), 'loss_vgg': float(loss_vgg), self_name+'_1': float(Reg_1), self_name+'_2': float(Reg_2),self_name+'_t_pose': float(pose_t_loss), self_name+'_r_pose': float(pose_r_loss), 'trajectory_gt_t_first': float(trajectory_gt_t[:,0].abs().mean()), 'trajectory_pred_t_first': float(trajectory_pred_t[:,0].abs().mean()), self_name+'_image': float(loss_image), 'images': images_org,  'images_gt':images_gt * render_mask.float(), 'images_gt_geo': images_gt_geo, self_name+'_1_pre': float(Reg_1_pre), self_name+'_2_pre': float(Reg_2_pre), 'image_2d': image_2d}
        return Sum((l1, mask1), (l2, mask2), (pose_r_loss, None), (pose_t_loss, None), (0, None), (loss_image, None), (loss_2d, None), (0, None), (0, None), (0, None)), details
        #  ((loss1, msk1), (loss2, msk2), (pose_r_loss, _), (pose_t_loss, _), (pose_f_loss, _), (loss_image, _), (loss_2d, _), (loss1_coarse, msk1_coarse), (loss2_coarse, msk2_coarse))

def sample_points_from_mask_torch(mask, N):
    valid_indices = torch.nonzero(mask, as_tuple=False)
    
    # Random sampling with replacement if needed
    if valid_indices.size(0) < N:
        # Allow replacement when there are not enough valid points
        sampled_indices = valid_indices[torch.randint(0, valid_indices.size(0), (N,))]
    else:
        # No replacement needed when there are enough valid points
        sampled_indices = valid_indices[torch.randperm(valid_indices.size(0))[:N]]
    
    return sampled_indices


def triangulate_one_scene(pts, intrinsics, w2cs, conf, rgb=None, num_points_3d=4096, conf_threshold=3,):
    # pts point map:  S, H, W, 3 
    # intrinsics:  S, 3, 3
    # w2cs:  S, 4, 4
    # conf: reference view confidence map :  S, H, W
    # rgb:  S, 3, H, W

    S, H, W, _ = pts.shape
    conf_reference = conf[0]
    conf_reference_valid = conf_reference > conf_threshold

    sampled_indices = sample_points_from_mask_torch(conf_reference_valid, num_points_3d)
    
    pts_reference_sampled = pts[0,sampled_indices[:,0], sampled_indices[:,1]]            
    rgb_reference_sampled = rgb.permute(0,2,3,1)[0, sampled_indices[:,0], sampled_indices[:,1]]
    
    # Reshape for computation: Sx(H*W)x3
    pts_reshaped = pts.view(S, H*W, 3)

    track_matches = []
    track_matches.append(sampled_indices)
    
    track_confs = []
    track_confs.append(conf_reference[sampled_indices[:,0], sampled_indices[:,1]])

    original_pts = []
    original_pts.append(pts_reference_sampled)
    
    
    for idxS in range(1, S):
        with autocast(dtype=torch.half):
            dist = pts_reshaped[idxS].unsqueeze(0) - pts_reference_sampled.unsqueeze(1)
            dist = (dist**2).sum(dim=-1)
            min_dist_indices = dist.argmin(dim=1)
            
        y_indices = min_dist_indices // W  
        x_indices = min_dist_indices % W   

        # Stack indices to get the final Nx2 tensor
        matches = torch.stack((y_indices, x_indices), dim=1)
        
        track_matches.append(matches)
        track_confs.append(conf[idxS, y_indices, x_indices])
        original_pts.append(pts[idxS, y_indices, x_indices])
    track_matches= torch.stack(track_matches, dim=0)
    track_confs= torch.stack(track_confs, dim=0)
    original_pts = torch.stack(original_pts, dim=0)
    
    # track_matches: SxNx2
    # track_confs: SxN
    
    assert (intrinsics[:, 0, 1] <= 1e-6).all(), "intrinsics should not have skew"
    assert (intrinsics[:, 1, 0] <= 1e-6).all(), "intrinsics should not have skew"
    
    
    principal_point = intrinsics[:, [0, 1], [2, 2]].unsqueeze(-2)
    focal_length = intrinsics[:, [0, 1], [0, 1]].unsqueeze(-2)

    # NOTE
    # Very careful here!
    # switch from yx to xy to fit the codebase of vggsfm
    track_matches = track_matches[..., [1,0]]
    # original_pts = original_pts[..., [1,0]]
    
    tracks_normalized = (track_matches - principal_point) / focal_length

    track_confs_vggsfm = track_confs.clone()
    track_confs_vggsfm = track_confs_vggsfm - conf_threshold
    track_confs_vggsfm = track_confs_vggsfm + 0.5
    
    
    
    with autocast(dtype=torch.double):
        triangulated_points, inlier_num, inlier_mask = triangulate_tracks(w2cs[:, :3], tracks_normalized, 
                        max_ransac_iters=64, lo_num=32, 
                        track_vis=torch.ones_like(tracks_normalized[..., 0]),
                        track_score=track_confs_vggsfm)
            
    # triangulated_points: num_points_3d, 3
    # rgb_reference_sampled: num_points_3d, 3
    # inlier_num: num_points_3d
    # inlier_mask: num_points_3d, S
    # original_pts: S, num_points_3d, 3
    # track_matches: S, num_points_3d, 2
    # track_confs: S,num_points_3d
    
    return triangulated_points, rgb_reference_sampled, inlier_num, inlier_mask, original_pts, track_matches, track_confs

def rgb_to_sh(rgb):
    C0 = 0.28209479177387814
    return (rgb - 0.5) / C0

def sh_to_rgb(sh):
    C0 = 0.28209479177387814
    return sh * C0 + 0.5

def pnp():
    pred_poses = []
    for i in range(pts3d.shape[1]):
        # Create mesh grid
        shape_input_each = shape_input[:, i]
        H, W = shape_input_each[0]
        x = np.arange(W)
        y = np.arange(H)
        X, Y = np.meshgrid(x, y)
        mesh_grid = np.stack((Y, X), axis=-1)
        conf_each = conf[0,i]
        conf_thres = torch.quantile(conf_each.flatten(), 0.6)
        cur_inlier = conf_each > conf_thres
        cur_inlier = cur_inlier.detach().cpu().numpy()
        ransac_thres = 10
        cur_intri = intrinsics[0,i].detach().cpu().numpy()
        camera = {'model': 'PINHOLE', 'width': W, 'height': H, 'params': [cur_intri[0, 0], cur_intri[1, 1], cur_intri[0, 2], cur_intri[1, 2] ]}
        cur_pts3d = pts3d[0,i].detach().cpu().numpy()
        # pose, info = poselib.estimate_absolute_pose(mesh_grid[cur_inlier], 
        #                                             cur_pts3d[cur_inlier], 
        #                                             camera, {'max_reproj_error': ransac_thres}, {})

        # inliers = np.array(info["inliers"])
        cam_from_world = pose.Rt
        # Do the inverse. You can pick any way you want
        cam_from_world_pycolmap = pycolmap.Rigid3d( pycolmap.Rotation3d(cam_from_world[:3, :3]), cam_from_world[:3, 3], )
        cam_to_world_pycolmap = cam_from_world_pycolmap.inverse()
        pred_pose = cam_to_world_pycolmap.matrix()
        pred_poses.append(pred_pose)
    pred_poses = np.stack(pred_poses, axis=0)
    pred_poses = torch.tensor(pred_poses).to(images_gt)
    pred_poses = pred_poses[None]
    triangluation_points = None

def p2p_triangulation(extrinsics, intrinsics, pts3ds, confs, images_list):
    pixel_tol = 0
    subsample = 4

    pts_list = []
    col_list = []
    pts_pred_list = []
    conf_list = []
    w2cs = extrinsics
    allcam_Ps_list = torch.einsum('bnij,bnjk->bnik', intrinsics, w2cs[...,:3,:])
    for b in range(len(pts3ds)):
        pts3d = pts3ds[b]
        conf = confs[b]
        images = images_list[b]
        allcam_Ps = allcam_Ps_list[b]
        colors_list = []
        points_list = []
        points_pred_list = []
        conf_corr_list = []
        for i in range(len(pts3d)-1):
            corres = extract_correspondences_nonsym(pts3d[i], pts3d[i+1], conf[i], conf[i+1],
                                            device=pts3d.device, subsample=subsample, pixel_tol=pixel_tol,
                                            ptmap_key='3d')
            conf_corr = corres[2]
            conf_thr = 1.01
            mask = conf_corr >= conf_thr
            if mask.sum() < 10:
                mask = conf_corr >= 1.
            matches_im0 = corres[0][mask]
            matches_im1 = corres[1][mask]
            conf_corr = conf_corr[mask]
            allpoints = torch.cat([matches_im0.reshape([1,1,-1,2]), matches_im1.reshape([1,1,-1,2])],dim=1) # [BxNv, 2, HxW, 2]
            colors1 = images[i, :,  matches_im0[:,1], matches_im0[:,0]]
            colors2 = images[i+1, :,  matches_im1[:,1], matches_im1[:,0]]
            pts3d1 = pts3d[i, matches_im0[:,1], matches_im0[:,0], :]
            pts3d2 = pts3d[i+1, matches_im1[:,1], matches_im1[:,0], :]
            # pts3ds = (pts3d1 + pts3d2) / 2
            # colors = (colors1.T + colors2.T) / 4 + 0.5 #torch.cat([colors1.T, colors2.T], dim=0)
            cam_Ps1, cam_Ps2 = allcam_Ps[[i]], allcam_Ps[[i+1]] # [B, Nv, 3, 4]
            formatted_camPs = torch.cat([cam_Ps1.reshape([1,1,3,4]), cam_Ps2.reshape([1,1,3,4])],dim=1) # [BxNv, 2, 3, 4]
            points_3d_world = batched_triangulate(allpoints, formatted_camPs) # [BxNv, HxW, three] 
            # colors_list.append(colors.reshape(-1,3))
            colors_list.append(colors1.T.reshape(-1,3)/2+0.5)
            colors_list.append(colors2.T.reshape(-1,3)/2+0.5)
            points_list.append(points_3d_world.reshape(-1,3))
            points_list.append(points_3d_world.reshape(-1,3))
            # points_pred_list.append(pts3ds)
            points_pred_list.append(pts3d1)
            points_pred_list.append(pts3d2)
            conf_corr_list.append(conf_corr)
            conf_corr_list.append(conf_corr)
        # mask = conf > 3
        # image_list = [view['img_org'] for view in gt1+gt2]
        # image_list = torch.cat(image_list,0).permute(0,2,3,1).detach().cpu().numpy()
        # imgs = image_list / 2 + 0.5
        # imgs = imgs
        # scene = trimesh.Scene()
        # meshes = []
        # import imageio
        # import os        
        # outfile = os.path.join('./data/mesh', view1[0]['instance'][0])
        # os.makedirs(outfile, exist_ok=True)
        conf = torch.cat(conf_corr_list, 0).reshape(-1, 1)
        pts = torch.cat(points_list, 0).reshape(-1, 3)
        col = torch.cat(colors_list, 0).reshape(-1, 3)
        pts_pred = torch.cat(points_pred_list, 0).reshape(-1, 3)
        pts_list.append(pts)
        col_list.append(col)
        pts_pred_list.append(pts_pred)
        conf_list.append(conf)
    pts_pred = pts_pred_list
    pts = pts_list
    col = col_list
    confidence = conf_list
    return pts, col, pts_pred, confidence

def convert_to_buffer(module: nn.Module, persistent: bool = True):
    # Recurse over child modules.
    for name, child in list(module.named_children()):
        convert_to_buffer(child, persistent)

    # Also re-save buffers to change persistence.
    for name, parameter_or_buffer in (
        *module.named_parameters(recurse=False),
        *module.named_buffers(recurse=False),
    ):
        value = parameter_or_buffer.detach().clone()
        delattr(module, name)
        module.register_buffer(name, value, persistent=persistent)


class Regr3D_gs_unsupervised_2v(Regr3D_clean):
    """ Ensure that all 3D points are correct.
        Asymmetric loss: view1 is supposed to be the anchor.

        P1 = RT1 @ D1
        P2 = RT2 @ D2
        loss1 = (I @ pred_D1) - (RT1^-1 @ RT1 @ D1)
        loss2 = (RT21 @ pred_D2) - (RT1^-1 @ P2)
              = (RT21 @ pred_D2) - (RT1^-1 @ RT2 @ D2)
    """

    def __init__(self, criterion, alpha, multiview_triangulation = True, disable_rayloss = False, scaling_mode='interp_5e-4_0.1_3', norm_mode='avg_dis', gt_scale=False, test=False):
        super().__init__(criterion)
        self.norm_mode = norm_mode
        self.disable_rayloss = disable_rayloss
        self.multiview_triangulation = multiview_triangulation
        self.scaling_mode = {'type':scaling_mode.split('_')[0], 'min_scaling': eval(scaling_mode.split('_')[1]), 'max_scaling': eval(scaling_mode.split('_')[2]), 'shift': eval(scaling_mode.split('_')[3])}
        self.gt_scale = gt_scale
        from lpips import LPIPS
        self.lpips = LPIPS(net="vgg", model_path='checkpoints/vgg16-397923af.pth')# , model_path='vgg.pth'
        convert_to_buffer(self.lpips, persistent=False)
        self.lpips = self.lpips.cuda()
        self.epoch = 0
        self.test = test
        self.gt_num_image = 0
        self.render_landscape = transpose_to_landscape_render(self.render)
        self.loss2d_landscape = transpose_to_landscape(self.loss2d)
        self.alpha = alpha
        self.sm = nn.SmoothL1Loss()
        self.disp_loss =  ScaleAndShiftInvariantLoss()

    def render(self, render_params, image_size):
        latent, output_fxfycxcy, output_c2ws = render_params
        (H_org, W_org) = image_size
        H = H_org
        W = W_org
        # if 'scaling_factor' in latent.keys():
        #     scaling_factor = latent.pop('scaling_factor')
        #     if self.scaling_mode['type'] == 'precomp':
        #         x = torch.clip(latent['pre_scaling'] - self.scaling_mode['shift'], max=np.log(3))
        #         latent['scaling'] = torch.exp(x).clamp(min= self.scaling_mode['min_scaling'], max=self.scaling_mode['max_scaling']) / scaling_factor
        # else:
        #     scaling_factor = torch.tensor(1.0).to(latent['xyz'])
        new_latent = {}
        if self.scaling_mode['type'] == 'precomp': 
            scaling_factor = latent['scaling_factor']
            x = torch.clip(latent['pre_scaling'] - self.scaling_mode['shift'], max=np.log(0.3))
            new_latent['scaling'] = torch.exp(x).clamp(min= self.scaling_mode['min_scaling'], max=self.scaling_mode['max_scaling'])
            skip = ['pre_scaling', 'scaling', 'scaling_factor']
        else:
            skip = ['pre_scaling', 'scaling_factor']
        for key in latent.keys():
            if key not in skip:
                new_latent[key] = latent[key]
        gs_render = GaussianRenderer(H_org, W_org, patch_size=128, gs_kwargs={'type':self.scaling_mode['type'], 'min_scaling': self.scaling_mode['min_scaling'], 'max_scaling': self.scaling_mode['max_scaling'], 'scaling_factor': scaling_factor}) # scaling_factor
        results = gs_render(new_latent, output_fxfycxcy.reshape(1,-1,4), output_c2ws.reshape(1,-1,4,4))
        images = results['image']
        image_mask = results['alpha']
        depth = results['depth']
        depth = depth.reshape(-1,1,H,W).permute(0,2,3,1)
        image_mask = image_mask.reshape(-1,1,H,W).permute(0,2,3,1)
        images = images.reshape(-1,3,H,W).permute(0,2,3,1)
        return {'images': images, 'image_mask': image_mask, 'depth':depth}
    
    def loss2d(self, input, image_size):
        pr_pts, fxfycxcy, c2ws = input
        H, W = image_size
        loss_2d = reproj2d(fxfycxcy, c2ws, pr_pts, image_size)
        return {'loss_2d': loss_2d}
    
    def get_all_pts3d(self, gt1, gt2, pred1, pred2, trajectory_pred, dist_clip=None, render_gt=None):
        # everything is normalized w.r.t. camera of view1        
        pr_pts1 = pred1['pts3d']
        pr_pts2 = pred2['pts3d']
        B, H, W, _= pr_pts1.shape
    
        pr_pts1 = pr_pts1.reshape(B,H,W,3)
        pr_pts2 = pr_pts2.reshape(B,H,W,3)
        # generate the gt camera trajectory
        with torch.no_grad():
            trajectory_R_prior = trajectory_pred[:4][-1]['R'].reshape(B, -1, 3, 3)
            trajectory_R_post = trajectory_pred[-1]['R'].reshape(B, -1, 3, 3)
        sh_dim = pred1['feature'].shape[-1]
        feature1 = pred1['feature'].reshape(B,H,W,sh_dim)
        feature2 = pred2['feature'].reshape(B,-1,W,sh_dim)
        feature = torch.cat((feature1, feature2), dim=1).float()
        image_2d = torch.cat((feature1.reshape(B,-1, H,W,sh_dim)[..., :3], feature2.reshape(B,-1, H,W,sh_dim)[..., :3]), dim=1)
        if sh_dim > 3:
            image_2d = sh_to_rgb(image_2d).clamp(0,1)
        else:
            image_2d = torch.sigmoid(image_2d)
        opacity1 = pred1['opacity'].reshape(B,H,W,1)
        opacity2 = pred2['opacity'].reshape(B,-1,W,1)
        opacity = torch.cat((opacity1, opacity2), dim=1).float()
        scaling1 = pred1['scaling'].reshape(B,H,W,3)
        scaling2 = pred2['scaling'].reshape(B,-1,W,3)
        scaling = torch.cat((scaling1, scaling2), dim=1).float()
        rotation1 = pred1['rotation'].reshape(B,H,W,4)
        rotation2 = pred2['rotation'].reshape(B,-1,W,4)
        rotation = torch.cat((rotation1, rotation2), dim=1).float()
        output_fxfycxcy = torch.stack([gt['fxfycxcy'] for gt in render_gt], dim=1)
        render_mask = torch.stack([gt['render_mask'] for gt in render_gt], dim=1)
        shape = torch.stack([gt['true_shape'] for gt in render_gt], dim=1)
        pr_pts = torch.cat((pr_pts1.reshape(B,-1,H,W,3), pr_pts2.reshape(B,-1,H,W,3)), dim=1)
        # TODO rewrite transform cameras

        images_gt = torch.stack([gt['img_org'] for gt in render_gt], dim=1)
        images_gt = images_gt / 2 + 0.5
        norm_factor_gt = 1
        output_c2ws = torch.stack([gt['camera_pose'] for gt in render_gt], dim=1)
        camera_pose = torch.stack([gt1_per['camera_pose'] for gt1_per in gt1], dim=1)
        in_camera1 = inv(camera_pose)
        output_c2ws = torch.einsum('bnjk,bnkl->bnjl', in_camera1.repeat(1,output_c2ws.shape[1],1,1), output_c2ws)
        depth_gt = torch.stack([gt['depth_anything'] for gt in render_gt], dim=1)
        if self.test:
            with torch.no_grad():
                trajectory_pred_t = output_c2ws[0,:, :3, 3]
                trajectory_pred_r = matrix_to_quaternion(output_c2ws[0,:, :3,:3])
                trajectory_pred = torch.cat([trajectory_pred_t, trajectory_pred_r], dim=-1)
                render_pose = interpolate_poses(list(range(len(trajectory_pred))), trajectory_pred.detach().cpu().numpy(), list(np.linspace(0,2,280)), 0)
                render_pose = np.stack(render_pose)
                render_pose = torch.tensor(render_pose).cuda()[:200]
                i = 0
                clip = 8
                # pr_pts1_pre, pr_pts2_pre, norm_factor_pr = normalize_pointcloud(pr_pts1_pre, pr_pts2_pre, self.norm_mode, valid1, valid2, ret_factor=True)
                latent = {'xyz': xyz.reshape(B, -1, 3)[i:i+1], 'feature': feature.reshape(B, -1, sh_dim)[i:i+1], 'opacity': opacity.reshape(B, -1, 1)[i:i+1], 'pre_scaling': scaling.reshape(B, -1, 3)[i:i+1].clone(), 'rotation': rotation.reshape(B, -1, 4)[i:i+1], 'scaling_factor': 1}
                # ret = self.render_landscape([latent, output_fxfycxcy[i:i+1][:1].reshape(-1,4).repeat(render_pose.reshape(-1,4,4).shape[0],1), render_pose.reshape(-1,4,4)], shape[i:i+1].reshape(-1,2))
                ret = self.render_landscape([latent, output_fxfycxcy[i:i+1][:1].reshape(-1,4).repeat(render_pose.reshape(-1,4,4).shape[0],1), render_pose.reshape(-1,4,4)], shape[i:i+1].reshape(-1,2))

                import viser
                # import 
                camera_poses = torch.stack([gt['camera_pose'] for gt in gt1+gt2], dim=1)[0]
                camera_poses = torch.einsum('njk,nkl->njl', in_camera1[0].repeat(camera_poses.shape[0],1,1), camera_poses)
                camera_poses[:, :3, 3] = camera_poses[:, :3, 3]
                # camera_poses = camera_poses.inverse()
                images = torch.stack([gt['img_org'] for gt in gt1+gt2], dim=1)
                fxfycxcy = torch.stack([gt['fxfycxcy'] for gt in gt1+gt2], dim=1)[0]
                images = images.permute(0,1,3,4,2)[0] / 2 + 0.5
                colors = images.reshape(B, -1, 3).detach().cpu().numpy() 
                points3d = pr_pts.reshape(B, -1, 3)[0]

                video = ret['images'].detach().cpu().numpy()
                import imageio
                import os
                i = 0
                output_path = f'test/output_video_{i}.mp4'
                os.makedirs('test', exist_ok=True)
                while os.path.exists(output_path):
                    i = i  + 1
                    output_path = f'test/output_video_{i}.mp4'
                # pc.save_ply_v2(f'test/{i}.ply')
                # import matplotlib.pyplot as plt
                # plt.imshow(images[0,0].detach().cpu().numpy().transpose(1,2,0))
                # plt.savefig('test/output.png')
                writer = imageio.get_writer(output_path, fps=30)  # You can specify the filename and frames per second
                for frame in video:  # Example for 100 frames
                    # Add the frame to the video
                    frame = (frame * 255).clip(0,255).astype(np.uint8)
                    writer.append_data(frame)
                # Close the writer to finalize the video
                writer.close()
                for i in range(len(images)):
                    imageio.imwrite(f'test/output_{i}.png', (images[i] * 255).clip(0,255).detach().cpu().numpy().astype(np.uint8))
                vis(camera_poses, images, points3d, colors, fxfycxcy, H, W)

        images_list = []
        image_mask_list = []
        depths_rendered_list = []
        for i in range(B):
            # try:
            latent = {'xyz': pr_pts.reshape(B, -1, 3)[i:i+1], 'feature': feature.reshape(B, -1, sh_dim)[i:i+1], 'opacity': opacity.reshape(B, -1, 1)[i:i+1], 'pre_scaling': scaling.reshape(B, -1, 3)[i:i+1].clone(), 'rotation': rotation.reshape(B, -1, 4)[i:i+1], 'scaling_factor': 1}
            # if 'scaling_factor' not in latent.keys():
            ret = self.render_landscape([latent, output_fxfycxcy[i:i+1].reshape(-1,4), output_c2ws[i:i+1].reshape(-1,4,4)], shape[i:i+1].reshape(-1,2))
            # except:
            #     import ipdb; ipdb.set_trace()
            images = ret['images']
            image_mask = ret['image_mask']
            images_list.append(images)
            image_mask_list.append(image_mask)
            depths_rendered_list.append(ret['depth'])
        images = torch.stack(images_list, dim=0).permute(0,1,4,2,3)
        image_masks = torch.stack(image_mask_list, dim=0).permute(0,1,4,2,3)
        depths_rendered_list = torch.stack(depths_rendered_list, dim=0).permute(0,1,4,2,3)
        
        image_mask_gt_geo = image_masks.clone()
        images_gt_geo = images.clone()
        shape_input = torch.stack([gt['true_shape'] for gt in gt1+gt2], dim=1)
        input_fxfycxcy = output_fxfycxcy[:,:4]
        input_c2ws = output_c2ws[:,:4]
        shape_input = shape_input[:,:4]

        metric_conf1 = pred1['conf']
        metric_conf2 = pred2['conf']
        return pr_pts1, pr_pts2, metric_conf1, metric_conf2, images, images_gt, image_masks, images_gt_geo, image_mask_gt_geo, render_mask, trajectory_R_prior, trajectory_R_post, image_2d, depth_gt, depths_rendered_list, {}


    def get_conf_log(self, x):
        return x, torch.log(x)

    
    def gt_loss(self, gt1, gt2, pred1, pred2, trajectory_pred, render_gt=None, **kw):
        pr_pts1, pr_pts2, metric_conf1, metric_conf2, images, images_gt, image_mask, images_gt_geo, image_mask_gt_geo, render_mask, trajectory_R_prior, trajectory_R_post, image_2d, depth_gt, depths, monitoring = self.get_all_pts3d(gt1, gt2, pred1, pred2, trajectory_pred, render_gt=render_gt, **kw)
        # loss on img1 side
        # loss on gt2 side
        render_mask = render_mask[:,:,None].repeat(1,1,3,1,1)
        render_mask = render_mask.float()
        image_2d = image_2d.permute(0,1,4,2,3)
        images_org = images.clone()
        
        loss_image_pred_geo = ((images.reshape(*render_mask.shape) - images_gt.reshape(*render_mask.shape)) ** 2)
        loss_image_pred_geo = loss_image_pred_geo.mean() if loss_image_pred_geo.numel() > 0 else 0
        loss_image_pred_geo = loss_image_pred_geo
        # depth = (depths - depths.min(dim)) / (depths - near)
        depth_dx = depths.diff(dim=-1)
        depth_dy = depths.diff(dim=-2)
        lossd = 0.25 * (depth_dx.abs().mean() + depth_dy.abs().mean())
        images_masked = images
        loss_vgg = self.lpips.forward(
                            images_masked.reshape(-1,*images_gt.shape[-3:]),
                            images_gt.reshape(-1,*images_gt.shape[-3:]),
                            normalize=True,
                        )    
        loss_vgg = loss_vgg.mean()
        loss_vgg = loss_vgg * 0.05    
        loss_image = loss_vgg + loss_image_pred_geo + lossd
        # image_2d
        mse = torch.mean(((images_org.reshape(*render_mask.shape)[render_mask > 0.] - images_gt.reshape(*render_mask.shape)[render_mask > 0.]) * 255) ** 2)
        PSNR = 20 * torch.log10(255/torch.sqrt(mse))
        loss_2d = 0
        self_name = type(self).__name__
        details = {'pr_pts2_scale': float(pr_pts2.abs().mean()), 'PSNR': float(PSNR), 'loss_2d': float(loss_2d), 'loss_pred_geo': float(loss_image_pred_geo), 'loss_vgg': float(loss_vgg), self_name+'_image': float(loss_image), 'images': images_org,  'images_gt':images_gt, 'images_gt_geo': images_gt_geo, 'image_2d': image_2d, 'lossd': float(lossd)}
        return (loss_image, None), (loss_2d, None), details

    def compute_loss(self, gt1, gt2, pred1, pred2, trajectory_pred, render_gt=None, **kw):
        (pose_r_loss, _), (pose_t_loss, _), (pose_f_loss, _), details_pose = self.compute_pose_loss(gt1, gt2, pred1, pred2, trajectory_pred, render_gt=render_gt)
        valids = torch.stack([gt1_per['valid_mask'] for gt1_per in gt1+gt2], dim=1)
        triangluation_mask = valids.flatten(1,3).sum(1) > -1
        gt1_use_gtloss = []
        gt2_use_gtloss = []
        render_gt_use_gtloss = []
        gt1_use_unsupervised = []
        gt2_use_unsupervised = []
        render_gt_use_unsupervised = []

        B = valids.shape[0]
        nviews = len(gt2)
        key_gt_list = ['camera_pose', 'true_shape', 'fxfycxcy', 'render_mask', 'img_org', 'valid_mask', 'pts3d', 'camera_intrinsics', 'depth_anything']
        for i in range(len(gt1)):
            gt1_dict = {}
            gt1_dict_unsupervised = {}
            for key in gt1[0].keys():
                if key in key_gt_list:
                    gt1_dict[key] = gt1[i][key][triangluation_mask.to(gt1[i][key].device)]
                    gt1_dict_unsupervised[key] = gt1[i][key][~triangluation_mask.to(gt1[i][key].device)]
                    # gt1_dict_unsupervised[key] = gt1[i][key][triangluation_mask.to(gt1[i][key].device)]
            gt1_use_gtloss.append(gt1_dict)
            gt1_use_unsupervised.append(gt1_dict_unsupervised)

        for i in range(len(gt2)):
            gt2_dict = {}
            gt2_dict_unsupervised = {}
            for key in gt2[0].keys():
                if key in key_gt_list:
                    gt2_dict[key] = gt2[i][key][triangluation_mask.to(gt2[i][key].device)]
                    gt2_dict_unsupervised[key] = gt2[i][key][~triangluation_mask.to(gt2[i][key].device)]
                    # gt2_dict_unsupervised[key] = gt2[i][key][triangluation_mask.to(gt2[i][key].device)]
            gt2_use_gtloss.append(gt2_dict)
            gt2_use_unsupervised.append(gt2_dict_unsupervised)
        for i in range(len(render_gt)):
            render_gt_dict = {}
            render_gt_dict_unsupervised = {}
            for key in render_gt[0].keys():
                if key in key_gt_list:
                    render_gt_dict[key] = render_gt[i][key][triangluation_mask.to(render_gt[i][key].device)]
                    render_gt_dict_unsupervised[key] = render_gt[i][key][~triangluation_mask.to(render_gt[i][key].device)]
                    # render_gt_dict_unsupervised[key] = render_gt[i][key][triangluation_mask.to(render_gt[i][key].device)]
            render_gt_use_gtloss.append(render_gt_dict)
            render_gt_use_unsupervised.append(render_gt_dict_unsupervised)
        key_pred_list = ['pts3d', 'feature', 'opacity', 'scaling', 'rotation', 'conf', 'pts3d_pre']
        pred1_use_gtloss = {}
        pred2_use_gtloss = {}
        pred1_unsupervised = {}
        pred2_unsupervised = {}

        for key in pred1.keys():
            if key in key_pred_list:
                pred1_use_gtloss[key] = pred1[key][triangluation_mask]
                pred1_unsupervised[key] = pred1[key][~triangluation_mask]
                # pred1_unsupervised[key] = pred1[key][triangluation_mask]
                pred2_use_gtloss[key] = pred2[key].unflatten(0, (B, nviews))[triangluation_mask].flatten(0,1)
                # pred2_unsupervised[key] = pred2[key].unflatten(0, (B, nviews))[triangluation_mask].flatten(0,1)
                pred2_unsupervised[key] = pred2[key].unflatten(0, (B, nviews))[~triangluation_mask].flatten(0,1)

        # if len(gt1_use_gtloss[0]['pts3d']) != 0: (conf_loss1, _), (conf_loss2, _), 
        (loss_image, _), (loss_2d, _),  details_gtloss = self.gt_loss(gt1_use_gtloss, gt2_use_gtloss, pred1_use_gtloss, pred2_use_gtloss, trajectory_pred, render_gt=render_gt_use_gtloss, **kw)
        loss_image_unsupervised = loss_2d_unsupervised = 0
        details_unsupervised = {}
        
        details = {**details_pose, **details_gtloss, **details_unsupervised}
        details['loss_image'] = loss_image
        details['pose_r_loss'] = float(pose_r_loss)
        details['pose_t_loss'] = float(pose_t_loss)
        final_loss = loss_image * triangluation_mask.float().mean() + loss_2d * triangluation_mask.float().mean() + pose_r_loss + pose_t_loss * 0.1 + pose_f_loss
        return final_loss, details


class Regr3D_gs_unsupervised(Regr3D_clean):
    """ Ensure that all 3D points are correct.
        Asymmetric loss: view1 is supposed to be the anchor.

        P1 = RT1 @ D1
        P2 = RT2 @ D2
        loss1 = (I @ pred_D1) - (RT1^-1 @ RT1 @ D1)
        loss2 = (RT21 @ pred_D2) - (RT1^-1 @ P2)
              = (RT21 @ pred_D2) - (RT1^-1 @ RT2 @ D2)
    """

    def __init__(self, criterion, alpha, multiview_triangulation = True, disable_rayloss = False, scaling_mode='interp_5e-4_0.1_3', norm_mode='avg_dis', gt_scale=False, test=False):
        super().__init__(criterion)
        self.norm_mode = norm_mode
        self.disable_rayloss = disable_rayloss
        self.multiview_triangulation = multiview_triangulation
        self.scaling_mode = {'type':scaling_mode.split('_')[0], 'min_scaling': eval(scaling_mode.split('_')[1]), 'max_scaling': eval(scaling_mode.split('_')[2]), 'shift': eval(scaling_mode.split('_')[3])}
        self.gt_scale = gt_scale
        from .perceptual import PerceptualLoss
        self.perceptual_loss = PerceptualLoss().cuda().eval()
        for param in self.perceptual_loss.parameters():
            param.requires_grad = False
        self.random_crop = RandomCrop(224)
        self.epoch = 0
        self.test = test
        self.gt_num_image = 0
        self.render_landscape = transpose_to_landscape_render(self.render)
        self.loss2d_landscape = transpose_to_landscape(self.loss2d)
        self.alpha = alpha
        self.sm = nn.SmoothL1Loss()
        self.disp_loss =  ScaleAndShiftInvariantLoss()
        from lpips import LPIPS
        self.lpips = LPIPS(net="vgg")
        convert_to_buffer(self.lpips, persistent=False)


    def render(self, render_params, image_size):
        latent, output_fxfycxcy, output_c2ws = render_params
        (H_org, W_org) = image_size
        H = H_org
        W = W_org
        # if 'scaling_factor' in latent.keys():
        #     scaling_factor = latent.pop('scaling_factor')
        #     if self.scaling_mode['type'] == 'precomp':
        #         x = torch.clip(latent['pre_scaling'] - self.scaling_mode['shift'], max=np.log(3))
        #         latent['scaling'] = torch.exp(x).clamp(min= self.scaling_mode['min_scaling'], max=self.scaling_mode['max_scaling']) / scaling_factor
        # else:
        #     scaling_factor = torch.tensor(1.0).to(latent['xyz'])
        new_latent = {}
        if self.scaling_mode['type'] == 'precomp': 
            scaling_factor = latent['scaling_factor']
            x = torch.clip(latent['pre_scaling'] - self.scaling_mode['shift'], max=np.log(0.3))
            new_latent['scaling'] = torch.exp(x).clamp(min= self.scaling_mode['min_scaling'], max=self.scaling_mode['max_scaling']) / scaling_factor
            skip = ['pre_scaling', 'scaling', 'scaling_factor']
        else:
            skip = ['pre_scaling', 'scaling_factor']
        for key in latent.keys():
            if key not in skip:
                new_latent[key] = latent[key]
        gs_render = GaussianRenderer(H_org, W_org, patch_size=128, gs_kwargs={'type':self.scaling_mode['type'], 'min_scaling': self.scaling_mode['min_scaling'], 'max_scaling': self.scaling_mode['max_scaling'], 'scaling_factor': scaling_factor.detach()}) # scaling_factor
        results = gs_render(new_latent, output_fxfycxcy.reshape(1,-1,4), output_c2ws.reshape(1,-1,4,4))
        images = results['image']
        image_mask = results['alpha']
        depth = results['depth']
        depth = depth.reshape(-1,1,H,W).permute(0,2,3,1)
        image_mask = image_mask.reshape(-1,1,H,W).permute(0,2,3,1)
        images = images.reshape(-1,3,H,W).permute(0,2,3,1)
        return {'images': images, 'image_mask': image_mask, 'depth':depth}
    
    def loss2d(self, input, image_size):
        pr_pts, fxfycxcy, c2ws = input
        H, W = image_size
        loss_2d = reproj2d(fxfycxcy, c2ws, pr_pts, image_size)
        return {'loss_2d': loss_2d}
    
    def get_all_pts3d(self, gt1, gt2, pred1, pred2, trajectory_pred, dist_clip=None, render_gt=None):
        # everything is normalized w.r.t. camera of view1        
        gt_pts1, gt_pts2, pr_pts1, pr_pts2, valid1, valid2, conf1, conf2, trajectory_gt, trajectory_pred, R, trajectory_R_prior, trajectory_R_post, focal_length_gt, focal_length_pred, se3_gt, se3_pred, pr_pts2_transform, monitoring = super().get_all_pts3d(gt1, gt2, pred1, pred2, trajectory_pred, dist_clip=None, render_gt=render_gt, wop=True)
        norm_factor_gt, norm_factor_pr = monitoring
        B, H, W, _ = gt_pts1.shape
        nviews = len(gt2)
        # pr_pts1_pre, pr_pts2_pre = pred1['pts3d_pre'].reshape(*pr_pts1.shape), pred2['pts3d_pre'].reshape(*pr_pts2.shape)
        # pr_pts1_pre, pr_pts2_pre, _ = normalize_pointcloud(pr_pts1_pre, pr_pts2_pre, self.norm_mode, valid1, valid2, ret_factor=True)
        sh_dim = pred1['feature'].shape[-1]
        feature1 = pred1['feature'].reshape(B,H,W,sh_dim)
        feature2 = pred2['feature'].reshape(B,-1,W,sh_dim)
        feature = torch.cat((feature1, feature2), dim=1).float()
        image_2d = torch.cat((feature1.reshape(B,-1, H,W,sh_dim)[..., :3], feature2.reshape(B,-1, H,W,sh_dim)[..., :3]), dim=1)
        if sh_dim > 3:
            image_2d = sh_to_rgb(image_2d).clamp(0,1)
        else:
            image_2d = torch.sigmoid(image_2d)
        opacity1 = pred1['opacity'].reshape(B,H,W,1)
        opacity2 = pred2['opacity'].reshape(B,-1,W,1)
        opacity = torch.cat((opacity1, opacity2), dim=1).float()
        scaling1 = pred1['scaling'].reshape(B,H,W,3)
        scaling2 = pred2['scaling'].reshape(B,-1,W,3)
        scaling = torch.cat((scaling1, scaling2), dim=1).float()
        rotation1 = pred1['rotation'].reshape(B,H,W,4)
        rotation2 = pred2['rotation'].reshape(B,-1,W,4)
        rotation = torch.cat((rotation1, rotation2), dim=1).float()
        output_fxfycxcy = torch.stack([gt['fxfycxcy'] for gt in render_gt], dim=1)
        render_mask = torch.stack([gt['render_mask'] for gt in render_gt], dim=1)
        shape = torch.stack([gt['true_shape'] for gt in render_gt], dim=1)
        xyz = torch.cat((pr_pts1, pr_pts2), dim=1)
        pr_pts = torch.cat((pr_pts1.reshape(B,-1,H,W,3), pr_pts2.reshape(B,-1,H,W,3)), dim=1)
        xyz_gt = torch.cat((gt_pts1.reshape(B,-1,H,W,3), gt_pts2.reshape(B,-1,H,W,3)), dim=1).detach()
        # TODO rewrite transform cameras


        images_gt = torch.stack([gt['img_org'] for gt in render_gt], dim=1)
        images_gt = images_gt / 2 + 0.5
        norm_factor_gt = monitoring[0]
        output_c2ws = torch.stack([gt['camera_pose'] for gt in render_gt], dim=1)
        camera_pose = torch.stack([gt1_per['camera_pose'] for gt1_per in gt1], dim=1)
        in_camera1 = inv(camera_pose)
        output_c2ws = torch.einsum('bnjk,bnkl->bnjl', in_camera1.repeat(1,output_c2ws.shape[1],1,1), output_c2ws)
        output_c2ws[..., :3, 3:] = output_c2ws[..., :3, 3:] / (norm_factor_gt.detach() + 1e-5)
        depth_gt = torch.stack([gt['depth_anything'] for gt in render_gt], dim=1) / (norm_factor_gt.detach() + 1e-5)
        if self.test:
            with torch.no_grad():
                trajectory_pred_t = trajectory_pred[-1, 4:]
                trajectory_pred_r = trajectory_pred[-1, :4]
                trajectory_pred = torch.cat([trajectory_pred_t, trajectory_pred_r], dim=-1)
                render_pose = interpolate_poses(list(range(len(trajectory_pred))), trajectory_pred.detach().cpu().numpy(), list(np.linspace(0,7,280)), 0)
                render_pose = np.stack(render_pose)
                render_pose = torch.tensor(render_pose).cuda()[:200]
                i = 0
                clip = 8
                # pr_pts1_pre, pr_pts2_pre, norm_factor_pr = normalize_pointcloud(pr_pts1_pre, pr_pts2_pre, self.norm_mode, valid1, valid2, ret_factor=True)
                latent = {'xyz': xyz.reshape(B, -1, 3)[i:i+1], 'feature': feature.reshape(B, -1, sh_dim)[i:i+1], 'opacity': opacity.reshape(B, -1, 1)[i:i+1], 'pre_scaling': scaling.reshape(B, -1, 3)[i:i+1].clone(), 'rotation': rotation.reshape(B, -1, 4)[i:i+1], 'scaling_factor': norm_factor_pr.reshape(B, -1)[i:i+1]}
                # ret = self.render_landscape([latent, output_fxfycxcy[i:i+1][:1].reshape(-1,4).repeat(render_pose.reshape(-1,4,4).shape[0],1), render_pose.reshape(-1,4,4)], shape[i:i+1].reshape(-1,2))
                ret = self.render_landscape([latent, output_fxfycxcy[i:i+1][:1].reshape(-1,4).repeat(render_pose.reshape(-1,4,4).shape[0],1), render_pose.reshape(-1,4,4)], shape[i:i+1].reshape(-1,2))

                import viser
                # import 
                camera_poses = torch.stack([gt['camera_pose'] for gt in gt1+gt2], dim=1)[0]
                camera_poses = torch.einsum('njk,nkl->njl', in_camera1[0].repeat(camera_poses.shape[0],1,1), camera_poses)
                camera_poses[:, :3, 3] = camera_poses[:, :3, 3] / (norm_factor_gt[0].reshape(1, -1) + 1e-5)
                # camera_poses = camera_poses.inverse()
                images = torch.stack([gt['img_org'] for gt in gt1+gt2], dim=1)
                fxfycxcy = torch.stack([gt['fxfycxcy'] for gt in gt1+gt2], dim=1)[0]
                images = images.permute(0,1,3,4,2)[0] / 2 + 0.5
                colors = images.reshape(B, -1, 3).detach().cpu().numpy() / 2 + 0.5
                points3d = pr_pts.reshape(B, -1, 3)[0]

                video = ret['images'].detach().cpu().numpy()
                import imageio
                import os
                i = 0
                output_path = f'test/output_video_{i}.mp4'
                os.makedirs('test', exist_ok=True)
                while os.path.exists(output_path):
                    i = i  + 1
                    output_path = f'test/output_video_{i}.mp4'
                # pc.save_ply_v2(f'test/{i}.ply')
                # import matplotlib.pyplot as plt
                # plt.imshow(images[0,0].detach().cpu().numpy().transpose(1,2,0))
                # plt.savefig('test/output.png')
                writer = imageio.get_writer(output_path, fps=30)  # You can specify the filename and frames per second
                for frame in video:  # Example for 100 frames
                    # Add the frame to the video
                    frame = (frame * 255).clip(0,255).astype(np.uint8)
                    writer.append_data(frame)
                # Close the writer to finalize the video
                writer.close()
                for i in range(len(images)):
                    imageio.imwrite(f'test/output_{i}.png', (images[i] * 255).clip(0,255).detach().cpu().numpy().astype(np.uint8))
                # vis(camera_poses, images, points3d, colors, fxfycxcy, H, W)

        images_list = []
        image_mask_list = []
        depths_rendered_list = []
        for i in range(B):
            # try:
            latent = {'xyz': xyz.reshape(B, -1, 3)[i:i+1], 'feature': feature.reshape(B, -1, sh_dim)[i:i+1], 'opacity': opacity.reshape(B, -1, 1)[i:i+1], 'pre_scaling': scaling.reshape(B, -1, 3)[i:i+1].clone(), 'rotation': rotation.reshape(B, -1, 4)[i:i+1], 'scaling_factor': norm_factor_pr.reshape(B, -1)[i:i+1]}
            # if 'scaling_factor' not in latent.keys():
            ret = self.render_landscape([latent, output_fxfycxcy[i:i+1].reshape(-1,4), output_c2ws[i:i+1].reshape(-1,4,4)], shape[i:i+1].reshape(-1,2))
            # except:
            #     import ipdb; ipdb.set_trace()
            images = ret['images']
            image_mask = ret['image_mask']
            images_list.append(images)
            image_mask_list.append(image_mask)
            depths_rendered_list.append(ret['depth'])
        images = torch.stack(images_list, dim=0).permute(0,1,4,2,3)
        image_masks = torch.stack(image_mask_list, dim=0).permute(0,1,4,2,3)
        depths_rendered_list = torch.stack(depths_rendered_list, dim=0).permute(0,1,4,2,3)
        
        image_mask_gt_geo = image_masks.clone()
        images_gt_geo = images.clone()
        shape_input = torch.stack([gt['true_shape'] for gt in gt1+gt2], dim=1)
        input_fxfycxcy = output_fxfycxcy[:,:4]
        input_c2ws = output_c2ws[:,:4]
        shape_input = shape_input[:,:4]
        pr_pts = pr_pts[:,:4]
        loss_2d = self.loss2d_landscape([pr_pts.reshape(-1, H, W, 3), input_fxfycxcy.reshape(-1,4), input_c2ws.reshape(-1,4,4)], shape_input.reshape(-1,2))
        # import ipdb; ipdb.set_trace()
        # masks = torch.cat((valid1.reshape(B, -1, H, W), valid2.reshape(B, -1, H, W)), 1)
        loss_2d = loss_2d['loss_2d'].mean() + opacity.abs().mean() * 0.001
        if self.disable_rayloss:
            loss_2d = torch.zeros_like(loss_2d).to(loss_2d.device)

        metric_conf1 = pred1['conf']
        metric_conf2 = pred2['conf']
        pr_pts2_transform = pr_pts2_transform.reshape(*gt_pts2.shape)
        return gt_pts1, gt_pts2, pr_pts1, pr_pts2, valid1, valid2, metric_conf1, metric_conf2, trajectory_gt, trajectory_pred, images, images_gt, image_masks, images_gt_geo, image_mask_gt_geo, render_mask, R, trajectory_R_prior, trajectory_R_post, loss_2d, image_2d, depth_gt, depths_rendered_list, {}


    def get_conf_log(self, x):
        return x, torch.log(x)

    
    def gt_loss(self, gt1, gt2, pred1, pred2, trajectory_pred, render_gt=None, **kw):
        gt_pts1, gt_pts2, pr_pts1, pr_pts2,  mask1, mask2, metric_conf1, metric_conf2, trajectory, trajectory_pred, images, images_gt, image_mask, images_gt_geo, image_mask_gt_geo, render_mask,  R, trajectory_R_prior, trajectory_R_post, loss_2d, image_2d, depth_gt, depths, monitoring = self.get_all_pts3d(gt1, gt2, pred1, pred2, trajectory_pred, render_gt=render_gt, **kw)
        # loss on img1 side
        # loss on gt2 side
        l1 = self.criterion(pr_pts1[mask1], gt_pts1[mask1])
        l2 = self.criterion(pr_pts2[mask2], gt_pts2[mask2])
        
        render_mask_depth = render_mask.clone()
        depth_mask = (depth_gt > 0) & (render_mask_depth > 0)
        # disp = 1 / depths.clip(0.1, 100)
        # disp = disp.reshape(*depth_mask.shape)
        # depth_gt_clip = depth_gt.clip(0.0001, 0.1)
        # depth_clip = depths.clip(0.0001, 0.1)
        # depth_clip = depth_clip.reshape(*depth_mask.shape)
        
        # if depth_mask.sum() > 0:
        #     depth_loss = self.sm(disp[depth_mask], disp_gt[depth_mask]) * 0.5 + self.sm(depth_clip[depth_mask], depth_gt_clip[depth_mask])
        # else:
        #     depth_loss = 0
        
        depths = depths.reshape(*depth_mask.shape)
        depth_mask_far = depth_mask.clone()
        depth_mask_far[depths<=0.1] = 0
        depth_mask_far[depth_gt<=0.1] = 0
        disp = 1 / depths.clip(0.1, 100)
        disp_gt = 1 / depth_gt.clip(0.1, 100)
        depth_loss = self.disp_loss(prediction = disp.flatten(0,1), target = disp_gt.flatten(0,1), mask = depth_mask_far.flatten(0,1))
        
        render_mask = render_mask[:,:,None].repeat(1,1,3,1,1)
        render_mask = render_mask.float()
        image_2d = image_2d.permute(0,1,4,2,3)
        images_org = images.clone()
        
        loss_image_pred_geo = ((images.reshape(*render_mask.shape)[render_mask> 0.] - images_gt.reshape(*render_mask.shape)[render_mask > 0.]) ** 2)
        loss_image_pred_geo = loss_image_pred_geo.mean() if loss_image_pred_geo.numel() > 0 else 0
        # loss_image_pred_geo_novel = ((images.reshape(*render_mask.shape)[:,4:][render_mask[:,4:] > 0.] - images_gt.reshape(*render_mask.shape)[:,4:][render_mask[:,4:] > 0.]) ** 2)
        # loss_image_pred_geo_novel = loss_image_pred_geo_novel.mean() if loss_image_pred_geo_novel.numel() > 0 else 0
        
        # loss_image_pred_self = ((image_2d[:,:4] - images_gt[:,:4]) ** 2).mean()
        loss_image_pred_geo = loss_image_pred_geo * 4 +  depth_loss * 0.1 #+ loss_image_pred_self
        
        images_masked = images
        # images_masked[:,:4] = (images_masked[:,:4] * 0.5 + image_2d[:,:4] * 0.5)
        images_masked = images_masked * render_mask + images_gt * (1 - render_mask)
        images_masked, images_gt_1 = self.random_crop(images_masked, images_gt)
        images_masked = images_masked.reshape(-1,*images_gt_1.shape[-3:])
        images_gt_1 = images_gt_1.reshape(-1,*images_gt_1.shape[-3:])
        # loss_vgg = self.perceptual_loss(images_masked, images_gt_1) * 0.2
        loss_vgg = self.lpips.forward(
                images_masked,
                images_gt_1,
                normalize=True,
            ) * 0.2
        loss_vgg = loss_vgg.mean()
        loss_image = loss_vgg + loss_image_pred_geo
        # image_2d
        mse = torch.mean(((images_org.reshape(*render_mask.shape)[render_mask > 0.] - images_gt.reshape(*render_mask.shape)[render_mask > 0.]) * 255) ** 2)
        PSNR = 20 * torch.log10(255/torch.sqrt(mse))
        Reg_1 = l1.mean() if l1.numel() > 0 else 0
        Reg_2 = l2.mean() if l2.numel() > 0 else 0
        msk1 = mask1
        msk2 = mask2
        conf1, log_conf1 = self.get_conf_log(pred1['conf'][msk1])
        conf2, log_conf2 = self.get_conf_log(pred2['conf'][msk2.view(*pred2['conf'].shape)])
        conf_loss1 = l1 * conf1 - self.alpha * log_conf1
        conf_loss2 = l2 * conf2 - self.alpha * log_conf2
        # average + nan protection (in case of no valid pixels at all)
        conf_loss1 = conf_loss1.mean() if conf_loss1.numel() > 0 else 0
        conf_loss2 = conf_loss2.mean() if conf_loss2.numel() > 0 else 0
        self_name = type(self).__name__
        details = {'depth_loss': float(depth_loss), 'conf_loss1': float(conf_loss1), 'conf_loss2': float(conf_loss2), 'PSNR': float(PSNR), 'loss_2d': float(loss_2d), 'loss_pred_geo': float(loss_image_pred_geo), 'loss_vgg': float(loss_vgg), 'gt_1': float(Reg_1), 'gt_2': float(Reg_2), self_name+'_image': float(loss_image), 'images': images_org,  'images_gt':images_gt * render_mask.float(), 'images_gt_geo': images_gt_geo, 'image_2d': image_2d, 'depths': disp, 'depth_org': disp_gt}
        return (conf_loss1, None), (conf_loss2, None), (loss_image, None), (loss_2d, None), details

    def compute_loss(self, gt1, gt2, pred1, pred2, trajectory_pred, render_gt=None, **kw):
        (pose_r_loss, _), (pose_t_loss, _), (pose_f_loss, _), details_pose = self.compute_pose_loss(gt1, gt2, pred1, pred2, trajectory_pred, render_gt=render_gt)
        valids = torch.stack([gt1_per['valid_mask'] for gt1_per in gt1+gt2], dim=1)
        triangluation_mask = valids.flatten(1,3).sum(1) > -1
        gt1_use_gtloss = []
        gt2_use_gtloss = []
        render_gt_use_gtloss = []
        gt1_use_unsupervised = []
        gt2_use_unsupervised = []
        render_gt_use_unsupervised = []

        B = valids.shape[0]
        nviews = len(gt2)
        key_gt_list = ['camera_pose', 'true_shape', 'fxfycxcy', 'render_mask', 'img_org', 'valid_mask', 'pts3d', 'camera_intrinsics', 'depth_anything']
        for i in range(len(gt1)):
            gt1_dict = {}
            gt1_dict_unsupervised = {}
            for key in gt1[0].keys():
                if key in key_gt_list:
                    gt1_dict[key] = gt1[i][key][triangluation_mask.to(gt1[i][key].device)]
                    gt1_dict_unsupervised[key] = gt1[i][key][~triangluation_mask.to(gt1[i][key].device)]
                    # gt1_dict_unsupervised[key] = gt1[i][key][triangluation_mask.to(gt1[i][key].device)]
            gt1_use_gtloss.append(gt1_dict)
            gt1_use_unsupervised.append(gt1_dict_unsupervised)

        for i in range(len(gt2)):
            gt2_dict = {}
            gt2_dict_unsupervised = {}
            for key in gt2[0].keys():
                if key in key_gt_list:
                    gt2_dict[key] = gt2[i][key][triangluation_mask.to(gt2[i][key].device)]
                    gt2_dict_unsupervised[key] = gt2[i][key][~triangluation_mask.to(gt2[i][key].device)]
                    # gt2_dict_unsupervised[key] = gt2[i][key][triangluation_mask.to(gt2[i][key].device)]
            gt2_use_gtloss.append(gt2_dict)
            gt2_use_unsupervised.append(gt2_dict_unsupervised)
        for i in range(len(render_gt)):
            render_gt_dict = {}
            render_gt_dict_unsupervised = {}
            for key in render_gt[0].keys():
                if key in key_gt_list:
                    render_gt_dict[key] = render_gt[i][key][triangluation_mask.to(render_gt[i][key].device)]
                    render_gt_dict_unsupervised[key] = render_gt[i][key][~triangluation_mask.to(render_gt[i][key].device)]
                    # render_gt_dict_unsupervised[key] = render_gt[i][key][triangluation_mask.to(render_gt[i][key].device)]
            render_gt_use_gtloss.append(render_gt_dict)
            render_gt_use_unsupervised.append(render_gt_dict_unsupervised)
        key_pred_list = ['pts3d', 'feature', 'opacity', 'scaling', 'rotation', 'conf', 'pts3d_pre']
        pred1_use_gtloss = {}
        pred2_use_gtloss = {}
        pred1_unsupervised = {}
        pred2_unsupervised = {}

        for key in pred1.keys():
            if key in key_pred_list:
                pred1_use_gtloss[key] = pred1[key][triangluation_mask]
                pred1_unsupervised[key] = pred1[key][~triangluation_mask]
                # pred1_unsupervised[key] = pred1[key][triangluation_mask]
                pred2_use_gtloss[key] = pred2[key].unflatten(0, (B, nviews))[triangluation_mask].flatten(0,1)
                # pred2_unsupervised[key] = pred2[key].unflatten(0, (B, nviews))[triangluation_mask].flatten(0,1)
                pred2_unsupervised[key] = pred2[key].unflatten(0, (B, nviews))[~triangluation_mask].flatten(0,1)

        # if len(gt1_use_gtloss[0]['pts3d']) != 0:
        (conf_loss1, _), (conf_loss2, _), (loss_image, _), (loss_2d, _),  details_gtloss = self.gt_loss(gt1_use_gtloss, gt2_use_gtloss, pred1_use_gtloss, pred2_use_gtloss, trajectory_pred, render_gt=render_gt_use_gtloss, **kw)
        # else:
        #     conf_loss1 = conf_loss2 = loss_image = loss_2d = 0
        #     details_gtloss = {}

        # if len(gt1_use_unsupervised[0]['pts3d']) != 0:
        #     # i = 0
        #     (loss_image_unsupervised, _), (loss_2d_unsupervised, _), details_unsupervised = self.unsupervised_loss(gt1_use_unsupervised, gt2_use_unsupervised, pred1_unsupervised, pred2_unsupervised, trajectory_pred, render_gt=render_gt_use_unsupervised, **kw)
        #     # # 'images': images_org,  'images_gt':images_gt * render_mask.float()
        #     # for num in range(details_unsupervised['images_gt'].shape[1]):
        #     #     plt.subplot(4,4,num+1)
        #     #     plt.imshow(details_unsupervised['images_gt'][0,num].permute(1,2,0).detach().cpu().numpy())
            
        #     # for num in range(details_unsupervised['images'].shape[1]):
        #     #     plt.subplot(4,4,num+9)
        #     #     plt.imshow(details_unsupervised['images'][0,num].permute(1,2,0).detach().cpu().numpy())

        #     # output_path = f'test/output_video_{i}.png'
        #     # import os
        #     # os.makedirs('test', exist_ok=True)
        #     # while os.path.exists(output_path):
        #     #     i = i  + 1
        #     #     output_path = f'test/output_video_{i}.png'
        #     # plt.savefig(output_path)

        #     if 'images' in details_gtloss.keys():
        #         details_gtloss.pop('images')
        #         details_gtloss.pop('images_gt')
        #         details_gtloss.pop('images_gt_geo')
        #         details_gtloss.pop('image_2d')
        # else:
        loss_image_unsupervised = loss_2d_unsupervised = 0
        details_unsupervised = {}
        
        details = {**details_pose, **details_gtloss, **details_unsupervised}
       

        final_loss = conf_loss1 + conf_loss2 + loss_image * triangluation_mask.float().mean() + loss_2d * triangluation_mask.float().mean() + pose_r_loss  * 0.25 + pose_t_loss * 0.1 + pose_f_loss
        return final_loss, details



class Regr3D_gs_SA(Regr3D_clean):
    """ Ensure that all 3D points are correct.
        Asymmetric loss: view1 is supposed to be the anchor.

        P1 = RT1 @ D1
        P2 = RT2 @ D2
        loss1 = (I @ pred_D1) - (RT1^-1 @ RT1 @ D1)
        loss2 = (RT21 @ pred_D2) - (RT1^-1 @ P2)
              = (RT21 @ pred_D2) - (RT1^-1 @ RT2 @ D2)
    """

    def __init__(self, criterion, alpha, multiview_triangulation = True, disable_rayloss = False, scaling_mode='interp_5e-4_0.1_3', norm_mode='avg_dis', gt_scale=False, test=False):
        super().__init__(criterion)
        self.norm_mode = norm_mode
        self.disable_rayloss = disable_rayloss
        self.multiview_triangulation = multiview_triangulation
        self.scaling_mode = {'type':scaling_mode.split('_')[0], 'min_scaling': eval(scaling_mode.split('_')[1]), 'max_scaling': eval(scaling_mode.split('_')[2]), 'shift': eval(scaling_mode.split('_')[3])}
        self.gt_scale = gt_scale
        from .perceptual import PerceptualLoss
        self.perceptual_loss = PerceptualLoss().cuda().eval()
        for param in self.perceptual_loss.parameters():
            param.requires_grad = False
        self.random_crop = RandomCrop(224)
        self.epoch = 0
        self.test = test
        self.gt_num_image = 0
        self.render_landscape = transpose_to_landscape_render(self.render)
        self.loss2d_landscape = transpose_to_landscape(self.loss2d)
        self.alpha = alpha
        

    def render(self, render_params, image_size):
        latent, output_fxfycxcy, output_c2ws = render_params
        (H_org, W_org) = image_size
        H = H_org
        W = W_org
        
        scaling_factor = latent.pop('scaling_factor')

        if self.scaling_mode['type'] == 'precomp':
            x = torch.clip(latent['scaling'] - self.scaling_mode['shift'], max=np.log(0.3))
            latent['scaling'] = torch.exp(x).clamp(min= self.scaling_mode['min_scaling'], max=self.scaling_mode['max_scaling']) / scaling_factor
        gs_render = GaussianRenderer(H_org, W_org, patch_size=128, gs_kwargs={'type':self.scaling_mode['type'], 'min_scaling': self.scaling_mode['min_scaling'], 'max_scaling': self.scaling_mode['max_scaling'], 'scaling_factor': scaling_factor.detach()}) # scaling_factor
        results = gs_render(latent, output_fxfycxcy.reshape(1,-1,4), output_c2ws.reshape(1,-1,4,4))
        images = results['image']
        image_mask = results['alpha']
        image_mask = image_mask.reshape(-1,1,H,W).permute(0,2,3,1)
        images = images.reshape(-1,3,H,W).permute(0,2,3,1)
        depths = results['depth'].reshape(-1,1,H,W).permute(0,2,3,1)
        return {'images': images, 'image_mask': image_mask, 'depths': depths}
    
    def loss2d(self, input, image_size):
        pr_pts, fxfycxcy, c2ws = input
        H, W = image_size
        loss_2d = reproj2d(fxfycxcy, c2ws, pr_pts, image_size)
        return {'loss_2d': loss_2d}
    
    def get_all_pts3d(self, gt1, gt2, pred1, pred2, trajectory_pred, dist_clip=None, render_gt=None):
        # everything is normalized w.r.t. camera of view1        
        gt_pts1, gt_pts2, pr_pts1, pr_pts2, valid1, valid2, conf1, conf2, trajectory_gt, trajectory_pred, R, trajectory_R_prior, trajectory_R_post, focal_length_gt, focal_length_pred, se3_gt, se3_pred, pr_pts2_transform, monitoring = super().get_all_pts3d(gt1, gt2, pred1, pred2, trajectory_pred, dist_clip=None, render_gt=render_gt, wop=True)
        norm_factor_gt, norm_factor_pr = monitoring
        B, H, W, _ = gt_pts1.shape
        nviews = len(gt2)
        pr_pts1_pre, pr_pts2_pre = pred1['pts3d_pre'].reshape(*pr_pts1.shape), pred2['pts3d_pre'].reshape(*pr_pts2.shape)
        pr_pts1_pre, pr_pts2_pre, _ = normalize_pointcloud(pr_pts1_pre, pr_pts2_pre, self.norm_mode, valid1, valid2, ret_factor=True)
        sh_dim = pred1['feature'].shape[-1]
        feature1 = pred1['feature'].reshape(B,H,W,sh_dim)
        feature2 = pred2['feature'].reshape(B,-1,W,sh_dim)
        feature = torch.cat((feature1, feature2), dim=1).float()
        image_2d = torch.cat((feature1.reshape(B,-1, H,W,sh_dim)[..., :3], feature2.reshape(B,-1, H,W,sh_dim)[..., :3]), dim=1)
        if sh_dim > 3:
            image_2d = sh_to_rgb(image_2d).clamp(0,1)
        else:
            image_2d = torch.sigmoid(image_2d)
        opacity1 = pred1['opacity'].reshape(B,H,W,1)
        opacity2 = pred2['opacity'].reshape(B,-1,W,1)
        opacity = torch.cat((opacity1, opacity2), dim=1).float()
        scaling1 = pred1['scaling'].reshape(B,H,W,3)
        scaling2 = pred2['scaling'].reshape(B,-1,W,3)
        scaling = torch.cat((scaling1, scaling2), dim=1).float()
        rotation1 = pred1['rotation'].reshape(B,H,W,4)
        rotation2 = pred2['rotation'].reshape(B,-1,W,4)
        rotation = torch.cat((rotation1, rotation2), dim=1).float()
        output_fxfycxcy = torch.stack([gt['fxfycxcy'] for gt in render_gt], dim=1)
        render_mask = torch.stack([gt['render_mask'] for gt in render_gt], dim=1)
        shape = torch.stack([gt['true_shape'] for gt in render_gt], dim=1)
        xyz = torch.cat((pr_pts1, pr_pts2), dim=1)
        pr_pts = torch.cat((pr_pts1.reshape(B,-1,H,W,3), pr_pts2.reshape(B,-1,H,W,3)), dim=1)
        xyz_gt = torch.cat((gt_pts1.reshape(B,-1,H,W,3), gt_pts2.reshape(B,-1,H,W,3)), dim=1).detach()
        # TODO rewrite transform cameras

        images_list = []
        image_mask_list = []
        images_gt = torch.stack([gt['img_org'] for gt in render_gt], dim=1)
        images_gt = images_gt / 2 + 0.5
        feature = feature.reshape(B, -1, 3)
        norm_factor_gt = monitoring[0]
        output_c2ws = torch.stack([gt['camera_pose'] for gt in render_gt], dim=1)
        camera_pose = torch.stack([gt1_per['camera_pose'] for gt1_per in gt1], dim=1)
        in_camera1 = inv(camera_pose)
        output_c2ws = torch.einsum('bnjk,bnkl->bnjl', in_camera1.repeat(1,output_c2ws.shape[1],1,1), output_c2ws)
        output_c2ws[..., :3, 3:] = output_c2ws[..., :3, 3:] / (norm_factor_gt.detach() + 1e-5)
        for i in range(B):
            # try:
            latent = {'xyz': xyz.reshape(B, -1, 3)[i:i+1], 'feature': feature.reshape(B, -1, sh_dim)[i:i+1], 'opacity': opacity.reshape(B, -1, 1)[i:i+1], 'scaling': scaling.reshape(B, -1, 3)[i:i+1], 'rotation': rotation.reshape(B, -1, 4)[i:i+1], 'scaling_factor': norm_factor_pr.reshape(B, -1)[i:i+1]}
            ret = self.render_landscape([latent, output_fxfycxcy[i:i+1].reshape(-1,4), output_c2ws[i:i+1].reshape(-1,4,4)], shape[i:i+1].reshape(-1,2))
            # except:
            #     import ipdb; ipdb.set_trace()
            images = ret['images']
            image_mask = ret['image_mask']
            images_list.append(images)
            image_mask_list.append(image_mask)
        images = torch.stack(images_list, dim=0).permute(0,1,4,2,3)
        image_mask = torch.stack(image_mask_list, dim=0).permute(0,1,4,2,3)
        # with torch.no_grad():
        #     images_gt_geo, image_mask_gt_geo = [], []
        #     for i in range(B):
        #         latent = {'xyz': xyz_gt.reshape(B, -1, 3)[i:i+1], 'feature': feature.reshape(B, -1, 3)[i:i+1], 'opacity': opacity.reshape(B, -1, 1)[i:i+1], 'scaling': scaling.reshape(B, -1, 3)[i:i+1], 'rotation': rotation.reshape(B, -1, 4)[i:i+1]}
        #         ret = self.render_landscape([latent, output_fxfycxcy[i:i+1].reshape(-1,4), output_c2ws[i:i+1].reshape(-1,4,4)], shape[i:i+1].reshape(-1,2))
        #         images_gt_render = ret['images']
        #         image_mask_gt = ret['image_mask']
        #         images_gt_geo.append(images_gt_render)
        #         image_mask_gt_geo.append(image_mask_gt)
        #     images_gt_geo = torch.stack(images_gt_geo, dim=0).permute(0,1,4,2,3)
        #     image_mask_gt_geo = torch.stack(image_mask_gt_geo, dim=0).permute(0,1,4,2,3)
        image_mask_gt_geo = image_mask.clone()
        images_gt_geo = images.clone()
        shape_input = torch.stack([gt['true_shape'] for gt in gt1+gt2], dim=1)
        input_fxfycxcy = output_fxfycxcy[:,:4]
        input_c2ws = output_c2ws[:,:4]
        shape_input = shape_input[:,:4]
        pr_pts = pr_pts[:,:4]
        loss_2d = self.loss2d_landscape([pr_pts.reshape(-1, H, W, 3), input_fxfycxcy.reshape(-1,4), input_c2ws.reshape(-1,4,4)], shape_input.reshape(-1,2))
        # import ipdb; ipdb.set_trace()
        # masks = torch.cat((valid1.reshape(B, -1, H, W), valid2.reshape(B, -1, H, W)), 1)
        loss_2d = loss_2d['loss_2d'].mean()
        if self.disable_rayloss:
            loss_2d = torch.zeros_like(loss_2d).to(loss_2d.device)

        if self.test:
            with torch.no_grad():
                render_pose = interpolate_poses([0,1,2,3,4,5,6,7], trajectory_pred.detach().cpu().numpy()[0,0], list(np.linspace(0,2,280)), 0)
                render_pose = np.stack(render_pose)
                render_pose = torch.tensor(render_pose).cuda()[:200]
                i=0
                B = 1 
                clip = 8
                # pr_pts1_pre, pr_pts2_pre, norm_factor_pr = normalize_pointcloud(pr_pts1_pre, pr_pts2_pre, self.norm_mode, valid1, valid2, ret_factor=True)
                latent = {'xyz': xyz.reshape(B, -1, 3)[i:i+1], 'feature': feature.reshape(B, -1, 3)[i:i+1], 'opacity': opacity.reshape(B, -1, 1)[i:i+1], 'scaling': scaling.reshape(B, -1, 3)[i:i+1], 'rotation': rotation.reshape(B, -1, 4)[i:i+1], 'scaling_factor': norm_factor_pr.reshape(B, -1)[i:i+1]}
                ret = self.render_landscape([latent, output_fxfycxcy[i:i+1][:1].reshape(-1,4).repeat(render_pose.reshape(-1,4,4).shape[0],1), render_pose.reshape(-1,4,4)], shape[i:i+1].reshape(-1,2))
                pc = GRMGaussianModel(sh_degree=0,
                        scaling_kwargs={'type':'interp', 'min_scaling': 5e-4, 'max_scaling': 0.1, 'scaling_factor': norm_factor_pr[i:i+1].reshape(B, -1)}, **latent)
                
                # results = gs_render.render(latent, output_fxfycxcy[:, :1].repeat(1,len(render_pose),1).to(xyz), render_pose.reshape(B,-1,4,4).to(xyz))
                video = ret['images'].detach().cpu().numpy()
                import imageio
                import os
                i = 0
                output_path = f'test/output_video_{i}.mp4'
                os.makedirs('test', exist_ok=True)
                while os.path.exists(output_path):
                    i = i  + 1
                    output_path = f'test/output_video_{i}.mp4'
                
                pc.save_ply_v2(f'test/{i}.ply')
                import matplotlib.pyplot as plt
                plt.imshow(images[0,0].detach().cpu().numpy().transpose(1,2,0))
                plt.savefig('test/output.png')
                writer = imageio.get_writer(output_path, fps=30)  # You can specify the filename and frames per second
                for frame in video:  # Example for 100 frames
                    # Add the frame to the video
                    frame = (frame * 255).astype(np.uint8)
                    writer.append_data(frame)
                # Close the writer to finalize the video
                writer.close()

        metric_conf1 = pred1['conf']
        metric_conf2 = pred2['conf']
        return gt_pts1, gt_pts2, pr_pts1, pr_pts2, valid1, valid2, metric_conf1, metric_conf2, trajectory_gt, trajectory_pred, images, images_gt, image_mask, images_gt_geo, image_mask_gt_geo, render_mask, R, trajectory_R_prior, trajectory_R_post, pr_pts1_pre, pr_pts2_pre, loss_2d, image_2d, {}


    def get_conf_log(self, x):
        return x, torch.log(x)

    def gt_loss(self, gt1, gt2, pred1, pred2, trajectory_pred, render_gt=None, **kw):
        gt_pts1, gt_pts2, pr_pts1, pr_pts2,  mask1, mask2, metric_conf1, metric_conf2, trajectory, trajectory_pred, images, images_gt, image_mask, images_gt_geo, image_mask_gt_geo, render_mask,  R, trajectory_R_prior, trajectory_R_post, pr_pts1_pre, pr_pts2_pre, loss_2d, image_2d, monitoring = self.get_all_pts3d(gt1, gt2, pred1, pred2, trajectory_pred, render_gt=render_gt, **kw)
        # loss on img1 side
        # loss on gt2 side
        l1 = self.criterion(pr_pts1[mask1], gt_pts1[mask1])
        l2 = self.criterion(pr_pts2[mask2], gt_pts2[mask2])
        render_mask = render_mask[:,:,None].repeat(1,1,3,1,1)
        render_mask = render_mask.float()
        image_2d = image_2d.permute(0,1,4,2,3)
        images_org = images.clone()
        
        loss_image_pred_geo = ((images.reshape(*render_mask.shape)[:,:4][render_mask[:,:4] > 0.] - images_gt.reshape(*render_mask.shape)[:,:4][render_mask[:,:4] > 0.]) ** 2)
        loss_image_pred_geo = loss_image_pred_geo.mean() if loss_image_pred_geo.numel() > 0 else 0
        loss_image_pred_geo_novel = ((images.reshape(*render_mask.shape)[:,4:][render_mask[:,4:] > 0.] - images_gt.reshape(*render_mask.shape)[:,4:][render_mask[:,4:] > 0.]) ** 2)
        loss_image_pred_geo_novel = loss_image_pred_geo_novel.mean() if loss_image_pred_geo_novel.numel() > 0 else 0
        # loss_image_pred_self = ((image_2d[:,:4] - images_gt[:,:4]) ** 2).mean()
        loss_image_pred_geo = loss_image_pred_geo  + loss_image_pred_geo_novel * 2#+ loss_image_pred_self
        images_masked = images 
        # images_masked[:,:4] = (images_masked[:,:4] * 0.5 + image_2d[:,:4] * 0.5)
        images_masked = images_masked * render_mask + images_gt * (1 - render_mask)
        images_masked, images_gt_1 = self.random_crop(images_masked, images_gt)
        images_masked = images_masked.reshape(-1,*images_gt_1.shape[-3:])
        images_gt_1 = images_gt_1.reshape(-1,*images_gt_1.shape[-3:])
        loss_vgg = self.perceptual_loss(images_masked, images_gt_1) * 0.1
        loss_image = loss_vgg + loss_image_pred_geo
        # image_2d
        mse = torch.mean(((images_org.reshape(*render_mask.shape)[render_mask > 0.] - images_gt.reshape(*render_mask.shape)[render_mask > 0.]) * 255) ** 2)
        PSNR = 20 * torch.log10(255/torch.sqrt(mse))
        Reg_1 = l1.mean() if l1.numel() > 0 else 0
        Reg_2 = l2.mean() if l2.numel() > 0 else 0

        msk1 = mask1
        msk2 = mask2
        conf1, log_conf1 = self.get_conf_log(pred1['conf'][msk1])
        conf2, log_conf2 = self.get_conf_log(pred2['conf'][msk2.view(*pred2['conf'].shape)])
        conf_loss1 = l1 * conf1 - self.alpha * log_conf1
        conf_loss2 = l2 * conf2 - self.alpha * log_conf2

        # average + nan protection (in case of no valid pixels at all)
        conf_loss1 = conf_loss1.mean() if conf_loss1.numel() > 0 else 0
        conf_loss2 = conf_loss2.mean() if conf_loss2.numel() > 0 else 0
        self_name = type(self).__name__
        details = {'conf_loss1': float(conf_loss1), 'conf_loss2': float(conf_loss2), 'PSNR': float(PSNR), 'loss_2d': float(loss_2d), 'loss_pred_geo': float(loss_image_pred_geo), 'loss_vgg': float(loss_vgg), 'gt_1': float(Reg_1), 'gt_2': float(Reg_2), self_name+'_image': float(loss_image), 'images': images_org,  'images_gt':images_gt * render_mask.float(), 'images_gt_geo': images_gt_geo, 'image_2d': image_2d}
        return (conf_loss1, None), (conf_loss2, None), (loss_image, None), (loss_2d, None), details

    def get_all_pts3d_unsupervised(self, gt1, gt2, pred1, pred2, trajectory_pred, dist_clip=None, render_gt=None):
        view1 = gt1
        view2 = gt2
        nviews = len(view2)
        pr_pts1 = pred1['depth']
        pr_pts2 = pred2['depth']
        B, H, W, _ = pr_pts1.shape
        pts3d = torch.cat([pr_pts1.reshape(B,1,H,W,3), pr_pts2.reshape(B,-1,H,W,3)], dim=1)
        # conf1 = pred1['conf']
        # conf2 = pred2['conf']
        # conf = torch.cat([conf1.reshape(B,1,H,W), conf2.reshape(B,-1,H,W)], dim=1)
        intrinsics = torch.stack([view['camera_intrinsics'] for view in gt1+gt2],1)
        extrinsics = torch.stack([view['camera_pose'] for view in gt1+gt2],1).clone()
        camera_pose = torch.stack([gt1_per['camera_pose'] for gt1_per in gt1], dim=1)
        in_camera1 = inv(camera_pose)
        feature1 = pred1['feature'].reshape(B,H,W,-1)
        feature2 = pred2['feature'].reshape(B,nviews * H,W,-1)
        feature = torch.cat((feature1, feature2), dim=1).float()
        sh_dim = pred1['feature'].shape[-1]
        feature = feature.reshape(B, -1, sh_dim)
        
        image_2d = torch.cat((feature1.reshape(B,-1, H,W,sh_dim)[..., :3], feature2.reshape(B,-1, H,W,sh_dim)[..., :3]), dim=1)
        if sh_dim > 3:
            image_2d = sh_to_rgb(image_2d).clamp(0,1)
        else:
            image_2d = torch.sigmoid(image_2d)
        opacity1 = pred1['opacity'].reshape(B,H,W,1)
        opacity2 = pred2['opacity'].reshape(B,-1,W,1)
        opacity = torch.cat((opacity1, opacity2), dim=1).float()
        scaling1 = pred1['scaling'].reshape(B,H,W,3)
        scaling2 = pred2['scaling'].reshape(B,-1,W,3)
        scaling = torch.cat((scaling1, scaling2), dim=1).float()
        rotation1 = pred1['rotation'].reshape(B,H,W,4)
        rotation2 = pred2['rotation'].reshape(B,-1,W,4)
        rotation = torch.cat((rotation1, rotation2), dim=1).float()

        output_fxfycxcy = torch.stack([gt['fxfycxcy'] for gt in render_gt], dim=1)
        render_mask = torch.stack([gt['render_mask'] for gt in render_gt], dim=1)
        shape = torch.stack([gt['true_shape'] for gt in render_gt], dim=1)
        shape_input = torch.stack([gt['true_shape'] for gt in gt1+gt2], dim=1)

        images_list = []
        image_mask_list = []
        depth_list = []
        images_gt = torch.stack([gt['img_org'] for gt in render_gt], dim=1)
        images_gt = images_gt / 2 + 0.5
        # gt_pts3d = torch.cat([gt['pts3d'].reshape(B,1,H,W,3) for gt in gt1+gt2], dim=1) 
        # gt_pts3d = geotrf(in_camera1.repeat(1, gt_pts3d.shape[1],1,1).flatten(0,1), gt_pts3d.flatten(0,1)) / size[:, None].repeat(1,len(gt2)+1,1,1,1).flatten(0,1)
        # gt_pts3d = gt_pts3d.reshape(B, -1, H, W, 3)
        gt_pts3d = None
        pr_pts1_grm = pred1['depth'].float()
        pr_pts2_grm = pred2['depth'].float()
        pts3d_grm = torch.cat([pr_pts1_grm.reshape(B,1,H,W,3), pr_pts2_grm.reshape(B,-1,H,W,3)], dim=1)
        scaling_grm = torch.cat([pred1['scaling'].reshape(B,1,H,W,3), pred2['scaling'].reshape(B,-1,H,W,3)], dim=1)
        output_c2ws = torch.stack([gt['camera_pose'] for gt in render_gt], dim=1)
        camera_pose = torch.stack([gt1_per['camera_pose'] for gt1_per in gt1], dim=1)
        in_camera1 = inv(camera_pose)
        output_c2ws = torch.einsum('bnjk,bnkl->bnjl', in_camera1.repeat(1,output_c2ws.shape[1],1,1), output_c2ws)
        size = pred1['size'].float()
        # import ipdb; ipdb.set_trace()
        output_c2ws[..., :3, 3:] = output_c2ws[..., :3, 3:] / (size.detach().unsqueeze(-1))
        for i in range(B):
            latent = {'xyz': pts3d_grm.reshape(B, -1, 3)[i:i+1], 'feature': feature[i:i+1], 'opacity': opacity.reshape(B, -1, 1)[i:i+1], 'scaling': scaling_grm.reshape(B, -1, 3)[i:i+1], 'rotation': rotation.reshape(B, -1, 4)[i:i+1], 'scaling_factor': torch.ones((1,1)).to(opacity)}
            ret = self.render_landscape([latent, output_fxfycxcy[i:i+1].reshape(-1,4), output_c2ws[i:i+1].reshape(-1,4,4)], shape[i:i+1].reshape(-1,2))
            images = ret['images']
            image_mask = ret['image_mask']
            images_list.append(images)
            depth_list.append(ret['depths'])
            image_mask_list.append(image_mask)
        images = torch.stack(images_list, dim=0).permute(0,1,4,2,3)
        image_mask = torch.stack(image_mask_list, dim=0).permute(0,1,4,2,3)
        depth_list = torch.stack(depth_list, dim=0).permute(0,1,4,2,3).squeeze(2)
        image_mask_gt_geo = image_mask.clone()
        images_gt_geo = images.clone()
        shape_input = torch.stack([gt['true_shape'] for gt in gt1+gt2], dim=1)
        input_fxfycxcy = output_fxfycxcy[:,:4]
        input_c2ws = output_c2ws[:,:4]
        shape_input = shape_input[:,:4]
        pr_pts = pts3d[:,:4]

        # valid1 = torch.stack([gt1_per['valid_mask'] for gt1_per in gt1], dim=1).view(B,-1,W).clone()
        # valid2 = torch.stack([gt2_per['valid_mask'] for gt2_per in gt2], dim=1).view(B,-1,W).clone()
        # valid1 = torch.ones_like(valid1).to(valid1) > 0
        # valid2 = torch.ones_like(valid2).to(valid2) > 0
        # pr_pts1_grm, pr_pts2_grm, _ = normalize_pointcloud(pr_pts1_grm.reshape(B, -1,W,3), pr_pts2_grm.reshape(B, -1,W,3), self.norm_mode, valid1, valid2, ret_factor=True)
        # pr_pts1, pr_pts2, _ = normalize_pointcloud(pr_pts1.reshape(B, -1,W,3), pr_pts2.reshape(B, -1,W,3), self.norm_mode, valid1, valid2, ret_factor=True)
        loss_2d = 0 # (pr_pts1_grm.detach()-pr_pts1).abs().mean() + (pr_pts2_grm.detach()-pr_pts2).abs().mean() + (scaling.reshape(-1,3) - scaling_grm.reshape(-1,3).detach()).abs().mean()
        
        # loss_2d = self.loss2d_landscape([pr_pts.reshape(-1, H, W, 3), input_fxfycxcy.reshape(-1,4), input_c2ws.reshape(-1,4,4)], shape_input.reshape(-1,2))
        # loss_2d = loss_2d['loss_2d'].mean()
        # if self.disable_rayloss:
        #     loss_2d = torch.zeros_like(loss_2d).to(loss_2d.device)
        return  images, images_gt, image_mask, images_gt_geo, image_mask_gt_geo, render_mask, loss_2d, image_2d, depth_list, {}


    def unsupervised_loss(self, gt1, gt2, pred1, pred2, trajectory_pred, render_gt=None, **kw):
        images, images_gt, image_mask, images_gt_geo, image_mask_gt_geo, render_mask, loss_2d, image_2d, depth_list, monitoring = self.get_all_pts3d_unsupervised(gt1, gt2, pred1, pred2, trajectory_pred, render_gt=render_gt, **kw)
        # loss on img1 side
        depth_org1 = pred1['depth_org']
        depth_org2 = pred2['depth_org']
        B, H, W, _ = depth_org1.shape
        depth_org = torch.cat([depth_org1.reshape(B, -1, H, W), depth_org2.reshape(B, -1, H, W)], dim=1)
        # loss on gt2 side
        render_mask = render_mask[:,:,None].repeat(1,1,3,1,1)
        render_mask = render_mask.float()
        image_2d = image_2d.permute(0,1,4,2,3)
        images_org = images.clone()
        
        loss_image_pred_geo = ((images.reshape(*render_mask.shape)[:,:4][render_mask[:,:4] > 0.] - images_gt.reshape(*render_mask.shape)[:,:4][render_mask[:,:4] > 0.]) ** 2)
        loss_image_pred_geo = loss_image_pred_geo.mean() if loss_image_pred_geo.numel() > 0 else 0
        loss_image_pred_geo_novel = ((images.reshape(*render_mask.shape)[:,4:][render_mask[:,4:] > 0.] - images_gt.reshape(*render_mask.shape)[:,4:][render_mask[:,4:] > 0.]) ** 2)
        loss_image_pred_geo_novel = loss_image_pred_geo_novel.mean() if loss_image_pred_geo_novel.numel() > 0 else 0
        # loss_image_pred_self = ((image_2d[:,:4] - images_gt[:,:4]) ** 2).mean()
        loss_image_pred_geo = loss_image_pred_geo  + loss_image_pred_geo_novel * 2 #+ loss_image_pred_self
        images_masked = images 
        # images_masked[:,:4] = (images_masked[:,:4] * 0.5 + image_2d[:,:4] * 0.5)
        images_masked = images_masked * render_mask + images_gt * (1 - render_mask)
        images_masked, images_gt_1 = self.random_crop(images_masked, images_gt)
        images_masked = images_masked.reshape(-1,*images_gt_1.shape[-3:])
        images_gt_1 = images_gt_1.reshape(-1,*images_gt_1.shape[-3:])
        loss_vgg = self.perceptual_loss(images_masked, images_gt_1) * 0.1
        loss_image = loss_vgg + loss_image_pred_geo
        # image_2d
        mse = torch.mean(((images_org.reshape(*render_mask.shape)[render_mask > 0.] - images_gt.reshape(*render_mask.shape)[render_mask > 0.]) * 255) ** 2)
        PSNR = 20 * torch.log10(255/torch.sqrt(mse))
        self_name = type(self).__name__
        details = {'PSNR_unsupervised': float(PSNR), 'loss_2d_unsupervised': float(loss_2d), 'loss_pred_geo_unsupervised': float(loss_image_pred_geo), 'loss_vgg_unsupervised': float(loss_vgg), self_name+'_image': float(loss_image), 'images': images_org,  'images_gt':images_gt * render_mask.float(), 'images_gt_geo': images_gt_geo, 'image_2d': image_2d, 'depths': depth_list, 'depth_org':depth_org}
        # 'images': images_org,  'images_gt':images_gt * render_mask.float(), 'images_gt_geo': images_gt_geo, 'image_2d': image_2d
        return (loss_image, None), (loss_2d, None), details

    def compute_loss(self, gt1, gt2, pred1, pred2, trajectory_pred, render_gt=None, **kw):
        (pose_r_loss, _), (pose_t_loss, _), (pose_f_loss, _), details_pose = self.compute_pose_loss(gt1, gt2, pred1, pred2, trajectory_pred, render_gt=render_gt)
        valids = torch.stack([gt1_per['valid_mask'] for gt1_per in gt1+gt2], dim=1)
        triangluation_mask = valids.flatten(1,3).sum(1) != 0
        # triangluation_mask = torch.zeros_like(triangluation_mask).to(triangluation_mask) > 0
        gt1_use_gtloss = []
        gt2_use_gtloss = []
        render_gt_use_gtloss = []
        gt1_use_unsupervised = []
        gt2_use_unsupervised = []
        render_gt_use_unsupervised = []

        B = valids.shape[0]
        nviews = len(gt2)
        key_gt_list = ['camera_pose', 'true_shape', 'fxfycxcy', 'render_mask', 'img_org', 'valid_mask', 'pts3d', 'camera_intrinsics', 'size']
        for i in range(len(gt1)):
            gt1_dict = {}
            gt1_dict_unsupervised = {}
            for key in gt1[0].keys():
                if key in key_gt_list:
                    gt1_dict[key] = gt1[i][key][triangluation_mask.to(gt1[i][key].device)]
                    gt1_dict_unsupervised[key] = gt1[i][key][~triangluation_mask.to(gt1[i][key].device)]
                    # gt1_dict_unsupervised[key] = gt1[i][key][triangluation_mask.to(gt1[i][key].device)]
            gt1_use_gtloss.append(gt1_dict)
            gt1_use_unsupervised.append(gt1_dict_unsupervised)

        for i in range(len(gt2)):
            gt2_dict = {}
            gt2_dict_unsupervised = {}
            for key in gt2[0].keys():
                if key in key_gt_list:
                    gt2_dict[key] = gt2[i][key][triangluation_mask.to(gt2[i][key].device)]
                    gt2_dict_unsupervised[key] = gt2[i][key][~triangluation_mask.to(gt2[i][key].device)]
                    # gt2_dict_unsupervised[key] = gt2[i][key][triangluation_mask.to(gt2[i][key].device)]
            gt2_use_gtloss.append(gt2_dict)
            gt2_use_unsupervised.append(gt2_dict_unsupervised)
        for i in range(len(render_gt)):
            render_gt_dict = {}
            render_gt_dict_unsupervised = {}
            for key in render_gt[0].keys():
                if key in key_gt_list:
                    render_gt_dict[key] = render_gt[i][key][triangluation_mask.to(render_gt[i][key].device)]
                    render_gt_dict_unsupervised[key] = render_gt[i][key][~triangluation_mask.to(render_gt[i][key].device)]
                    # render_gt_dict_unsupervised[key] = render_gt[i][key][triangluation_mask.to(render_gt[i][key].device)]
            render_gt_use_gtloss.append(render_gt_dict)
            render_gt_use_unsupervised.append(render_gt_dict_unsupervised)
        key_pred_list = ['pts3d', 'feature', 'opacity', 'scaling', 'rotation', 'conf', 'pts3d_pre', 'depth',  'depth_conf', 'size', 'depth_org']
        pred1_use_gtloss = {}
        pred2_use_gtloss = {}
        pred1_unsupervised = {}
        pred2_unsupervised = {}

        for key in pred1.keys():
            if key in key_pred_list:
                pred1_use_gtloss[key] = pred1[key][triangluation_mask]
                pred1_unsupervised[key] = pred1[key][~triangluation_mask]
                # pred1_unsupervised[key] = pred1[key][triangluation_mask]
                pred2_use_gtloss[key] = pred2[key].unflatten(0, (B, nviews))[triangluation_mask].flatten(0,1)
                # pred2_unsupervised[key] = pred2[key].unflatten(0, (B, nviews))[triangluation_mask].flatten(0,1)
                pred2_unsupervised[key] = pred2[key].unflatten(0, (B, nviews))[~triangluation_mask].flatten(0,1)

        if len(gt1_use_gtloss[0]['pts3d']) != 0:
            (conf_loss1, _), (conf_loss2, _), (loss_image, _), (loss_2d, _), details_gtloss = self.gt_loss(gt1_use_gtloss, gt2_use_gtloss, pred1_use_gtloss, pred2_use_gtloss, trajectory_pred, render_gt=render_gt_use_gtloss, **kw)
            # (loss_image, _), (loss_2d, _), details_gtloss_test = self.unsupervised_loss(gt1_use_gtloss, gt2_use_gtloss, pred1_use_gtloss, pred2_use_gtloss, trajectory_pred, render_gt=render_gt_use_gtloss, **kw)
            # import matplotlib.pyplot as plt
            # plt.subplot(2,2,1)
            # plt.imshow(details_gtloss_test['images'][0,1].permute(1,2,0).detach().cpu().numpy())
            # plt.subplot(2,2,2)
            # plt.imshow(details_gtloss['images'][0,1].permute(1,2,0).detach().cpu().numpy())
            # plt.subplot(2,2,3)
            # plt.imshow(details_gtloss_test['images_gt'][0,1].permute(1,2,0).detach().cpu().numpy())
            # plt.subplot(2,2,4)
            # plt.imshow(details_gtloss['images_gt'][0,1].permute(1,2,0).detach().cpu().numpy())
            # plt.savefig('test.png')
            # import ipdb; ipdb.set_trace()
            # conf_loss1 = conf_loss2 = 0
        else:
            conf_loss1 = conf_loss2 = loss_image = loss_2d = 0
            details_gtloss = {}

        if len(gt1_use_unsupervised[0]['pts3d']) != 0:
            # i = 0
            (loss_image_unsupervised, _), (loss_2d_unsupervised, _), details_unsupervised = self.unsupervised_loss(gt1_use_unsupervised, gt2_use_unsupervised, pred1_unsupervised, pred2_unsupervised, trajectory_pred, render_gt=render_gt_use_unsupervised, **kw)
            # # 'images': images_org,  'images_gt':images_gt * render_mask.float()
            # for num in range(details_unsupervised['images_gt'].shape[1]):
            #     plt.subplot(4,4,num+1)
            #     plt.imshow(details_unsupervised['images_gt'][0,num].permute(1,2,0).detach().cpu().numpy())
            
            # for num in range(details_unsupervised['images'].shape[1]):
            #     plt.subplot(4,4,num+9)
            #     plt.imshow(details_unsupervised['images'][0,num].permute(1,2,0).detach().cpu().numpy())

            # output_path = f'test/output_video_{i}.png'
            # import os
            # os.makedirs('test', exist_ok=True)
            # while os.path.exists(output_path):
            #     i = i  + 1
            #     output_path = f'test/output_video_{i}.png'
            # plt.savefig(output_path)

            if 'images' in details_gtloss.keys():
                details_gtloss.pop('images')
                details_gtloss.pop('images_gt')
                details_gtloss.pop('images_gt_geo')
                details_gtloss.pop('image_2d')
        else:
            loss_image_unsupervised = loss_2d_unsupervised = 0
            details_unsupervised = {}
        
        details = {**details_pose, **details_gtloss, **details_unsupervised}
       

        final_loss = conf_loss1 + conf_loss2 + loss_image * triangluation_mask.float().mean() + loss_2d * triangluation_mask.float().mean() + loss_image_unsupervised * (~triangluation_mask).float().mean() + loss_2d_unsupervised * (~triangluation_mask).float().mean() + pose_r_loss + pose_t_loss + pose_f_loss
        if not math.isfinite(final_loss):
            import ipdb; ipdb.set_trace()
        return final_loss, details



class ConfLoss (MultiLoss):
    """ Weighted regression by learned confidence.
        Assuming the input pixel_loss is a pixel-level regression loss.

    Principle:
        high-confidence means high conf = 0.1 ==> conf_loss = x / 10 + alpha*log(10)
        low  confidence means low  conf = 10  ==> conf_loss = x * 10 - alpha*log(10) 

        alpha: hyperparameter
    """

    def __init__(self, pixel_loss, alpha=1):
        super().__init__()
        assert alpha > 0
        self.alpha = alpha
        self.pixel_loss = pixel_loss.with_reduction('none')

    def get_name(self):
        return f'ConfLoss({self.pixel_loss})'

    def get_conf_log(self, x):
        return x, torch.log(x)

    def compute_loss(self, gt1, gt2, pred1, pred2, trajectory_pred, **kw):
        # compute per-pixel loss
        ((loss1, msk1), (loss2, msk2), (pose_r_loss, _), (pose_t_loss, _), (pose_f_loss, _), (loss_image, _), (loss_2d, _), (loss1_coarse, msk1_coarse), (loss2_coarse, msk2_coarse), (l2_rigid, _)), details, monitoring = self.pixel_loss(gt1, gt2, pred1, pred2, trajectory_pred, **kw)

        # if loss1.numel() == 0:
        #     print('NO VALID POINTS in img1', force=True)
        # if loss2.numel() == 0:
        #     print('NO VALID POINTS in img2', force=True)

        # weight by confidence
        conf1, log_conf1 = self.get_conf_log(pred1['conf'][msk1])
        conf2, log_conf2 = self.get_conf_log(pred2['conf'][msk2.view(*pred2['conf'].shape)])
        # if msk1_coarse is not None:
        #     conf1_coarse, log_conf1_coarse = self.get_conf_log(pred1['conf_coarse'][msk1_coarse])
        #     conf2_coarse, log_conf2_coarse = self.get_conf_log(pred2['conf_coarse'][msk2_coarse.view(*pred2['conf_coarse'].shape)])
        #     conf_loss1_coarse = loss1_coarse * conf1_coarse - self.alpha * log_conf1_coarse
        #     conf_loss2_coarse = loss2_coarse * conf2_coarse - self.alpha * log_conf2_coarse
        # else:
        conf_loss1_coarse = 0
        conf_loss2_coarse = 0
        conf_loss1 = loss1 * conf1 - self.alpha * log_conf1
        conf_loss2 = loss2 * conf2 - self.alpha * log_conf2
        if type(l2_rigid) != int:
            conf_loss2_rigid = l2_rigid * conf2 - self.alpha * log_conf2
            conf_loss2_rigid = conf_loss2_rigid.mean() if conf_loss2_rigid.numel() > 0 else 0
        else:
            conf_loss2_rigid = 0 
        # average + nan protection (in case of no valid pixels at all)
        conf_loss1 = conf_loss1.mean() if conf_loss1.numel() > 0 else 0
        conf_loss2 = conf_loss2.mean() if conf_loss2.numel() > 0 else 0
        if conf_loss2 == 0 and conf_loss1 == 0:
            conf_loss1_coarse = conf_loss2_coarse = conf_loss1 = conf_loss2 = pose_f_loss = loss_image = loss_2d = 0
        if type(conf_loss1_coarse) != int:
            conf_loss1_coarse = conf_loss1_coarse.mean() if conf_loss1_coarse.numel() > 0 else 0
        if type(conf_loss2_coarse) != int:
            conf_loss2_coarse = conf_loss2_coarse.mean() if conf_loss2_coarse.numel() > 0 else 0
        if loss_image is None:
            loss_image = 0
        # if pose_f_loss is None:
        pose_f_loss = 0
        return conf_loss1_coarse + conf_loss2_coarse + conf_loss1 + conf_loss2 + pose_f_loss + pose_r_loss + pose_t_loss  + loss_image + loss_2d + conf_loss2_rigid, dict(conf_loss2_rigid = float(conf_loss2_rigid),conf_loss_1=float(conf_loss1), conf_loss2=float(conf_loss2), pose_r_loss=float(pose_r_loss), pose_t_loss=float(pose_t_loss), pose_f_loss=float(pose_f_loss), image_loss=float(loss_image) ,conf_loss1_coarse=float(conf_loss1_coarse),  conf_loss2_coarse=float(conf_loss2_coarse), **details)


class PoseLoss (MultiLoss):
    """ Weighted regression by learned confidence.
        Assuming the input pixel_loss is a pixel-level regression loss.

    Principle:
        high-confidence means high conf = 0.1 ==> conf_loss = x / 10 + alpha*log(10)
        low  confidence means low  conf = 10  ==> conf_loss = x * 10 - alpha*log(10) 

        alpha: hyperparameter
    """

    def __init__(self, ):
        super().__init__()

    def get_name(self):
        return f'ConfLoss({self.pixel_loss})'

    def get_conf_log(self, x):
        return x, torch.log(x)

    def compute_loss(self, gt1, gt2, pred1, pred2, trajectory_pred, **kw):
        # compute per-pixel loss
        camera_pose = torch.stack([gt1_per['camera_pose'] for gt1_per in gt1], dim=1)
        B = camera_pose.shape[0]
        in_camera1 = inv(camera_pose)
        trajectory = torch.stack([view['camera_pose'] for view in gt2], dim=1)
        trajectory = torch.cat((camera_pose, trajectory), dim=1)
        trajectory = torch.einsum('bnjk,bnkl->bnjl', in_camera1.repeat(1, trajectory.shape[1], 1, 1), trajectory)
        # normalize point map
        trajectory_t_gt = trajectory[..., :3, 3]
        trajectory_t_gt = trajectory_t_gt / (trajectory_t_gt.norm(dim=-1, keepdim=True).mean(dim=1, keepdim=True) + 1e-5)
        quaternion_R = matrix_to_quaternion(trajectory[...,:3,:3])
        trajectory_gt = torch.cat([trajectory_t_gt, quaternion_R], dim=-1)[None]
        fxfycxcy1 = torch.stack([view['fxfycxcy'] for view in gt1], dim=1).float()
        fxfycxcy2 = torch.stack([view['fxfycxcy'] for view in gt2], dim=1).float()
        fxfycxcy = torch.cat((fxfycxcy1, fxfycxcy2), dim=1).to(fxfycxcy1)
        focal_length_gt = fxfycxcy[...,:2]

        trajectory_t_pred = torch.stack([view["T"].reshape(B, -1, 3) for view in trajectory_pred], dim=1)
        trajectory_r_pred = torch.stack([view["quaternion_R"].reshape(B, -1, 4) for view in trajectory_pred], dim=1)
        focal_length_pred = torch.stack([view["focal_length"].reshape(B, -1, 2) for view in trajectory_pred], dim=1)
        trajectory_t_pred = trajectory_t_pred / (trajectory_t_pred.norm(dim=-1, keepdim=True).mean(dim=-2, keepdim=True) + 1e-5)
        trajectory_pred = torch.cat([trajectory_t_pred, trajectory_r_pred], dim=-1)
        trajectory_pred = trajectory_pred.permute(1,0,2,3)
        focal_length_pred = focal_length_pred.permute(1,0,2,3)

        n_predictions = len(trajectory_pred)
        pose_r_loss = 0.0
        pose_t_loss = 0.0
        pose_f_loss = 0.0
        gamma = 0.8
        for i in range(n_predictions):
            i_weight = gamma ** (n_predictions - i % 4 - 1)
            trajectory_pred_iter = trajectory_pred[i]
            trajectory_gt_iter = trajectory_gt[0]
            trajectory_pred_r = trajectory_pred_iter[..., 3:]
            trajectory_pred_t = trajectory_pred_iter[..., :3]
            trajectory_gt_r = trajectory_gt_iter[..., 3:]
            trajectory_gt_t = trajectory_gt_iter[..., :3]
            pose_r_loss += i_weight * (trajectory_gt_r - trajectory_pred_r).abs().mean() * 10
            pose_t_loss += i_weight * (trajectory_gt_t - trajectory_pred_t).abs().mean() * 10
        pose_t_loss = pose_t_loss / n_predictions
        pose_r_loss = pose_r_loss / n_predictions
        with torch.no_grad():
            batch_size = trajectory_pred.shape[1]
            R = quaternion_to_matrix(trajectory_pred[-1][..., 3:].reshape(-1, 4)).reshape(batch_size, -1, 3, 3)
            if n_predictions > 4:
                R_prior = quaternion_to_matrix(trajectory_pred[:4][-1][..., 3:].reshape(-1, 4)).reshape(batch_size, -1, 3, 3)
            else:
                R_prior = None
            R_gt = quaternion_to_matrix(trajectory_gt[..., 3:].reshape(-1, 4)).reshape(batch_size, -1, 3, 3)
            if R_prior is not None:
                se3_pred_prior = torch.cat((R_prior, trajectory_pred[:4][-1][..., :3][..., None]), dim=-1).reshape(batch_size, -1 , 3, 4)
            else:
                se3_pred_prior = None
            se3_gt = torch.cat((R_gt, trajectory_gt[0,..., :3, None]), dim=-1).reshape(batch_size, -1 , 3, 4)
            se3_pred = torch.cat((R, trajectory_pred[-1][..., :3][..., None]), dim=-1).reshape(batch_size, -1 , 3, 4)
            # se3_gt = se3_gt[0]
            # se3_pred = se3_pred[0]
            nviews = se3_pred.shape[1]
            pair_idx_i1, pair_idx_i2 = batched_all_pairs(B, nviews)
            bottom_ = torch.tensor([[[0,0,0,1]]]).to(se3_gt.device)
            se3_gt = se3_gt.reshape(-1, 3, 4)
            se3_pred = se3_pred.reshape(-1, 3, 4)
            if se3_pred_prior is not None:
                se3_pred_prior = se3_pred_prior.reshape(-1, 3, 4)
            se3_gt = torch.cat((se3_gt, bottom_.repeat(se3_gt.shape[0],1,1)), dim=1)
            se3_pred = torch.cat((se3_pred, bottom_.repeat(se3_pred.shape[0],1,1)), dim=1)
            if se3_pred_prior is not None:
                se3_pred_prior = torch.cat((se3_pred_prior, bottom_.repeat(se3_pred_prior.shape[0],1,1)), dim=1)
            relative_pose_gt = closed_form_inverse(se3_gt[pair_idx_i1]).bmm(se3_gt[pair_idx_i2])
            relative_pose_pred = closed_form_inverse(se3_pred[pair_idx_i1]).bmm(se3_pred[pair_idx_i2])
            if se3_pred_prior is not None:
                relative_pose_pred_prior = closed_form_inverse(se3_pred_prior[pair_idx_i1]).bmm(se3_pred_prior[pair_idx_i2])
            rel_rangle_deg = rotation_angle(relative_pose_gt[:, :3, :3], relative_pose_pred[:, :3, :3]).mean()
            rel_tangle_deg = translation_angle(relative_pose_gt[:, :3, 3:], relative_pose_pred[:, :3, 3:]).mean()
            if se3_pred_prior is not None:
                rel_rangle_deg_prior = rotation_angle(relative_pose_gt[:, :3, :3], relative_pose_pred_prior[:, :3, :3]).mean()
                rel_tangle_deg_prior = translation_angle(relative_pose_gt[:, :3, 3:], relative_pose_pred_prior[:, :3, 3:]).mean()
            else:
                rel_rangle_deg_prior = rel_tangle_deg_prior = 0
        return pose_r_loss + pose_f_loss + pose_t_loss, dict(pose_r_loss=float(pose_r_loss), pose_f_loss=float(pose_f_loss), pose_t_loss=float(pose_t_loss), rel_tangle_deg=float(rel_tangle_deg), rel_rangle_deg=float(rel_rangle_deg), rel_rangle_deg_prior=float(rel_rangle_deg_prior), rel_tangle_deg_prior=float(rel_tangle_deg_prior))

class Regr3D_ShiftInv (Regr3D_gs):
    """ Same than Regr3D but invariant to depth shift.
    """

    def get_all_pts3d(self, gt1, gt2, pred1, pred2):
        # compute unnormalized points
        gt_pts1, gt_pts2, pred_pts1, pred_pts2, mask1, mask2, monitoring = \
            super().get_all_pts3d(gt1, gt2, pred1, pred2)

        # compute median depth
        gt_z1, gt_z2 = gt_pts1[..., 2], gt_pts2[..., 2]
        pred_z1, pred_z2 = pred_pts1[..., 2], pred_pts2[..., 2]
        gt_shift_z = get_joint_pointcloud_depth(gt_z1, gt_z2, mask1, mask2)[:, None, None]
        pred_shift_z = get_joint_pointcloud_depth(pred_z1, pred_z2, mask1, mask2)[:, None, None]

        # subtract the median depth
        gt_z1 -= gt_shift_z
        gt_z2 -= gt_shift_z
        pred_z1 -= pred_shift_z
        pred_z2 -= pred_shift_z

        # monitoring = dict(monitoring, gt_shift_z=gt_shift_z.mean().detach(), pred_shift_z=pred_shift_z.mean().detach())
        return gt_pts1, gt_pts2, pred_pts1, pred_pts2, mask1, mask2, monitoring


class Regr3D_ScaleInv (Regr3D_gs):
    """ Same than Regr3D but invariant to depth shift.
        if gt_scale == True: enforce the prediction to take the same scale than GT
    """

    def get_all_pts3d(self, gt1, gt2, pred1, pred2):
        # compute depth-normalized points
        gt_pts1, gt_pts2, pred_pts1, pred_pts2, mask1, mask2, monitoring = super().get_all_pts3d(gt1, gt2, pred1, pred2)

        # measure scene scale
        _, gt_scale = get_joint_pointcloud_center_scale(gt_pts1, gt_pts2, mask1, mask2)
        _, pred_scale = get_joint_pointcloud_center_scale(pred_pts1, pred_pts2, mask1, mask2)

        # prevent predictions to be in a ridiculous range
        pred_scale = pred_scale.clip(min=1e-3, max=1e3)

        # subtract the median depth
        if self.gt_scale:
            pred_pts1 *= gt_scale / pred_scale
            pred_pts2 *= gt_scale / pred_scale
            # monitoring = dict(monitoring, pred_scale=(pred_scale/gt_scale).mean())
        else:
            gt_pts1 /= gt_scale
            gt_pts2 /= gt_scale
            pred_pts1 /= pred_scale
            pred_pts2 /= pred_scale
            # monitoring = dict(monitoring, gt_scale=gt_scale.mean(), pred_scale=pred_scale.mean().detach())

        return gt_pts1, gt_pts2, pred_pts1, pred_pts2, mask1, mask2, monitoring


class Regr3D_ScaleShiftInv (Regr3D_ScaleInv, Regr3D_ShiftInv):
    # calls Regr3D_ShiftInv first, then Regr3D_ScaleInv
    pass




class Regr3D_clean_scale(Regr3D_clean):
    """ Ensure that all 3D points are correct.
        Asymmetric loss: view1 is supposed to be the anchor.

        P1 = RT1 @ D1
        P2 = RT2 @ D2
        loss1 = (I @ pred_D1) - (RT1^-1 @ RT1 @ D1)
        loss2 = (RT21 @ pred_D2) - (RT1^-1 @ P2)
              = (RT21 @ pred_D2) - (RT1^-1 @ RT2 @ D2)
    """

    def __init__(self, criterion, norm_mode='avg_dis', gt_scale=False, test=False):
        super().__init__(criterion)
        self.norm_mode = norm_mode
        self.gt_scale = gt_scale
        self.epoch = 0
        self.test = test
        self.gt_num_image = 0

    def compute_loss(self, gt1, gt2, pred1, pred2, trajectory_pred, render_gt=None, **kw):
        ((l1, msk1), (loss2, msk2), (pose_r_loss, _), (pose_t_loss, _), (pose_f_loss, _), (loss_image, _), (loss_2d, _), (loss1_coarse, msk1_coarse), (loss2_coarse, msk2_coarse)), details, monitoring =  super().compute_loss(gt1, gt2, pred1, pred2, trajectory_pred, render_gt=render_gt, **kw)
        gt1_pts3d = torch.stack([gt1_per['pts3d'] for gt1_per in gt1], dim=1)
        pr_pts1 = pred1['pts3d']
        pr_pts2 = pred2['pts3d']

        B, _, H, W, _ = gt1_pts3d.shape
        num_views = pr_pts2.shape[0] // B
        pr_pts1 = pr_pts1.reshape(B, 1, H, W, 3)
        pr_pts2 = pr_pts2.reshape(B, num_views, H, W, 3)
        pr_pts = torch.cat((pr_pts1, pr_pts2), dim=1)
        # caculate the number of views and read the gt2
        gt2_pts3d = torch.stack([gt2_per['pts3d'] for gt2_per in gt2], dim=1)
        # camera trajectory

        camera_pose = torch.stack([gt1_per['camera_pose'] for gt1_per in gt1], dim=1)
        in_camera1 = closed_form_inverse_OpenCV(camera_pose.reshape(-1,4,4)).reshape(B, -1, 4, 4)
        trajectory = torch.stack([view['camera_pose'] for view in gt2], dim=1)

        camera_intrinsics1 = torch.stack([gt1_per['camera_intrinsics'] for gt1_per in gt1], dim=1)
        camera_intrinsics2 = torch.stack([gt2_per['camera_intrinsics'] for gt2_per in gt2], dim=1)
        camera_intrinsics = torch.cat((camera_intrinsics1, camera_intrinsics2), dim=1)

        norm_factor_gt, norm_factor_pr = monitoring
        trajectory = torch.cat((camera_pose, trajectory), dim=1)
        trajectory = torch.einsum('bnjk,bnkl->bnjl', in_camera1.repeat(1, trajectory.shape[1],1,1), trajectory)
        # gt_pts = geotrf(in_camera1.repeat(1,num_views+1,1,1).view(-1,4,4), gt_pts.view(-1,H,W,3))  # B,H,W,3

        gt_R = trajectory[...,:3,:3]
        pred_T = trajectory_pred[-1]['T'].reshape(B, -1, 3)
        SE3_pred = torch.cat((gt_R, pred_T[..., None]), dim=-1).reshape(-1,3,4)
        bottom_ = torch.tensor([[[0,0,0,1]]]).to(SE3_pred.device)
        SE3_pred = torch.cat((SE3_pred, bottom_.repeat(SE3_pred.shape[0],1,1)), dim=1)
        SE3_pred = closed_form_inverse_OpenCV(SE3_pred)
        SE3_gt = closed_form_inverse_OpenCV(trajectory.reshape(-1,4,4))
        
        trajectory_t = trajectory[..., :3, 3] / (norm_factor_gt[1] + 1e-7)
        trajectory_t_pred = torch.stack([view["T"].reshape(B, -1, 3) for view in trajectory_pred], dim=1)
        trajectory_t_pred = trajectory_t_pred[:, trajectory_t_pred.shape[1]//2:]
        trajectory_t_pred_scaled = trajectory_t_pred / (norm_factor_pr + 1e-7)
        conf1 = pred1['conf']
        conf2 = pred2['conf']
        confs = torch.cat((conf1.reshape(B, 1, H, W), conf2.reshape(B, -1, H, W)), dim=1)
        sorted_indices = torch.argsort(confs.reshape(-1, H * W).float(), axis=1)
        top_1024_indices = sorted_indices[:, -1024:]  # 形状为 [16, 1024]
        # gt_pts = gt_pts.reshape(-1, H*W,3)
        top_1024_indices = top_1024_indices[..., None].repeat(1,1,3)
        # pr_pts = torch.cat((gt1_pts3d, gt2_pts3d), dim=1)
        pr_pts = torch.gather(pr_pts.reshape(-1, H*W,3).detach(), dim=1, index=top_1024_indices)
        confidence = torch.gather(confs.reshape(-1, H*W), dim=1, index=top_1024_indices[..., 0])
        pr_pts = pr_pts.reshape(B, -1, 3)
        confidence = confidence.reshape(B,1,-1,1).repeat(1,num_views+1,1,1)
        projected_points = reproject(pr_pts, camera_intrinsics, SE3_pred.reshape(B,-1,4,4)) # pred
        allcam_Ps = torch.einsum('bnij,bnjk->bnik', camera_intrinsics, SE3_gt.reshape(B,-1,4,4)[...,:3,:])
        points_3d_world_gt = triangulate_batch_of_points(allcam_Ps, projected_points) # gt
        scale = (points_3d_world_gt.norm(dim=-1, keepdim=True) / pr_pts.norm(dim=-1, keepdim=True)).mean(dim=1, keepdim=True)  # gt/pred
        n_predictions = len(trajectory_pred) // 2
        pose_t_scale_loss = 0.0
        pose_t_align_loss = 0.0
        gamma = 0.8
        for i in range(n_predictions):
            i_weight = gamma ** (n_predictions - i % 4 - 1)
            pose_t_scale_loss += i_weight * (trajectory_t - trajectory_t_pred_scaled[:, i]).abs().mean()
            pose_t_align_loss += i_weight * ((trajectory.reshape(B,-1,4,4)[..., :3, 3] - trajectory_t_pred[:, i] * scale)/trajectory[..., :3, 3].norm(dim=-1, keepdim=True).mean(dim=1, keepdim=True)).abs().mean()
            if torch.isinf(pose_t_scale_loss) or torch.isnan(pose_t_scale_loss) or torch.isinf(pose_t_align_loss) or torch.isnan(pose_t_align_loss):
                pose_t_align_loss = 0
                pose_t_scale_loss = 0

        details['pose_t_align_loss'] = float(pose_t_align_loss)
        details['pose_t_scale_loss'] = float(pose_t_scale_loss)
        pose_t_loss = pose_t_loss + pose_t_scale_loss
        details['scale'] = float(scale.mean())
        return Sum((l1, msk1), (loss2, msk2), (pose_r_loss, _), (pose_t_loss, _), (pose_f_loss, _), (loss_image, _), (loss_2d, _), (loss1_coarse, msk1_coarse), (loss2_coarse, msk2_coarse)), details
        #  ((loss1, msk1), (loss2, msk2), (pose_r_loss, _), (pose_t_loss, _), (pose_f_loss, _), (loss_image, _), (loss_2d, _), (loss1_coarse, msk1_coarse), (loss2_coarse, msk2_coarse))

class Regr3D_gs_unsupervised_2v_old(Regr3D_clean):
    """ Ensure that all 3D points are correct.
        Asymmetric loss: view1 is supposed to be the anchor.

        P1 = RT1 @ D1
        P2 = RT2 @ D2
        loss1 = (I @ pred_D1) - (RT1^-1 @ RT1 @ D1)
        loss2 = (RT21 @ pred_D2) - (RT1^-1 @ P2)
              = (RT21 @ pred_D2) - (RT1^-1 @ RT2 @ D2)
    """

    def __init__(self, criterion, alpha, multiview_triangulation = True, disable_rayloss = False, scaling_mode='interp_5e-4_0.1_3', norm_mode='avg_dis', gt_scale=False, test=False):
        super().__init__(criterion)
        self.norm_mode = norm_mode
        self.disable_rayloss = disable_rayloss
        self.multiview_triangulation = multiview_triangulation
        self.scaling_mode = {'type':scaling_mode.split('_')[0], 'min_scaling': eval(scaling_mode.split('_')[1]), 'max_scaling': eval(scaling_mode.split('_')[2]), 'shift': eval(scaling_mode.split('_')[3])}
        self.gt_scale = gt_scale
        from .perceptual import PerceptualLoss
        self.perceptual_loss = PerceptualLoss().cuda().eval()
        for param in self.perceptual_loss.parameters():
            param.requires_grad = False
        self.random_crop = RandomCrop(224)
        self.epoch = 0
        self.test = test
        self.gt_num_image = 0
        self.render_landscape = transpose_to_landscape_render(self.render)
        self.loss2d_landscape = transpose_to_landscape(self.loss2d)
        self.alpha = alpha
        self.sm = nn.SmoothL1Loss()
        self.disp_loss =  ScaleAndShiftInvariantLoss()

    def render(self, render_params, image_size):
        latent, output_fxfycxcy, output_c2ws = render_params
        (H_org, W_org) = image_size
        H = H_org
        W = W_org
        # if 'scaling_factor' in latent.keys():
        #     scaling_factor = latent.pop('scaling_factor')
        #     if self.scaling_mode['type'] == 'precomp':
        #         x = torch.clip(latent['pre_scaling'] - self.scaling_mode['shift'], max=np.log(3))
        #         latent['scaling'] = torch.exp(x).clamp(min= self.scaling_mode['min_scaling'], max=self.scaling_mode['max_scaling']) / scaling_factor
        # else:
        #     scaling_factor = torch.tensor(1.0).to(latent['xyz'])
        new_latent = {}
        if self.scaling_mode['type'] == 'precomp': 
            scaling_factor = latent['scaling_factor']
            x = torch.clip(latent['pre_scaling'] - self.scaling_mode['shift'], max=np.log(0.3))
            new_latent['scaling'] = torch.exp(x).clamp(min= self.scaling_mode['min_scaling'], max=self.scaling_mode['max_scaling']) / scaling_factor
            skip = ['pre_scaling', 'scaling', 'scaling_factor']
        else:
            skip = ['pre_scaling', 'scaling_factor']
        for key in latent.keys():
            if key not in skip:
                new_latent[key] = latent[key]
        gs_render = GaussianRenderer(H_org, W_org, patch_size=128, gs_kwargs={'type':self.scaling_mode['type'], 'min_scaling': self.scaling_mode['min_scaling'], 'max_scaling': self.scaling_mode['max_scaling'], 'scaling_factor': scaling_factor}) # scaling_factor
        results = gs_render(new_latent, output_fxfycxcy.reshape(1,-1,4), output_c2ws.reshape(1,-1,4,4))
        images = results['image']
        image_mask = results['alpha']
        depth = results['depth']
        depth = depth.reshape(-1,1,H,W).permute(0,2,3,1)
        image_mask = image_mask.reshape(-1,1,H,W).permute(0,2,3,1)
        images = images.reshape(-1,3,H,W).permute(0,2,3,1)
        return {'images': images, 'image_mask': image_mask, 'depth':depth}
    
    def loss2d(self, input, image_size):
        pr_pts, fxfycxcy, c2ws = input
        H, W = image_size
        loss_2d = reproj2d(fxfycxcy, c2ws, pr_pts, image_size)
        return {'loss_2d': loss_2d}
    
    def get_all_pts3d(self, gt1, gt2, pred1, pred2, trajectory_pred, dist_clip=None, render_gt=None):
        # everything is normalized w.r.t. camera of view1        
        pr_pts1 = pred1['pts3d']
        pr_pts2 = pred2['pts3d']
        # GT pts3d
        gt1_pts3d = torch.stack([gt1_per['pts3d'] for gt1_per in gt1], dim=1)
        B, _, H, W, _ = gt1_pts3d.shape
        # caculate the number of views and read the gt2
        num_views = pr_pts2.shape[0] // B
        gt2_pts3d = torch.stack([gt2_per['pts3d'] for gt2_per in gt2[:num_views]], dim=1)
        # camera trajectory
        camera_pose = torch.stack([gt1_per['camera_pose'] for gt1_per in gt1], dim=1)
        in_camera1 = inv(camera_pose)
        trajectory = torch.stack([view['camera_pose'] for view in gt2], dim=1)
        trajectory = torch.cat((camera_pose, trajectory), dim=1)
        # import ipdb; ipdb.set_trace()

        # trajectory_1 = closed_form_inverse(camera_pose.repeat(1, trajectory.shape[1],1,1).reshape(-1,4,4)).bmm(trajectory.reshape(-1,4,4))
        trajectory = torch.einsum('bnjk,bnkl->bnjl', in_camera1.repeat(1, trajectory.shape[1],1,1), trajectory)
        # trajectory_1 - trajectory.reshape(-1,4,4)
        # transform the gt points to the coordinate of the first view
        Rs = []
        Ts = []


        # valid mask
        valid1 = torch.stack([gt1_per['valid_mask'] for gt1_per in gt1], dim=1).view(B,-1,H,W).clone()
        valid2 = torch.stack([gt2_per['valid_mask'] for gt2_per in gt2[:num_views]], dim=1).view(B,-1,H,W).clone()

        gt_pts1 = gt1_pts3d
        gt_pts2 = gt2_pts3d
        # reshape point map
        gt_pts1 = gt_pts1.reshape(B,H,W,3)
        gt_pts2 = gt_pts2.reshape(B,H*num_views,W,3)
        pr_pts1 = pr_pts1.reshape(B,H,W,3)
        pr_pts2 = pr_pts2.reshape(B,H*num_views,W,3)
        valid1 = valid1.view(B,H,W)
        valid2 = valid2.view(B,H*num_views,W)
        
        if valid1.sum() == 0 and valid2.sum() == 0:
            valid1 = torch.ones_like(valid1).to(valid1) > 0
            valid2 = torch.ones_like(valid2).to(valid2) > 0
            # import ipdb; ipdb.set_trace()

        # generate the gt camera trajectory
        trajectory_t_gt = trajectory[..., :3, 3]
        size =  (trajectory_t_gt[:,0:1] - trajectory_t_gt[:,1:2]).norm(dim=-1, keepdim=True) + 1e-8 
        trajectory_t_gt = trajectory_t_gt / size
        quaternion_R = matrix_to_quaternion(trajectory[...,:3,:3])
        trajectory_gt = torch.cat([trajectory_t_gt, quaternion_R], dim=-1)[None]
        R = trajectory[...,:3,:3]
        fxfycxcy1 = torch.stack([view['fxfycxcy'] for view in gt1], dim=1).float()
        fxfycxcy2 = torch.stack([view['fxfycxcy'] for view in gt2], dim=1).float()
        fxfycxcy = torch.cat((fxfycxcy1, fxfycxcy2), dim=1).to(fxfycxcy1)
        focal_length_gt = fxfycxcy[...,:2]

        # generate the predicted camera trajectory
        with torch.no_grad():
            trajectory_R_prior = trajectory_pred[:4][-1]['R'].reshape(B, -1, 3, 3)
            trajectory_R_post = trajectory_pred[-1]['R'].reshape(B, -1, 3, 3)

        trajectory_t_pred = torch.stack([view["T"].reshape(B, -1, 3) for view in trajectory_pred], dim=1)
        trajectory_r_pred = torch.stack([view["quaternion_R"].reshape(B, -1, 4) for view in trajectory_pred], dim=1)
        focal_length_pred = torch.stack([view["focal_length"].reshape(B, -1, 2) for view in trajectory_pred], dim=1)
        size_pred = (trajectory_t_pred[:,:, 0:1] - trajectory_t_pred[:,:, 1:2]).norm(dim=-1, keepdim=True) + 1e-8 
        trajectory_t_pred = trajectory_t_pred / size_pred
        trajectory_pred = torch.cat([trajectory_t_pred, trajectory_r_pred], dim=-1)
        trajectory_pred = trajectory_pred.permute(1,0,2,3)
        focal_length_pred = focal_length_pred.permute(1,0,2,3)
        B, H, W, _ = gt_pts1.shape
        nviews = len(gt2)
        sh_dim = pred1['feature'].shape[-1]
        feature1 = pred1['feature'].reshape(B,H,W,sh_dim)
        feature2 = pred2['feature'].reshape(B,-1,W,sh_dim)
        feature = torch.cat((feature1, feature2), dim=1).float()
        image_2d = torch.cat((feature1.reshape(B,-1, H,W,sh_dim)[..., :3], feature2.reshape(B,-1, H,W,sh_dim)[..., :3]), dim=1)
        if sh_dim > 3:
            image_2d = sh_to_rgb(image_2d).clamp(0,1)
        else:
            image_2d = torch.sigmoid(image_2d)
        opacity1 = pred1['opacity'].reshape(B,H,W,1)
        opacity2 = pred2['opacity'].reshape(B,-1,W,1)
        opacity = torch.cat((opacity1, opacity2), dim=1).float()
        scaling1 = pred1['scaling'].reshape(B,H,W,3)
        scaling2 = pred2['scaling'].reshape(B,-1,W,3)
        scaling = torch.cat((scaling1, scaling2), dim=1).float()
        rotation1 = pred1['rotation'].reshape(B,H,W,4)
        rotation2 = pred2['rotation'].reshape(B,-1,W,4)
        rotation = torch.cat((rotation1, rotation2), dim=1).float()
        output_fxfycxcy = torch.stack([gt['fxfycxcy'] for gt in render_gt], dim=1)
        render_mask = torch.stack([gt['render_mask'] for gt in render_gt], dim=1)
        shape = torch.stack([gt['true_shape'] for gt in render_gt], dim=1)
        xyz = torch.cat((pr_pts1, pr_pts2), dim=1)
        pr_pts = torch.cat((pr_pts1.reshape(B,-1,H,W,3), pr_pts2.reshape(B,-1,H,W,3)), dim=1)
        xyz_gt = torch.cat((gt_pts1.reshape(B,-1,H,W,3), gt_pts2.reshape(B,-1,H,W,3)), dim=1).detach()
        # TODO rewrite transform cameras

        images_gt = torch.stack([gt['img_org'] for gt in render_gt], dim=1)
        images_gt = images_gt / 2 + 0.5
        norm_factor_gt = 1
        output_c2ws = torch.stack([gt['camera_pose'] for gt in render_gt], dim=1)
        camera_pose = torch.stack([gt1_per['camera_pose'] for gt1_per in gt1], dim=1)
        in_camera1 = inv(camera_pose)
        output_c2ws = torch.einsum('bnjk,bnkl->bnjl', in_camera1.repeat(1,output_c2ws.shape[1],1,1), output_c2ws)
        output_c2ws[..., :3, 3:] = output_c2ws[..., :3, 3:]
        depth_gt = torch.stack([gt['depth_anything'] for gt in render_gt], dim=1)
        if self.test:
            with torch.no_grad():
                trajectory_pred_t = output_c2ws[0,:, :3, 3]
                trajectory_pred_r = matrix_to_quaternion(output_c2ws[0,:, :3,:3])
                trajectory_pred = torch.cat([trajectory_pred_t, trajectory_pred_r], dim=-1)
                render_pose = interpolate_poses(list(range(len(trajectory_pred))), trajectory_pred.detach().cpu().numpy(), list(np.linspace(0,2,280)), 0)
                render_pose = np.stack(render_pose)
                render_pose = torch.tensor(render_pose).cuda()[:200]
                i = 0
                clip = 8
                # pr_pts1_pre, pr_pts2_pre, norm_factor_pr = normalize_pointcloud(pr_pts1_pre, pr_pts2_pre, self.norm_mode, valid1, valid2, ret_factor=True)
                latent = {'xyz': xyz.reshape(B, -1, 3)[i:i+1], 'feature': feature.reshape(B, -1, sh_dim)[i:i+1], 'opacity': opacity.reshape(B, -1, 1)[i:i+1], 'pre_scaling': scaling.reshape(B, -1, 3)[i:i+1].clone(), 'rotation': rotation.reshape(B, -1, 4)[i:i+1], 'scaling_factor': 1}
                # ret = self.render_landscape([latent, output_fxfycxcy[i:i+1][:1].reshape(-1,4).repeat(render_pose.reshape(-1,4,4).shape[0],1), render_pose.reshape(-1,4,4)], shape[i:i+1].reshape(-1,2))
                ret = self.render_landscape([latent, output_fxfycxcy[i:i+1][:1].reshape(-1,4).repeat(render_pose.reshape(-1,4,4).shape[0],1), render_pose.reshape(-1,4,4)], shape[i:i+1].reshape(-1,2))

                import viser
                # import 
                camera_poses = torch.stack([gt['camera_pose'] for gt in gt1+gt2], dim=1)[0]
                camera_poses = torch.einsum('njk,nkl->njl', in_camera1[0].repeat(camera_poses.shape[0],1,1), camera_poses)
                camera_poses[:, :3, 3] = camera_poses[:, :3, 3]
                # camera_poses = camera_poses.inverse()
                images = torch.stack([gt['img_org'] for gt in gt1+gt2], dim=1)
                fxfycxcy = torch.stack([gt['fxfycxcy'] for gt in gt1+gt2], dim=1)[0]
                images = images.permute(0,1,3,4,2)[0] / 2 + 0.5
                colors = images.reshape(B, -1, 3).detach().cpu().numpy() 
                points3d = pr_pts.reshape(B, -1, 3)[0]

                video = ret['images'].detach().cpu().numpy()
                import imageio
                import os
                i = 0
                output_path = f'test/output_video_{i}.mp4'
                os.makedirs('test', exist_ok=True)
                while os.path.exists(output_path):
                    i = i  + 1
                    output_path = f'test/output_video_{i}.mp4'
                # pc.save_ply_v2(f'test/{i}.ply')
                # import matplotlib.pyplot as plt
                # plt.imshow(images[0,0].detach().cpu().numpy().transpose(1,2,0))
                # plt.savefig('test/output.png')
                writer = imageio.get_writer(output_path, fps=30)  # You can specify the filename and frames per second
                for frame in video:  # Example for 100 frames
                    # Add the frame to the video
                    frame = (frame * 255).clip(0,255).astype(np.uint8)
                    writer.append_data(frame)
                # Close the writer to finalize the video
                writer.close()
                for i in range(len(images)):
                    imageio.imwrite(f'test/output_{i}.png', (images[i] * 255).clip(0,255).detach().cpu().numpy().astype(np.uint8))
                vis(camera_poses, images, points3d, colors, fxfycxcy, H, W)

        images_list = []
        image_mask_list = []
        depths_rendered_list = []
        for i in range(B):
            # try:
            latent = {'xyz': xyz.reshape(B, -1, 3)[i:i+1], 'feature': feature.reshape(B, -1, sh_dim)[i:i+1], 'opacity': opacity.reshape(B, -1, 1)[i:i+1], 'pre_scaling': scaling.reshape(B, -1, 3)[i:i+1].clone(), 'rotation': rotation.reshape(B, -1, 4)[i:i+1], 'scaling_factor': 1}
            # if 'scaling_factor' not in latent.keys():
            ret = self.render_landscape([latent, output_fxfycxcy[i:i+1].reshape(-1,4), output_c2ws[i:i+1].reshape(-1,4,4)], shape[i:i+1].reshape(-1,2))
            # except:
            #     import ipdb; ipdb.set_trace()
            images = ret['images']
            image_mask = ret['image_mask']
            images_list.append(images)
            image_mask_list.append(image_mask)
            depths_rendered_list.append(ret['depth'])
        images = torch.stack(images_list, dim=0).permute(0,1,4,2,3)
        image_masks = torch.stack(image_mask_list, dim=0).permute(0,1,4,2,3)
        depths_rendered_list = torch.stack(depths_rendered_list, dim=0).permute(0,1,4,2,3)
        
        image_mask_gt_geo = image_masks.clone()
        images_gt_geo = images.clone()
        shape_input = torch.stack([gt['true_shape'] for gt in gt1+gt2], dim=1)
        input_fxfycxcy = output_fxfycxcy[:,:4]
        input_c2ws = output_c2ws[:,:4]
        shape_input = shape_input[:,:4]
        pr_pts = pr_pts[:,:4]
        # loss_2d = self.loss2d_landscape([pr_pts.reshape(-1, H, W, 3), input_fxfycxcy.reshape(-1,4), input_c2ws.reshape(-1,4,4)], shape_input.reshape(-1,2))
        # import ipdb; ipdb.set_trace()
        # masks = torch.cat((valid1.reshape(B, -1, H, W), valid2.reshape(B, -1, H, W)), 1)

        metric_conf1 = pred1['conf']
        metric_conf2 = pred2['conf']
        pr_pts2_transform = gt_pts2.reshape(*gt_pts2.shape)
        return gt_pts1, gt_pts2, pr_pts1, pr_pts2, valid1, valid2, metric_conf1, metric_conf2, trajectory_gt, trajectory_pred, images, images_gt, image_masks, images_gt_geo, image_mask_gt_geo, render_mask, R, trajectory_R_prior, trajectory_R_post, image_2d, depth_gt, depths_rendered_list, {}


    def get_conf_log(self, x):
        return x, torch.log(x)

    
    def gt_loss(self, gt1, gt2, pred1, pred2, trajectory_pred, render_gt=None, **kw):
        gt_pts1, gt_pts2, pr_pts1, pr_pts2,  mask1, mask2, metric_conf1, metric_conf2, trajectory, trajectory_pred, images, images_gt, image_mask, images_gt_geo, image_mask_gt_geo, render_mask,  R, trajectory_R_prior, trajectory_R_post, image_2d, depth_gt, depths, monitoring = self.get_all_pts3d(gt1, gt2, pred1, pred2, trajectory_pred, render_gt=render_gt, **kw)
        # loss on img1 side
        # loss on gt2 side
        render_mask = render_mask[:,:,None].repeat(1,1,3,1,1)
        render_mask = render_mask.float()
        image_2d = image_2d.permute(0,1,4,2,3)
        images_org = images.clone()
        
        loss_image_pred_geo = ((images.reshape(*render_mask.shape)[:,:4][render_mask[:,:4] > 0.] - images_gt.reshape(*render_mask.shape)[:,:4][render_mask[:,:4] > 0.]) ** 2)
        loss_image_pred_geo = loss_image_pred_geo.mean() if loss_image_pred_geo.numel() > 0 else 0
        loss_image_pred_geo_novel = ((images.reshape(*render_mask.shape)[:,4:][render_mask[:,4:] > 0.] - images_gt.reshape(*render_mask.shape)[:,4:][render_mask[:,4:] > 0.]) ** 2)
        loss_image_pred_geo_novel = loss_image_pred_geo_novel.mean() if loss_image_pred_geo_novel.numel() > 0 else 0
        
        # loss_image_pred_self = ((image_2d[:,:4] - images_gt[:,:4]) ** 2).mean()
        loss_image_pred_geo = loss_image_pred_geo * 4 + loss_image_pred_geo_novel * 4 
        
        images_masked = images
        # images_masked[:,:4] = (images_masked[:,:4] * 0.5 + image_2d[:,:4] * 0.5)
        images_masked = images_masked * render_mask + images_gt * (1 - render_mask)
        images_masked, images_gt_1 = self.random_crop(images_masked, images_gt)
        images_masked = images_masked.reshape(-1,*images_gt_1.shape[-3:])
        images_gt_1 = images_gt_1.reshape(-1,*images_gt_1.shape[-3:])
        loss_vgg = self.perceptual_loss(images_masked, images_gt_1) * 0.2
        
        loss_image = loss_vgg + loss_image_pred_geo
        # image_2d
        mse = torch.mean(((images_org.reshape(*render_mask.shape)[render_mask > 0.] - images_gt.reshape(*render_mask.shape)[render_mask > 0.]) * 255) ** 2)
        PSNR = 20 * torch.log10(255/torch.sqrt(mse))

        msk1 = mask1
        msk2 = mask2
        loss_2d = 0
        self_name = type(self).__name__
        details = {'PSNR': float(PSNR), 'loss_2d': float(loss_2d), 'loss_pred_geo': float(loss_image_pred_geo), 'loss_vgg': float(loss_vgg), self_name+'_image': float(loss_image), 'images': images_org,  'images_gt':images_gt * render_mask.float(), 'images_gt_geo': images_gt_geo, 'image_2d': image_2d}
        return (loss_image, None), (loss_2d, None), details

    def compute_loss(self, gt1, gt2, pred1, pred2, trajectory_pred, render_gt=None, **kw):
        (pose_r_loss, _), (pose_t_loss, _), (pose_f_loss, _), details_pose = self.compute_pose_loss(gt1, gt2, pred1, pred2, trajectory_pred, render_gt=render_gt)
        valids = torch.stack([gt1_per['valid_mask'] for gt1_per in gt1+gt2], dim=1)
        triangluation_mask = valids.flatten(1,3).sum(1) > -1
        gt1_use_gtloss = []
        gt2_use_gtloss = []
        render_gt_use_gtloss = []
        gt1_use_unsupervised = []
        gt2_use_unsupervised = []
        render_gt_use_unsupervised = []

        B = valids.shape[0]
        nviews = len(gt2)
        key_gt_list = ['camera_pose', 'true_shape', 'fxfycxcy', 'render_mask', 'img_org', 'valid_mask', 'pts3d', 'camera_intrinsics', 'depth_anything']
        for i in range(len(gt1)):
            gt1_dict = {}
            gt1_dict_unsupervised = {}
            for key in gt1[0].keys():
                if key in key_gt_list:
                    gt1_dict[key] = gt1[i][key][triangluation_mask.to(gt1[i][key].device)]
                    gt1_dict_unsupervised[key] = gt1[i][key][~triangluation_mask.to(gt1[i][key].device)]
                    # gt1_dict_unsupervised[key] = gt1[i][key][triangluation_mask.to(gt1[i][key].device)]
            gt1_use_gtloss.append(gt1_dict)
            gt1_use_unsupervised.append(gt1_dict_unsupervised)

        for i in range(len(gt2)):
            gt2_dict = {}
            gt2_dict_unsupervised = {}
            for key in gt2[0].keys():
                if key in key_gt_list:
                    gt2_dict[key] = gt2[i][key][triangluation_mask.to(gt2[i][key].device)]
                    gt2_dict_unsupervised[key] = gt2[i][key][~triangluation_mask.to(gt2[i][key].device)]
                    # gt2_dict_unsupervised[key] = gt2[i][key][triangluation_mask.to(gt2[i][key].device)]
            gt2_use_gtloss.append(gt2_dict)
            gt2_use_unsupervised.append(gt2_dict_unsupervised)
        for i in range(len(render_gt)):
            render_gt_dict = {}
            render_gt_dict_unsupervised = {}
            for key in render_gt[0].keys():
                if key in key_gt_list:
                    render_gt_dict[key] = render_gt[i][key][triangluation_mask.to(render_gt[i][key].device)]
                    render_gt_dict_unsupervised[key] = render_gt[i][key][~triangluation_mask.to(render_gt[i][key].device)]
                    # render_gt_dict_unsupervised[key] = render_gt[i][key][triangluation_mask.to(render_gt[i][key].device)]
            render_gt_use_gtloss.append(render_gt_dict)
            render_gt_use_unsupervised.append(render_gt_dict_unsupervised)
        key_pred_list = ['pts3d', 'feature', 'opacity', 'scaling', 'rotation', 'conf', 'pts3d_pre']
        pred1_use_gtloss = {}
        pred2_use_gtloss = {}
        pred1_unsupervised = {}
        pred2_unsupervised = {}

        for key in pred1.keys():
            if key in key_pred_list:
                pred1_use_gtloss[key] = pred1[key][triangluation_mask]
                pred1_unsupervised[key] = pred1[key][~triangluation_mask]
                # pred1_unsupervised[key] = pred1[key][triangluation_mask]
                pred2_use_gtloss[key] = pred2[key].unflatten(0, (B, nviews))[triangluation_mask].flatten(0,1)
                # pred2_unsupervised[key] = pred2[key].unflatten(0, (B, nviews))[triangluation_mask].flatten(0,1)
                pred2_unsupervised[key] = pred2[key].unflatten(0, (B, nviews))[~triangluation_mask].flatten(0,1)

        # if len(gt1_use_gtloss[0]['pts3d']) != 0: (conf_loss1, _), (conf_loss2, _), 
        (loss_image, _), (loss_2d, _),  details_gtloss = self.gt_loss(gt1_use_gtloss, gt2_use_gtloss, pred1_use_gtloss, pred2_use_gtloss, trajectory_pred, render_gt=render_gt_use_gtloss, **kw)
        loss_image_unsupervised = loss_2d_unsupervised = 0
        details_unsupervised = {}
        
        details = {**details_pose, **details_gtloss, **details_unsupervised}
       

        final_loss = loss_image * triangluation_mask.float().mean() + loss_2d * triangluation_mask.float().mean() + pose_r_loss * 0.5 + pose_t_loss * 0.2 + pose_f_loss
        return final_loss, details



class Regr3D_gs_unsupervised_2v1(Regr3D_clean):
    """ Ensure that all 3D points are correct.
        Asymmetric loss: view1 is supposed to be the anchor.

        P1 = RT1 @ D1
        P2 = RT2 @ D2
        loss1 = (I @ pred_D1) - (RT1^-1 @ RT1 @ D1)
        loss2 = (RT21 @ pred_D2) - (RT1^-1 @ P2)
              = (RT21 @ pred_D2) - (RT1^-1 @ RT2 @ D2)
    """

    def __init__(self, criterion, alpha, multiview_triangulation = True, disable_rayloss = False, scaling_mode='interp_5e-4_0.1_3', norm_mode='avg_dis', gt_scale=False, test=False):
        super().__init__(criterion)
        self.norm_mode = norm_mode
        self.disable_rayloss = disable_rayloss
        self.multiview_triangulation = multiview_triangulation
        self.scaling_mode = {'type':scaling_mode.split('_')[0], 'min_scaling': eval(scaling_mode.split('_')[1]), 'max_scaling': eval(scaling_mode.split('_')[2]), 'shift': eval(scaling_mode.split('_')[3])}
        self.gt_scale = gt_scale
        from lpips import LPIPS
        self.lpips = LPIPS(net="vgg", model_path='checkpoints/vgg.pth') # , model_path='checkpoints/vgg16-397923af.pth'
        convert_to_buffer(self.lpips, persistent=False)
        self.lpips = self.lpips.cuda()
        self.epoch = 0
        self.test = test
        self.gt_num_image = 0
        self.render_landscape = transpose_to_landscape_render(self.render)
        self.render_landscape_wgrad = transpose_to_landscape_render(self.render_wgrad)

        self.loss2d_landscape = transpose_to_landscape(self.loss2d)
        self.alpha = alpha
        self.sm = nn.SmoothL1Loss()
        self.disp_loss =  ScaleAndShiftInvariantLoss()

    def render(self, render_params, image_size):
        latent, output_fxfycxcy, output_c2ws = render_params
        (H_org, W_org) = image_size
        H = H_org
        W = W_org
        # if 'scaling_factor' in latent.keys():
        #     scaling_factor = latent.pop('scaling_factor')
        #     if self.scaling_mode['type'] == 'precomp':
        #         x = torch.clip(latent['pre_scaling'] - self.scaling_mode['shift'], max=np.log(3))
        #         latent['scaling'] = torch.exp(x).clamp(min= self.scaling_mode['min_scaling'], max=self.scaling_mode['max_scaling']) / scaling_factor
        # else:
        #     scaling_factor = torch.tensor(1.0).to(latent['xyz'])
        new_latent = {}
        if self.scaling_mode['type'] == 'precomp': 
            scaling_factor = latent['scaling_factor']
            x = torch.clip(latent['pre_scaling'] - self.scaling_mode['shift'], max=np.log(0.3))
            new_latent['scaling'] = torch.exp(x).clamp(min= self.scaling_mode['min_scaling'], max=self.scaling_mode['max_scaling']) / scaling_factor
            skip = ['pre_scaling', 'scaling', 'scaling_factor']
        else:
            skip = ['pre_scaling', 'scaling_factor']
        for key in latent.keys():
            if key not in skip:
                new_latent[key] = latent[key]
        gs_render = GaussianRenderer(H_org, W_org, patch_size=128, gs_kwargs={'type':self.scaling_mode['type'], 'min_scaling': self.scaling_mode['min_scaling'], 'max_scaling': self.scaling_mode['max_scaling'], 'scaling_factor': scaling_factor}) # scaling_factor
        results = gs_render(new_latent, output_fxfycxcy.reshape(1,-1,4), output_c2ws.reshape(1,-1,4,4))
        images = results['image']
        image_mask = results['alpha']
        depth = results['depth']
        depth = depth.reshape(-1,1,H,W).permute(0,2,3,1)
        image_mask = image_mask.reshape(-1,1,H,W).permute(0,2,3,1)
        images = images.reshape(-1,3,H,W).permute(0,2,3,1)
        return {'images': images, 'image_mask': image_mask, 'depth':depth}
    
    def loss2d(self, input, image_size):
        pr_pts, fxfycxcy, c2ws = input
        H, W = image_size
        loss_2d = reproj2d(fxfycxcy, c2ws, pr_pts, image_size)
        return {'loss_2d': loss_2d}
    
    
    def render_wgrad(self, render_params, image_size):
        latent, output_fxfycxcy, output_c2ws = render_params
        (H_org, W_org) = image_size
        H = H_org
        W = W_org
        
        # if 'scaling_factor' in latent.keys():
        #     scaling_factor = latent.pop('scaling_factor')
        #     if self.scaling_mode['type'] == 'precomp':
        #         x = torch.clip(latent['pre_scaling'] - self.scaling_mode['shift'], max=np.log(3))
        #         latent['scaling'] = torch.exp(x).clamp(min= self.scaling_mode['min_scaling'], max=self.scaling_mode['max_scaling']) / scaling_factor
        # else:
        #     scaling_factor = torch.tensor(1.0).to(latent['xyz'])
        new_latent = {}
        if self.scaling_mode['type'] == 'precomp': 
            scaling_factor = latent['scaling_factor']
            x = torch.clip(latent['pre_scaling'] - self.scaling_mode['shift'], max=np.log(0.3))
            new_latent['scaling'] = torch.exp(x).clamp(min= self.scaling_mode['min_scaling'], max=self.scaling_mode['max_scaling']) / scaling_factor
            skip = ['pre_scaling', 'scaling', 'scaling_factor']
        else:
            skip = ['pre_scaling', 'scaling_factor']
        for key in latent.keys():
            if key not in skip:
                new_latent[key] = latent[key]
        gs_render = GaussianRenderer(H_org, W_org, patch_size=128, gs_kwargs={'type':self.scaling_mode['type'], 'min_scaling': self.scaling_mode['min_scaling'], 'max_scaling': self.scaling_mode['max_scaling'], 'scaling_factor': scaling_factor}, render_type='render_wodb') # scaling_factor
        results = gs_render(new_latent, output_fxfycxcy.reshape(1,-1,4), output_c2ws.reshape(1,-1,4,4))
        images = results['image']
        image_mask = results['alpha']
        depth = results['depth']
        depth = depth.reshape(-1,1,H,W).permute(0,2,3,1)
        image_mask = image_mask.reshape(-1,1,H,W).permute(0,2,3,1)
        images = images.reshape(-1,3,H,W).permute(0,2,3,1)
        return {'images': images, 'image_mask': image_mask, 'depth':depth}
    
    def get_all_pts3d(self, gt1, gt2, pred1, pred2, trajectory_pred, dist_clip=None, render_gt=None):
        # everything is normalized w.r.t. camera of view1        
        pr_pts1 = pred1['pts3d']
        pr_pts2 = pred2['pts3d']
        # GT pts3d
        gt1_pts3d = torch.stack([gt1_per['pts3d'] for gt1_per in gt1], dim=1)
        B, _, H, W, _ = gt1_pts3d.shape
        # caculate the number of views and read the gt2
        num_views = pr_pts2.shape[0] // B
        gt2_pts3d = torch.stack([gt2_per['pts3d'] for gt2_per in gt2[:num_views]], dim=1)
        # camera trajectory
        camera_pose = torch.stack([gt1_per['camera_pose'] for gt1_per in gt1], dim=1)
        in_camera1 = inv(camera_pose)
        trajectory = torch.stack([view['camera_pose'] for view in gt2], dim=1)
        trajectory = torch.cat((camera_pose, trajectory), dim=1)
        # import ipdb; ipdb.set_trace()
        # trajectory_1 = closed_form_inverse(camera_pose.repeat(1, trajectory.shape[1],1,1).reshape(-1,4,4)).bmm(trajectory.reshape(-1,4,4))
        trajectory = torch.einsum('bnjk,bnkl->bnjl', in_camera1.repeat(1, trajectory.shape[1],1,1), trajectory)
        # trajectory_1 - trajectory.reshape(-1,4,4)
        # transform the gt points to the coordinate of the first view
        Rs = []
        Ts = []


        # valid mask
        valid1 = torch.stack([gt1_per['valid_mask'] for gt1_per in gt1], dim=1).view(B,-1,H,W).clone()
        valid2 = torch.stack([gt2_per['valid_mask'] for gt2_per in gt2[:num_views]], dim=1).view(B,-1,H,W).clone()

        gt_pts1 = gt1_pts3d
        gt_pts2 = gt2_pts3d
        # reshape point map
        gt_pts1 = gt_pts1.reshape(B,H,W,3)
        gt_pts2 = gt_pts2.reshape(B,H*num_views,W,3)
        pr_pts1 = pr_pts1.reshape(B,H,W,3)
        pr_pts2 = pr_pts2.reshape(B,H*num_views,W,3)
        valid1 = valid1.view(B,H,W)
        valid2 = valid2.view(B,H*num_views,W)
        
        if valid1.sum() == 0 and valid2.sum() == 0:
            valid1 = torch.ones_like(valid1).to(valid1) > 0
            valid2 = torch.ones_like(valid2).to(valid2) > 0
            # import ipdb; ipdb.set_trace()

        # generate the gt camera trajectory
        trajectory_t_gt = trajectory[..., :3, 3]
        size =  (trajectory_t_gt[:,0:1] - trajectory_t_gt[:,1:2]).norm(dim=-1, keepdim=True) + 1e-8 
        trajectory_t_gt = trajectory_t_gt / size
        quaternion_R = matrix_to_quaternion(trajectory[...,:3,:3])
        trajectory_gt = torch.cat([trajectory_t_gt, quaternion_R], dim=-1)[None]
        R = trajectory[...,:3,:3]
        fxfycxcy1 = torch.stack([view['fxfycxcy'] for view in gt1], dim=1).float()
        fxfycxcy2 = torch.stack([view['fxfycxcy'] for view in gt2], dim=1).float()
        fxfycxcy = torch.cat((fxfycxcy1, fxfycxcy2), dim=1).to(fxfycxcy1)
        focal_length_gt = fxfycxcy[...,:2]

        # generate the predicted camera trajectory
        with torch.no_grad():
            trajectory_R_prior = trajectory_pred[:4][-1]['R'].reshape(B, -1, 3, 3)
            trajectory_R_post = trajectory_pred[-1]['R'].reshape(B, -1, 3, 3)

        trajectory_t_pred = torch.stack([view["T"].reshape(B, -1, 3) for view in trajectory_pred], dim=1)
        trajectory_r_pred = torch.stack([view["quaternion_R"].reshape(B, -1, 4) for view in trajectory_pred], dim=1)
        focal_length_pred = torch.stack([view["focal_length"].reshape(B, -1, 2) for view in trajectory_pred], dim=1)
        size_pred = (trajectory_t_pred[:,:, 0:1] - trajectory_t_pred[:,:, 1:2]).norm(dim=-1, keepdim=True) + 1e-8 
        trajectory_t_pred = trajectory_t_pred / size_pred
        trajectory_pred = torch.cat([trajectory_t_pred, trajectory_r_pred], dim=-1)
        trajectory_pred = trajectory_pred.permute(1,0,2,3)
        focal_length_pred = focal_length_pred.permute(1,0,2,3)
        B, H, W, _ = gt_pts1.shape
        nviews = len(gt2)
        sh_dim = pred1['feature'].shape[-1]
        feature1 = pred1['feature'].reshape(B,H,W,sh_dim)
        feature2 = pred2['feature'].reshape(B,-1,W,sh_dim)
        feature = torch.cat((feature1, feature2), dim=1).float()
        image_2d = torch.cat((feature1.reshape(B,-1, H,W,sh_dim)[..., :3], feature2.reshape(B,-1, H,W,sh_dim)[..., :3]), dim=1)
        if sh_dim > 3:
            image_2d = sh_to_rgb(image_2d).clamp(0,1)
        else:
            image_2d = torch.sigmoid(image_2d)
        opacity1 = pred1['opacity'].reshape(B,H,W,1)
        opacity2 = pred2['opacity'].reshape(B,-1,W,1)
        opacity = torch.cat((opacity1, opacity2), dim=1).float()
        scaling1 = pred1['scaling'].reshape(B,H,W,3)
        scaling2 = pred2['scaling'].reshape(B,-1,W,3)
        scaling = torch.cat((scaling1, scaling2), dim=1).float()
        rotation1 = pred1['rotation'].reshape(B,H,W,4)
        rotation2 = pred2['rotation'].reshape(B,-1,W,4)
        rotation = torch.cat((rotation1, rotation2), dim=1).float()
        output_fxfycxcy = torch.stack([gt['fxfycxcy'] for gt in render_gt], dim=1)
        render_mask = torch.stack([gt['render_mask'] for gt in render_gt], dim=1)
        shape = torch.stack([gt['true_shape'] for gt in render_gt], dim=1)
        xyz = torch.cat((pr_pts1, pr_pts2), dim=1)
        pr_pts = torch.cat((pr_pts1.reshape(B,-1,H,W,3), pr_pts2.reshape(B,-1,H,W,3)), dim=1)
        xyz_gt = torch.cat((gt_pts1.reshape(B,-1,H,W,3), gt_pts2.reshape(B,-1,H,W,3)), dim=1).detach()
        # TODO rewrite transform cameras

        images_gt = torch.stack([gt['img_org'] for gt in render_gt], dim=1)
        images_gt = images_gt / 2 + 0.5
        norm_factor_gt = 1
        output_c2ws = torch.stack([gt['camera_pose'] for gt in render_gt], dim=1)
        camera_pose = torch.stack([gt1_per['camera_pose'] for gt1_per in gt1], dim=1)
        in_camera1 = inv(camera_pose)
        output_c2ws = torch.einsum('bnjk,bnkl->bnjl', in_camera1.repeat(1,output_c2ws.shape[1],1,1), output_c2ws)
        # output_c2ws[..., :3, 3:] = output_c2ws[..., :3, 3:]
        depth_gt = torch.stack([gt['depth_anything'] for gt in render_gt], dim=1)
        
        if self.test:
            output_c2ws_input = torch.stack([gt['camera_pose'] for gt in gt1 + gt2], dim=1)
            output_c2ws_input = torch.einsum('bnjk,bnkl->bnjl', in_camera1.repeat(1,output_c2ws.shape[1],1,1), output_c2ws_input)
            output_fxfycxcy_input = torch.stack([gt['fxfycxcy'] for gt in gt1 + gt2], dim=1)
            shape_input = torch.stack([gt['true_shape'] for gt in gt1 + gt2], dim=1)

            with torch.set_grad_enabled(True):
                images_gt_input = torch.stack([gt['img_org'] for gt in gt1 + gt2], dim=1)
                images_gt_input = images_gt_input / 2 + 0.5
                B = output_c2ws.shape[0]
                v = output_c2ws.shape[1]
                cam_rot_delta = nn.Parameter(torch.zeros([B, v, 6], requires_grad=True, device=output_c2ws.device))
                cam_trans_delta = nn.Parameter(torch.zeros([B, v, 3], requires_grad=True, device=output_c2ws.device))
                opt_params = []
                self.register_buffer("identity", torch.tensor([1.0, 0.0, 0.0, 0.0, 1.0, 0.0]).to(output_c2ws))

                opt_params.append(
                    {
                        "params": [cam_rot_delta],
                        "lr": 0.005,
                    }
                )
                opt_params.append(
                    {
                        "params": [cam_trans_delta],
                        "lr": 0.005,
                    }
                )
                pose_optimizer = torch.optim.Adam(opt_params)
                i = 0
                latent = {'xyz': xyz.reshape(B, -1, 3)[i:i+1], 'feature': feature.reshape(B, -1, sh_dim)[i:i+1], 'opacity': opacity.reshape(B, -1, 1)[i:i+1], 'pre_scaling': scaling.reshape(B, -1, 3)[i:i+1].clone(), 'rotation': rotation.reshape(B, -1, 4)[i:i+1], 'scaling_factor': 1}
                extrinsics = output_c2ws_input
                for i in range(100):
                    pose_optimizer.zero_grad()
                    dx, drot = cam_trans_delta, cam_rot_delta
                    rot = rotation_6d_to_matrix(
                        drot + self.identity.expand(B, v, -1)
                    )  # (..., 3, 3)
                    transform = torch.eye(4, device=extrinsics.device).repeat((B, v, 1, 1))
                    transform[..., :3, :3] = rot
                    transform[..., :3, 3] = dx
                    new_extrinsics = torch.matmul(extrinsics, transform)
                    ret = self.render_landscape_wgrad([latent, output_fxfycxcy_input[:1].reshape(-1,4), new_extrinsics.reshape(-1,4,4)], shape_input[:1].reshape(-1,2))
                    color = ret['images']
                    color = color.permute(0,3,1,2)
                    # if i % 10 == 0:
                    #     import os
                    #     import matplotlib.pyplot as plt
                    #     os.makedirs('data/new', exist_ok=True)
                    #     plt.imshow(color[0].detach().cpu().numpy().clip(0,1).transpose(1,2,0))
                    #     plt.savefig(f'./data/new/test{i}_0.png')
                    #     plt.imshow(images_gt_input[0,0].detach().cpu().numpy().clip(0,1).transpose(1,2,0))
                    #     plt.savefig(f'./data/new/test_gt{i}_0.png')
                    #     plt.imshow(color[1].detach().cpu().numpy().clip(0,1).transpose(1,2,0))
                    #     plt.savefig(f'./data/new/test{i}_1.png')
                    #     plt.imshow(images_gt_input[0,1].detach().cpu().numpy().clip(0,1).transpose(1,2,0))
                    #     plt.savefig(f'./data/new/test_gt{i}_1.png')
                    loss_vgg1 = self.lpips.forward(
                        color,
                        images_gt_input[0],
                        normalize=True,
                    )
                    loss_vgg2 = self.lpips.forward(
                        color[1:],
                        images_gt_input[0,1:],
                        normalize=True,
                    )
                    total_loss = loss_vgg1.mean() * 0.1 + loss_vgg2.mean() * 2 + ((color - images_gt_input[0])**2).mean()
                    # print(total_loss)
                    #((color - images_gt[0])**2).mean()
                    # for loss_fn in self.losses:
                    # delta = prediction.color - batch["target"]["image"]
                    # loss = loss_fn.forward(output, batch, gaussians, self.global_step)
                    # total_loss = total_loss + loss
                    total_loss.backward()
                    pose_optimizer.step()
            with torch.no_grad():
                # output_c2ws[..., :3, 3:] = output_c2ws[..., :3, 3:]
                # trajectory_pred_t = output_c2ws[0,:, :3, 3]
                # trajectory_pred_r = matrix_to_quaternion(output_c2ws[0,:, :3,:3])
                # trajectory_pred = torch.cat([trajectory_pred_t, trajectory_pred_r], dim=-1)
                render_pose = []
                for i in range(120):
                    pose = interpolate_pose(new_extrinsics[0,0], new_extrinsics[0,1], i/120)
                    pose = torch.tensor(pose).to(output_c2ws_input)
                    render_pose.append(pose)
                # render_pose = interpolate_poses(list(range(len(trajectory_pred))), trajectory_pred.detach().cpu().numpy(), list(np.linspace(0,2,280)), 0)
                render_pose = torch.stack(render_pose)
                i = 0
                clip = 8
                # pr_pts1_pre, pr_pts2_pre, norm_factor_pr = normalize_pointcloud(pr_pts1_pre, pr_pts2_pre, self.norm_mode, valid1, valid2, ret_factor=True)
                latent = {'xyz': xyz.reshape(B, -1, 3)[i:i+1], 'feature': feature.reshape(B, -1, sh_dim)[i:i+1], 'opacity': opacity.reshape(B, -1, 1)[i:i+1], 'pre_scaling': scaling.reshape(B, -1, 3)[i:i+1].clone(), 'rotation': rotation.reshape(B, -1, 4)[i:i+1], 'scaling_factor': 1}
                # ret = self.render_landscape([latent, output_fxfycxcy[i:i+1][:1].reshape(-1,4).repeat(render_pose.reshape(-1,4,4).shape[0],1), render_pose.reshape(-1,4,4)], shape[i:i+1].reshape(-1,2))
                ret = self.render_landscape([latent, output_fxfycxcy[i:i+1][:1].reshape(-1,4).repeat(render_pose.reshape(-1,4,4).shape[0],1), render_pose.reshape(-1,4,4)], shape[i:i+1].reshape(-1,2))
                video = ret['images'].detach().cpu().numpy()
                import imageio
                import os
                i = 0
                output_path = f'test/output_video_{i}.mp4'
                os.makedirs('test', exist_ok=True)
                while os.path.exists(output_path):
                    i = i  + 1
                    output_path = f'test/output_video_{i}.mp4'
                print(output_path)
                writer = imageio.get_writer(output_path, fps=30)  # You can specify the filename and frames per second
                for frame in video:  
                    # Add the frame to the video
                    frame = (frame * 255).clip(0,255).astype(np.uint8)
                    writer.append_data(frame)
                # Close the writer to finalize the video
                writer.close()

        images_list = []
        image_mask_list = [] 
        depths_rendered_list = []
        for i in range(B):
            # try:
            latent = {'xyz': xyz.reshape(B, -1, 3)[i:i+1], 'feature': feature.reshape(B, -1, sh_dim)[i:i+1], 'opacity': opacity.reshape(B, -1, 1)[i:i+1], 'pre_scaling': scaling.reshape(B, -1, 3)[i:i+1].clone(), 'rotation': rotation.reshape(B, -1, 4)[i:i+1], 'scaling_factor': 1}
            # if 'scaling_factor' not in latent.keys():
            ret = self.render_landscape([latent, output_fxfycxcy[i:i+1].reshape(-1,4), output_c2ws[i:i+1].reshape(-1,4,4)], shape[i:i+1].reshape(-1,2))
            # except:
            #     import ipdb; ipdb.set_trace()
            images = ret['images']
            image_mask = ret['image_mask']
            images_list.append(images)
            image_mask_list.append(image_mask)
            depths_rendered_list.append(ret['depth'])
        images = torch.stack(images_list, dim=0).permute(0,1,4,2,3)
        image_masks = torch.stack(image_mask_list, dim=0).permute(0,1,4,2,3)
        depths_rendered_list = torch.stack(depths_rendered_list, dim=0).permute(0,1,4,2,3)
        
        image_mask_gt_geo = image_masks.clone()
        images_gt_geo = images.clone()
        shape_input = torch.stack([gt['true_shape'] for gt in gt1+gt2], dim=1)
        input_fxfycxcy = output_fxfycxcy[:,:4]
        input_c2ws = output_c2ws[:,:4]
        shape_input = shape_input[:,:4]
        pr_pts = pr_pts[:,:4]
        # loss_2d = self.loss2d_landscape([pr_pts.reshape(-1, H, W, 3), input_fxfycxcy.reshape(-1,4), input_c2ws.reshape(-1,4,4)], shape_input.reshape(-1,2))
        # import ipdb; ipdb.set_trace()
        # masks = torch.cat((valid1.reshape(B, -1, H, W), valid2.reshape(B, -1, H, W)), 1)
        metric_conf1 = pred1['conf']
        metric_conf2 = pred2['conf']
        pr_pts2_transform = gt_pts2.reshape(*gt_pts2.shape)
        return gt_pts1, gt_pts2, pr_pts1, pr_pts2, valid1, valid2, metric_conf1, metric_conf2, trajectory_gt, trajectory_pred, images, images_gt, image_masks, images_gt_geo, image_mask_gt_geo, render_mask, R, trajectory_R_prior, trajectory_R_post, image_2d, depth_gt, depths_rendered_list, {}


    def get_conf_log(self, x):
        return x, torch.log(x)

    
    def gt_loss(self, gt1, gt2, pred1, pred2, trajectory_pred, render_gt=None, **kw):
        gt_pts1, gt_pts2, pr_pts1, pr_pts2,  mask1, mask2, metric_conf1, metric_conf2, trajectory, trajectory_pred, images, images_gt, image_mask, images_gt_geo, image_mask_gt_geo, render_mask,  R, trajectory_R_prior, trajectory_R_post, image_2d, depth_gt, depths, monitoring = self.get_all_pts3d(gt1, gt2, pred1, pred2, trajectory_pred, render_gt=render_gt, **kw)
        # loss on img1 side
        # loss on gt2 side
        render_mask = render_mask[:,:,None].repeat(1,1,3,1,1)
        render_mask = render_mask.float()
        image_2d = image_2d.permute(0,1,4,2,3)
        images_org = images.clone()
        
        loss_image_pred_geo_l2 = ((images.reshape(*render_mask.shape) - images_gt.reshape(*render_mask.shape)) ** 2)
        loss_image_pred_geo_l2 = loss_image_pred_geo_l2.mean() if loss_image_pred_geo_l2.numel() > 0 else 0
        # loss_image_pred_geo_novel = ((images.reshape(*render_mask.shape)[:,4:][render_mask[:,4:] > 0.] - images_gt.reshape(*render_mask.shape)[:,4:][render_mask[:,4:] > 0.]) ** 2)
        # loss_image_pred_geo_novel = loss_image_pred_geo_novel.mean() if loss_image_pred_geo_novel.numel() > 0 else 0
        
        # loss_image_pred_self = ((image_2d[:,:4] - images_gt[:,:4]) ** 2).mean()
        loss_image_pred_geo = loss_image_pred_geo_l2 # + loss_image_pred_geo_novel * 4 
        
        # images_masked = images
        # images_masked[:,:4] = (images_masked[:,:4] * 0.5 + image_2d[:,:4] * 0.5)
        # images_masked = images_masked.reshape(-1,*images_gt.shape[-3:])
        # images_masked, images_gt_1 = self.random_crop(images_masked, images_gt)
        # images_gt = images_gt.reshape(-1,*images_gt.shape[-3:])
        # loss_vgg = self.perceptual_loss(images_masked, images_gt_1) * 0.2
        loss_vgg = self.lpips.forward(
                    images.flatten(0,1),
                    images_gt.flatten(0,1),
                    normalize=True,
                )  * 0.05
        loss_vgg = loss_vgg.mean()
        loss_image = loss_vgg + loss_image_pred_geo
        loss_image = loss_image * 4
        # image_2d
        mse = torch.mean(((images_org.reshape(*render_mask.shape)[render_mask > 0.] - images_gt.reshape(*render_mask.shape)[render_mask > 0.]) * 255) ** 2)
        PSNR = 20 * torch.log10(255/torch.sqrt(mse))

        msk1 = mask1
        msk2 = mask2
        loss_2d = 0
        self_name = type(self).__name__
        details = {'PSNR': float(PSNR), 'loss_2d': float(loss_2d), 'loss_pred_geo': float(loss_image_pred_geo_l2), 'loss_vgg': float(loss_vgg), self_name+'_image': float(loss_image), 'images': images_org,  'images_gt':images_gt * render_mask.float(), 'images_gt_geo': images_gt_geo, 'image_2d': image_2d}
        return (loss_image, None), (loss_2d, None), details

    def compute_loss(self, gt1, gt2, pred1, pred2, trajectory_pred, render_gt=None, **kw):
        (pose_r_loss, _), (pose_t_loss, _), (pose_f_loss, _), details_pose = self.compute_pose_loss(gt1, gt2, pred1, pred2, trajectory_pred, render_gt=render_gt)
        valids = torch.stack([gt1_per['valid_mask'] for gt1_per in gt1+gt2], dim=1)
        triangluation_mask = valids.flatten(1,3).sum(1) > -1
        gt1_use_gtloss = []
        gt2_use_gtloss = []
        render_gt_use_gtloss = []
        gt1_use_unsupervised = []
        gt2_use_unsupervised = []
        render_gt_use_unsupervised = []

        B = valids.shape[0]
        nviews = len(gt2)
        key_gt_list = ['camera_pose', 'true_shape', 'fxfycxcy', 'render_mask', 'img_org', 'valid_mask', 'pts3d', 'camera_intrinsics', 'depth_anything']
        for i in range(len(gt1)):
            gt1_dict = {}
            gt1_dict_unsupervised = {}
            for key in gt1[0].keys():
                if key in key_gt_list:
                    gt1_dict[key] = gt1[i][key][triangluation_mask.to(gt1[i][key].device)]
                    gt1_dict_unsupervised[key] = gt1[i][key][~triangluation_mask.to(gt1[i][key].device)]
                    # gt1_dict_unsupervised[key] = gt1[i][key][triangluation_mask.to(gt1[i][key].device)]
            gt1_use_gtloss.append(gt1_dict)
            gt1_use_unsupervised.append(gt1_dict_unsupervised)

        for i in range(len(gt2)):
            gt2_dict = {}
            gt2_dict_unsupervised = {}
            for key in gt2[0].keys():
                if key in key_gt_list:
                    gt2_dict[key] = gt2[i][key][triangluation_mask.to(gt2[i][key].device)]
                    gt2_dict_unsupervised[key] = gt2[i][key][~triangluation_mask.to(gt2[i][key].device)]
                    # gt2_dict_unsupervised[key] = gt2[i][key][triangluation_mask.to(gt2[i][key].device)]
            gt2_use_gtloss.append(gt2_dict)
            gt2_use_unsupervised.append(gt2_dict_unsupervised)
        for i in range(len(render_gt)):
            render_gt_dict = {}
            render_gt_dict_unsupervised = {}
            for key in render_gt[0].keys():
                if key in key_gt_list:
                    render_gt_dict[key] = render_gt[i][key][triangluation_mask.to(render_gt[i][key].device)]
                    render_gt_dict_unsupervised[key] = render_gt[i][key][~triangluation_mask.to(render_gt[i][key].device)]
                    # render_gt_dict_unsupervised[key] = render_gt[i][key][triangluation_mask.to(render_gt[i][key].device)]
            render_gt_use_gtloss.append(render_gt_dict)
            render_gt_use_unsupervised.append(render_gt_dict_unsupervised)
        key_pred_list = ['pts3d', 'feature', 'opacity', 'scaling', 'rotation', 'conf', 'pts3d_pre']
        pred1_use_gtloss = {}
        pred2_use_gtloss = {}
        pred1_unsupervised = {}
        pred2_unsupervised = {}

        for key in pred1.keys():
            if key in key_pred_list:
                pred1_use_gtloss[key] = pred1[key][triangluation_mask]
                pred1_unsupervised[key] = pred1[key][~triangluation_mask]
                # pred1_unsupervised[key] = pred1[key][triangluation_mask]
                pred2_use_gtloss[key] = pred2[key].unflatten(0, (B, nviews))[triangluation_mask].flatten(0,1)
                # pred2_unsupervised[key] = pred2[key].unflatten(0, (B, nviews))[triangluation_mask].flatten(0,1)
                pred2_unsupervised[key] = pred2[key].unflatten(0, (B, nviews))[~triangluation_mask].flatten(0,1)

        # if len(gt1_use_gtloss[0]['pts3d']) != 0: (conf_loss1, _), (conf_loss2, _), 
        (loss_image, _), (loss_2d, _),  details_gtloss = self.gt_loss(gt1_use_gtloss, gt2_use_gtloss, pred1_use_gtloss, pred2_use_gtloss, trajectory_pred, render_gt=render_gt_use_gtloss, **kw)
        loss_image_unsupervised = loss_2d_unsupervised = 0
        details_unsupervised = {}
        
        details = {**details_pose, **details_gtloss, **details_unsupervised}
       

        final_loss = loss_image * triangluation_mask.float().mean() + loss_2d * triangluation_mask.float().mean() + pose_r_loss * 0.5 + pose_t_loss * 0.2 + pose_f_loss
        return final_loss, details

