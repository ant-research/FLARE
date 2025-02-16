import os
import math
import functools
import torch
import torch.nn as nn
import numpy as np
from typing import Union, NamedTuple
# from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer

from plyfile import PlyData, PlyElement
from gsplat.rendering import rasterization
import torch
from einops import rearrange
from jaxtyping import Float
from torch import Tensor



def strip_lowerdiag(L):
    uncertainty = torch.zeros((L.shape[0], 6), dtype=torch.float, device="cuda")

    uncertainty[:, 0] = L[:, 0, 0]
    uncertainty[:, 1] = L[:, 0, 1]
    uncertainty[:, 2] = L[:, 0, 2]
    uncertainty[:, 3] = L[:, 1, 1]
    uncertainty[:, 4] = L[:, 1, 2]
    uncertainty[:, 5] = L[:, 2, 2]
    return uncertainty


def strip_symmetric(sym):
    return strip_lowerdiag(sym)


def build_rotation(r):
    norm = torch.sqrt(r[:, 0]*r[:, 0] + r[:, 1]*r[:, 1] + r[:, 2]*r[:, 2] + r[:, 3]*r[:, 3])

    q = r / norm[:, None]

    R = torch.zeros((q.size(0), 3, 3), device='cuda')

    r = q[:, 0]
    x = q[:, 1]
    y = q[:, 2]
    z = q[:, 3]
    R[:, 0, 0] = 1 - 2 * (y*y + z*z)
    R[:, 0, 1] = 2 * (x*y - r*z)
    R[:, 0, 2] = 2 * (x*z + r*y)
    R[:, 1, 0] = 2 * (x*y + r*z)
    R[:, 1, 1] = 1 - 2 * (x*x + z*z)
    R[:, 1, 2] = 2 * (y*z - r*x)
    R[:, 2, 0] = 2 * (x*z - r*y)
    R[:, 2, 1] = 2 * (y*z + r*x)
    R[:, 2, 2] = 1 - 2 * (x*x + y*y)
    return R


def build_scaling_rotation(s, r):
    s = s[0]
    r = r[0]
    L = torch.zeros((s.shape[0], 3, 3), dtype=torch.float, device="cuda")
    R = build_rotation(r)
    L[:, 0, 0] = s[:, 0]
    L[:, 1, 1] = s[:, 1]
    L[:, 2, 2] = s[:, 2]
    L = R @ L
    return L


def focal2fov(focal, pixels):
    return 2*math.atan(pixels/(2*focal))


def get_projection_matrix(znear, zfar, fovX, fovY, sx, sy, device):
    tanHalfFovY = math.tan((fovY / 2))
    tanHalfFovX = math.tan((fovX / 2))

    top = tanHalfFovY * znear
    bottom = -top
    right = tanHalfFovX * znear
    left = -right

    P = torch.zeros(4, 4, device=device)

    z_sign = 1.0

    P[0, 0] = 2.0 * znear / (right - left)
    P[1, 1] = 2.0 * znear / (top - bottom)
    P[0, 2] = (right + left) / (right - left) + sx
    P[1, 2] = (top + bottom) / (top - bottom) + sy
    P[3, 2] = z_sign
    P[2, 2] = z_sign * zfar / (zfar - znear)
    P[2, 3] = -(zfar * znear) / (zfar - znear)
    return P


class Camera(nn.Module):
    def __init__(self, K, RT, image_h, image_w):
        super(Camera, self).__init__()

        self.RT = RT 
        self.K = K 
        fx = K[0, 0]
        fy = K[1, 1]
        sx = 2*K[0, 2] - 1
        sy = 2*K[1, 2] - 1
         
        
        self.fovx = focal2fov(fx, 1)
        self.fovy = focal2fov(fy, 1)

        self.zfar = 100.0
        self.znear = 0.01

        self.view_trans_matrix = torch.linalg.inv(self.RT).transpose(0, 1)
        self.projection_trans_matrix = get_projection_matrix(self.znear, self.zfar, self.fovx, self.fovy, sx, sy, RT.device).transpose(0, 1)
        self.full_proj_transform = (self.view_trans_matrix.unsqueeze(0).bmm(self.projection_trans_matrix.unsqueeze(0))).squeeze(0)
        self.camera_center = self.RT[:3, 3]

        self.image_h = image_h
        self.image_w = image_w


def render_image(pc, K, RT, height, width, bg_color=(0.0, 0.0, 0.0), scaling_modifier=1.0,debug=False):
    screenspace_points = (torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0)
    bg_color = torch.tensor(bg_color, dtype=torch.float32, device=K.device)
    try:
        screenspace_points.retain_grad()
    except:
        pass
    # camera = Camera(K=K, RT=RT, image_h=height, image_w=width)

    # raster_settings = GaussianRasterizationSettings(
    #     image_height=height,
    #     image_width=width,
    #     tanfovx=math.tan(camera.fovx * 0.5),
    #     tanfovy=math.tan(camera.fovy * 0.5),
    #     bg=bg_color,
    #     scale_modifier=scaling_modifier,
    #     viewmatrix=camera.view_trans_matrix,
    #     projmatrix=camera.full_proj_transform,
    #     sh_degree=pc.sh_degree,
    #     campos=camera.camera_center,
    #     prefiltered=False,
    #     debug=debug
    # )

    # rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    means3D = pc.get_xyz
    means2D = screenspace_points
    opacity = pc.get_opacity
    scales = pc.get_scaling
    rotations = pc.get_rotation
    colors_precomp = pc.get_features
    K[:1] = K[:1] * width
    K[1:2] = K[1:2] * height
    if colors_precomp.shape[1] == 3:
        sh_degree = None
    else:
        sh_degree = int(math.sqrt(colors_precomp.shape[1])) - 1
    render_colors, render_alphas, meta = rasterization(means = means3D, quats= rotations, scales = scales, opacities = opacity.squeeze(), colors = colors_precomp, sh_degree=sh_degree, viewmats = RT[None].inverse(), Ks=K[None][:,:3,:3], width=width, height=height, near_plane=0.00001, render_mode="RGB+D", radius_clip=0.1)#, rasterize_mode="antialiased")
    # rendered_image, _, rendered_depth, rendered_alpha = rasterizer(
    #     shs=None,
    #     cov3D_precomp=None,
    #     means3D=means3D,
    #     means2D=means2D,
    #     opacities=opacity,
    #     colors_precomp=colors_precomp,
    #     scales=scales,
    #     rotations=rotations)
    render_depths = render_colors.permute(0, 3, 1, 2).squeeze()[3:4, :, :]
    render_colors = render_colors.permute(0, 3, 1, 2).squeeze()[:3, :, :]
    render_alphas = render_alphas.permute(0, 3, 1, 2)[0]
    return {
        "image": render_colors,
        "alpha": render_alphas,
        "depth": render_depths,
    }

C0 = 0.28209479177387814
def SH2RGB(sh):
    return sh * C0 + 0.5




# https://github.com/facebookresearch/pytorch3d/blob/main/pytorch3d/transforms/rotation_conversions.py
def quaternion_to_matrix(
    quaternions: Float[Tensor, "*batch 4"],
    eps: float = 1e-8,
) -> Float[Tensor, "*batch 3 3"]:
    # Order changed to match scipy format!
    i, j, k, r = torch.unbind(quaternions, dim=-1)
    two_s = 2 / ((quaternions * quaternions).sum(dim=-1) + eps)

    o = torch.stack(
        (
            1 - two_s * (j * j + k * k),
            two_s * (i * j - k * r),
            two_s * (i * k + j * r),
            two_s * (i * j + k * r),
            1 - two_s * (i * i + k * k),
            two_s * (j * k - i * r),
            two_s * (i * k - j * r),
            two_s * (j * k + i * r),
            1 - two_s * (i * i + j * j),
        ),
        -1,
    )
    return rearrange(o, "... (i j) -> ... i j", i=3, j=3)


def build_covariance(
    scale: Float[Tensor, "*#batch 3"],
    rotation_xyzw: Float[Tensor, "*#batch 4"],
) -> Float[Tensor, "*batch 3 3"]:
    scale = scale.diag_embed()
    rotation = quaternion_to_matrix(rotation_xyzw)
    return (
        rotation
        @ scale
        @ rearrange(scale, "... i j -> ... j i")
        @ rearrange(rotation, "... i j -> ... j i")
    )


class GRMGaussianModel:

    def __init__(self,
                 sh_degree,
                 xyz=None, 
                 feature=None,
                 opacity=None,
                 scaling=None,
                 rotation=None,
                 feature_rest=None,
                 scaling_kwargs=dict(type='exp'),
                 **kwargs):
        self.sh_degree = sh_degree

        # self.min_scale_activation = min_scale_activation
        # self.max_scale_activation = max_scale_activation

        self.feature_activation = torch.sigmoid
        self.rotation_activation = torch.nn.functional.normalize
        self.covariance_activation = build_covariance
        self.opacity_activation = torch.sigmoid
        self.scaling_activation = self.build_scaling_activation(scaling_kwargs)
        self._xyz = torch.empty(0) if xyz is None else xyz
        self._features_dc = torch.empty(0) if feature is None else feature
        self._features_rest = torch.empty(0) if feature_rest is None else feature_rest
        self._opacity = torch.empty(0) if opacity is None else opacity
        self._scaling = torch.empty(0) if scaling is None else scaling
        self._rotation = torch.empty(0) if rotation is None else rotation

    @staticmethod
    def build_scaling_activation(scaling_kwargs):
        stype = scaling_kwargs.get('type', None)
        if stype == 'exp':
            min_scaling = scaling_kwargs['min_scaling']
            max_scaling = scaling_kwargs['max_scaling']
            scaling_factor = scaling_kwargs['scaling_factor']
            def exp(x, min_scaling, max_scaling):
                x = torch.clip(x - 3, max=np.log(0.3))
                return torch.exp(x).clamp(min=min_scaling, max=max_scaling) / scaling_factor
            return functools.partial(exp, min_scaling=min_scaling, max_scaling=max_scaling)
        elif stype == 'interp':
            min_scaling = scaling_kwargs['min_scaling']
            max_scaling = scaling_kwargs['max_scaling']
            scaling_factor = scaling_kwargs['scaling_factor']
            def interp(x, min_scaling, max_scaling):
                ratio = torch.sigmoid(x - 2)
                return (((1-ratio)*min_scaling + ratio * max_scaling) / scaling_factor).clamp(max=max_scaling)
            return functools.partial(interp, min_scaling=min_scaling, max_scaling=max_scaling)
        elif stype == 'precomp':
            min_scaling = scaling_kwargs['min_scaling']
            max_scaling = scaling_kwargs['max_scaling']
            return lambda x: x.clamp(min=min_scaling, max=max_scaling)
        else:
            raise NotImplementedError

    @staticmethod
    def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
        L = build_scaling_rotation(scaling_modifier * scaling, rotation)
        actual_covariance = L @ L.transpose(1, 2)
        # symm = strip_symmetric(actual_covariance)
        return actual_covariance

    def set_xyz(self, xyz):
        self._xyz = xyz

    def set_scaling(self, scaling):
        self._scaling = scaling

    def set_rotation(self, rotation):
        self._rotation = rotation

    def set_opacity(self, opacity):
        self._opacity = opacity

    def set_features_dc(self, features_dc):
        self._features_dc = features_dc

    def set_device(self, device):
        self._xyz = self._xyz.to(device)
        self._features_dc = self._features_dc.to(device)
        self._scaling = self._scaling.to(device)
        self._rotation = self._rotation.to(device)
        self._opacity = self._opacity.to(device)

    @property
    def get_params(self):
        results = {'scaling': self._scaling,
                   'rotation': self._rotation,
                   'xyz': self._xyz,
                   'feature': self._features_dc,
                   'opacity': self._opacity}
        return results

    @property
    def get_scaling(self):
        return self.scaling_activation(self._scaling)
    

    @property
    def get_rotation(self):
        return self.rotation_activation(self._rotation)

    @property
    def get_xyz(self):
        return self._xyz

    @property
    def get_features(self):
        if self._features_dc.shape[-1] == 3:
            return self.feature_activation(self._features_dc)
        else:
            dc = self._features_dc[...,:3].unsqueeze(-2)
            rest = self._features_dc[...,3:].unflatten(-1, (-1, 3))
            return torch.cat([dc, rest], dim=-2)

    @property
    def get_opacity(self):
        return self.opacity_activation(self._opacity)

    def get_covariance(self, scaling_modifier=1):
        # build_covariance(scales, rotations)
        return self.covariance_activation(self.get_scaling, self._rotation)

    # def save_ply(self, path):
    #     os.makedirs(os.path.dirname(path), exist_ok=True)

    #     xyz = self._xyz.detach().cpu().numpy()
    #     f_dc = self._features_dc.detach().contiguous().cpu().numpy()
    #     opacities = self._opacity.detach().cpu().numpy()
    #     scale = self._scaling.detach().cpu().numpy()
    #     rotation = self._rotation.detach().cpu().numpy()

    #     dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]

    #     elements = np.empty(xyz.shape[1], dtype=dtype_full)
    #     attributes = np.concatenate((xyz.squeeze(0), f_dc.squeeze(0), opacities.squeeze(0), scale.squeeze(0), rotation.squeeze(0)), axis=1)
    #     elements[:] = list(map(tuple, attributes))
    #     el = PlyElement.describe(elements, 'vertex')
    #     PlyData([el]).write(path)

    def construct_list_of_attributes(self):
        l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        # All channels except the 3 DC
        for i in range(self._features_dc.shape[2]):
            l.append(f"f_dc_{i}")
        # for i in range(self._features_rest.shape[1]*self._features_rest.shape[2]):
        #     l.append('f_rest_{}'.format(i))
        l.append('opacity')
        for i in range(self._scaling.shape[2]):
            l.append('scale_{}'.format(i))
        for i in range(self._rotation.shape[2]):
            l.append('rot_{}'.format(i))
        return l



    def save_ply_v2(self, path, use_fp16=False, enable_gs_viewer=True, color_code=False):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        xyz = self._xyz.detach().cpu().numpy()
        if self.sh_degree > 0:
            f_dc = (
                self._features_dc.detach()
                .transpose(1, 2)
                .flatten(start_dim=1)
                .contiguous()
                .cpu()
                .numpy()
            )
        else:
            f_dc = (
                self._features_dc.detach()
                .contiguous()
                .cpu()
                .numpy()
            )
        if not color_code:
            rgb = (SH2RGB(f_dc) * 255.0).clip(0.0, 255.0).astype(np.uint8)
        else:
            # use an color map to color code the index of points
            index = np.linspace(0, 1, xyz.shape[0])
            rgb = matplotlib.colormaps["viridis"](index)[..., :3]
            rgb = (rgb * 255.0).clip(0.0, 255.0).astype(np.uint8)
        # rgb = self._features_dc.detach().cpu().numpy()
        opacities = self._opacity.detach().cpu().numpy()
        # scale = self._scaling.detach().cpu().numpy()
        # rotation = self._rotation.detach().cpu().numpy()
        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]
        elements = np.empty(xyz.shape[1], dtype=dtype_full)
        scale = self.get_scaling.detach().cpu().numpy()
        scale = np.log(scale)
        rotation = self.get_rotation.detach().cpu().numpy()

        f_rest = None
        if self.sh_degree > 0:
            f_rest = (
                self._features_rest.detach()
                .transpose(1, 2)
                .flatten(start_dim=1)
                .contiguous()
                .cpu()
                .numpy()
            )  # (3, (self.sh_degree + 1) ** 2 - 1)

        if enable_gs_viewer:
            if self.sh_degree > 0:
                sh_degree = 3
                if f_rest is None:
                    f_rest = np.zeros((xyz.shape[0], 3*((sh_degree + 1) ** 2 - 1)), dtype=np.float32)
                elif f_rest.shape[1] < 3*((sh_degree + 1) ** 2 - 1):
                    f_rest_pad = np.zeros((xyz.shape[0], 3*((sh_degree + 1) ** 2 - 1)), dtype=np.float32)
                    f_rest_pad[:, : f_rest.shape[1]] = f_rest
                    f_rest = f_rest_pad

        if f_rest is not None:
            attributes = np.concatenate(
                (xyz.squeeze(0), rgb.squeeze(0), f_dc.squeeze(0), f_rest.squeeze(0), opacities.squeeze(0), scale.squeeze(0), rotation.squeeze(0)), axis=1
            )
        else:
            attributes = np.concatenate(
                (xyz.squeeze(0), rgb.squeeze(0), f_dc.squeeze(0), opacities.squeeze(0), scale.squeeze(0), rotation.squeeze(0)), axis=1
            )
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, "vertex")
        PlyData([el]).write(path)

    def load_ply(self, path):
        plydata = PlyData.read(path)

        xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])), axis=1)
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

        features_dc = np.zeros((xyz.shape[0], 3, 1))
        features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
        features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scale_names = sorted(scale_names, key=lambda x: int(x.split('_')[-1]))
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
        rot_names = sorted(rot_names, key=lambda x: int(x.split('_')[-1]))
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])
        # import ipdb;ipdb.set_trace()
        # self.set_xyz(torch.from_numpy(xyz.astype(np.float32))
        # self.set_features_dc(torch.from_numpy(features_dc, dtype=torch.float32, device="cuda").squeeze(-1))
        # self.set_opacity(torch.from_numpy(opacities, dtype=torch.float32, device="cuda"))
        # self.set_scaling(torch.from_numpy(scales, dtype=torch.float32, device="cuda"))
        # self.set_rotation(torch.from_numpy(rots, dtype=torch.float32, device="cuda"))

        self.set_xyz(torch.from_numpy(xyz.astype(np.float32)).contiguous())
        # if self.sh_degree > 0 or True:
        #     self.set_features_dc(torch.from_numpy(features_dc.astype(np.float32)).transpose(1, 2).contiguous())
        # else:
        self.set_features_dc(torch.from_numpy(features_dc.astype(np.float32))[..., 0].contiguous())
        self.set_opacity(torch.from_numpy(opacities.astype(np.float32)).contiguous())
        self.set_scaling(torch.from_numpy(scales.astype(np.float32)).contiguous())
        self.set_rotation(torch.from_numpy(rots.astype(np.float32)).contiguous())
