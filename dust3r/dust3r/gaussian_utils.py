# conda install -y pytorch  pytorch-cuda=12.1 -c pytorch -c nvidia
# pip install Pillow plyfile opencv-python numpy git+https://github.com/graphdeco-inria/diff-gaussian-rasterization


import os
import torch
from torch import nn
import numpy as np
from diff_gaussian_rasterization import (
    GaussianRasterizationSettings,
    GaussianRasterizer,
)
from plyfile import PlyData, PlyElement
import cv2
# import matplotlib


# copied from: utils.general_utils
def strip_lowerdiag(L):
    uncertainty = torch.zeros((L.shape[0], 6), dtype=torch.float, device=L.device)

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
    norm = torch.sqrt(
        r[:, 0] * r[:, 0] + r[:, 1] * r[:, 1] + r[:, 2] * r[:, 2] + r[:, 3] * r[:, 3]
    )

    q = r / norm[:, None]

    R = torch.zeros((q.size(0), 3, 3), device=r.device)

    r = q[:, 0]
    x = q[:, 1]
    y = q[:, 2]
    z = q[:, 3]

    R[:, 0, 0] = 1 - 2 * (y * y + z * z)
    R[:, 0, 1] = 2 * (x * y - r * z)
    R[:, 0, 2] = 2 * (x * z + r * y)
    R[:, 1, 0] = 2 * (x * y + r * z)
    R[:, 1, 1] = 1 - 2 * (x * x + z * z)
    R[:, 1, 2] = 2 * (y * z - r * x)
    R[:, 2, 0] = 2 * (x * z - r * y)
    R[:, 2, 1] = 2 * (y * z + r * x)
    R[:, 2, 2] = 1 - 2 * (x * x + y * y)
    return R


def build_scaling_rotation(s, r):
    L = torch.zeros((s.shape[0], 3, 3), dtype=torch.float, device=s.device)
    R = build_rotation(r)

    L[:, 0, 0] = s[:, 0]
    L[:, 1, 1] = s[:, 1]
    L[:, 2, 2] = s[:, 2]

    L = R @ L
    return L


# copied from: utils.sh_utils
C0 = 0.28209479177387814
C1 = 0.4886025119029199
C2 = [
    1.0925484305920792,
    -1.0925484305920792,
    0.31539156525252005,
    -1.0925484305920792,
    0.5462742152960396,
]
C3 = [
    -0.5900435899266435,
    2.890611442640554,
    -0.4570457994644658,
    0.3731763325901154,
    -0.4570457994644658,
    1.445305721320277,
    -0.5900435899266435,
]
C4 = [
    2.5033429417967046,
    -1.7701307697799304,
    0.9461746957575601,
    -0.6690465435572892,
    0.10578554691520431,
    -0.6690465435572892,
    0.47308734787878004,
    -1.7701307697799304,
    0.6258357354491761,
]


def eval_sh(deg, sh, dirs):
    """
    Evaluate spherical harmonics at unit directions
    using hardcoded SH polynomials.
    Works with torch/np/jnp.
    ... Can be 0 or more batch dimensions.
    Args:
        deg: int SH deg. Currently, 0-3 supported
        sh: jnp.ndarray SH coeffs [..., C, (deg + 1) ** 2]
        dirs: jnp.ndarray unit directions [..., 3]
    Returns:
        [..., C]
    """
    assert deg <= 4 and deg >= 0
    coeff = (deg + 1) ** 2
    assert sh.shape[-1] >= coeff

    result = C0 * sh[..., 0]
    if deg > 0:
        x, y, z = dirs[..., 0:1], dirs[..., 1:2], dirs[..., 2:3]
        result = (
            result - C1 * y * sh[..., 1] + C1 * z * sh[..., 2] - C1 * x * sh[..., 3]
        )

        if deg > 1:
            xx, yy, zz = x * x, y * y, z * z
            xy, yz, xz = x * y, y * z, x * z
            result = (
                result
                + C2[0] * xy * sh[..., 4]
                + C2[1] * yz * sh[..., 5]
                + C2[2] * (2.0 * zz - xx - yy) * sh[..., 6]
                + C2[3] * xz * sh[..., 7]
                + C2[4] * (xx - yy) * sh[..., 8]
            )

            if deg > 2:
                result = (
                    result
                    + C3[0] * y * (3 * xx - yy) * sh[..., 9]
                    + C3[1] * xy * z * sh[..., 10]
                    + C3[2] * y * (4 * zz - xx - yy) * sh[..., 11]
                    + C3[3] * z * (2 * zz - 3 * xx - 3 * yy) * sh[..., 12]
                    + C3[4] * x * (4 * zz - xx - yy) * sh[..., 13]
                    + C3[5] * z * (xx - yy) * sh[..., 14]
                    + C3[6] * x * (xx - 3 * yy) * sh[..., 15]
                )

                if deg > 3:
                    result = (
                        result
                        + C4[0] * xy * (xx - yy) * sh[..., 16]
                        + C4[1] * yz * (3 * xx - yy) * sh[..., 17]
                        + C4[2] * xy * (7 * zz - 1) * sh[..., 18]
                        + C4[3] * yz * (7 * zz - 3) * sh[..., 19]
                        + C4[4] * (zz * (35 * zz - 30) + 3) * sh[..., 20]
                        + C4[5] * xz * (7 * zz - 3) * sh[..., 21]
                        + C4[6] * (xx - yy) * (7 * zz - 1) * sh[..., 22]
                        + C4[7] * xz * (xx - 3 * yy) * sh[..., 23]
                        + C4[8]
                        * (xx * (xx - 3 * yy) - yy * (3 * xx - yy))
                        * sh[..., 24]
                    )
    return result


def RGB2SH(rgb):
    return (rgb - 0.5) / C0


def SH2RGB(sh):
    return sh * C0 + 0.5


def create_video(image_folder, output_video_file, framerate=30):
    # Get all image file paths to a list.
    images = [img for img in os.listdir(image_folder) if img.endswith(".png")]
    images.sort()

    # Read the first image to know the height and width
    frame = cv2.imread(os.path.join(image_folder, images[0]))
    height, width, layers = frame.shape

    video = cv2.VideoWriter(
        output_video_file, cv2.VideoWriter_fourcc(*"mp4v"), framerate, (width, height)
    )

    # iterate over each image and add it to the video sequence
    for image in images:
        video.write(cv2.imread(os.path.join(image_folder, image)))

    cv2.destroyAllWindows()
    video.release()


class Camera(nn.Module):
    def __init__(self, C2W, fxfycxcy, h, w):
        """
        C2W: 4x4 camera-to-world matrix; opencv convention
        fxfycxcy: 4
        """
        super().__init__()
        self.C2W = C2W.clone().float()
        self.W2C = self.C2W.inverse()

        self.znear = 0.00001
        self.zfar = 100.0
        self.h = h
        self.w = w

        fx, fy, cx, cy = fxfycxcy[0], fxfycxcy[1], fxfycxcy[2], fxfycxcy[3]
        self.tanfovX = 1 / (2 * fx)
        self.tanfovY = 1 / (2 * fy)
        self.fovX = 2 * torch.atan(self.tanfovX)
        self.fovY = 2 * torch.atan(self.tanfovY)
        self.shiftX = 2 * cx - 1
        self.shiftY = 2 * cy - 1

        def getProjectionMatrix(znear, zfar, fovX, fovY, shiftX, shiftY):
            tanHalfFovY = torch.tan((fovY / 2))
            tanHalfFovX = torch.tan((fovX / 2))

            top = tanHalfFovY * znear
            bottom = -top
            right = tanHalfFovX * znear
            left = -right

            P = torch.zeros(4, 4, device=fovX.device)

            z_sign = 1.0

            P[0, 0] = 2.0 * znear / (right - left)
            P[1, 1] = 2.0 * znear / (top - bottom)
            P[0, 2] = (right + left) / (right - left) + shiftX
            P[1, 2] = (top + bottom) / (top - bottom) + shiftY
            P[3, 2] = z_sign
            P[2, 2] = z_sign * zfar / (zfar - znear)
            P[2, 3] = -(zfar * znear) / (zfar - znear)
            return P

        self.world_view_transform = self.W2C.transpose(0, 1)
        self.projection_matrix = getProjectionMatrix(
            znear=self.znear, zfar=self.zfar, fovX=self.fovX, fovY=self.fovY, shiftX=self.shiftX, shiftY=self.shiftY
        ).transpose(0, 1)
        self.full_proj_transform = (
            self.world_view_transform.unsqueeze(0).bmm(
                self.projection_matrix.unsqueeze(0)
            )
        ).squeeze(0)
        self.camera_center = self.C2W[:3, 3]


# modified from scene/gaussian_model.py
class GaussianModel:
    def setup_functions(self, scaling_activation_type='exp', scale_min_act=0.001, scale_max_act=0.3, scale_multi_act=0.1):
        def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
            L = build_scaling_rotation(scaling_modifier * scaling, rotation)
            actual_covariance = L @ L.transpose(1, 2)
            symm = strip_symmetric(actual_covariance)
            return symm
        if scaling_activation_type == 'exp':
            self.scaling_activation = torch.exp
        elif scaling_activation_type == 'softplus':
            self.scaling_activation = torch.nn.functional.softplus
            self.scale_multi_act = scale_multi_act
        elif scaling_activation_type == 'sigmoid':
            self.scale_min_act = scale_min_act
            self.scale_max_act = scale_max_act
            self.scaling_activation = torch.sigmoid
        else:
            raise NotImplementedError
        self.scaling_activation_type = scaling_activation_type

        self.rotation_activation = torch.nn.functional.normalize
        self.opacity_activation = torch.sigmoid
        self.feature_activation = torch.sigmoid
        self.covariance_activation = build_covariance_from_scaling_rotation

    def __init__(self, sh_degree: int, scaling_activation_type='exp', scale_min_act=0.001, scale_max_act=0.3, scale_multi_act=0.1):
        self.sh_degree = sh_degree
        self._xyz = torch.empty(0)
        self._features_dc = torch.empty(0)
        if self.sh_degree > 0:
            self._features_rest = torch.empty(0)
        else:
            self._features_rest = None
        self._scaling = torch.empty(0)
        self._rotation = torch.empty(0)
        self._opacity = torch.empty(0)
        self.setup_functions(scaling_activation_type=scaling_activation_type, scale_min_act=scale_min_act, scale_max_act=scale_max_act, scale_multi_act=scale_multi_act)
    
    # def set_data(self, xyz, colors, opacity, scaling, rotation, offset):
    #     """
    #     depth: N_views, H, W
    #     intrinsics: N_views, 3, 3
    #     c2ws: N_views, 4, 4 
    #     others: n_points, c_dim
    #     """
    #     self._xyz = unproject(depth.unsqueeze(0), intrinsics.unsqueeze(0), c2ws.unsqueeze(0)).squeeze(0) + offset
    #     self._features_dc = colors[:, :1, :].contiguous()
    #     if self.sh_degree > 0:
    #         self._features_rest = colors[:, 1:, :].contiguous()
    #     else:
    #         self._features_rest = None
    #     self._scaling = scaling
    #     self._rotation = rotation
    #     self._opacity = opacity

    def set_data(self, xyz, features, scaling, rotation, opacity):
        """
        xyz : torch.tensor of shape (N, 3)
        features : torch.tensor of shape (N, (self.sh_degree + 1) ** 2, 3)
        scaling : torch.tensor of shape (N, 3)
        rotation : torch.tensor of shape (N, 4)
        opacity : torch.tensor of shape (N, 1)
        """
        self._xyz = xyz
        # self._features_dc = features[:, :1, :].contiguous()
        self._features_dc = features.contiguous()
        if self.sh_degree > 0:
            self._features_rest = features[:, 1:, :].contiguous()
        else:
            self._features_rest = None
        self._scaling = scaling
        self._rotation = rotation
        self._opacity = opacity
        return self

    def to(self, device):
        self._xyz = self._xyz.to(device)
        self._features_dc = self._features_dc.to(device)
        if self.sh_degree > 0:
            self._features_rest = self._features_rest.to(device)
        self._scaling = self._scaling.to(device)
        self._rotation = self._rotation.to(device)
        self._opacity = self._opacity.to(device)
        return self

    def crop(self, crop_bbx=[-1, 1, -1, 1, -1, 1]):
        x_min, x_max, y_min, y_max, z_min, z_max = crop_bbx
        xyz = self._xyz
        invalid_mask = (
            (xyz[:, 0] < x_min)
            | (xyz[:, 0] > x_max)
            | (xyz[:, 1] < y_min)
            | (xyz[:, 1] > y_max)
            | (xyz[:, 2] < z_min)
            | (xyz[:, 2] > z_max)
        )
        valid_mask = ~invalid_mask

        self._xyz = self._xyz[valid_mask]
        self._features_dc = self._features_dc[valid_mask]
        if self.sh_degree > 0:
            self._features_rest = self._features_rest[valid_mask]
        self._scaling = self._scaling[valid_mask]
        self._rotation = self._rotation[valid_mask]
        self._opacity = self._opacity[valid_mask]
        return self

    def prune(self, opacity_thres=0.05):
        opacity = self.get_opacity.squeeze(1)
        valid_mask = opacity > opacity_thres

        self._xyz = self._xyz[valid_mask]
        self._features_dc = self._features_dc[valid_mask]
        if self.sh_degree > 0:
            self._features_rest = self._features_rest[valid_mask]
        self._scaling = self._scaling[valid_mask]
        self._rotation = self._rotation[valid_mask]
        self._opacity = self._opacity[valid_mask]
        return self

    def shrink_bbx(self, drop_ratio=0.05):
        xyz = self._xyz
        xyz_min, xyz_max = torch.quantile(
            xyz,
            torch.tensor([drop_ratio, 1 - drop_ratio]).float().to(xyz.device),
            dim=0,
        )  # [2, N]
        xyz_min = xyz_min.detach().cpu().numpy()
        xyz_max = xyz_max.detach().cpu().numpy()
        crop_bbx = [
            xyz_min[0],
            xyz_max[0],
            xyz_min[1],
            xyz_max[1],
            xyz_min[2],
            xyz_max[2],
        ]
        print(f"Shrinking bbx to {crop_bbx}")
        return self.crop(crop_bbx)

    def load_ply_new_defered(self, path, batch_size):
        plydata = PlyData.read(path)
        xyz = np.stack(
            (
                np.asarray(plydata.elements[0]["x"]),
                np.asarray(plydata.elements[0]["y"]),
                np.asarray(plydata.elements[0]["z"]),
            ),
            axis=1,
        )
        opacities = np.ones((batch_size,xyz.shape[0], 1)).astype(np.float32)
        rots = np.zeros((batch_size,xyz.shape[0], 4)).astype(np.float32)
        scales = np.ones((batch_size,xyz.shape[0], 3)).astype(np.float32)* -5
        colors = np.zeros((batch_size,xyz.shape[0], 3)).astype(np.float32)+0.5
        self._xyz=torch.from_numpy(xyz).cuda().repeat(batch_size,1,1)
        self._opacity = torch.from_numpy(opacities).cuda()
        self._scaling = torch.from_numpy(scales).cuda()
        self._rotation = torch.from_numpy(rots).cuda()
        self._features= torch.from_numpy(colors).cuda()

    def report_stats(self):
        print(
            f"xyz: {self._xyz.shape}, {self._xyz.min().item()}, {self._xyz.max().item()}"
        )
        print(
            f"features_dc: {self._features_dc.shape}, {self._features_dc.min().item()}, {self._features_dc.max().item()}"
        )
        if self.sh_degree > 0:
            print(
                f"features_rest: {self._features_rest.shape}, {self._features_rest.min().item()}, {self._features_rest.max().item()}"
            )
        print(
            f"scaling: {self._scaling.shape}, {self._scaling.min().item()}, {self._scaling.max().item()}"
        )
        print(
            f"rotation: {self._rotation.shape}, {self._rotation.min().item()}, {self._rotation.max().item()}"
        )
        print(
            f"opacity: {self._opacity.shape}, {self._opacity.min().item()}, {self._opacity.max().item()}"
        )

        print(
            f"after activation, xyz: {self.get_xyz.shape}, {self.get_xyz.min().item()}, {self.get_xyz.max().item()}"
        )
        print(
            f"after activation, features: {self.get_features.shape}, {self.get_features.min().item()}, {self.get_features.max().item()}"
        )
        print(
            f"after activation, scaling: {self.get_scaling.shape}, {self.get_scaling.min().item()}, {self.get_scaling.max().item()}"
        )
        print(
            f"after activation, rotation: {self.get_rotation.shape}, {self.get_rotation.min().item()}, {self.get_rotation.max().item()}"
        )
        print(
            f"after activation, opacity: {self.get_opacity.shape}, {self.get_opacity.min().item()}, {self.get_opacity.max().item()}"
        )
        print(
            f"after activation, covariance: {self.get_covariance().shape}, {self.get_covariance().min().item()}, {self.get_covariance().max().item()}"
        )

    @property
    def get_scaling(self):
        if self.scaling_activation_type == 'exp':
            scales = self.scaling_activation(self._scaling)
        elif self.scaling_activation_type == 'softplus':
            scales = self.scaling_activation(self._scaling) * self.scale_multi_act
        elif self.scaling_activation_type == 'sigmoid':
            scales = self.scale_min_act + (self.scale_max_act - self.scale_min_act) * self.scaling_activation(self._scaling) # self._scaling#
        return scales

    @property
    def get_rotation(self):
        return self.rotation_activation(self._rotation)

    @property
    def get_xyz(self):
        return self._xyz

    @property
    def get_features(self):
        if self.sh_degree > 0:
            features_dc = self._features_dc
            features_rest = self._features_rest
            return torch.cat((features_dc, features_rest), dim=1)
        else:
            return self.feature_activation(self._features_dc)

    @property
    def get_opacity(self):
        return self.opacity_activation(self._opacity)

    def get_covariance(self, scaling_modifier=1):
        return self.covariance_activation(
            self.get_scaling, scaling_modifier, self._rotation
        )

    def construct_dtypes(self, use_fp16=False, enable_gs_viewer=True):
        if not use_fp16:
            l = [
                ("x", "f4"),
                ("y", "f4"),
                ("z", "f4"),
                ("red", "u1"),
                ("green", "u1"),
                ("blue", "u1"),
            ]
            # All channels except the 3 DC
            if self.sh_degree > 0:
                for i in range(self._features_dc.shape[1] * self._features_dc.shape[2]):
                    l.append((f"f_dc_{i}", "f4"))
            else:
                for i in range(self._features_dc.shape[1]):
                    l.append((f"f_dc_{i}", "f4"))

            if enable_gs_viewer:
                assert self.sh_degree <= 3, "GS viewer only supports SH up to degree 3"
                if self.sh_degree > 0:
                    sh_degree = 3
                    for i in range(((sh_degree + 1) ** 2 - 1) * 3):
                        l.append((f"f_rest_{i}", "f4"))
            else:
                if self.sh_degree > 0:
                    for i in range(
                        self._features_rest.shape[1] * self._features_rest.shape[2]
                    ):
                        l.append((f"f_rest_{i}", "f4"))

            l.append(("opacity", "f4"))
            for i in range(self._scaling.shape[1]):
                l.append((f"scale_{i}", "f4"))
            for i in range(self._rotation.shape[1]):
                l.append((f"rot_{i}", "f4"))
        else:
            l = [
                ("x", "f2"),
                ("y", "f2"),
                ("z", "f2"),
                ("red", "u1"),
                ("green", "u1"),
                ("blue", "u1"),
            ]
            # All channels except the 3 DC
            for i in range(self._features_dc.shape[1] * self._features_dc.shape[2]):
                l.append((f"f_dc_{i}", "f2"))

            if self.sh_degree > 0:
                for i in range(
                    self._features_rest.shape[1] * self._features_rest.shape[2]
                ):
                    l.append((f"f_rest_{i}", "f2"))
            l.append(("opacity", "f2"))
            for i in range(self._scaling.shape[1]):
                l.append((f"scale_{i}", "f2"))
            for i in range(self._rotation.shape[1]):
                l.append((f"rot_{i}", "f2"))
        return l

    def save_ply(self, path, use_fp16=False, enable_gs_viewer=True, color_code=False):
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

        opacities = self._opacity.detach().cpu().numpy()
        if self.scaling_activation_type == 'exp':
            scale = self._scaling
        elif self.scaling_activation_type == 'softplus':
            scale = torch.log(self.scaling_activation(self._scaling) * self.scale_multi_act)
        elif self.scaling_activation_type == 'sigmoid':
            scale = self.scale_min_act + (self.scale_max_act - self.scale_min_act) * self.scaling_activation(self._scaling)
            scale = torch.log(scale)
        scale = scale.detach().cpu().numpy()

        rotation = self._rotation.detach().cpu().numpy()

        dtype_full = self.construct_dtypes(use_fp16, enable_gs_viewer)
        elements = np.empty(xyz.shape[0], dtype=dtype_full)

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
                (xyz, rgb, f_dc, f_rest, opacities, scale, rotation), axis=1
            )
        else:
            attributes = np.concatenate(
                (xyz, rgb, f_dc, opacities, scale, rotation), axis=1
            )
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, "vertex")
        PlyData([el]).write(path)

    def load_ply(self, path):
        plydata = PlyData.read(path)

        xyz = np.stack(
            (
                np.asarray(plydata.elements[0]["x"]),
                np.asarray(plydata.elements[0]["y"]),
                np.asarray(plydata.elements[0]["z"]),
            ),
            axis=1,
        )
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

        features_dc = np.zeros((xyz.shape[0], 3, 1))
        features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
        features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

        if self.sh_degree > 0:
            extra_f_names = [
                p.name
                for p in plydata.elements[0].properties
                if p.name.startswith("f_rest_")
            ]
            extra_f_names = sorted(extra_f_names, key=lambda x: int(x.split("_")[-1]))
            assert len(extra_f_names) == 3 * (self.sh_degree + 1) ** 2 - 3
            features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
            for idx, attr_name in enumerate(extra_f_names):
                features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
            # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
            features_extra = features_extra.reshape(
                (features_extra.shape[0], 3, (self.sh_degree + 1) ** 2 - 1)
            )

        scale_names = [
            p.name
            for p in plydata.elements[0].properties
            if p.name.startswith("scale_")
        ]
        scale_names = sorted(scale_names, key=lambda x: int(x.split("_")[-1]))
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [
            p.name for p in plydata.elements[0].properties if p.name.startswith("rot")
        ]
        rot_names = sorted(rot_names, key=lambda x: int(x.split("_")[-1]))
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

        self._xyz = torch.from_numpy(xyz.astype(np.float32))
        self._features_dc = (
            torch.from_numpy(features_dc.astype(np.float32))
            .transpose(1, 2)
            .contiguous()
        )
        if self.sh_degree > 0:
            self._features_rest = (
                torch.from_numpy(features_extra.astype(np.float32))
                .transpose(1, 2)
                .contiguous()
            )
        self._opacity = torch.from_numpy(
            np.copy(opacities).astype(np.float32)
        ).contiguous()
        self._scaling = torch.from_numpy(scales.astype(np.float32)).contiguous()
        self._rotation = torch.from_numpy(rots.astype(np.float32)).contiguous()


def render_opencv_cam(
    pc: GaussianModel,
    height: int,
    width: int,
    C2W: torch.Tensor,
    fxfycxcy: torch.Tensor,
    bg_color=(0.0, 0.0, 0.0),
    scaling_modifier=1.0,
):
    """
    Render the scene.

    Background tensor (bg_color) must be on GPU!
    """
    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = (
        torch.zeros_like(
            pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda"
        )
        + 0
    )
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # viewpoint_camera = Camera(C2W=C2W, fxfycxcy=fxfycxcy, h=height, w=width)
    viewpoint_camera = Camera(C2W=C2W, fxfycxcy=fxfycxcy, h=height, w=width)

    bg_color = torch.tensor(list(bg_color), dtype=torch.float32, device=C2W.device)

    # Set up rasterization configuration
    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.h),
        image_width=int(viewpoint_camera.w),
        tanfovx=viewpoint_camera.tanfovX,
        tanfovy=viewpoint_camera.tanfovY,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=pc.sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=False,
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    means3D = pc.get_xyz
    means2D = screenspace_points
    opacity = pc.get_opacity
    scales = pc.get_scaling
    rotations = pc.get_rotation
    shs = pc.get_features
    # Rasterize visible Gaussians to image, obtain their radii (on screen).
    rendered_image, radii, rendered_depth, rendered_alpha = rasterizer(
        means3D=means3D,
        means2D=means2D,
        shs=None,
        colors_precomp=shs,
        opacities=opacity,
        scales=scales,
        rotations=rotations,
        cov3D_precomp=None,
    )

    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    return {
        "render": rendered_image,
        "viewspace_points": screenspace_points,
        "visibility_filter": radii > 0,
        "radii": radii,
        "alpha": rendered_alpha,
        "depth": rendered_depth,
    }


if __name__ == "__main__":
    import json
    from PIL import Image
    from tqdm import tqdm

    out_dir = "/mnt/localssd/debug-3dgs"
    os.makedirs(out_dir, exist_ok=True)

    os.system(
        f"wget https://phidias.s3.us-west-2.amazonaws.com/kaiz/neural-capture/eval-3dgs-lowres/AWS_test_set/results/1.fashion_boots_rubber_boots__short__Feb_21__2023_at_5_19_25_PM_yf/point_cloud/iteration_30000_fg/point_cloud.ply -O {out_dir}/point_cloud.ply"
    )
    os.system(
        f"wget https://neural-capture.s3.us-west-2.amazonaws.com/data/AWS_test_set/preprocessed/1.fashion_boots_rubber_boots__short__Feb_21__2023_at_5_19_25_PM_yf/opencv_cameras_traj_norm.json -O {out_dir}/opencv_cameras_traj_norm.json"
    )

    device = "cuda:0"

    pc = GaussianModel(sh_degree=3)
    pc.load_ply(f"{out_dir}/point_cloud.ply")
    pc = pc.to(device)

    pc.save_ply(f"{out_dir}/point_cloud_shrink.ply")
    pc.load_ply(f"{out_dir}/point_cloud_shrink.ply")
    pc = pc.to(device)

    pc.prune(opacity_thres=0.05)
    pc.save_ply(f"{out_dir}/point_cloud_shrink_prune.ply")
    pc = pc.to(device)

    pc.shrink_bbx(drop_ratio=0.01)
    pc.save_ply(f"{out_dir}/point_cloud_shrink_prune.ply")
    pc = pc.to(device)

    pc.report_stats()

    with open(f"{out_dir}/opencv_cameras_traj_norm.json", "r") as f:
        cam_traj = json.load(f)

    for i, cam in tqdm(enumerate(cam_traj["frames"]), desc="Rendering progress"):
        w2c = np.array(cam["w2c"])
        c2w = np.linalg.inv(w2c)
        c2w = torch.from_numpy(c2w.astype(np.float32)).to(device)

        fx = cam["fx"]
        fy = cam["fy"]
        cx = cam["cx"]
        cy = cam["cy"]
        fxfycxcy = torch.tensor([fx, fy, cx, cy], dtype=torch.float32, device=device)

        h = cam["h"]
        w = cam["w"]

        im = render_opencv_cam(pc, h, w, c2w, fxfycxcy, bg_color=[0.0, 0.0, 0.0])[
            "render"
        ]
        im = im.detach().cpu().numpy().transpose(1, 2, 0)
        im = (im * 255).astype(np.uint8)
        Image.fromarray(im).save(f"{out_dir}/render_{i:08d}.png")

    create_video(out_dir, f"{out_dir}/render.mp4", framerate=30)
    print(f"Saved {out_dir}/render.mp4")
