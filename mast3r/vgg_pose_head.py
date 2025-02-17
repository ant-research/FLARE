# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


import logging
from collections import defaultdict
from dataclasses import field, dataclass

import math
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from typing import Any, Dict, List, Optional, Tuple, Union, Callable
from functools import partial
from models.blocks import DecoderBlock
from .modules import AttnBlock, CrossAttnBlock, Mlp, ResidualBlock, Mlp_res
from .util_vgg import PoseEmbedding, pose_encoding_to_camera, camera_to_pose_encoding
from torch.utils.checkpoint import checkpoint
import torch.nn.functional as F
from pytorch3d.transforms.rotation_conversions import matrix_to_quaternion, quaternion_to_matrix
import pytorch3d.transforms 
logger = logging.getLogger(__name__)
_RESNET_MEAN = [0.485, 0.456, 0.406]
_RESNET_STD = [0.229, 0.224, 0.225]

def standardize_quaternion(quaternions: torch.Tensor) -> torch.Tensor:
    """
    Convert a unit quaternion to a standard form: one in which the real
    part is non negative.

    Args:
        quaternions: Quaternions with real part first,
            as tensor of shape (..., 4).

    Returns:
        Standardized quaternions as tensor of shape (..., 4).
    """
    return torch.where(quaternions[..., 0:1] < 0, -quaternions, quaternions)

def rotation_distance(R1,R2,eps=1e-7):
    # http://www.boris-belousov.net/2016/12/01/quat-dist/
    R_diff = R1@R2.transpose(-2,-1)
    trace = R_diff[...,0,0]+R_diff[...,1,1]+R_diff[...,2,2]
    angle = ((trace-1)/2).clamp(-1+eps,1-eps).acos_() # numerical stability near -1/+1
    return angle

class SimpleVQAutoEncoder(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.encoder = nn.ModuleList(
            [Mlp(7, hidden_size*2, hidden_size*2, drop=0)]+ [Mlp_res(hidden_size*2, hidden_size*2, hidden_size*2, drop=0) for _ in range(4)] + \
            [Mlp(hidden_size*2, hidden_size*2, 256, drop=0)]
        )
        self.decoder = nn.ModuleList(
            [Mlp(256, hidden_size*2, hidden_size*2, drop=0)] + [Mlp_res(hidden_size*2, hidden_size*2, hidden_size*2, drop=0) for _ in range(4)] + [Mlp(hidden_size*2, hidden_size*2, 7, drop=0)])
        
    def forward(self, xs):
        z_e = self.encode(xs)
        out = self.decode(z_e)
        return out

    def encode(self, x):
        for encoder in self.encoder:
            x = encoder(x)
        # z_e = x.permute(0, 2, 3, 1).contiguous()
        return x

    def decode(self, z_q):
        # z_q = z_q.permute(0, 3, 1, 2).contiguous()
        for decoder in self.decoder:
            z_q = decoder(z_q)
        # out = z_q
        # out_rot = out[..., 3:]
        # out = torch.cat([out[..., :3], out_rot], dim=-1)
        return z_q

    @torch.no_grad()
    def get_codes(self, xs):
        z_e = self.encode(xs)
        _, _, code = self.quantizer(z_e)
        return code

    @torch.no_grad()
    def get_soft_codes(self, xs, temp=1.0, stochastic=False):
        assert hasattr(self.quantizer, 'get_soft_codes')

        z_e = self.encode(xs)
        soft_code, code = self.quantizer.get_soft_codes(z_e, temp=temp, stochastic=stochastic)
        return soft_code, code

    @torch.no_grad()
    def decode_code(self, code):
        z_q = self.quantizer.embed_code(code)
        decoded = self.decode(z_q)
        return decoded

    def get_recon_imgs(self, xs_real, xs_recon):

        xs_real = xs_real * 0.5 + 0.5
        xs_recon = xs_recon * 0.5 + 0.5
        xs_recon = torch.clamp(xs_recon, 0, 1)

        return xs_real, xs_recon

    def compute_loss(self, out, xs=None, valid=False):

        # if self.loss_type == 'mse':
        loss_recon = F.mse_loss(out, xs, reduction='mean') 
        # elif self.loss_type == 'l1':
        #     loss_recon = F.l1_loss(out, xs, reduction='mean')
        # else:
        #     raise ValueError('incompatible loss type')
        if valid:
            loss_recon = loss_recon * xs.shape[0] * xs.shape[1]
        loss_total = loss_recon 
        return {
            'loss_total': loss_total,
            'loss_recon': loss_recon,
        }

    def get_last_layer(self):
        return self.decoder.conv_out.weight

    @torch.no_grad()
    def get_code_emb_with_depth(self, code):
        return self.quantizer.embed_code_with_depth(code)

    @torch.no_grad()
    def decode_partial_code(self, code, code_idx, decode_type='select'):
        r"""
        Use partial codebooks and decode the codebook features.
        If decode_type == 'select', the (code_idx)-th codebook features are decoded.
        If decode_type == 'add', the [0,1,...,code_idx]-th codebook features are added and decoded.
        """
        z_q = self.quantizer.embed_partial_code(code, code_idx, decode_type)
        decoded = self.decode(z_q)
        return decoded

    @torch.no_grad()
    def forward_partial_code(self, xs, code_idx, decode_type='select'):
        r"""
        Reconstuct an input using partial codebooks.
        """
        code = self.get_codes(xs)
        out = self.decode_partial_code(code, code_idx, decode_type)
        return out

class CameraPredictor(nn.Module):
    def __init__(
        self,
        hooks_idx,
        hidden_size=768,
        num_heads=8,
        mlp_ratio=4,
        z_dim: int = 768,
        z_dim_input: int = 768,
        down_size=336,
        att_depth=8,
        trunk_depth=4,
        pose_encoding_type="absT_quaR_logFL",
        cfg=None,
        rope=None
    ):
        super().__init__()
        self.cfg = cfg
        self.hooks_idx = hooks_idx

        self.att_depth = att_depth
        self.down_size = down_size
        self.pose_encoding_type = pose_encoding_type
        self.rope = rope
        if self.pose_encoding_type == "absT_quaR_OneFL":
            self.target_dim = 8
        if self.pose_encoding_type == "absT_quaR_logFL":
            self.target_dim = 9

        # self.backbone = self.get_backbone(backbone)

        # for param in self.backbone.parameters():
        #     param.requires_grad = False
        self.norm = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.norm_input = nn.LayerNorm(hidden_size, elementwise_affine=True, eps=1e-6)

        # sine and cosine embed for camera parameters
        self.embed_pose = PoseEmbedding(
            target_dim=self.target_dim, n_harmonic_functions=(hidden_size // self.target_dim) // 2, append_input=True
        )
        self.pose_proj = nn.Linear(756 + 9, hidden_size)
        self.pose_token = nn.Parameter(torch.zeros(1, 1, 1, hidden_size))  # register
        self.pose_token_ref = nn.Parameter(torch.zeros(1, 1, 1, hidden_size))  # register
        self.feat0_token = nn.Parameter(torch.zeros(1, 1, 1, hidden_size))  # register
        self.feat1_token = nn.Parameter(torch.zeros(1, 1, 1, hidden_size))  # register

        self.input_transform = Mlp(in_features=z_dim_input, hidden_features=hidden_size, out_features=hidden_size, drop=0)
        self.pose_branch = Mlp(
            in_features=hidden_size, hidden_features=hidden_size * 2, out_features=hidden_size + self.target_dim, drop=0
        )

        self.ffeat_updater = nn.Sequential(nn.Linear(hidden_size, hidden_size), nn.GELU())
        self.self_att = nn.ModuleList(
            [
                AttnBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio, attn_class=nn.MultiheadAttention)
                for _ in range(self.att_depth)
            ]
        )
        self.cross_att = nn.ModuleList(
            [CrossAttnBlock(hidden_size, hidden_size, num_heads, mlp_ratio=mlp_ratio) for _ in range(self.att_depth)]
        )
        
        self.dec_blocks = nn.ModuleList([
            DecoderBlock(hidden_size, 12, mlp_ratio=mlp_ratio, qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), norm_mem=True, rope=self.rope)
            for i in range(1)])
        
        self.trunk = nn.Sequential(
            *[
                AttnBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio, attn_class=nn.MultiheadAttention)
                for _ in range(trunk_depth)
            ]
        )

        self.gamma = 0.8

        nn.init.normal_(self.pose_token, std=1e-6)

        for name, value in (("_resnet_mean", _RESNET_MEAN), ("_resnet_std", _RESNET_STD)):
            self.register_buffer(name, torch.FloatTensor(value).view(1, 3, 1, 1), persistent=False)

    def forward(self, batch_size, iters=4, pos_encoding=None, interm_feature1=None, interm_feature2=None, enabled=True, dtype=torch.bfloat16):
        """
        reshaped_image: Bx3xHxW. The values of reshaped_image are within [0, 1]
        preliminary_cameras: PyTorch3D cameras.

        TODO: dropping the usage of PyTorch3D cameras.
        """
        # if rgb_feat_init is None:
        #     # Get the 2D image features
        rgb_feat_init1 = interm_feature1
        rgb_feat_init2 = interm_feature2
        rgb_feat_init1[0] =  self.norm_input(self.input_transform(rgb_feat_init1[0]))
        rgb_feat_init2[0] =  self.norm_input(self.input_transform(rgb_feat_init2[0]))
        rgb_feat, B, S, C = self.get_2D_image_features(batch_size, rgb_feat_init1, rgb_feat_init2, pos_encoding, dtype)
        B, S, C = rgb_feat.shape
        # if preliminary_cameras is not None:
        #     # Init the pred_pose_enc by preliminary_cameras
        #     pred_pose_enc = (
        #         camera_to_pose_encoding(preliminary_cameras, pose_encoding_type=self.pose_encoding_type)
        #         .reshape(B, S, -1)
        #         .to(rgb_feat.dtype)
        #     )
        # else:
        # Or you can use random init for the poses
        pred_pose_enc = torch.zeros(B, S, self.target_dim).to(rgb_feat.device)
        rgb_feat_init = rgb_feat.clone()
        pred_cameras_list = []
        for iter_num in range(iters):
            pred_pose_enc = pred_pose_enc.detach()
            # Embed the camera parameters and add to rgb_feat
            pose_embed = self.embed_pose(pred_pose_enc)
            pose_embed = self.pose_proj(pose_embed)
            rgb_feat = rgb_feat + pose_embed
            rgb_feat[:,:1] = self.pose_token_ref[:, 0] + rgb_feat[:,:1]
            # Run trunk transformers on rgb_feat
            rgb_feat = self.trunk(rgb_feat)
            # Predict the delta feat and pose encoding at each iteration
            delta = self.pose_branch(rgb_feat)
            delta_pred_pose_enc = delta[..., : self.target_dim]
            delta_feat = delta[..., self.target_dim :]
            rgb_feat = self.ffeat_updater(self.norm(delta_feat)) + rgb_feat
            pred_pose_enc = pred_pose_enc + delta_pred_pose_enc
            # Residual connection
            rgb_feat = (rgb_feat + rgb_feat_init) / 2
            # Pose encoding to Cameras
            with torch.cuda.amp.autocast(enabled=False, dtype=torch.float32): 
                pred_cameras = pose_encoding_to_camera(pred_pose_enc, pose_encoding_type='train')
                pred_cameras_list = pred_cameras_list + [pred_cameras]
        # pose_predictions = {
        #     "pred_pose_enc": pred_pose_enc,
        #     "pred_cameras": pred_cameras,
        #     "rgb_feat_init": rgb_feat_init,
        # }

        return pred_cameras_list, rgb_feat

    def get_backbone(self, backbone):
        """
        Load the backbone model.
        """
        if backbone == "dinov2s":
            return torch.hub.load("facebookresearch/dinov2", "dinov2_vits14_reg")
        elif backbone == "dinov2b":
            return torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14_reg")
        else:
            raise NotImplementedError(f"Backbone '{backbone}' not implemented")

    def _resnet_normalize_image(self, img: torch.Tensor) -> torch.Tensor:
        return (img - self._resnet_mean) / self._resnet_std

    def get_2D_image_features(self, batch_size, rgb_feat_init1, rgb_feat_init2, pos_encoding, dtype):
        # Get the 2D image features
        # if reshaped_image.shape[-1] != self.down_size:
        #     reshaped_image = F.interpolate(
        #         reshaped_image, (self.down_size, self.down_size), mode="bilinear", align_corners=True
        #     )
        rgb_feat0 = torch.cat([rgb_feat_init1[0], rgb_feat_init2[0]], dim=0).to(dtype) + self.feat0_token[0].to(dtype)
        rgb_feat1 = torch.cat([rgb_feat_init1[1], rgb_feat_init2[1]], dim=0).to(dtype) + self.feat1_token[0].to(dtype)
        rgb_feat0 = rgb_feat0.reshape(-1,*rgb_feat0.shape[1:])
        rgb_feat1 = rgb_feat1.reshape(-1,*rgb_feat1.shape[1:])
        rgb_feat1, _ = self.dec_blocks[0](rgb_feat1, rgb_feat0, pos_encoding, pos_encoding)
        rgb_feat = rgb_feat1.reshape(batch_size, -1, *rgb_feat1.shape[1:])
        # B, N, P, C = rgb_feat.shape
        # add embedding of 2D spaces
        # pos_embed = get_2d_sincos_pos_embed(C, pos_encoding).reshape(B, S, P, C)
        x = rgb_feat.reshape(-1, *rgb_feat1.shape[-2:])
        B, N, C = x.shape
        x = x.reshape(B, N, -1, 64)
        x = x.permute(0, 2, 1, 3)
        x = x + self.rope(torch.ones_like(x).to(x), pos_encoding).to(dtype)
        x = x.permute(0, 2, 1, 3)
        x = x.reshape(B, N, -1)
        rgb_feat = x.reshape(batch_size, -1, N, C)
        # register for pose
        B, S, P, C = rgb_feat.shape
        pose_token = self.pose_token.expand(B, S-1, -1, -1)
        pose_token = torch.cat((self.pose_token_ref.expand(B, 1, -1, -1), pose_token), dim=1).to(dtype)
        rgb_feat = torch.cat([pose_token, rgb_feat], dim=-2)
        B, S, P, C = rgb_feat.shape
        for idx in range(self.att_depth):
            # self attention
            rgb_feat = rearrange(rgb_feat, "b s p c -> (b s) p c", b=B, s=S)
            rgb_feat = self.self_att[idx](rgb_feat)
            rgb_feat = rearrange(rgb_feat, "(b s) p c -> b s p c", b=B, s=S)
            feat_0 = rgb_feat[:, 0]
            feat_others = rgb_feat[:, 1:]
            # cross attention
            feat_others = rearrange(feat_others, "b m p c -> b (m p) c", m=S - 1, p=P)
            feat_others = self.cross_att[idx](feat_others, feat_0)
            feat_others = rearrange(feat_others, "b (m p) c -> b m p c", m=S - 1, p=P)
            rgb_feat = torch.cat([rgb_feat[:, 0:1], feat_others], dim=1)

        rgb_feat = rgb_feat[:, :, 0]
        return rgb_feat, B, S, C
    

class CameraPredictor_light(nn.Module):
    def __init__(
        self,
        hood_idx,
        hidden_size=768,
        num_heads=8,
        mlp_ratio=4,
        down_size=336,
        att_depth=8,
        trunk_depth=4,
        pose_encoding_type="absT_quaR_logFL",
        cfg=None,
        rope=None
    ):
        super().__init__()
        self.cfg = cfg
        self.hood_idx = hood_idx
        self.att_depth = att_depth
        self.down_size = down_size
        self.pose_encoding_type = pose_encoding_type
        self.rope = rope
        if self.pose_encoding_type == "absT_quaR_OneFL":
            self.target_dim = 8
        if self.pose_encoding_type == "absT_quaR_logFL":
            self.target_dim = 9

        # self.backbone = self.get_backbone(backbone)

        # for param in self.backbone.parameters():
        #     param.requires_grad = False
        self.norm = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        # sine and cosine embed for camera parameters
        self.embed_pose = PoseEmbedding(
            target_dim=self.target_dim, n_harmonic_functions=(hidden_size // self.target_dim) // 2, append_input=True
        )
        self.pose_proj = nn.Linear(756 + 9, hidden_size)
        self.time_proj = nn.Linear(1, hidden_size)

        self.pose_token_ref = nn.Parameter(torch.zeros(1, 1, 1, hidden_size))  # register

        self.pose_branch = Mlp(
            in_features=hidden_size, hidden_features=hidden_size * 2, out_features=hidden_size + self.target_dim, drop=0
        )

        self.ffeat_updater = nn.Sequential(nn.Linear(hidden_size, hidden_size), nn.GELU())

        
        self.trunk = nn.Sequential(
            *[
                AttnBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio, attn_class=nn.MultiheadAttention)
                for _ in range(trunk_depth)
            ]
        )
        self.gamma = 0.8
        self.cam_token_encoder = nn.ModuleList([AttnBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio, attn_class=nn.MultiheadAttention)
                for _ in range(2)])
        
        nn.init.normal_(self.pose_token_ref, std=1e-6)
        self.hidden_size = hidden_size
        for name, value in (("_resnet_mean", _RESNET_MEAN), ("_resnet_std", _RESNET_STD)):
            self.register_buffer(name, torch.FloatTensor(value).view(1, 3, 1, 1), persistent=False)

    def forward(self, batch_size, iters=4, interm_feature1=None, interm_feature2=None, enabled=True, dtype=torch.bfloat16):
        """
        reshaped_image: Bx3xHxW. The values of reshaped_image are within [0, 1]
        preliminary_cameras: PyTorch3D cameras.

        TODO: dropping the usage of PyTorch3D cameras.
        """
        # if rgb_feat_init is None:
        #     # Get the 2D image features
        import ipdb; ipdb.set_trace()
        rgb_feat_init1 = [interm_feature1[i-1].reshape(batch_size, 1, self.hidden_size) for i in self.hood_idx[1:]]
        rgb_feat_init2 = [interm_feature2[i-1].reshape(batch_size, 1, self.hidden_size) for i in self.hood_idx[1:]]
        rgb_feat_init1 = torch.cat(rgb_feat_init1, dim=1)
        rgb_feat_init2 = torch.cat(rgb_feat_init2, dim=1)
        rgb_feat = torch.cat([rgb_feat_init1, rgb_feat_init2], dim=0).to(dtype)
        for cam_token_encoder in self.cam_token_encoder:
            rgb_feat = rgb_feat + cam_token_encoder(rgb_feat)
            
        rgb_feat = rgb_feat[:, 2:]
        rgb_feat = rgb_feat.reshape(batch_size, -1, rgb_feat.shape[-1])
        B, S, C = rgb_feat.shape
        pred_pose_enc = torch.zeros(B, S, self.target_dim).to(rgb_feat)
        rgb_feat_init = rgb_feat.clone()
        pred_cameras_list = []
        for iter_num in range(iters):
            pred_pose_enc = pred_pose_enc.detach()
            # Embed the camera parameters and add to rgb_feat
            pose_embed_time = self.time_proj(torch.tensor([iter_num]).to(rgb_feat))[None, None]
            pose_embed = self.embed_pose(pred_pose_enc)
            pose_embed = self.pose_proj(pose_embed)
            rgb_feat = rgb_feat + pose_embed + pose_embed_time
            rgb_feat[:,:1] = self.pose_token_ref[:, 0] + rgb_feat[:,:1]
            # Run trunk transformers on rgb_feat
            rgb_feat = self.trunk(rgb_feat)
            # Predict the delta feat and pose encoding at each iteration
            delta = self.pose_branch(rgb_feat)
            delta_pred_pose_enc = delta[..., : self.target_dim]
            delta_feat = delta[..., self.target_dim :]
            rgb_feat = self.ffeat_updater(self.norm(delta_feat)) + rgb_feat
            pred_pose_enc = pred_pose_enc + delta_pred_pose_enc
            # Residual connection
            rgb_feat = (rgb_feat + rgb_feat_init) / 2
            # Pose encoding to Cameras
            with torch.cuda.amp.autocast(enabled=False, dtype=torch.float32): 
                pred_cameras = pose_encoding_to_camera(pred_pose_enc.float(), pose_encoding_type='train')
                pred_cameras_list = pred_cameras_list + [pred_cameras]
        return pred_cameras_list, rgb_feat

    def get_backbone(self, backbone):
        """
        Load the backbone model.
        """
        if backbone == "dinov2s":
            return torch.hub.load("facebookresearch/dinov2", "dinov2_vits14_reg")
        elif backbone == "dinov2b":
            return torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14_reg")
        else:
            raise NotImplementedError(f"Backbone '{backbone}' not implemented")

    def _resnet_normalize_image(self, img: torch.Tensor) -> torch.Tensor:
        return (img - self._resnet_mean) / self._resnet_std

    

class CameraPredictor_clean(nn.Module):
    def __init__(
        self,
        hood_idx,
        hidden_size=768,
        num_heads=8,
        mlp_ratio=4,
        down_size=336,
        att_depth=8,
        trunk_depth=4,
        pose_encoding_type="absT_quaR_logFL",
        cfg=None,
        rope=None
    ):
        super().__init__()
        self.cfg = cfg
        self.hood_idx = hood_idx
        self.att_depth = att_depth
        self.down_size = down_size
        self.pose_encoding_type = pose_encoding_type
        self.rope = rope
        if self.pose_encoding_type == "absT_quaR_OneFL":
            self.target_dim = 8
        if self.pose_encoding_type == "absT_quaR_logFL":
            self.target_dim = 9
        self.norm = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        # sine and cosine embed for camera parameters
        self.embed_pose = PoseEmbedding(
            target_dim=self.target_dim, n_harmonic_functions=(hidden_size // self.target_dim) // 2, append_input=True
        )
        self.pose_proj = nn.Linear(756 + 9, hidden_size)
        self.pose_token_ref = nn.Parameter(torch.zeros(1, 1, 1, hidden_size))  # register
        self.pose_branch = Mlp(
            in_features=hidden_size, hidden_features=hidden_size * 2, out_features=hidden_size + self.target_dim, drop=0
        )
        self.ffeat_updater = nn.Sequential(nn.Linear(hidden_size, hidden_size), nn.GELU())
        self.trunk = nn.Sequential(
            *[
                AttnBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio, attn_class=nn.MultiheadAttention)
                for _ in range(trunk_depth)
            ]
        )
        self.gamma = 0.8
        nn.init.normal_(self.pose_token_ref, std=1e-6)
        for name, value in (("_resnet_mean", _RESNET_MEAN), ("_resnet_std", _RESNET_STD)):
            self.register_buffer(name, torch.FloatTensor(value).view(1, 3, 1, 1), persistent=False)

    def forward(self, batch_size, iters=4, interm_feature1=None, interm_feature2=None, enabled=True, dtype=torch.bfloat16):
        """
        reshaped_image: Bx3xHxW. The values of reshaped_image are within [0, 1]
        preliminary_cameras: PyTorch3D cameras.

        TODO: dropping the usage of PyTorch3D cameras.
        """
        # if rgb_feat_init is None:
        #     # Get the 2D image features
        rgb_feat_init1 = interm_feature1[-1].reshape(batch_size, -1, interm_feature1[-1].shape[-1])
        rgb_feat_init2 = interm_feature2[-1].reshape(batch_size, -1, interm_feature2[-1].shape[-1])
        rgb_feat = torch.cat([rgb_feat_init1, rgb_feat_init2], dim=1).to(dtype)
        B, S, C = rgb_feat.shape
        pred_pose_enc = torch.zeros(B, S, self.target_dim).to(rgb_feat)
        rgb_feat_init = rgb_feat.clone()
        pred_cameras_list = []
        for iter_num in range(iters):
            pred_pose_enc = pred_pose_enc.detach()
            # Embed the camera parameters and add to rgb_feat
            pose_embed = self.embed_pose(pred_pose_enc)
            pose_embed = self.pose_proj(pose_embed)
            rgb_feat = rgb_feat + pose_embed
            rgb_feat[:,:1] = self.pose_token_ref[:, 0] + rgb_feat[:,:1]
            # Run trunk transformers on rgb_feat
            # rgb_feat = self.trunk(rgb_feat)
            rgb_feat = checkpoint(self.trunk, rgb_feat)
            # Predict the delta feat and pose encoding at each iteration
            delta = self.pose_branch(rgb_feat)
            delta_pred_pose_enc = delta[..., : self.target_dim]
            delta_feat = delta[..., self.target_dim :]
            rgb_feat = self.ffeat_updater(self.norm(delta_feat)) + rgb_feat
            pred_pose_enc = pred_pose_enc + delta_pred_pose_enc
            # Residual connection
            rgb_feat = (rgb_feat + rgb_feat_init) / 2
            # Pose encoding to Cameras
            with torch.cuda.amp.autocast(enabled=False, dtype=torch.float32): 
                pred_cameras = pose_encoding_to_camera(pred_pose_enc.float(), pose_encoding_type='train')
                pred_cameras_list = pred_cameras_list + [pred_cameras]
        return pred_cameras_list, rgb_feat

    def get_backbone(self, backbone):
        """
        Load the backbone model.
        """
        if backbone == "dinov2s":
            return torch.hub.load("facebookresearch/dinov2", "dinov2_vits14_reg")
        elif backbone == "dinov2b":
            return torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14_reg")
        else:
            raise NotImplementedError(f"Backbone '{backbone}' not implemented")

    def _resnet_normalize_image(self, img: torch.Tensor) -> torch.Tensor:
        return (img - self._resnet_mean) / self._resnet_std

