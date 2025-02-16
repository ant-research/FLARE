# Copyright (C) 2024-present Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).
#
# --------------------------------------------------------
# MASt3R heads
# --------------------------------------------------------
import torch
import torch.nn.functional as F

import mast3r.utils.path_to_dust3r  # noqa
from dust3r.heads.postprocess import reg_dense_depth, reg_dense_conf  # noqa
from dust3r.heads.dpt_head import PixelwiseTaskWithDPT  # noqa
import dust3r.utils.path_to_croco  # noqa
from models.blocks import Mlp  # noqa
import torch.nn as nn 


def reg_desc(desc, mode):
    if 'norm' in mode:
        desc = desc / desc.norm(dim=-1, keepdim=True)
    else:
        raise ValueError(f"Unknown desc mode {mode}")
    return desc


def postprocess(out, depth_mode, conf_mode, desc_dim=None, desc_mode='norm', two_confs=False, desc_conf_mode=None):
    if desc_conf_mode is None:
        desc_conf_mode = conf_mode
    fmap = out.permute(0, 2, 3, 1)  # B,H,W,D
    res = dict(pts3d=reg_dense_depth(fmap[..., 0:3], mode=depth_mode))
    if conf_mode is not None:
        res['conf'] = reg_dense_conf(fmap[..., 3], mode=conf_mode)
    if desc_dim is not None:
        start = 3 + int(conf_mode is not None)
        res['desc'] = fmap[..., start:]
        # if two_confs:
        #     res['desc_conf'] = reg_dense_conf(fmap[..., start + desc_dim], mode=desc_conf_mode)
        # else:
        #     res['desc_conf'] = res['conf'].clone()
    return res


class Cat_MLP_LocalFeatures_DPT_Pts3d(PixelwiseTaskWithDPT):
    """ Mixture between MLP and DPT head that outputs 3d points and local features (with MLP).
    The input for both heads is a concatenation of Encoder and Decoder outputs
    """

    def __init__(self, net, has_conf=False, local_feat_dim=16, hidden_dim_factor=4., hooks_idx=None, dim_tokens=None,
                 num_channels=1, postprocess=None, feature_dim=256, last_dim=32, depth_mode=None, conf_mode=None, head_type="regression", **kwargs):
        super().__init__(num_channels=num_channels, feature_dim=feature_dim, last_dim=last_dim, hooks_idx=hooks_idx,
                         dim_tokens=dim_tokens, depth_mode=depth_mode, postprocess=postprocess, conf_mode=conf_mode, head_type=head_type)
        self.local_feat_dim = local_feat_dim

        patch_size = net.patch_embed.patch_size
        if isinstance(patch_size, tuple):
            assert len(patch_size) == 2 and isinstance(patch_size[0], int) and isinstance(
                patch_size[1], int), "What is your patchsize format? Expected a single int or a tuple of two ints."
            assert patch_size[0] == patch_size[1], "Error, non square patches not managed"
            patch_size = patch_size[0]
        self.patch_size = patch_size

        self.desc_mode = net.desc_mode
        self.has_conf = has_conf
        self.two_confs = net.two_confs  # independent confs for 3D regr and descs
        self.desc_conf_mode = net.desc_conf_mode
        idim = net.enc_embed_dim + net.dec_embed_dim

        self.head_local_features = Mlp(in_features=idim,
                                       hidden_features=int(hidden_dim_factor * idim),
                                       out_features=(self.local_feat_dim + self.two_confs) * self.patch_size**2)

    def forward(self, decout, img_shape):
        # pass through the heads
        pts3d = self.dpt(decout, image_size=(img_shape[0], img_shape[1]))

        # recover encoder and decoder outputs
        enc_output, dec_output = decout[0], decout[-1]
        cat_output = torch.cat([enc_output, dec_output], dim=-1)  # concatenate
        H, W = img_shape
        B, S, D = cat_output.shape

        # extract local_features
        local_features = self.head_local_features(cat_output)  # B,S,D
        local_features = local_features.transpose(-1, -2).view(B, -1, H // self.patch_size, W // self.patch_size)
        local_features = F.pixel_shuffle(local_features, self.patch_size)  # B,d,H,W

        # post process 3D pts, descriptors and confidences
        out = torch.cat([pts3d, local_features], dim=1)
        if self.postprocess:
            out = self.postprocess(out,
                                   depth_mode=self.depth_mode,
                                   conf_mode=self.conf_mode,
                                   desc_dim=self.local_feat_dim,
                                   desc_mode=self.desc_mode,
                                   two_confs=self.two_confs,
                                   desc_conf_mode=self.desc_conf_mode)
        # out.update({'local_token': local_token})
        return out


class DPT_depth(PixelwiseTaskWithDPT):
    """ Mixture between MLP and DPT head that outputs 3d points and local features (with MLP).
    The input for both heads is a concatenation of Encoder and Decoder outputs
    """

    def __init__(self, net, has_conf=False, hidden_dim_factor=4., hooks_idx=None, dim_tokens=None,
                 num_channels=1, postprocess=None, feature_dim=256, last_dim=32, depth_mode=None, conf_mode=None, head_type="regression", **kwargs):
        super().__init__(num_channels=num_channels, feature_dim=feature_dim, last_dim=last_dim, hooks_idx=hooks_idx,
                         dim_tokens=dim_tokens, depth_mode=depth_mode, postprocess=postprocess, conf_mode=conf_mode, head_type=head_type)

        patch_size = net.patch_embed.patch_size
        if isinstance(patch_size, tuple):
            assert len(patch_size) == 2 and isinstance(patch_size[0], int) and isinstance(
                patch_size[1], int), "What is your patchsize format? Expected a single int or a tuple of two ints."
            assert patch_size[0] == patch_size[1], "Error, non square patches not managed"
            patch_size = patch_size[0]
        self.patch_size = patch_size

        self.desc_mode = net.desc_mode
        self.has_conf = has_conf
        self.two_confs = net.two_confs  # independent confs for 3D regr and descs
        self.desc_conf_mode = net.desc_conf_mode
        idim = net.enc_embed_dim + net.dec_embed_dim
        # self.conf_mode = conf_mode

    def forward(self, decout, img_shape):
        # pass through the heads
        pts3d = self.dpt(decout, image_size=(img_shape[0], img_shape[1]))
        out = pts3d
        # post process 3D pts, descriptors and confidences
        # out = torch.cat([pts3d, local_features], dim=1)
        fmap = out.permute(0, 2, 3, 1)  # B,H,W,3
        res = {}
        res['depth'] = torch.exp(fmap[...,:1]-1).clamp(0.0001, 1000.)
        # res['depth_scaling'] = fmap[...,1:4]
        res['depth_conf'] = reg_dense_conf(fmap[..., -1:], mode=self.conf_mode)
        res['desc'] = fmap[..., 1:]
        # out.update({'local_token': local_token})
        return res


class Cat_GS_LocalFeatures_DPT_Pts3d(PixelwiseTaskWithDPT):
    """ Mixture between MLP and DPT head that outputs 3d points and local features (with MLP).
    The input for both heads is a concatenation of Encoder and Decoder outputs
    """

    def __init__(self, net, has_conf=False, local_feat_dim=16, hidden_dim_factor=4., hooks_idx=None, dim_tokens=None,
                 num_channels=1, postprocess=None, feature_dim=256, last_dim=32, depth_mode=None, conf_mode=None, head_type="regression", **kwargs):
        super().__init__(num_channels=num_channels, feature_dim=feature_dim, last_dim=last_dim, hooks_idx=hooks_idx,
                         dim_tokens=dim_tokens, depth_mode=depth_mode, postprocess=postprocess, conf_mode=conf_mode, head_type=head_type)
        self.local_feat_dim = local_feat_dim

        patch_size = net.patch_embed.patch_size
        if isinstance(patch_size, tuple):
            assert len(patch_size) == 2 and isinstance(patch_size[0], int) and isinstance(
                patch_size[1], int), "What is your patchsize format? Expected a single int or a tuple of two ints."
            assert patch_size[0] == patch_size[1], "Error, non square patches not managed"
            patch_size = patch_size[0]
        self.patch_size = patch_size

        self.desc_mode = net.desc_mode
        self.has_conf = has_conf
        self.two_confs = net.two_confs  # independent confs for 3D regr and descs
        self.desc_conf_mode = net.desc_conf_mode
        idim = net.enc_embed_dim + net.dec_embed_dim

    def forward(self, decout, img_shape):
        # pass through the heads
        out = self.dpt(decout, image_size=(img_shape[0], img_shape[1]))

        # recover encoder and decoder outputs
        # enc_output, dec_output = decout[0], decout[-1]
        # cat_output = torch.cat([enc_output, dec_output], dim=-1)  # concatenate
        # H, W = img_shape
        # B, S, D = cat_output.shape
        # post process 3D pts, descriptors and confidences
        # out = torch.cat([pts3d, local_features], dim=1)
        if self.postprocess:
            out = self.postprocess(out,
                                   depth_mode=self.depth_mode,
                                   conf_mode=self.conf_mode,
                                   desc_dim=self.local_feat_dim,
                                   desc_mode=self.desc_mode,
                                   two_confs=self.two_confs,
                                   desc_conf_mode=self.desc_conf_mode)
        # out.update({'local_token': local_token})
        return out

class UNet(nn.Module):
    def __init__(self, in_channels, out_channels, hidden_dim):
        super(UNet, self).__init__()
        # 编码器
        self.enc1 = self.conv_block(in_channels, hidden_dim)
        # self.downsample = nn.Conv2d(hidden_dim, hidden_dim * 2, kernel_size=2, stride=2)  # 下采样
        # 解码器
        # self.dec1 = self.upconv_block(hidden_dim * 2, hidden_dim)
        self.dec2 = nn.Conv2d(hidden_dim, out_channels, kernel_size=3, padding=1)

    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.GELU()
        )

    def upconv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2),
            nn.GELU()
        )

    def forward(self, x):
        # 编码
        enc1 = self.enc1(x)
        dec2 = self.dec2(enc1)
        return dec2

class gs_head_heavy(nn.Module):
    def __init__(self,
                 feature_dim, 
                 last_dim,
                 high_feature,
                 sh_degree = 2,
                 ):
        super().__init__()
        self.high_feature = high_feature
        self.high_feature_fusion = UNet(high_feature, high_feature, high_feature)
        sh_degree = sh_degree
        self.feat_sh = nn.Sequential(
                nn.Conv2d(feature_dim, last_dim, kernel_size=3, stride=1, padding=1),
                nn.GELU(),
            )
        self.color = nn.Conv2d(last_dim, 3, kernel_size=1, stride=1, padding=0)
        self.sh_high_fre = nn.Conv2d(last_dim, (sh_degree + 1) ** 2 * 3 - 3, kernel_size=1, stride=1, padding=0)
        self.feat_opacity = nn.Sequential(
                nn.Conv2d(feature_dim, last_dim, kernel_size=3, stride=1, padding=1),
                nn.GELU(),
                nn.Conv2d(last_dim, 1, kernel_size=1, stride=1, padding=0)
            )
        self.feat_scaling = nn.Sequential(
                nn.Conv2d(high_feature, last_dim, kernel_size=3, stride=1, padding=1),
                nn.GELU(),
                nn.Conv2d(last_dim, 3, kernel_size=1, stride=1, padding=0)
            )
        
        self.feat_rotation = nn.Sequential(
                nn.Conv2d(high_feature, last_dim, kernel_size=3, stride=1, padding=1),
                nn.GELU(),
                nn.Conv2d(last_dim, 4, kernel_size=1, stride=1, padding=0)
            )
        self.feat_scaling[-1].weight.data.normal_(mean=0, std=0.02)
        self.feat_opacity[-1].weight.data.normal_(mean=0, std=0.02)
        self.color.weight.data.normal_(mean=0, std=0.88)
        self.sh_high_fre.weight.data.normal_(mean=0, std=0.02)

    def forward(self, x, true_shape):
        # H, W = x.shape[-2:]
        # if H != H_org or W != W_org:
        #     x = x.permute(0, 1, 3, 2)
        x = x[0]
        x = x.permute(0,3,1,2)  # B,H,W,D
        assert x.shape[-1] == true_shape[-1]
        high_feature =  self.high_feature_fusion(x[:, :self.high_feature])
        fusion_feature = torch.cat([high_feature, x[:, self.high_feature:]], dim=1)
        feat_opacity = self.feat_opacity(fusion_feature)
        feat_scaling = self.feat_scaling(high_feature)
        feat_rotation = self.feat_rotation(high_feature)
        featuresh = self.feat_sh(fusion_feature)
        feat_color = self.color(featuresh)
        feat_sh = self.sh_high_fre(featuresh)
        feat_feature = torch.cat([feat_color, feat_sh], dim=1)
        feat_feature = feat_feature.permute(0, 2, 3, 1)  # B,H,W,3
        feat_opacity = feat_opacity - 2
        feat_opacity = feat_opacity.permute(0, 2, 3, 1)  # B,H,W,1
        feat_scaling = feat_scaling.permute(0, 2, 3, 1)  # B,H,W,1
        feat_rotation = feat_rotation.permute(0, 2, 3, 1)  # B,H,W,1
        res = dict(feature=feat_feature, opacity=feat_opacity, scaling=feat_scaling, rotation=feat_rotation)
        return res

class gs_head(nn.Module):
    def __init__(self,
                 feature_dim, 
                 last_dim,
                 high_feature,
                 ):
        super().__init__()
        self.high_feature = high_feature
        self.feat_feature = nn.Sequential(
                nn.Conv2d(feature_dim, last_dim, kernel_size=3, stride=1, padding=1),
                nn.GELU(),
                nn.Conv2d(last_dim, 3, kernel_size=1, stride=1, padding=0)
            )
        self.feat_opacity = nn.Sequential(
                nn.Conv2d(feature_dim, last_dim, kernel_size=3, stride=1, padding=1),
                nn.GELU(),
                nn.Conv2d(last_dim, 1, kernel_size=1, stride=1, padding=0)
            )
        self.feat_scaling = nn.Sequential(
                nn.Conv2d(high_feature, last_dim, kernel_size=3, stride=1, padding=1),
                nn.GELU(),
                nn.Conv2d(last_dim, 3, kernel_size=1, stride=1, padding=0)
            )
        
        self.feat_rotation = nn.Sequential(
                nn.Conv2d(high_feature, last_dim, kernel_size=3, stride=1, padding=1),
                nn.GELU(),
                nn.Conv2d(last_dim, 4, kernel_size=1, stride=1, padding=0)
            )
        self.feat_scaling[-1].weight.data.normal_(mean=0, std=0.02)
        self.feat_opacity[-1].weight.data.normal_(mean=0, std=0.02)
        self.feat_feature[-1].weight.data.normal_(mean=0, std=0.5)

    def forward(self, x, true_shape):
        # H, W = x.shape[-2:]
        # if H != H_org or W != W_org:
        #     x = x.permute(0, 1, 3, 2)
        x = x[0]
        x = x.permute(0,3,1,2)  # B,H,W,D
        assert x.shape[-1] == true_shape[-1]
        feat_opacity = self.feat_opacity(x)
        feat_scaling = self.feat_scaling(x[:, :self.high_feature])
        feat_rotation = self.feat_rotation(x[:, :self.high_feature])
        feat_feature = self.feat_feature(x)
        feat_feature = feat_feature.permute(0, 2, 3, 1)  # B,H,W,3
        feat_opacity = feat_opacity - 2
        feat_opacity = feat_opacity.permute(0, 2, 3, 1)  # B,H,W,1
        feat_scaling = feat_scaling.permute(0, 2, 3, 1)  # B,H,W,1
        feat_rotation = feat_rotation.permute(0, 2, 3, 1)  # B,H,W,1
        res = dict(feature=feat_feature, opacity=feat_opacity, scaling=feat_scaling, rotation=feat_rotation)
        return res

def mast3r_head_factory(head_type, output_mode, net, has_conf=False, sh_degree=2):
    """" build a prediction head for the decoder 
    """
    if head_type == 'catmlp+dpt' and output_mode.startswith('pts3d+desc'):
        local_feat_dim = int(output_mode[10:])
        assert net.dec_depth > 9
        l2 = net.dec_depth
        feature_dim = 256
        last_dim = feature_dim // 2
        out_nchan = 3
        ed = net.enc_embed_dim
        dd = net.dec_embed_dim
        return Cat_MLP_LocalFeatures_DPT_Pts3d(net, local_feat_dim=local_feat_dim, has_conf=has_conf,
                                               num_channels=out_nchan + has_conf,
                                               feature_dim=feature_dim,
                                               last_dim=last_dim,
                                               hooks_idx=[0, l2 * 2 // 4, l2 * 3 // 4, l2],
                                               dim_tokens=[ed, dd, dd, dd],
                                               postprocess=postprocess,
                                               depth_mode=net.depth_mode,
                                               conf_mode=net.conf_mode,
                                               head_type='regression')

    elif output_mode=='depth_conf_scaling':
        local_feat_dim = 24
        assert net.dec_depth > 9
        l2 = net.dec_depth
        feature_dim = 256
        last_dim = feature_dim // 2
        out_nchan = 1
        ed = net.enc_embed_dim
        dd = net.dec_embed_dim
        return DPT_depth(net,  has_conf=has_conf,
                        num_channels=out_nchan + local_feat_dim + net.two_confs,
                        feature_dim=feature_dim,
                        last_dim=last_dim,
                        hooks_idx=[0, l2 * 2 // 4, l2 * 3 // 4, l2],
                        dim_tokens=[ed, dd, dd, dd],
                        postprocess=postprocess,
                        depth_mode=net.depth_mode,
                        conf_mode=net.conf_mode,
                        head_type='regression')

    elif head_type == 'dpt_gs' and output_mode.startswith('pts3d+desc'):
        local_feat_dim = int(output_mode[10:])
        assert net.dec_depth > 9
        l2 = net.dec_depth
        feature_dim = 256
        last_dim = feature_dim // 2
        out_nchan = 3
        ed = net.enc_embed_dim
        dd = net.dec_embed_dim
        return Cat_GS_LocalFeatures_DPT_Pts3d(net, has_conf=has_conf,
                                               num_channels=out_nchan + has_conf + local_feat_dim + net.two_confs,
                                               feature_dim=feature_dim,
                                               last_dim=last_dim,
                                               hooks_idx=[0, l2 * 2 // 4, l2 * 3 // 4, l2],
                                               dim_tokens=[ed, dd, dd, dd],
                                               postprocess=postprocess,
                                               depth_mode=net.depth_mode,
                                               conf_mode=net.conf_mode,
                                               head_type='regression')
    elif head_type == 'gs':
        local_feat_dim = int(output_mode[10:]) + 1 + 16
        return gs_head(feature_dim=local_feat_dim, last_dim=local_feat_dim//2, high_feature=int(output_mode[10:]) + 1)
    elif head_type == 'sh':
        local_feat_dim = int(output_mode[10:]) + 1 + 16
        return gs_head_heavy(feature_dim=local_feat_dim, last_dim=local_feat_dim//2, high_feature=int(output_mode[10:]) + 1, sh_degree=sh_degree)
    else:
        raise NotImplementedError(
            f"unexpected {head_type=} and {output_mode=}")
