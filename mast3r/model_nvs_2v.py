# Copyright (C) 2024-present Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).
#
# --------------------------------------------------------
# MASt3R model class
# --------------------------------------------------------
import torch
import torch.nn.functional as F
import os
from dust3r.utils.geometry import inv, geotrf, normalize_pointcloud
from mast3r.catmlp_dpt_head import mast3r_head_factory
from mast3r.vgg_pose_head import CameraPredictor, CameraPredictor_clean, Mlp
from mast3r.shallow_cnn import FeatureNet
import mast3r.utils.path_to_dust3r  # noqa
from dust3r.model import AsymmetricCroCo3DStereo  # noqa
from dust3r.utils.misc import transpose_to_landscape, freeze_all_params  # noqa
import torch.nn as nn
import torchvision.models as tvm
inf = float('inf')
import torchvision.transforms.functional as functional
from dust3r.patch_embed import get_patch_embed
from torch.utils.checkpoint import checkpoint
from pytorch3d.transforms.rotation_conversions import matrix_to_quaternion, quaternion_to_matrix
import copy

def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)

def load_model(model_path, device, verbose=True):
    if verbose:
        print('... loading model from', model_path)
    ckpt = torch.load(model_path, map_location='cpu')
    args = ckpt['args'].model.replace("ManyAR_PatchEmbed", "PatchEmbedDust3R")
    if 'landscape_only' not in args:
        args = args[:-1] + ', landscape_only=False)'
    else:
        args = args.replace(" ", "").replace('landscape_only=True', 'landscape_only=False')
    assert "landscape_only=False" in args
    if verbose:
        print(f"instantiating : {args}")
    net = eval(args)
    s = net.load_state_dict(ckpt['model'], strict=False)
    if verbose:
        print(s)
    return net.to(device)
import torch

class AsymmetricMASt3R(AsymmetricCroCo3DStereo):
    def __init__(self, desc_mode=('norm'), two_confs=False, desc_conf_mode=None, **kwargs):
        self.desc_mode = desc_mode
        self.two_confs = two_confs
        self.desc_conf_mode = desc_conf_mode
        super().__init__(**kwargs)
        self.dec_blocks_point = copy.deepcopy(self.dec_blocks_fine)
        self.cam_cond_encoder_fine = copy.deepcopy(self.cam_cond_encoder)
        self.cam_cond_encoder_point = copy.deepcopy(self.cam_cond_encoder)
        self.adaLN_modulation = nn.ModuleList([nn.Sequential(
            nn.SiLU(inplace=False),
            nn.Linear(self.dec_embed_dim, 3 * self.dec_embed_dim, bias=True)
        ) for _ in range(len(self.dec_blocks_fine))])
        self.enc_norm_coarse = copy.deepcopy(self.enc_norm)
        for block in self.adaLN_modulation:
            nn.init.constant_(block[-1].weight, 0)
            nn.init.constant_(block[-1].bias, 0)
        self.decoder_embed_fine = copy.deepcopy(self.decoder_embed)
        self.decoder_embed_point = copy.deepcopy(self.decoder_embed)
        self.enc_norm_coarse = copy.deepcopy(self.enc_norm)
        self.embed_pose = Mlp(7, hidden_features=self.dec_embed_dim, out_features=self.dec_embed_dim)
        self.cnn_wobn = FeatureNet()
        self.cnn_proj = nn.Conv2d(64, 16, 3, 1, 1)
        self.cnn_fusion = nn.Conv2d(32*3, 64, 3, 1, 1)
        self.dec_cam_norm_fine = copy.deepcopy(self.dec_cam_norm)
        self.dec_norm_fine = copy.deepcopy(self.dec_norm)
        self.dec_norm_point = copy.deepcopy(self.dec_norm)
        self.pose_token_ref_fine = copy.deepcopy(self.pose_token_ref)
        self.pose_token_ref_point = copy.deepcopy(self.pose_token_ref)
        self.pose_token_source_fine = copy.deepcopy(self.pose_token_source)
        self.pose_token_source_point = copy.deepcopy(self.pose_token_source)
        self.cam_cond_embed_fine = copy.deepcopy(self.cam_cond_embed)
        self.cam_cond_embed_point = copy.deepcopy(self.cam_cond_embed)
        self.cam_cond_embed_point_pre = copy.deepcopy(self.cam_cond_embed)
        self.inject_stage2 = nn.ModuleList([nn.Linear(self.enc_embed_dim, self.dec_embed_dim, bias=False) for i in range(3)])
        self.inject_stage3 = nn.ModuleList([nn.Linear(self.enc_embed_dim, self.dec_embed_dim, bias=False) for i in range(3)])
        self.enc_inject_stage2  = nn.ModuleList([copy.deepcopy(self.enc_norm) for i in range(3)])
        self.enc_inject_stage3  = nn.ModuleList([copy.deepcopy(self.enc_norm) for i in range(3)])
        for i in range(3):
            nn.init.constant_(self.inject_stage2[i].weight, 0.)
            nn.init.constant_(self.inject_stage3[i].weight, 0.)
        self.idx_hook = [2, 5, 8]
        self.encode_feature_landscape = transpose_to_landscape(self.encode_feature, activate=True)
        self.decoder_embed_stage2 = copy.deepcopy(self.decoder_embed)
        nn.init.constant_(self.decoder_embed_stage2.weight, 0.)
        self.decoder_embed_fxfycxcy = Mlp(4, hidden_features=self.dec_embed_dim, out_features=self.dec_embed_dim)
        nn.init.constant_(self.decoder_embed_fxfycxcy.fc2.weight, 0.)
        nn.init.constant_(self.decoder_embed_fxfycxcy.fc2.bias, 0.)

    def load_state_dict(self, ckpt, **kw):
        # duplicate all weights for the second decoder if not present
        new_ckpt = dict(ckpt)
        if self.head_type == 'dpt_gs':
            for key, value in ckpt.items():
                if 'dpt.head.4' in key:
                    state_dict = self.state_dict()
                    state_dict[key][:value.shape[0]] = value
                    new_ckpt[key] = state_dict[key]
                    
        for key, value in ckpt.items():
            if 'sh_high_fre' in key:
                state_dict = self.state_dict()
                state_dict[key][:value.shape[0]] = value
                state_dict[key][value.shape[0]:] = 0
                new_ckpt[key] = state_dict[key]
        return super().load_state_dict(new_ckpt, **kw)

    def encode_feature(self, imgs_vgg, image_size):
        H, W = image_size
        imgs_vgg = imgs_vgg[0].permute(0,3,1,2)
        feat_vgg3, feat_vgg2, feat_vgg1 = self.cnn_wobn(imgs_vgg)
        feat_vgg2 = F.interpolate(feat_vgg2.float(), (H, W), mode='bilinear', align_corners=True)
        feat_vgg3 = F.interpolate(feat_vgg3.float(), (H, W), mode='bilinear', align_corners=True)
        feat_vgg = self.cnn_fusion(torch.cat((feat_vgg1.float(), feat_vgg2, feat_vgg3), 1))
        feat_vgg_detail = self.cnn_proj(feat_vgg)
        N, C, h, w = feat_vgg.shape
        imgs_vgg = feat_vgg.reshape(N, C, -1).permute(0,2,1)
        N, P, C = imgs_vgg.shape
        imgs_vgg = imgs_vgg.reshape(N, P, -1, 64)
        imgs_vgg = imgs_vgg.permute(0, 2, 1, 3)
        x = torch.arange(w).to(imgs_vgg)
        y = torch.arange(h).to(imgs_vgg)
        xy = torch.meshgrid(x, y, indexing='xy')
        pos_full = torch.cat((xy[0].unsqueeze(-1), xy[1].unsqueeze(-1)), -1).unsqueeze(0)
        imgs_vgg = imgs_vgg + self.rope(torch.ones_like(imgs_vgg).to(imgs_vgg), pos_full.reshape(1,-1,2).repeat(N, 1, 1).long()).to(imgs_vgg)
        imgs_vgg = imgs_vgg.permute(0, 2, 1, 3)
        imgs_vgg = imgs_vgg.reshape(N, -1, C).permute(0, 2, 1)
        imgs_vgg = imgs_vgg.reshape(N, C, h, w)
        return {'imgs_vgg': imgs_vgg.permute(0, 2, 3, 1), 'feat_vgg_detail': feat_vgg_detail.permute(0, 2, 3, 1)}

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kw):
        if os.path.isfile(pretrained_model_name_or_path):
            return load_model(pretrained_model_name_or_path, device='cpu')
        else:
            return super(AsymmetricMASt3R, cls).from_pretrained(pretrained_model_name_or_path, **kw)

    def _encode_image(self, image, true_shape):
        # embed the image into patches  (x has size B x Npatches x C)
        interm_features = []
        x, pos = self.patch_embed(image, true_shape=true_shape)
        # add positional embedding without cls token
        assert self.enc_pos_embed is None
        # now apply the transformer encoder and normalization
        for blk in self.enc_blocks:
            interm_features.append(x)
            x = blk(x, pos)
        x = self.enc_norm(x)
        return x, pos, interm_features
        
    def _encode_symmetrized(self, views):
        imgs = [view['img'] for view in views]
        shapes = [view['true_shape'] for view in views]
        imgs  = torch.stack((imgs), dim=1)
        B, views, _, H, W = imgs.shape
        dtype = imgs.dtype
        imgs = imgs.view(-1, *imgs.shape[2:])
        shapes = torch.stack((shapes), dim=1)
        shapes = shapes.view(-1, *shapes.shape[2:])
        out, pos, interm_features = self._encode_image(imgs, shapes)
        out = out.to(dtype)
        for i in range(len(interm_features)):
            interm_features[i] = interm_features[i].to(dtype)
            interm_features[i] = interm_features[i].reshape(B, views, *out.shape[1:])
        true_shape = shapes
        W //= 64
        H //= 64
        n_tokens = H * W
        x_coarse = out.new_zeros((B*views, n_tokens, self.patch_embed_coarse2.embed_dim)).to(dtype)
        pos_coarse = out.new_zeros((B*views, n_tokens, 2), dtype=torch.int64)
        height, width = true_shape.T
        is_landscape = (width >= height)
        is_portrait = ~is_landscape
        fine_token = out.view(B*views, H * 4, W * 4, -1).permute(0, 3, 1, 2)
        x_coarse[is_landscape] = self.patch_embed_coarse2.proj(fine_token[is_landscape]).permute(0, 2, 3, 1).flatten(1, 2)
        x_coarse[is_portrait] = self.patch_embed_coarse2.proj(fine_token[is_portrait].swapaxes(-1, -2)).permute(0, 2, 3, 1).flatten(1, 2)
        pos_coarse[is_landscape] = self.patch_embed_test_.position_getter(1, H, W, pos.device)
        pos_coarse[is_portrait] = self.patch_embed_test_.position_getter(1, W, H, pos.device)
        x_coarse = self.enc_norm_coarse(x_coarse)
        out_coarse = x_coarse.reshape(B, views, *x_coarse.shape[1:]).to(dtype)
        pos_coarse = pos_coarse.reshape(B, views, *pos_coarse.shape[1:])
        shapes_coarse = shapes.reshape(B, views, *shapes.shape[1:]) // 4
        out = out.reshape(B, views, *out.shape[1:])
        pos = pos.reshape(B, views, *pos.shape[1:])
        shapes = shapes.reshape(B, views, *shapes.shape[1:])
        return shapes_coarse, out_coarse, pos_coarse, shapes, out, pos, interm_features
    
    def _set_patch_embed(self, img_size=224, patch_size=16, enc_embed_dim=768):
        self.patch_embed = get_patch_embed(self.patch_embed_cls, img_size, patch_size, enc_embed_dim)
        self.patch_embed_coarse2 = get_patch_embed(self.patch_embed_cls, img_size, 4, enc_embed_dim, input_dim=enc_embed_dim)
        self.patch_embed_test_ = get_patch_embed(self.patch_embed_cls, img_size, 4 * patch_size, enc_embed_dim)
        self.patch_embed_fine = get_patch_embed(self.patch_embed_cls, img_size, patch_size, enc_embed_dim, input_dim=64)

    def set_downstream_head(self, output_mode, head_type, landscape_only, depth_mode, conf_mode, patch_size, img_size, **kw):
        assert img_size[0] % patch_size == 0 and img_size[
            1] % patch_size == 0, f'{img_size=} must be multiple of {patch_size=}'
        self.output_mode = output_mode
        self.head_type = head_type
        self.depth_mode = depth_mode
        self.conf_mode = conf_mode
        if self.desc_conf_mode is None:
            self.desc_conf_mode = conf_mode
        # allocate heads
        self.downstream_head1 = mast3r_head_factory(head_type, output_mode, self, has_conf=bool(conf_mode))
        self.downstream_head2 = mast3r_head_factory(head_type, output_mode, self, has_conf=bool(conf_mode))
        # magic wrapper
        self.head1 = transpose_to_landscape(self.downstream_head1, activate=landscape_only)
        self.head2 = transpose_to_landscape(self.downstream_head2, activate=landscape_only)
        self.pose_head = CameraPredictor_clean(hood_idx=self.downstream_head2.dpt.hooks, trunk_depth=4, rope=self.rope)
        self.pose_head_stage2 = CameraPredictor_clean(hood_idx=self.downstream_head2.dpt.hooks, trunk_depth=4, rope=self.rope)
        self.downstream_head4 = mast3r_head_factory('sh', output_mode, self, has_conf=bool(conf_mode), sh_degree=4)
        self.head4 = transpose_to_landscape(self.downstream_head4, activate=landscape_only)

    def _encode_image_fine(self, imgs_vgg, true_shape, dtype):
        x, pos = self.patch_embed_fine(imgs_vgg, true_shape=true_shape)
        x = x.to(dtype)
        # add positional embedding without cls token
        # now apply the transformer encoder and normalization
        for blk in self.enc_blocks_stage2:
            x = blk(x, pos)
        x = self.enc_norm_stage2(x)
        x = x.to(dtype)
        return x, pos, None
  
    def _decoder_stage2(self, f1, pos1, f2, pos2, pose1, pose2, low_token=None):
        f = torch.cat((f1, f2), 1)
        pos = torch.cat((pos1, pos2), 1)
        final_output = [f]  # before projection
        # project to decoder dim
        f = self.decoder_embed_fine(f)
        B, views, P, C = f.shape
        f = f.view(B, -1 ,C)
        pos = pos.view(B, -1, pos.shape[-1])
        cam_tokens = []
        final_output.append(f)
        pose1_embed = self.embed_pose(pose1)
        pose2_embed = self.embed_pose(pose2)
        pose_embed = torch.cat((pose1_embed, pose2_embed), 1)
        views = views - 1
        pose_token_ref, pose_token_source = self.pose_token_ref_fine.to(f1.dtype).repeat(B,1,1).view(B, -1, C), self.pose_token_source_fine.to(f1.dtype).repeat(B*views,1,1).view(B*views, -1, C)
        dtype = f.dtype
        hook_idx = 0
        for i, (blk1, cam_cond, cam_cond_embed_fine, adaLN_modulation) in enumerate(zip(self.dec_blocks_fine, self.cam_cond_encoder_fine, self.cam_cond_embed_fine, self.adaLN_modulation)):
            shift_msa, scale_msa, gate_msa = adaLN_modulation(pose_embed).chunk(3, dim=-1)
            pose_token_ref = modulate(pose_token_ref.reshape(B, -1, C), shift_msa[:,:1].reshape(B,-1), scale_msa[:,:1].reshape(B,-1))
            pose_token_source =  modulate(pose_token_source.reshape(B*views, -1, C), shift_msa[:,1:].reshape(B*views,-1), scale_msa[:,1:].reshape(B*views,-1))
            feat = checkpoint(blk1, f, pos)
            feat = feat.view(B, views+1, -1, C)
            f1 = feat[:,:1].view(B, -1, C)
            f2 = feat[:,1:].reshape(B*views, -1, C)
            f1_cam = torch.cat((pose_token_ref, f1.view(B, -1, C)), 1)
            f2_cam = torch.cat((pose_token_source, f2.view(B*views, -1, C)), 1)
            f_cam = torch.cat((f1_cam, f2_cam), 0)
            f_cam = checkpoint(cam_cond, f_cam) # torch.Size([64, 769, 768])
            f_delta = f_cam[:,1:]
            f_cam = f_cam[:,:1]
            f_delta1 = f_delta[:B].view(B, -1, C)
            f_delta2 = f_delta[B:].view(B*views, -1, C)
            pose_token_ref = pose_token_ref.view(B, -1, C) + f_cam[:B].view(B, -1, C)
            pose_token_source = pose_token_source.view(B*views, -1, C) + f_cam[B:].view(B*views, -1, C)
            cam_tokens.append((pose_token_ref, pose_token_source))
            f1 = f1.view(B, -1, C) +  cam_cond_embed_fine(f_delta1) 
            f2 = f2.view(B*views, -1, C) + cam_cond_embed_fine(f_delta2) 
            if i in self.idx_hook:
                f1 = f1.view(B, -1, C) + self.inject_stage2[hook_idx](self.enc_inject_stage2[hook_idx](low_token[hook_idx][:,:1].view(B, -1, 1024)))
                f2 = f2.view(B*views, -1, C) + self.inject_stage2[hook_idx](self.enc_inject_stage2[hook_idx](low_token[hook_idx][:,1:].reshape(B*views, -1, 1024)))
                hook_idx += 1
            f1 = f1.view(B, 1, -1 ,C)
            f2 = f2.view(B, views, -1 ,C)
            f = torch.cat((f1, f2), 1)
            final_output.append(f)
            f = f.view(B, -1 ,C)

        # normalize last output
        del final_output[1]  # duplicate with final_output[0]
        final_output[-1] = self.dec_norm_fine(final_output[-1])
        cam_tokens[-1] = tuple(map(self.dec_cam_norm_fine, cam_tokens[-1]))
        return final_output, zip(*cam_tokens)
    

    def _decoder_stage3(self, feat_ref, pos1, pos2, pose1, pose2, low_token=None, feat_stage2=None, fxfycxcy1=None, fxfycxcy2=None):
        final_output = [feat_ref[0]]  # before projection
        # project to decoder dim
        final_output.append(feat_ref[1])
        with torch.cuda.amp.autocast(enabled=False,dtype=torch.float32):
            pose1_embed = self.embed_pose(pose1)
            pose2_embed = self.embed_pose(pose2)
        pose_embed = torch.cat((pose1_embed, pose2_embed), 1)
        B, views, P, C = feat_ref[-1].shape
        if feat_stage2 is None:
            f = self.decoder_embed_point(feat_ref[0])
        else:
            f = self.decoder_embed_point(feat_ref[0]) + self.decoder_embed_stage2(feat_stage2)
        views = views - 1
        dtype = f.dtype
        pose_token_ref, pose_token_source = self.pose_token_ref_point.to(dtype).repeat(B,1,1).view(B, -1, C), self.pose_token_source_point.to(dtype).repeat(B*views,1,1).view(B*views, -1, C)
        pos = torch.cat((pos1, pos2), 1)
        if fxfycxcy1 is not None:
            with torch.cuda.amp.autocast(enabled=False,dtype=torch.float32):
                fxfycxcy1 = self.decoder_embed_fxfycxcy(fxfycxcy1)
                fxfycxcy2 = self.decoder_embed_fxfycxcy(fxfycxcy2)
                pose1_embed = pose1_embed + fxfycxcy1
                pose2_embed = pose2_embed + fxfycxcy2
        pose1_embed = pose1_embed.to(dtype)
        pose2_embed = pose2_embed.to(dtype)
        pose_token_ref = pose_token_ref + pose1_embed
        pose_token_source = pose_token_source + pose2_embed.view(B*views, -1, C)
        hook_idx = 0
        for i, (blk, blk_cross, cam_cond, cam_cond_embed_point, cam_cond_embed_point_pre) in enumerate(zip(self.dec_blocks_point, self.dec_blocks_point_cross, self.cam_cond_encoder_point, self.cam_cond_embed_point, self.cam_cond_embed_point_pre)):
            f1_pre = feat_ref[i+1].reshape(B, (views+1), -1, C)[:,:1].view(B, -1, C)
            f2_pre = feat_ref[i+1].reshape(B, (views+1), -1, C)[:,1:].reshape(B*views, -1, C)
            f1_pre = f1_pre + cam_cond_embed_point_pre(pose_token_ref)
            f2_pre = f2_pre + cam_cond_embed_point_pre(pose_token_source)
            f_pre = torch.cat((f1_pre.view(B, 1, -1, C), f2_pre.view(B, views, -1, C)), 1)
            feat, _ = checkpoint(blk_cross, f.reshape(B*(views+1), -1, C), f_pre.reshape(B*(views+1), -1, C), pos.reshape(B*(views+1), -1, 2), pos.reshape(B*(views+1), -1, 2))
            feat = feat.view(B, views+1, -1, C).reshape(B, -1, C)
            feat = checkpoint(blk, feat, pos.reshape(B, -1, 2))
            feat = feat.view(B, views+1, -1, C)
            f1 = feat[:,:1].view(B, -1, C)
            f2 = feat[:,1:].reshape(B*views, -1, C)
            f1_cam = torch.cat((pose_token_ref, f1.view(B, -1, C)), 1)
            f2_cam = torch.cat((pose_token_source, f2.view(B*views, -1, C)), 1)
            f_cam = torch.cat((f1_cam, f2_cam), 0)
            f_cam = checkpoint(cam_cond, f_cam) 
            f_delta = f_cam[:,1:]
            f_cam = f_cam[:,:1]
            f_delta1 = f_delta[:B].view(B, -1, C)
            f_delta2 = f_delta[B:].view(B*views, -1, C)
            pose_token_ref = pose_token_ref.view(B, -1, C) + f_cam[:B].view(B, -1, C)
            pose_token_source = pose_token_source.view(B*views, -1, C) + f_cam[B:].view(B*views, -1, C)
            f1 = f1.view(B, -1, C) +  cam_cond_embed_point(f_delta1) 
            f2 = f2.view(B*views, -1, C) + cam_cond_embed_point(f_delta2)
            if i in self.idx_hook:
                f1 = f1.view(B, -1, C) + self.inject_stage3[hook_idx](self.enc_inject_stage3[hook_idx](low_token[hook_idx][:,:1].view(B, -1, 1024)))
                f2 = f2.view(B*views, -1, C) + self.inject_stage3[hook_idx](self.enc_inject_stage3[hook_idx](low_token[hook_idx][:,1:].reshape(B*views, -1, 1024)))
                hook_idx += 1
            f1 = f1.view(B, 1, -1 ,C)
            f2 = f2.view(B, views, -1 ,C)
            f = torch.cat((f1, f2), 1)
            final_output.append(f)
            f = f.view(B, -1 ,C)

        # normalize last output
        del final_output[1]  # duplicate with final_output[0]
        final_output[-1] = self.dec_norm_point(final_output[-1])
        return final_output
  
    def _decoder(self, f1, pos1, f2, pos2):
        final_output = [(f1, f2)]  # before projection
        # project to decoder dim
        f1 = self.decoder_embed(f1)
        f2 = self.decoder_embed(f2)
        B, views, P, C = f2.shape
        f1 = f1.view(B, -1 ,C)
        f2 = f2.view(B, -1 ,C)
        pos1 = pos1.view(B, -1, pos1.shape[-1])
        pos2 = pos2.view(B, -1, pos2.shape[-1])
        cam_tokens = []
        final_output.append((f1, f2))
        pose_token_ref, pose_token_source = self.pose_token_ref.to(f1.dtype).repeat(B,1,1).view(B, -1, C), self.pose_token_source.to(f1.dtype).repeat(B*views,1,1).view(B*views, -1, C)
        for i, (blk1, blk2, cam_cond, cam_cond_embed) in enumerate(zip(self.dec_blocks, self.dec_blocks2, self.cam_cond_encoder,  self.cam_cond_embed)):
            f1, _ = checkpoint(blk1, *final_output[-1][::+1], pos1, pos2)
            f2, _ = checkpoint(blk2, *final_output[-1][::-1], pos2, pos1)
            f1_cam = torch.cat((pose_token_ref, f1.view(B, -1, C)), 1)
            f2_cam = torch.cat((pose_token_source, f2.view(B*views, -1, C)), 1)
            f_cam = torch.cat((f1_cam, f2_cam), 0)
            f_cam = checkpoint(cam_cond, f_cam)
            # f_cam = cam_cond(f_cam)
            f_delta = f_cam[:,1:]
            f_cam = f_cam[:,:1]
            f_delta1 = f_delta[:B].view(B, -1, C)
            f_delta2 = f_delta[B:].view(B*views, -1, C)
            pose_token_ref = pose_token_ref.view(B, -1, C) + f_cam[:B].view(B, -1, C)
            pose_token_source = pose_token_source.view(B*views, -1, C) + f_cam[B:].view(B*views, -1, C)
            cam_tokens.append((pose_token_ref, pose_token_source))
            f1 = f1.view(B, -1, C) + cam_cond_embed(f_delta1)
            f2 = f2.view(B*views, -1, C) + cam_cond_embed(f_delta2)
            f1 = f1.view(B, -1 ,C)
            f2 = f2.view(B, -1 ,C)
            # store the result
            final_output.append((f1, f2))

        # normalize last output
        del final_output[1]  # duplicate with final_output[0]
        cam_tokens[-1] = tuple(map(self.dec_cam_norm, cam_tokens[-1]))
        return zip(*cam_tokens)


    def forward_coarse_pose(self, view1, view2, enabled=True, dtype=torch.float32):
        # encode the two images --> B,S,D
        batch_size, _, _, _  = view1[0]['img'].shape
        view_num = len(view2)
        with torch.cuda.amp.autocast(enabled=False, dtype=torch.float32):
            shapes, feat, pos, shape_stage2, feat_stage2, pos_stage2, interm_features = self._encode_symmetrized(view1+view2)
        feat1 = feat[:, :1].to(dtype)
        feat2 = feat[:, 1:].to(dtype)
        pos1 = pos[:, :1]
        pos2 = pos[:, 1:]
        shape1 = shapes[:, :1]
        shape2 = shapes[:, 1:]
        shape1_stage2 = shape_stage2[:, :1]
        shape2_stage2 = shape_stage2[:, 1:]
        feat1_stage2 = feat_stage2[:, :1]
        feat2_stage2 = feat_stage2[:, 1:]
        pos1_stage2 = pos_stage2[:, :1]
        pos2_stage2 = pos_stage2[:, 1:]
        (pose_token1, pose_token2)  = self._decoder(feat1, pos1, feat2, pos2)
        pred_cameras, _ = self.pose_head(batch_size, interm_feature1=pose_token1, interm_feature2=pose_token2, enabled=True, dtype=dtype)
        return feat1_stage2, pos1_stage2, feat2_stage2, pos2_stage2, pred_cameras, shape1_stage2, shape2_stage2, None, None, pose_token1, pose_token2, interm_features

    def _encode_symmetrized_stage2(self, views, dtype):
        imgs = [view['img_org'] for view in views]
        shapes = [view['true_shape'] for view in views]
        imgs  = torch.stack((imgs), dim=1).contiguous()
        B, num_views, _, H, W = imgs.shape
        shapes = torch.stack((shapes), dim=1).contiguous()
        imgs = imgs.view(-1, *imgs.shape[2:])
        shapes = shapes.view(-1, *shapes.shape[2:])
        with torch.cuda.amp.autocast(enabled=True, dtype=dtype):
            imgs_vgg = imgs.to(dtype)
            ret = self.encode_feature_landscape([imgs_vgg.permute(0,2,3,1)], shapes.reshape(-1,2))
            feat_vgg_detail = ret['feat_vgg_detail']
            imgs_vgg = ret['imgs_vgg']
            imgs_vgg = imgs_vgg.permute(0, 3, 1, 2)
            feat_stage2, pos, _ = self._encode_image_fine(imgs_vgg.to(dtype), shapes, dtype)
            feat_stage2 = feat_stage2.view(B, num_views, *feat_stage2.shape[1:])
            pos = pos.view(B, num_views, *pos.shape[1:])
            shapes = shapes.view(B, num_views, 2)
            feat_vgg_detail = feat_vgg_detail.view(B, num_views, *feat_vgg_detail.shape[1:])
        return shapes, feat_vgg_detail, feat_stage2, pos

    
    def forward(self, view1, view2, enabled=True, dtype=torch.float32):
        res1, res2, pred_cameras = self.forward_gs(view1, view2, enabled=enabled, dtype=dtype)
        return res1, res2, pred_cameras
    
    
    def forward_gs(self, view1, view2, enabled=True, dtype=torch.float32):
        # encode the two images --> B,S,D
        batch_size, _, _, _  = view1[0]['img'].shape
        view_num = len(view2)
        feat1, pos1, feat2, pos2, pred_cameras_coarse, shape1, shape2, res1_stage1, res2_stage1, pose_token1, pose_token2, interm_features = self.forward_coarse_pose(view1, view2, enabled=enabled, dtype=dtype)
        trans_pred = pred_cameras_coarse[-1]['T'].float().detach().clone()
        trans_pred = trans_pred.reshape(batch_size, -1, 3)
        quaternion_R_pred = pred_cameras_coarse[-1]['quaternion_R'].reshape(batch_size, -1, 4).float().detach().clone()
        quaternion_R_noise = quaternion_R_pred
        trans_noise = trans_pred
        size =  (trans_noise.norm(dim=-1, keepdim=True).mean(dim=-2, keepdim=True) + 1e-8)
        trans_noise = trans_noise / size
        camera_embed = torch.cat((quaternion_R_noise, trans_noise), -1)
        camera_embed1 = camera_embed[:, :1].to(dtype)
        camera_embed2 = camera_embed[:, 1:].to(dtype)
        dec_fine, (pose_token1_fine, pose_token2_fine) = self._decoder_stage2(feat1, pos1, feat2, pos2, camera_embed1, camera_embed2, interm_features)
        shape = torch.cat((shape1, shape2), 1)
        pred_cameras, _ = self.pose_head_stage2(batch_size, interm_feature1=pose_token1_fine, interm_feature2=pose_token2_fine, enabled=True, dtype=dtype)
        
        trans = pred_cameras[-1]['T'].float().detach().clone()
        trans = trans.reshape(batch_size, -1, 3)
        size =  (trans[:,0:1,:] - trans[:,1:2,:]).norm(dim=-1, keepdim=True) + 1e-8
        trans_pred = trans / size
        quaternion_R_pred = pred_cameras[-1]['quaternion_R'].reshape(batch_size, -1, 4).float().detach().clone()
        quaternion_R_noise = quaternion_R_pred
        trans_noise = trans_pred
        camera_embed = torch.cat((quaternion_R_noise, trans_noise), -1)
        camera_embed1 = camera_embed[:, :1]
        camera_embed2 = camera_embed[:, 1:]
        c2ws = [view['camera_pose'] for view in view1 + view2]
        c2ws = torch.stack(c2ws, dim=1).clone()

        real_pose = torch.einsum('bnjk,bnkl->bnjl', c2ws[:,:1].repeat(1,c2ws.shape[1], 1, 1).inverse(), c2ws)
        trans_gt = real_pose[..., :3, 3]
        size_gt =  (trans_gt[:,0:1] - trans_gt[:,1:2]).norm(dim=-1, keepdim=True) + 1e-8 
        trans_gt = trans_gt / size_gt
        quaternion_R_gt = matrix_to_quaternion(real_pose[...,:3,:3])
        camera_embed_gt = torch.cat((quaternion_R_gt, trans_gt), -1)
        camera_embed1_gt = camera_embed_gt[:, :1]
        camera_embed2_gt = camera_embed_gt[:, 1:]
        pred_cameras = pred_cameras_coarse + pred_cameras

        fxfycxcy_unorm = [view['fxfycxcy_unorm'] for view in view1 + view2]
        fxfycxcy_unorm = torch.stack(fxfycxcy_unorm, dim=1).clone()
        fxfycxcy_unorm = fxfycxcy_unorm / 2000
        fxfycxcy_unorm = fxfycxcy_unorm.reshape(batch_size, -1, 4)
        fxfycxcy_unorm1 = fxfycxcy_unorm[:,:1]
        fxfycxcy_unorm2 = fxfycxcy_unorm[:,1:]
        shapes, feat_vgg_detail, feat_stage, pos = self._encode_symmetrized_stage2(view1+view2, dtype=dtype)
        dec_fine_stage2 = self._decoder_stage3(dec_fine, pos1, pos2, camera_embed1, camera_embed2, interm_features, feat_stage, fxfycxcy_unorm1, fxfycxcy_unorm2)
        with torch.cuda.amp.autocast(enabled=False, dtype=torch.float32):
            res2 = self._downstream_head(2, [tok.float().reshape(-1, tok.shape[-2], tok.shape[-1]) for tok in dec_fine_stage2], shape.reshape(-1, 2))
        for key in res2.keys():
            res2[key] = res2[key].unflatten(0, (batch_size, view_num+1))
        desc2 = torch.cat((res2['desc'].to(dtype), feat_vgg_detail), -1)
        gs2 = self.head4([desc2.flatten(0,1)], shape.reshape(-1,2))
        for key in gs2.keys():
            gs2[key] = gs2[key].unflatten(0, (batch_size, view_num+1))
        res2.update(gs2)
        res2_tmp = {}
        res1 = {}
        for key in res2.keys():
            res1[key] = res2[key][:,:1].flatten(0,1)
            res2_tmp[key] = res2[key][:,1:].flatten(0,1)
        res2 = res2_tmp
        return res1, res2, pred_cameras
