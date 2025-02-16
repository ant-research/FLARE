# Copyright (C) 2024-present Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).
#
# --------------------------------------------------------
# DUSt3R model class
# --------------------------------------------------------
from copy import deepcopy
import torch
import os
from packaging import version
import huggingface_hub
import torch.nn as nn
import torchvision.models as tvm
from .utils.misc import fill_default_args, freeze_all_params, is_symmetrized, interleave, transpose_to_landscape
from .heads import head_factory
from dust3r.patch_embed import get_patch_embed

import dust3r.utils.path_to_croco  # noqa: F401
from models.croco_self_old import CroCoNet  # noqa

inf = float('inf')
import torch.nn as nn
import torch.nn.functional as F

class ConvBnReLU(nn.Module):
    def __init__(self, in_channels, out_channels,
                 kernel_size=3, stride=1, pad=1):
        super(ConvBnReLU, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels,
                              kernel_size, stride=stride, padding=pad, bias=False)
        self.relu = nn.ReLU(inplace=True)
    def forward(self, x):
        return self.relu(self.conv(x))
    
class FeatureNet(nn.Module):
    def __init__(self, norm_act=nn.BatchNorm2d):
        super(FeatureNet, self).__init__()
        self.conv0 = nn.Sequential(
                        ConvBnReLU(3, 8, 3, 1, 1),
                        ConvBnReLU(8, 8, 3, 1, 1))
        self.conv1 = nn.Sequential(
                        ConvBnReLU(8, 16, 5, 2, 2),
                        ConvBnReLU(16, 16, 3, 1, 1))
        self.conv2 = nn.Sequential(
                        ConvBnReLU(16, 32, 5, 2, 2),
                        ConvBnReLU(32, 32, 3, 1, 1))

        self.toplayer = nn.Conv2d(32, 32, 1)
        self.lat1 = nn.Conv2d(16, 32, 1)
        self.lat0 = nn.Conv2d(8, 32, 1)

        self.smooth1 = nn.Conv2d(32, 32, 3, padding=1)
        self.smooth0 = nn.Conv2d(32, 32, 3, padding=1)

    def _upsample_add(self, x, y):
        return F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True) + y

    def forward(self, x):
        conv0 = self.conv0(x)
        conv1 = self.conv1(conv0)
        conv2 = self.conv2(conv1)
        feat2 = self.toplayer(conv2)
        feat1 = self._upsample_add(feat2, self.lat1(conv1))
        feat0 = self._upsample_add(feat1, self.lat0(conv0))
        feat1 = self.smooth1(feat1)
        feat0 = self.smooth0(feat0)
        return feat2, feat1, feat0

    
hf_version_number = huggingface_hub.__version__
assert version.parse(hf_version_number) >= version.parse("0.22.0"), "Outdated huggingface_hub version, please reinstall requirements.txt"

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


class VGG19(nn.Module):
    def __init__(self, pretrained=False, amp = False, amp_dtype = torch.float16) -> None:
        super().__init__()
        self.layers = nn.ModuleList(tvm.vgg19_bn(pretrained=pretrained).features[:40])
        self.amp = amp
        self.amp_dtype = amp_dtype

    def forward(self, x):
        feats = {}
        scale = 1
        for layer in self.layers:
            if isinstance(layer, nn.MaxPool2d):
                feats[scale] = x[:, :64]
                scale = scale*2
                if scale > 4:
                    break
            x = layer(x)
        return feats[1], feats[2], feats[4]
    
class AsymmetricCroCo3DStereo (
    CroCoNet,
    huggingface_hub.PyTorchModelHubMixin,
    library_name="dust3r",
    repo_url="https://github.com/naver/dust3r",
    tags=["image-to-3d"],
):
    """ Two siamese encoders, followed by two decoders.
    The goal is to output 3d points directly, both images in view1's frame
    (hence the asymmetry).   
    """
    def __init__(self,
                 output_mode='pts3d',
                 head_type='linear',
                 depth_mode=('exp', -inf, inf),
                 conf_mode=('exp', 1, inf),
                 freeze='none',
                 landscape_only=True,
                 patch_embed_cls='PatchEmbedDust3R',  # PatchEmbedDust3R or ManyAR_PatchEmbed
                 **croco_kwargs):
        self.patch_embed_cls = patch_embed_cls
        self.croco_args = fill_default_args(croco_kwargs, super().__init__)
        super().__init__(**croco_kwargs)

        # dust3r specific initialization
        self.dec_blocks2 = deepcopy(self.dec_blocks)
        self.set_downstream_head(output_mode, head_type, landscape_only, depth_mode, conf_mode, **croco_kwargs)
        # self.set_freeze(freeze)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kw):
        if os.path.isfile(pretrained_model_name_or_path):
            return load_model(pretrained_model_name_or_path, device='cpu')
        else:
            return super(AsymmetricCroCo3DStereo, cls).from_pretrained(pretrained_model_name_or_path, **kw)

    def _set_patch_embed(self, img_size=224, patch_size=16, enc_embed_dim=768):
        self.patch_embed = get_patch_embed(self.patch_embed_cls, img_size, patch_size, enc_embed_dim)

    def load_state_dict(self, ckpt, **kw):
        # duplicate all weights for the second decoder if not present
        new_ckpt = dict(ckpt)
        if not any(k.startswith('dec_blocks2') for k in ckpt):
            for key, value in ckpt.items():
                if key.startswith('dec_blocks'):
                    new_ckpt[key.replace('dec_blocks', 'dec_blocks2')] = value
        return super().load_state_dict(new_ckpt, **kw)

    def set_freeze(self, freeze):  # this is for use by downstream models
        self.freeze = freeze
        to_be_frozen = {
            'none':     [],
            'mask':     [self.mask_token],
            'encoder':  [self.mask_token, self.patch_embed, self.enc_blocks],
            'decoder':  [self.downstream_head1, self.downstream_head2, self.mask_token, self.patch_embed, self.enc_blocks],
        }
        freeze_all_params(to_be_frozen[freeze])

    def _set_prediction_head(self, *args, **kwargs):
        """ No prediction head """
        return

    def set_downstream_head(self, output_mode, head_type, landscape_only, depth_mode, conf_mode, patch_size, img_size,
                            **kw):
        assert img_size[0] % patch_size == 0 and img_size[1] % patch_size == 0, \
            f'{img_size=} must be multiple of {patch_size=}'
        self.output_mode = output_mode
        self.head_type = head_type
        self.depth_mode = depth_mode
        self.conf_mode = conf_mode
        # allocate heads
        self.downstream_head1 = head_factory(head_type, output_mode, self, has_conf=bool(conf_mode))
        self.downstream_head2 = head_factory(head_type, output_mode, self, has_conf=bool(conf_mode))
        # magic wrapper
        self.head1 = transpose_to_landscape(self.downstream_head1, activate=landscape_only)
        self.head2 = transpose_to_landscape(self.downstream_head2, activate=landscape_only)

    def _encode_image(self, image, true_shape):
        # embed the image into patches  (x has size B x Npatches x C)
        x, pos = self.patch_embed(image, true_shape=true_shape)

        # add positional embedding without cls token
        assert self.enc_pos_embed is None

        # now apply the transformer encoder and normalization
        for blk in self.enc_blocks:
            x = blk(x, pos)

        x = self.enc_norm(x)
        return x, pos, None

    def _encode_image_pairs(self, img1, img2, true_shape1, true_shape2):
        if img1.shape[-2:] == img2.shape[-2:]:
            out, pos, _ = self._encode_image(torch.cat((img1, img2), dim=0),
                                             torch.cat((true_shape1, true_shape2), dim=0))
            out, out2 = out.chunk(2, dim=0)
            pos, pos2 = pos.chunk(2, dim=0)
        else:
            out, pos, _ = self._encode_image(img1, true_shape1)
            out2, pos2, _ = self._encode_image(img2, true_shape2)
        return out, out2, pos, pos2

    def _encode_symmetrized(self, view1, view2):
        img1 = view1['img']
        img2 = view2['img']
        B = img1.shape[0]
        # Recover true_shape when available, otherwise assume that the img shape is the true one
        shape1 = view1.get('true_shape', torch.tensor(img1.shape[-2:])[None].repeat(B, 1))
        shape2 = view2.get('true_shape', torch.tensor(img2.shape[-2:])[None].repeat(B, 1))
        # warning! maybe the images have different portrait/landscape orientations

        if is_symmetrized(view1, view2):
            # computing half of forward pass!'
            feat1, feat2, pos1, pos2 = self._encode_image_pairs(img1[::2], img2[::2], shape1[::2], shape2[::2])
            feat1, feat2 = interleave(feat1, feat2)
            pos1, pos2 = interleave(pos1, pos2)
        else:
            feat1, feat2, pos1, pos2 = self._encode_image_pairs(img1, img2, shape1, shape2)

        return (shape1, shape2), (feat1, feat2), (pos1, pos2)

    def _decoder(self, f1, pos1, f2, pos2):
        final_output = [(f1, f2)]  # before projection

        # project to decoder dim
        f1 = self.decoder_embed(f1)
        f2 = self.decoder_embed(f2)
        B, views, P, C = f2.shape
        f1_orig = f1
        f2_orig = f2
        f1 = f1.view(B, -1 ,C)
        f2 = f2.view(B, -1 ,C)
        pos1 = pos1.view(B, -1, pos1.shape[-1])
        pos2 = pos2.view(B, -1, pos2.shape[-1])
        final_output.append((f1, f2))
        for i, (blk1, blk2, cam_cond, cam_cond_embed) in enumerate(zip(self.dec_blocks, self.dec_blocks2, self.cam_cond_encoder,  self.cam_cond_embed)):
            # img1 side
            f1, _ = blk1(*final_output[-1][::+1], pos1, pos2)
            # img2 side
            f2, _ = blk2(*final_output[-1][::-1], pos2, pos1)
            if i < len(self.dec_blocks)-1:
                f1_cam, _= cam_cond(f1.view(B, -1, C), f1_orig.view(B, -1, C), pos1, pos1)
                f1 = f1.view(B, -1, C) + cam_cond_embed(f1_cam)
                f2_cam, _ = cam_cond(f2.view(B*views, -1, C), f2_orig.view(B*views, -1, C), pos2.view(B*views, -1, 2), pos2.view(B*views, -1, 2))
                f2 = f2.view(B*views, -1, C) + cam_cond_embed(f2_cam)
            f1 = f1.view(B, -1 ,C)
            f2 = f2.view(B, -1 ,C)
            # store the result
            final_output.append((f1, f2))

        # normalize last output
        del final_output[1]  # duplicate with final_output[0]
        final_output[-1] = tuple(map(self.dec_norm, final_output[-1]))
        return zip(*final_output)

    def _downstream_head(self, head_num, decout, img_shape):
        B, S, D = decout[-1].shape
        # img_shape = tuple(map(int, img_shape))
        head = getattr(self, f'head{head_num}')
        return head(decout, img_shape)

    # def forward(self, view1, view2, enabled=True, dtype=torch.bfloat16):
    #     # encode the two images --> B,S,D
    #     batch_size, _, _, _  = view1[0]['img'].shape
    #     view_num = len(view2)

    #     shapes, feat, feat_ray, pos, real_pose, scale, valid1, valid2 = self._encode_symmetrized(view1+view2)

    #     feat1 = feat[:batch_size].view(batch_size, -1, feat.shape[1], feat.shape[2])
    #     feat2 = feat[batch_size:].view(batch_size, -1, feat.shape[1], feat.shape[2])
    #     pos1 = pos[:batch_size].view(batch_size, -1, pos.shape[1], pos.shape[2])
    #     pos2 = pos[batch_size:].view(batch_size, -1, pos.shape[1], pos.shape[2])
    #     shape1 = shapes[:batch_size].view(batch_size, -1, shapes.shape[1])
    #     shape2 = shapes[batch_size:].view(batch_size, -1, shapes.shape[1])
    #     feat_ray1 = feat_ray[:batch_size].view(batch_size, -1, feat_ray.shape[1], feat_ray.shape[2])
    #     feat_ray2 = feat_ray[batch_size:].view(batch_size, -1, feat_ray.shape[1], feat_ray.shape[2])
    #     dec1, dec2 = self._decoder(feat1, pos1, feat2, pos2, feat_ray1, feat_ray2)
    #     feat1 = [tok.float().view(-1, tok.shape[-2], tok.shape[-1]) for tok in dec1]
    #     feat2 = [tok.float().view(batch_size*view_num, -1, tok.shape[-1]) for tok in dec2]
    #     # combine all ref images into object-centric representation
    #     dec1, dec2 = self._decoder(feat1, pos1, feat2, pos2)
    #     with torch.cuda.amp.autocast(enabled=False, dtype=torch.float32):
    #         res1 = self._downstream_head(1, [tok.float().view(-1, tok.shape[-2], tok.shape[-1]) for tok in dec1], shape1)
    #         res2 = self._downstream_head(2, [tok.float().view(batch_size*view_num, -1, tok.shape[-1]) for tok in dec2], shape2)

    #     res2['pts3d_in_other_view'] = res2.pop('pts3d')  # predict view2's pts3d in view1's frame
    #     pred_cameras, _ = self.pose_head(batch_size, pos_encoding=pos, interm_feature1=feat1, interm_feature2=feat2)
    #     return res1, res2