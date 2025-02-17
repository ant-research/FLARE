import os
from collections import OrderedDict
from typing import Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import nn, einsum
from torch.utils.checkpoint import checkpoint

from einops import rearrange, repeat

from inspect import isfunction
try:
    from flash_attn import flash_attn_qkvpacked_func, flash_attn_func, flash_attn_varlen_qkvpacked_func
    from flash_attn.bert_padding import unpad_input, pad_input
except:
    flash_attn_qkvpacked_func, flash_attn_func, flash_attn_varlen_qkvpacked_func = None, None, None
    unpad_input, pad_input = None, None
from .x_transformer import AttentionLayers, BasicEncoder
import math



def _init_weights(module):
    if isinstance(module, nn.Linear):
        torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        if module.bias is not None:
            torch.nn.init.zeros_(module.bias)

def exists(val):
    return val is not None

def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d

def zero_module(module):
    """
    Zero out the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module

# Copy from CLIP GitHub
class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)

class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)

def modulate(x, shift, scale):
    # from https://github.com/facebookresearch/DiT/blob/796c29e532f47bba17c5b9c5eb39b9354b8b7c64/models.py#L19
    return x * (1 + scale.unsqueeze(0)) + shift.unsqueeze(0)


class MultiheadAttentionFlashV2(nn.Module):
    def __init__(self, embed_dim, n_head, bias=False, shift_group=None, qkv_packed=False, window_size=None):
        super().__init__()

        self.head_dim = embed_dim// n_head
        self.embed_dim = embed_dim
        self.n_head = n_head
        self.to_q = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.to_k = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.to_v = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.shift_group = shift_group
        self.qkv_packed = qkv_packed
        self.window_size = window_size


    def forward(self, q, k, v, dropout_p=0.0, softmax_scale=None, causal=False, need_weights=False, attn_mask=None):
        q = q.permute(1, 0, 2)
        k = k.permute(1, 0, 2)
        v = v.permute(1, 0, 2)

        h = self.n_head
        q = self.to_q(q)
        k = self.to_k(k)
        v = self.to_v(v)

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b n h d', h=h), (q, k, v))
        # print(q.dtype, k.dtype, v.dtype)
        if self.qkv_packed:
            bsz, q_len, heads, head_dim = q.shape
            group_size = self.shift_group
            nheads = self.n_head
            qkv = torch.stack([q,k,v], dim=2)
            qkv = qkv.reshape(bsz, q_len, 3, 2, nheads // 2, self.head_dim).permute(0, 3, 1, 2, 4, 5).reshape(bsz * 2,
                                                                                                              q_len, 3,
                                                                                                              nheads // 2,
                                                                                                              self.head_dim)

            x = rearrange(qkv, "b s three h d -> b s (three h d)")
            key_padding_mask = torch.ones(x.shape[0], x.shape[1], device=x.device, dtype=x.dtype)
            x_unpad, indices, cu_q_lens, max_s = unpad_input(x, key_padding_mask)
            cu_q_len_tmp = torch.arange(0, max_s, group_size, device=key_padding_mask.device, dtype=cu_q_lens.dtype)
            cu_q_len_tmp2 = cu_q_len_tmp + group_size // 2
            cu_q_len_tmp2[cu_q_len_tmp2 >= max_s] = torch.iinfo(cu_q_len_tmp2.dtype).min
            cu_q_len_tmp = torch.stack([cu_q_len_tmp, cu_q_len_tmp2]).repeat(bsz, 1) + cu_q_lens[:-1].unsqueeze(-1)
            cu_q_lens = torch.cat([cu_q_len_tmp, cu_q_lens[1:].unsqueeze(-1)], dim=-1).view(-1)
            cu_q_lens = cu_q_lens[cu_q_lens >= 0]
            x_unpad = rearrange(
                x_unpad, "nnz (three h d) -> nnz three h d", three=3, h=nheads // 2
            )
            output_unpad = flash_attn_varlen_qkvpacked_func(
                x_unpad, cu_q_lens, group_size, 0.0, softmax_scale=None, causal=False,
            )
            output = rearrange(
                   pad_input(
                       rearrange(output_unpad, "nnz h d -> nnz (h d)"), indices, bsz * 2, q_len
                   ),
                   "b s (h d) -> b s h d",
                   h=nheads // 2,
               )
            r_out = output.reshape(bsz, 2, q_len, nheads // 2, self.head_dim).transpose(1, 2).reshape(bsz, q_len, nheads,
                                                                                               self.head_dim)
        else:
            if self.shift_group is not None:
                 bsz, q_len, heads, head_dim = q.shape
                 assert q_len % self.shift_group == 0

                 def shift(qkv, bsz, q_len, group_size, num_heads, head_dim):
                     qkv[:, num_heads // 2:] = qkv[:, num_heads // 2:].roll(-group_size // 2, dims=2)
                     qkv = qkv.transpose(1, 2).reshape(bsz * (q_len // group_size), group_size, num_heads, head_dim).transpose(1, 2)
                     return qkv

                 q = shift(q, bsz, q_len, self.shift_group, h, self.head_dim)
                 k = shift(k, bsz, q_len, self.shift_group, h, self.head_dim)
                 v = shift(v, bsz, q_len, self.shift_group, h, self.head_dim)
            if self.window_size:
                out = flash_attn_func(q, k, v, dropout_p=dropout_p, softmax_scale=softmax_scale, causal=causal, window_size=(self.window_size // 2, self.window_size // 2))
            else:
                out = flash_attn_func(q, k, v, dropout_p=dropout_p, softmax_scale=softmax_scale, causal=causal)

            if self.shift_group is not None:
                out = out.transpose(1, 2).contiguous()
                out = rearrange(out, '(b l) g h d -> b (l g) h d', l=q_len // self.shift_group)
                r_out = out.clone()
                r_out[:, :, h//2:] = r_out[:, :, h//2:].roll(h//2, dims=1)    
            else:
                r_out = out

        r_out = rearrange(r_out, 'b n h d -> b n (h d)')
        r_out = r_out.permute(1, 0, 2)
        return (r_out,)

class PSUpsamplerBlock(nn.Module):
    def __init__(self, d_model: int, d_model_out: int, scale_factor: int):
        super().__init__()

        # self.mlp = nn.Sequential(OrderedDict([
        #     ("c_fc", nn.Linear(d_model, d_model_out * scale_factor**2)),
        #     ("gelu", QuickGELU()),
        #     ("c_proj", nn.Linear(d_model_out * scale_factor**2, d_model_out * scale_factor**2))
        # ]))
        # self.ln_2 = LayerNorm(d_model)
        self.scale_factor = scale_factor
        self.d_model_out = d_model_out
        self.residual_fc = nn.Linear(d_model, d_model_out * (scale_factor**2))
        self.pixelshuffle = nn.PixelShuffle(scale_factor)

    def forward(self, x: torch.Tensor):
        # mlp block
        # x.shape b, l, d
        # y = self.ln_2(x)
        # y = self.mlp(y)
        # For here we have two cases:
        # 1. If we have a modulation function for the MLP, we use it to modulate the output of the MLP
        # 2. If we don't have a modulation function for the MLP, we use the modulation function for the attention
        x = self.residual_fc(x)# .repeat(1, 1, self.scale_factor**2)
        # x = x + y
        bs, l, c = x.shape
        resolution = int(np.sqrt(l))
        x = x.permute(0, 2, 1).reshape(bs, c, resolution, resolution)
        x = self.pixelshuffle(x)
        x = x.reshape(bs, self.d_model_out, resolution*self.scale_factor*resolution*self.scale_factor)
        x = x.permute(0, 2, 1)
        # x = rearrange(x, 'b l (s c) -> b (l s) c', s=self.scale_factor**2)
        return x

class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, 
            n_head: int, 
            attn_mask: torch.Tensor = None, 
            modulate_feature_size: int = None, 
            modulate_act_type: str = 'gelu', 
            cross_att: bool = None, 
            flash_v2: bool = None, 
            qkv_packed: bool = None, 
            shift_group: int = None,
            window_size: int = None,):
        super().__init__()

        print('vit flashv2', flash_v2)

        self.flash_v2 = flash_v2
        self.window_size = window_size
        if self.flash_v2:
            self.attn = MultiheadAttentionFlashV2(d_model, n_head, shift_group=shift_group, qkv_packed=qkv_packed, window_size=window_size)
        else:
            self.attn = nn.MultiheadAttention(d_model, n_head)

        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask
        self.window_size = window_size

        if modulate_feature_size is not None:
            act_dict = {'gelu': QuickGELU,
                    'silu': nn.SiLU}
            self.modulation_fn = nn.Sequential(
                LayerNorm(modulate_feature_size),
                act_dict[modulate_act_type](),
                nn.Linear(modulate_feature_size, 3 * d_model, bias=True)
            )
            self.mlp_modulation_fn = nn.Sequential(
                LayerNorm(modulate_feature_size),
                act_dict[modulate_act_type](),
                nn.Linear(modulate_feature_size, 3 * d_model, bias=True)
            )
        else:
            self.modulation_fn = None
            self.mlp_modulation_fn = None

        self.cross_att = cross_att
        if self.cross_att:
            self.cross_att = CrossAttention(query_dim=d_model, context_dim=d_model,
                                    heads=n_head, dim_head=int(d_model//n_head), dropout=0)    
            self.ln_1_5 = LayerNorm(d_model)

    def attention(self, x: torch.Tensor, index):
        if self.attn_mask is not None:
            self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device)
            length = x.shape[0]
            attn_mask = self.attn_mask[:length, :length]
        else:
            attn_mask = None
        if self.window_size is not None:
            x = x.permute(1, 0, 2)
            b, l, c = x.shape
            # print(x.shape)
            assert l % self.window_size == 0
            if index % 2 == 0:
                x = rearrange(x, 'b (p w) c -> (b p) w c', w=self.window_size)
                x = x.permute(1, 0, 2) # w, bp, c
                x = self.attn(x, x, x, need_weights=False, attn_mask=attn_mask)[0] 
                x = x.permute(1, 0, 2) # bp, w, c
                x = rearrange(x, '(b l) w c -> b (l w) c', l=l//self.window_size, w=self.window_size)
                x = x.permute(1, 0, 2) # w, bp, c
            else:
                x = torch.roll(x, shifts=self.window_size//2, dims=1)
                x = rearrange(x, 'b (p w) c -> (b p) w c', w=self.window_size)
                x = x.permute(1, 0, 2) # w, bp, c
                x = self.attn(x, x, x, need_weights=False, attn_mask=attn_mask)[0] 
                x = x.permute(1, 0, 2) # w, bp, c
                x = rearrange(x, '(b l) w c -> b (l w) c', l=l//self.window_size, w=self.window_size)
                x = torch.roll(x, shifts=-self.window_size//2, dims=1)
                x = x.permute(1, 0, 2)
        else:
            x = self.attn(x, x, x, need_weights=False, attn_mask=attn_mask)[0]

        return x

    def forward(self, x: torch.Tensor, modulation: torch.Tensor = None, context: torch.Tensor = None, index=None):
        # self attention block
        y = self.ln_1(x)
        if self.modulation_fn is not None:
            shift, scale, gate = self.modulation_fn(modulation).chunk(3, dim=1)
            y = modulate(y, shift, scale)
        y = self.attention(y, index)
        # If we have modulation func for mlp as well, we will just use the gate for the attention
        if self.modulation_fn is not None and self.mlp_modulation_fn is not None:
            y = y * gate.unsqueeze(0)
        x = x + y

        # cross attention block
        if self.cross_att:
            y = self.cross_att(self.ln_1_5(x), context=context) 
            # print(y.mean().item())
            x = x + y

        # mlp block
        y = self.ln_2(x)
        if self.mlp_modulation_fn is not None:
            shift, scale, gate = self.mlp_modulation_fn(modulation).chunk(3, dim=1)
            y = modulate(y, shift, scale)
        y = self.mlp(y)
        # For here we have two cases:
        # 1. If we have a modulation function for the MLP, we use it to modulate the output of the MLP
        # 2. If we don't have a modulation function for the MLP, we use the modulation function for the attention
        if self.modulation_fn is not None:
            y = y * gate.unsqueeze(0)
        x = x + y

        return x


class Transformer(nn.Module):
    def __init__(self, 
            width: int, 
            layers: int, 
            heads: int, 
            attn_mask: torch.Tensor = None, 
            modulate_feature_size: int = None, 
            modulate_act_type: str = 'gelu', 
            cross_att_layers: int = 0, 
            return_all_layers=False, 
            flash_v2=True, 
            qkv_packed=False,
            shift_group=None,
            window_size=None):

        super().__init__()
        self.width = width
        self.layers = layers

        blocks = []
        for _ in range(layers):
            layer = ResidualAttentionBlock(width, 
                        heads, 
                        attn_mask, 
                        modulate_feature_size=modulate_feature_size, 
                        modulate_act_type=modulate_act_type, 
                        cross_att = (_ + cross_att_layers)>=layers, 
                        flash_v2=flash_v2, 
                        qkv_packed=qkv_packed,
                        shift_group=shift_group,
                        window_size=window_size) 
            blocks.append(layer)

        self.resblocks = nn.Sequential(*blocks)

        self.grad_checkpointing = False
        self.return_all_layers = return_all_layers 
        self.flash_v2 = flash_v2

    def set_grad_checkpointing(self, flag=True):
        self.grad_checkpointing = flag

    def forward(self, 
            x: torch.Tensor, 
            modulation: torch.Tensor = None, 
            context: torch.Tensor = None, 
            additional_residuals = None):

        all_x = []
        if additional_residuals is not None:
            assert len(additional_residuals) == self.layers
        for res_i, module in enumerate(self.resblocks):
            if self.grad_checkpointing:
                # print("Grad checkpointing")
                x = checkpoint(module, x, modulation, context, res_i)
            else:
                x = module(x, modulation, context, res_i)
            if additional_residuals is not None:
                add_res = additional_residuals[res_i]
                x[:, :add_res.shape[1]] = x[:, :add_res.shape[1]] + add_res
            all_x.append(x)
        if self.return_all_layers:
            return all_x
        else:
            return x

class GaussianUpsampler(nn.Module):
    def __init__(self, width,
                 up_ratio,
                 ch_decay=1,
                 low_channels=64,
                 window_size=False,
                 with_additional_inputs=False):

        super().__init__()
        self.up_ratio = up_ratio
        self.low_channels = low_channels
        self.window_size = window_size
        self.base_width = width
        self.with_additional_inputs = with_additional_inputs
        for res_log2 in range(int(np.log2(up_ratio))):
            _width = width
            width = max(width // ch_decay, 64)
            heads = int(width / 64)
            width = heads * 64
            if self.with_additional_inputs: 
                self.add_module(f'upsampler_{res_log2}', PSUpsamplerBlock(_width+self.base_width, width, 2))
            else:
                self.add_module(f'upsampler_{res_log2}', PSUpsamplerBlock(_width, width, 2))
            encoder = Transformer(width, 2, heads,
                                  modulate_feature_size=None,
                                  modulate_act_type=None,
                                  cross_att_layers=0,
                                  return_all_layers=False,
                                  flash_v2=False,
                                  qkv_packed=False,
                                  shift_group=False,
                                  window_size=window_size)
            self.add_module(f'attention_{res_log2}', encoder)
        self.out_channels = width
        self.ln_post = LayerNorm(width)

    def forward(self, x, additional_inputs=None):
        if self.with_additional_inputs:
            assert len(additional_inputs) == int(np.log2(self.up_ratio))
        for res_log2 in range(int(np.log2(self.up_ratio))):
            if self.with_additional_inputs:
                add_input = additional_inputs[res_log2]
                scale = x.shape[1] // add_input.shape[1] 
                add_input = add_input.repeat_interleave(scale, 1)
                x = torch.cat([x, add_input], dim=2)
            x = getattr(self, f'upsampler_{res_log2}')(x)
            x = x.permute(1, 0, 2)
            x = getattr(self, f'attention_{res_log2}')(x)
            x = x.permute(1, 0, 2)
        x = self.ln_post(x)
        return x



class HyperGaussianUpsampler(nn.Module):
    def __init__(self, width,
                 resolution,
                 up_ratio,
                 ch_decay=1,
                 window_size=False,
                 with_additional_inputs=False,
                 upsampler_kwargs={}):

        super().__init__()
        self.up_ratio = up_ratio
        self.window_size = window_size
        self.base_width = width
        self.with_additional_inputs = with_additional_inputs
        self.resolution = resolution
        for res_log2 in range(int(np.log2(up_ratio))):
            if res_log2 == 0:
                _width = width
                width = width
                heads = int(width / 64)
                width = heads * 64
                if self.with_additional_inputs: 
                    self.add_module(f'upsampler_{res_log2}', PSUpsamplerBlock(_width+self.base_width, width, 2))
                else:
                    self.add_module(f'upsampler_{res_log2}', PSUpsamplerBlock(_width, width, 2))
                encoder = Transformer(width, 2, heads,
                                    modulate_feature_size=None,
                                    modulate_act_type=None,
                                    cross_att_layers=0,
                                    return_all_layers=False,
                                    flash_v2=False,
                                    qkv_packed=False,
                                    shift_group=False,
                                    window_size=window_size)
                self.add_module(f'attention_{res_log2}', encoder)
                self.resolution = self.resolution * 2
            else:
                self.resolution = self.resolution * 2
                self.add_module(f'upsample_{res_log2}',
                                UpsamplerLayers_conv(in_channels=width,
                                                     out_channels=width,
                                                     resolution=self.resolution,
                                                     conv_block_type = 'convnext',
                                                     **upsampler_kwargs))
        self.out_channels = width
        # self.ln_post = LayerNorm(width)
        self.ln_post = LayerNorm([self.resolution, self.resolution, width])

    def forward(self, x, additional_inputs=None):
        if self.with_additional_inputs:
            assert len(additional_inputs) == int(np.log2(self.up_ratio))
        for res_log2 in range(int(np.log2(self.up_ratio))):
            if res_log2 == 0:
                if self.with_additional_inputs:
                    add_input = additional_inputs[res_log2]
                    scale = x.shape[1] // add_input.shape[1] 
                    add_input = add_input.repeat_interleave(scale, 1)
                    x = torch.cat([x, add_input], dim=2)
                x = getattr(self, f'upsampler_{res_log2}')(x)
                x = x.permute(1, 0, 2)
                x = getattr(self, f'attention_{res_log2}')(x)
                x = x.permute(1, 0, 2)
                x = x.reshape(x.shape[0], int(math.sqrt(x.shape[1])), int(math.sqrt(x.shape[1])), -1).permute(0, 3, 1, 2)
            else:
                x = getattr(self, f'upsample_{res_log2}')(x)
        x = self.ln_post(x.permute(0, 2, 3, 1))
        return x

class VisionTransformer(nn.Module):
    def __init__(self, 
                 # transformer params
                 in_channels: int, 
                 patch_size: int, 
                 width: int, 
                 layers: int, 
                 heads: int, 
                 weight: str = None,
                 encode_layers: int = 0,
                 shift_group = False,
                 flash_v2 = False,
                 qkv_packed = False,
                 window_size = False,
                 use_pe = False,
                 # modualtion params 
                 modulate_feature_size: int = None, 
                 modulate_act_type: str = 'gelu', 
                 # camera condition
                 camera_condition: str = 'plucker',
                 # init params
                 disable_dino=False,
                 error_weight_init_mode='mean', 
                 # other params
                 add_zero_conv=False, 
                 return_all_layers=False, 
                 disable_post_ln=False,
                 rope=None):
        super().__init__()
        self.patch_size = patch_size
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=width, kernel_size=patch_size, stride=patch_size, bias=False)
        self.use_pe = use_pe
        self.rope = rope
        self.disable_dino = disable_dino
        # if not self.disable_dino:
        #     scale = width ** -0.5
        #     self.class_embedding = nn.Parameter(scale * torch.randn(width))
        #     self.positional_embedding = nn.Parameter(scale * torch.randn((input_res// patch_size) ** 2 + 1, width))
        # else:
        #     if self.use_pe:
        # self.positional_embedding = nn.Parameter(torch.zeros(1, (input_res// patch_size) ** 2, width))
        # nn.init.trunc_normal_(self.positional_embedding, std=0.02)
        self.ln_pre = LayerNorm(width)
        self.add_zero_conv = add_zero_conv
        self.return_all_layers = return_all_layers
        self.disable_post_ln = disable_post_ln
        self.flash_v2 = flash_v2
        self.qkv_packed = qkv_packed

        self.camera_condition = camera_condition
        if self.camera_condition == 'plucker': assert modulate_feature_size is None

        if self.add_zero_conv:
            assert self.return_all_layers
            self.zero_convs = nn.ModuleList([zero_module(nn.Conv1d(in_channels=width, out_channels=width, kernel_size=1, stride=1, bias=True)) for _ in range(layers)])

        self.encode_layers = encode_layers
        if self.encode_layers > 0:
            self.encoder = Transformer(width, encode_layers, heads, 
                                       modulate_feature_size=modulate_feature_size, 
                                       modulate_act_type=modulate_act_type, 
                                       cross_att_layers=0, 
                                       return_all_layers=return_all_layers, 
                                       flash_v2=flash_v2, 
                                       qkv_packed=qkv_packed,
                                       shift_group=shift_group,
                                       window_size=window_size)
        self.transformer = Transformer(width, layers-encode_layers, heads, 
                                       modulate_feature_size=modulate_feature_size, 
                                       modulate_act_type=modulate_act_type, 
                                       cross_att_layers=0, 
                                       return_all_layers=return_all_layers, 
                                       flash_v2=flash_v2, 
                                       qkv_packed=qkv_packed,
                                       shift_group=shift_group,
                                       window_size=window_size)

        if not self.disable_post_ln:
            self.ln_post = LayerNorm(width)

        if weight is not None:
            if not self.disable_dino:
                if "clip" in weight:
                    raise NotImplementedError()
                elif weight.startswith("vit_b_16"):
                    load_timm_to_clip(self, config_name=weight, init_mode=error_weight_init_mode)
                elif weight.startswith("vit_b_8"):
                    load_timm_to_clip(self, config_name=weight, init_mode=error_weight_init_mode)
                else:
                    raise NotImplementedError()
            else:
                self.apply(_init_weights)

            # Init the weight and bias of modulation_fn to zero
            if modulate_feature_size != 0:
                for block in self.transformer.resblocks:
                    if block.modulation_fn is not None:
                        block.modulation_fn[2].weight.data.zero_()
                        block.modulation_fn[2].bias.data.zero_()
                        if block.mlp_modulation_fn is not None:
                            block.mlp_modulation_fn[2].weight.data.zero_()
                            block.mlp_modulation_fn[2].bias.data.zero_()
            for block in self.transformer.resblocks:
                if block.cross_att:
                    zero_module(block.cross_att.to_out)

    def set_grad_checkpointing(self, flag=True):
        self.transformer.set_grad_checkpointing(flag)

    def forward(self, 
            x: torch.Tensor, 
            modulation: torch.Tensor = None, 
            context: torch.Tensor = None, 
            additional_residuals=None,
            abla_crossview=False,
            pos=None):

        # image tokenization
        bs, vs = x.shape[:2]
        x = rearrange(x, 'b v c h w -> (b v) c h w')
        pos = rearrange(pos, 'b v c h -> (b v) c h')
        if self.camera_condition == 'plucker' and modulation is not None:
            modulation = rearrange(modulation, 'b v c h w -> (b v) c h w')   
            x = torch.cat([x, modulation], dim=1)
        modulation = None
        
        x = self.conv1(x)  # shape = [*, width, grid, grid]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
       
        # pre-normalization
        x = self.ln_pre(x)
        B, N, C = x.shape
        x = x.reshape(B, N, -1, 64)
        x = x.permute(0, 2, 1, 3)
        # print('pre x mean: ', x.mean().item())
        # print('pre x var: ', x.var().item())
        x = x + self.rope(torch.ones_like(x).to(x), pos)
        # print('x mean: ', x.mean().item())
        # print('x var: ', x.var().item())

        x = x.permute(0, 2, 1, 3)
        x = x.reshape(B, N, -1)
        # use encode to extract features
        if self.encode_layers > 0:
            x = x.permute(1, 0, 2)  # NLD -> LND
            x = self.encoder(x, modulation, context, additional_residuals=additional_residuals)
            x = x.permute(1, 0, 2)  # LND -> NLD
        if not self.disable_dino:
            x = x.permute(1, 0, 2)  # NLD -> LND
        else:    
            if not abla_crossview:
                # flatten x along the video dimension
                x = rearrange(x, '(b v) n d -> b (v n) d', v=vs)
                # print(x.shape)
                x = x.permute(1, 0, 2)  # NLD -> LND
            else:
                x = x.permute(1, 0, 2)
        x = self.transformer(x, modulation, context, additional_residuals=additional_residuals)


        if self.add_zero_conv:
            assert isinstance(x, (list, tuple))
            assert len(x) == len(self.zero_convs)
            new_x = []
            for sub_x, sub_zero_conv in zip(x, self.zero_convs):
                sub_x_out = sub_zero_conv(sub_x.permute(1, 2, 0))
                new_x.append(sub_x_out.permute(2, 0, 1))
            x = new_x

        if self.return_all_layers:
            assert isinstance(x, (list, tuple))
            if not self.disable_post_ln:
                x_final = x[-1].permute(1, 0, 2)  # LND -> NLD
                x_final = self.ln_post(x_final)
                x_final = rearrange(x_final, 'b (v n) d -> b v n d', v=vs)
            x = [s.permute(1, 0, 2) for s in x]
            x.append(x_final)
            return x

        if not self.disable_post_ln:
            x = x.permute(1, 0, 2)  # LND -> NLD
            x = self.ln_post(x)
        if not self.disable_dino:
            x = rearrange(x, '(b v) n d -> b v n d', b=bs, v=vs)
        else:
            if not abla_crossview:
                # reshape x back to video dimension
                x = rearrange(x, 'b (v n) d -> b v n d', v=vs)
            else:
                x = rearrange(x, '(b v) n d -> b v n d', v=vs)
        return x

    def extra_repr(self) -> str:
        pass


class VisionTransformer_fusion(nn.Module):
    def __init__(self, 
                 # transformer params
                 in_channels: int, 
                 patch_size: int, 
                 width: int, 
                 layers: int, 
                 heads: int, 
                 weight: str = None,
                 encode_layers: int = 0,
                 shift_group = False,
                 flash_v2 = False,
                 qkv_packed = False,
                 window_size = False,
                 use_pe = False,
                 # modualtion params 
                 modulate_feature_size: int = None, 
                 modulate_act_type: str = 'gelu', 
                 # camera condition
                 camera_condition: str = 'plucker',
                 # init params
                 disable_dino=False,
                 error_weight_init_mode='mean', 
                 # other params
                 add_zero_conv=False, 
                 return_all_layers=False, 
                 disable_post_ln=False,
                 rope=None):
        super().__init__()
        self.patch_size = patch_size
        self.use_pe = use_pe
        self.rope = rope
        self.disable_dino = disable_dino
        # if not self.disable_dino:
        #     scale = width ** -0.5
        #     self.class_embedding = nn.Parameter(scale * torch.randn(width))
        #     self.positional_embedding = nn.Parameter(scale * torch.randn((input_res// patch_size) ** 2 + 1, width))
        # else:
        #     if self.use_pe:
        # self.positional_embedding = nn.Parameter(torch.zeros(1, (input_res// patch_size) ** 2, width))
        # nn.init.trunc_normal_(self.positional_embedding, std=0.02)
        self.ln_pre = LayerNorm(width)
        self.add_zero_conv = add_zero_conv
        self.return_all_layers = return_all_layers
        self.disable_post_ln = disable_post_ln
        self.flash_v2 = flash_v2
        self.qkv_packed = qkv_packed

        self.camera_condition = camera_condition
        if self.camera_condition == 'plucker': assert modulate_feature_size is None

        if self.add_zero_conv:
            assert self.return_all_layers
            self.zero_convs = nn.ModuleList([zero_module(nn.Conv1d(in_channels=width, out_channels=width, kernel_size=1, stride=1, bias=True)) for _ in range(layers)])

        self.encode_layers = encode_layers
        if self.encode_layers > 0:
            self.encoder = Transformer(width, encode_layers, heads, 
                                       modulate_feature_size=modulate_feature_size, 
                                       modulate_act_type=modulate_act_type, 
                                       cross_att_layers=0, 
                                       return_all_layers=return_all_layers, 
                                       flash_v2=flash_v2, 
                                       qkv_packed=qkv_packed,
                                       shift_group=shift_group,
                                       window_size=window_size)
        self.transformer = Transformer(width, layers-encode_layers, heads, 
                                       modulate_feature_size=modulate_feature_size, 
                                       modulate_act_type=modulate_act_type, 
                                       cross_att_layers=0, 
                                       return_all_layers=return_all_layers, 
                                       flash_v2=flash_v2, 
                                       qkv_packed=qkv_packed,
                                       shift_group=shift_group,
                                       window_size=window_size)

        if not self.disable_post_ln:
            self.ln_post = LayerNorm(width)

        if weight is not None:
            if not self.disable_dino:
                if "clip" in weight:
                    raise NotImplementedError()
                elif weight.startswith("vit_b_16"):
                    load_timm_to_clip(self, config_name=weight, init_mode=error_weight_init_mode)
                elif weight.startswith("vit_b_8"):
                    load_timm_to_clip(self, config_name=weight, init_mode=error_weight_init_mode)
                else:
                    raise NotImplementedError()
            else:
                self.apply(_init_weights)

            # Init the weight and bias of modulation_fn to zero
            if modulate_feature_size != 0:
                for block in self.transformer.resblocks:
                    if block.modulation_fn is not None:
                        block.modulation_fn[2].weight.data.zero_()
                        block.modulation_fn[2].bias.data.zero_()
                        if block.mlp_modulation_fn is not None:
                            block.mlp_modulation_fn[2].weight.data.zero_()
                            block.mlp_modulation_fn[2].bias.data.zero_()
            for block in self.transformer.resblocks:
                if block.cross_att:
                    zero_module(block.cross_att.to_out)

    def set_grad_checkpointing(self, flag=True):
        self.transformer.set_grad_checkpointing(flag)

    def forward(self, 
            x: torch.Tensor, 
            modulation: torch.Tensor = None, 
            context: torch.Tensor = None, 
            additional_residuals=None,
            abla_crossview=False,
            pos=None):

        # image tokenization
        bs, vs = x.shape[:2]
        x = rearrange(x, 'b v h g -> (b v) h g')  # shape = [*, grid ** 2, width]
        pos = rearrange(pos, 'b v c h -> (b v) c h')
       
        # pre-normalization
        B, N, C = x.shape
        x = x.reshape(B, N, -1, 64)
        x = x.permute(0, 2, 1, 3)
        # print('pre x mean: ', x.mean().item())
        # print('pre x var: ', x.var().item())
        x = x + self.rope(torch.ones_like(x).to(x), pos)
        # print('x mean: ', x.mean().item())
        # print('x var: ', x.var().item())
        x = x.permute(0, 2, 1, 3)
        x = x.reshape(B, N, -1)
        # use encode to extract features
        if self.encode_layers > 0:
            x = x.permute(1, 0, 2)  # NLD -> LND
            x = self.encoder(x, modulation, context, additional_residuals=additional_residuals)
            x = x.permute(1, 0, 2)  # LND -> NLD
        if not self.disable_dino:
            x = x.permute(1, 0, 2)  # NLD -> LND
        else:    
            if not abla_crossview:
                # flatten x along the video dimension
                x = rearrange(x, '(b v) n d -> b (v n) d', v=vs)
                # print(x.shape)
                x = x.permute(1, 0, 2)  # NLD -> LND
            else:
                x = x.permute(1, 0, 2)
        x = self.transformer(x, modulation, context, additional_residuals=additional_residuals)

        if self.add_zero_conv:
            assert isinstance(x, (list, tuple))
            assert len(x) == len(self.zero_convs)
            new_x = []
            for sub_x, sub_zero_conv in zip(x, self.zero_convs):
                sub_x_out = sub_zero_conv(sub_x.permute(1, 2, 0))
                new_x.append(sub_x_out.permute(2, 0, 1))
            x = new_x

        if self.return_all_layers:
            assert isinstance(x, (list, tuple))
            if not self.disable_post_ln:
                x_final = x[-1].permute(1, 0, 2)  # LND -> NLD
                x_final = self.ln_post(x_final)
                x_final = rearrange(x_final, 'b (v n) d -> b v n d', v=vs)
            x = [s.permute(1, 0, 2) for s in x]
            x.append(x_final)
            return x

        if not self.disable_post_ln:
            x = x.permute(1, 0, 2)  # LND -> NLD
            x = self.ln_post(x)
        if not self.disable_dino:
            x = rearrange(x, '(b v) n d -> b v n d', b=bs, v=vs)
        else:
            if not abla_crossview:
                # reshape x back to video dimension
                x = rearrange(x, 'b (v n) d -> b v n d', v=vs)
            else:
                x = rearrange(x, '(b v) n d -> b v n d', v=vs)
        return x

    def extra_repr(self) -> str:
        pass


def resize_pos_embed(state_dict, model, interpolation: str = 'bicubic'):
    """
    Resize positional embeddings, implementation from google/simclr and open_clip.
    """
    # Rescale the grid of position embeddings when loading from state_dict
    old_pos_embed = state_dict.get('positional_embedding', None)
    if old_pos_embed is None:
        return

    # Compute the grid size and extra tokens
    old_pos_len = state_dict["positional_embedding"].shape[0]
    old_grid_size = round((state_dict["positional_embedding"].shape[0]) ** 0.5)
    grid_size = round((model.positional_embedding.shape[0]) ** 0.5)
    if old_grid_size == grid_size:
        return
    extra_tokens = old_pos_len - (old_grid_size ** 2)

    if extra_tokens:
        pos_emb_tok, pos_emb_img = old_pos_embed[:extra_tokens], old_pos_embed[extra_tokens:]
    else:
        pos_emb_tok, pos_emb_img = None, old_pos_embed

    # Only interpolate the positional emb part, not the extra token part.
    pos_emb_img = pos_emb_img.reshape(1, old_grid_size, old_grid_size, -1).permute(0, 3, 1, 2)
    pos_emb_img = F.interpolate(
        pos_emb_img,
        size=grid_size,
        mode=interpolation,
        align_corners=True,
    )
    pos_emb_img = pos_emb_img.permute(0, 2, 3, 1).reshape(1, grid_size * grid_size, -1)[0]

    # Concatenate back the
    if pos_emb_tok is not None:
        new_pos_embed = torch.cat([pos_emb_tok, pos_emb_img], dim=0)
    else:
        new_pos_embed = pos_emb_img
    state_dict['positional_embedding'] = new_pos_embed


myname2timmname = {
    "vit_b_16_mae": None,
    "vit_b_16_in": "vit_base_patch16_224",
    "vit_b_16_in21k": 'vit_base_patch16_224_in21k',
    "vit_b_16_sam": 'vit_base_patch16_224_sam',
    "vit_b_16_dino": 'vit_base_patch16_224_dino',
    "vit_b_16_mill_in21k": 'vit_base_patch16_224_miil_in21k',
    "vit_b_16_mill": 'vit_base_patch16_224_miil',
    "vit_b_8_dino": 'vit_base_patch16_224_dino',
}

def load_timm_to_clip(module, config_name="vit_b_16_mae", init_mode='zero'):
    from torch import nn
    from clip.model import LayerNorm as CLIPLayerNorm
    from clip.model import QuickGELU

    from torch.nn import GELU
    from torch.nn import LayerNorm

    import json
    now_dir = os.path.abspath(os.path.dirname(__file__))
    timm2clip = json.load(open(f"{now_dir}/timm2clip_vit_b_16.json"))

    assert config_name in myname2timmname, f"The name {config_name} is not one of {list(myname2timmname.keys())}"
    try:
        timm_weight = torch.load(f"/sensei-fs/users/hatan/model/{config_name}.pth")["model"]
    except Exception as e:
        try:
            print(f"/input/yhxu/models/dino_weights/{config_name}.pth")
            timm_weight = torch.load(f"/input/yhxu/models/dino_weights/{config_name}.pth")["model"]
        except Exception as e:
            try:
                print(f"/home/yhxu/models/dino_weights/{config_name}.pth")
                timm_weight = torch.load(f"/home/yhxu/models/dino_weights/{config_name}.pth")["model"]
            except:
                try:
                    timm_weight = torch.load(f"/nas2/zifan/checkpoint/dino_weights/{config_name}.pth")["model"]
                except Exception as e:
                    print("Please download weight with support/dump_timm_weights.py. \n"
                        "If using mae weight, please check https://github.com/facebookresearch/mae,"
                        "and download the weight as vit_b_16_mae.pth")
                    assert False

    # Build model's state dict
    clipname2timmweight = {}
    for timm_key, clip_key in timm2clip.items():
        timm_value = timm_weight[timm_key]
        clipname2timmweight[clip_key[len("visual."):]] = timm_value.squeeze()

    # Resize positional embedding
    resize_pos_embed(clipname2timmweight, module)

    # Load weight to model.
    model_visual_keys = set(module.state_dict().keys())
    load_keys = set(clipname2timmweight.keys())
    # print(f"Load not in model: {load_keys - model_visual_keys}")
    # print(f"Model not in load: {model_visual_keys - load_keys}")
    # status = module.load_state_dict(clipname2timmweight, strict=False)
    try:
        status = module.load_state_dict(clipname2timmweight, strict=False)
    except:
        print('conv.weight has error!')
        if init_mode == 'zero':
            new_weight = torch.zeros_like(clipname2timmweight['conv1.weight'])
            new_weight = new_weight.repeat(1, 2, 1, 1)
            new_weight[:,:3] = clipname2timmweight['conv1.weight']
        elif init_mode == 'mean':
            new_weight = torch.zeros_like(clipname2timmweight['conv1.weight'])
            new_weight = new_weight.repeat(1, 3, 1, 1)
            new_weight = ((clipname2timmweight['conv1.weight']).repeat(1, 3, 1, 1))/3

        clipname2timmweight['conv1.weight'] = new_weight
        status = module.load_state_dict(clipname2timmweight, strict=False)

    # Since timm model has bias, we add it back here.
    module.conv1.bias = nn.Parameter(clipname2timmweight['conv1.bias'])

    # Reinit the visual weights that not covered by timm
    module.ln_pre.reset_parameters()

    def convert_clip_to_timm(module):
        """Copy from detectron2, frozen BN"""
        res = module
        if isinstance(module, CLIPLayerNorm):
            # Timm uses eps=1e-6 while CLIP uses eps=1e-5
            res = LayerNorm(module.normalized_shape, eps=1e-6, elementwise_affine=module.elementwise_affine)
            if module.elementwise_affine:
                res.weight.data = module.weight.data.clone().detach()
                res.bias.data = module.bias.data.clone().detach()
        elif isinstance(module, QuickGELU):
            # Timm uses GELU while CLIP uses QuickGELU
            res = GELU()
        else:
            for name, child in module.named_children():
                new_child = convert_clip_to_timm(child)
                if new_child is not child:
                    res.add_module(name, new_child)
        return res
