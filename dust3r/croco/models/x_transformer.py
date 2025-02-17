"""from https://github.com/lucidrains/x-transformers"""
import math
from random import random

import torch
from torch import nn, einsum
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from functools import partial, wraps
from inspect import isfunction

from einops import rearrange, repeat, reduce


# constants

DEFAULT_DIM_HEAD = 64


# helpers

def exists(val):
    return val is not None


def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d


def cast_tuple(val, depth):
    return val if isinstance(val, tuple) else (val,) * depth


# init helpers

def init_zero_(layer):
    nn.init.constant_(layer.weight, 0.)
    if exists(layer.bias):
        nn.init.constant_(layer.bias, 0.)


# keyword argument helpers

def pick_and_pop(keys, d):
    values = list(map(lambda key: d.pop(key), keys))
    return dict(zip(keys, values))


def group_dict_by_key(cond, d):
    return_val = [dict(), dict()]
    for key in d.keys():
        match = bool(cond(key))
        ind = int(not match)
        return_val[ind][key] = d[key]
    return (*return_val,)


def string_begins_with(prefix, str):
    return str.startswith(prefix)


def group_by_key_prefix(prefix, d):
    return group_dict_by_key(partial(string_begins_with, prefix), d)


def groupby_prefix_and_trim(prefix, d):
    kwargs_with_prefix, kwargs = group_dict_by_key(partial(string_begins_with, prefix), d)
    kwargs_without_prefix = dict(map(lambda x: (x[0][len(prefix):], x[1]), tuple(kwargs_with_prefix.items())))
    return kwargs_without_prefix, kwargs


# initializations

def deepnorm_init(
        transformer,
        beta,
        module_name_match_list=['.ff.', '.to_v', '.to_out']
):
    for name, module in transformer.named_modules():
        if type(module) != nn.Linear:
            continue

        needs_beta_gain = any(map(lambda substr: substr in name, module_name_match_list))
        gain = beta if needs_beta_gain else 1
        nn.init.xavier_normal_(module.weight.data, gain=gain)

        if exists(module.bias):
            nn.init.constant_(module.bias.data, 0)


# activations

class ReluSquared(nn.Module):
    def forward(self, x):
        return F.relu(x) ** 2


# norms

class Scale(nn.Module):
    def __init__(self, value, fn):
        super().__init__()
        self.value = value
        self.fn = fn

    def forward(self, x, **kwargs):
        out = self.fn(x, **kwargs)
        scale_fn = lambda t: t * self.value

        if not isinstance(out, tuple):
            return scale_fn(out)

        return (scale_fn(out[0]), *out[1:])


class ScaleNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.g = nn.Parameter(torch.ones(1) * (dim ** -0.5))

    def forward(self, x):
        norm = torch.norm(x, dim=-1, keepdim=True)
        return x / norm.clamp(min=self.eps) * self.g


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-8):
        super().__init__()
        self.scale = dim ** -0.5
        self.eps = eps
        self.g = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        norm = torch.norm(x, dim=-1, keepdim=True) * self.scale
        return x / norm.clamp(min=self.eps) * self.g


# residual and residual gates

class Residual(nn.Module):
    def __init__(self, dim, scale_residual=False, scale_residual_constant=1.):
        super().__init__()
        self.residual_scale = nn.Parameter(torch.ones(dim)) if scale_residual else None
        self.scale_residual_constant = scale_residual_constant

    def forward(self, x, residual):
        if exists(self.residual_scale):
            residual = residual * self.residual_scale

        if self.scale_residual_constant != 1:
            residual = residual * self.scale_residual_constant

        return x + residual


class GRUGating(nn.Module):
    def __init__(self, dim, scale_residual=False, **kwargs):
        super().__init__()
        self.gru = nn.GRUCell(dim, dim)
        self.residual_scale = nn.Parameter(torch.ones(dim)) if scale_residual else None

    def forward(self, x, residual):
        if exists(self.residual_scale):
            residual = residual * self.residual_scale

        gated_output = self.gru(
            rearrange(x, 'b n d -> (b n) d'),
            rearrange(residual, 'b n d -> (b n) d')
        )

        return gated_output.reshape_as(x)


# feedforward
class GLU(nn.Module):
    def __init__(self, dim_in, dim_out, activation):
        super().__init__()
        self.act = activation
        self.proj = nn.Linear(dim_in, dim_out * 2)

    def forward(self, x):
        x, gate = self.proj(x).chunk(2, dim=-1)
        return x * self.act(gate)


class FeedForward(nn.Module):
    def __init__(
            self,
            dim,
            dim_out=None,
            mult=4,
            glu=False,
            swish=False,
            relu_squared=False,
            post_act_ln=False,
            dropout=0.,
            no_bias=False,
            zero_init_output=False
    ):
        super().__init__()
        inner_dim = int(dim * mult)
        dim_out = default(dim_out, dim)

        if relu_squared:
            activation = ReluSquared()
        elif swish:
            activation = nn.SiLU()
        else:
            activation = nn.GELU()

        project_in = nn.Sequential(
            nn.Linear(dim, inner_dim, bias=not no_bias),
            activation
        ) if not glu else GLU(dim, inner_dim, activation)

        self.ff = nn.Sequential(
            project_in,
            nn.LayerNorm(inner_dim) if post_act_ln else nn.Identity(),
            nn.Dropout(dropout),
            nn.Linear(inner_dim, dim_out, bias=not no_bias)
        )

        # init last linear layer to 0
        if zero_init_output:
            init_zero_(self.ff[-1])

    def forward(self, x):
        return self.ff(x)


# attention.

class Attention(nn.Module):
    def __init__(
            self,
            dim,
            kv_dim=None,
            dim_head=DEFAULT_DIM_HEAD,
            heads=8,
            causal=False,
            dropout=0.,
            zero_init_output=False,
            shared_kv=False,
            value_dim_head=None,
            flash_attention=True,
    ):
        super().__init__()
        self.scale = dim_head ** -0.5
        if kv_dim is None:
            kv_dim = dim

        self.heads = heads
        self.causal = causal

        value_dim_head = default(value_dim_head, dim_head)
        q_dim = k_dim = dim_head * heads
        v_dim = out_dim = value_dim_head * heads

        self.to_q = nn.Linear(dim, q_dim, bias=False)
        self.to_k = nn.Linear(kv_dim, k_dim, bias=False)

        # shared key / values, for further memory savings during inference
        assert not (
                    shared_kv and value_dim_head != dim_head), 'key and value head dimensions must be equal for shared key / values'
        self.to_v = nn.Linear(kv_dim, v_dim, bias=False) if not shared_kv else None

        # Convert to output
        self.to_out = nn.Linear(out_dim, dim, bias=False)

        # dropout
        self.dropout_p = dropout
        self.dropout = nn.Dropout(dropout)

        # Flash Attention, needs PyTorch >= 1.13
        self.flash = flash_attention
        assert self.flash

        # Use torch.nn.functional.scaled_dot_product_attention if available
        # otherwise, we use the xformer library.
        # self.use_xformer = True
        self.use_xformer = not hasattr(torch.nn.functional, 'scaled_dot_product_attention')

        # init output projection 0
        if zero_init_output:
            init_zero_(self.to_out)

    def forward(
            self,
            x,
            context=None,
            mask=None,
            context_mask=None,
    ):
        # print("x", x.dtype)
        h = self.heads
        kv_input = default(context, x)

        q_input = x
        k_input = kv_input
        v_input = kv_input

        q = self.to_q(q_input)
        k = self.to_k(k_input)
        v = self.to_v(v_input) if exists(self.to_v) else k

        # print("q", q.dtype)
        # print("k", k.dtype)
        # print("v", v.dtype)

        if self.use_xformer:
            # Since xformers only accepts bf16/fp16, we need to convert qkv to bf16/fp16
            dtype = q.dtype
            q, k, v = map(lambda t: t.bfloat16() if t.dtype == torch.float32 else t, (q, k, v))

            q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b n h d', h=h), (q, k, v))
            try:
                import xformers.ops as xops
            except ImportError as e:
                print("Please install xformers to use flash attention for PyTorch < 2.0.0.")
                raise e

            # Use the flash attention support from the xformers library
            if self.causal:
                attention_bias = xops.LowerTriangularMask()
            else:
                attention_bias = None

            # The memory_efficient_attention takes the input as (batch, seq_len, heads, dim)
            out = xops.memory_efficient_attention(
                q, k, v, attn_bias=attention_bias,
                # op=(xops.fmha.flash.FwOp, xops.fmha.flash.BwOp),
            )

            out = out.to(dtype)

            out = rearrange(out, 'b n h d -> b n (h d)')
        else:
            q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), (q, k, v))
            # efficient attention using Flash Attention CUDA kernels
            out = torch.nn.functional.scaled_dot_product_attention(
                q, k, v, attn_mask=None, dropout_p=self.dropout_p, is_causal=self.causal,
            )
            out = rearrange(out, 'b h n d -> b n (h d)')

        out = self.to_out(out)

        if exists(mask):
            mask = rearrange(mask, 'b n -> b n 1')
            out = out.masked_fill(~mask, 0.)

        return out

    def extra_repr(self) -> str:
        return f"causal: {self.causal}, flash attention: {self.flash}, " \
               f"use_xformers (if False, use torch.nn.functional.scaled_dot_product_attention): {self.use_xformer}"


def modulate(x, shift, scale):
    # from https://github.com/facebookresearch/DiT/blob/796c29e532f47bba17c5b9c5eb39b9354b8b7c64/models.py#L19
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class AttentionLayers(nn.Module):
    def __init__(
            self,
            dim,
            depth,
            heads=8,
            ctx_dim=None,
            causal=False,
            cross_attend=False,
            only_cross=False,
            use_scalenorm=False,
            use_rmsnorm=False,
            residual_attn=False,
            cross_residual_attn=False,
            macaron=False,
            pre_norm=True,
            gate_residual=False,
            scale_residual=False,
            scale_residual_constant=1.,
            deepnorm=False,
            sandwich_norm=False,
            zero_init_branch_output=False,
            layer_dropout=0.,
            # Below are the arguments used for this img2nerf projects
            modulate_feature_size=-1,
            checkpointing=False,
            checkpoint_every=1,
            **kwargs
    ):
        super().__init__()

        # Add checkpointing
        self.checkpointing = checkpointing
        self.checkpoint_every = checkpoint_every

        ff_kwargs, kwargs = groupby_prefix_and_trim('ff_', kwargs)
        attn_kwargs, kwargs = groupby_prefix_and_trim('attn_', kwargs)

        self.dim = dim
        self.depth = depth
        self.layers = nn.ModuleList([])

        # determine deepnorm and residual scale
        if deepnorm:
            assert scale_residual_constant == 1, 'scale residual constant is being overridden by deep norm settings'
            pre_norm = sandwich_norm = False
            scale_residual = True
            scale_residual_constant = (2 * depth) ** 0.25

        assert not (not pre_norm and sandwich_norm), 'sandwich norm cannot be used when not using prenorm'
        self.pre_norm = pre_norm
        self.sandwich_norm = sandwich_norm

        self.residual_attn = residual_attn
        self.cross_residual_attn = cross_residual_attn
        self.cross_attend = cross_attend

        norm_class = ScaleNorm if use_scalenorm else nn.LayerNorm
        norm_class = RMSNorm if use_rmsnorm else norm_class
        norm_fn = partial(norm_class, dim)

        if cross_attend and not only_cross:
            default_block = ('a', 'c', 'f')
        elif cross_attend and only_cross:
            default_block = ('c', 'f')
        else:
            default_block = ('a', 'f')

        if macaron:
            default_block = ('f',) + default_block

        # zero init

        if zero_init_branch_output:
            attn_kwargs = {**attn_kwargs, 'zero_init_output': True}
            ff_kwargs = {**ff_kwargs, 'zero_init_output': True}

        # calculate layer block order
        layer_types = default_block * depth

        self.layer_types = layer_types

        # stochastic depth
        self.layer_dropouts = cast_tuple(layer_dropout, len(layer_types))

        # iterate and construct layers
        for ind, layer_type in enumerate(self.layer_types):
            is_last_layer = ind == (len(self.layer_types) - 1)

            if layer_type == 'a':
                layer = Attention(dim, heads=heads, causal=causal, **attn_kwargs)
            elif layer_type == 'c':
                layer = Attention(dim, kv_dim=ctx_dim, heads=heads, **attn_kwargs)
            elif layer_type == 'f':
                layer = FeedForward(dim, **ff_kwargs)
                layer = layer if not macaron else Scale(0.5, layer)
            else:
                raise Exception(f'invalid layer type {layer_type}')

            residual_fn = GRUGating if gate_residual else Residual
            residual = residual_fn(dim, scale_residual=scale_residual, scale_residual_constant=scale_residual_constant)

            pre_branch_norm = norm_fn() if pre_norm else None
            post_branch_norm = norm_fn() if sandwich_norm else None
            post_main_norm = norm_fn() if not pre_norm and not is_last_layer else None

            # The whole modulation part is copied from DiT
            # https://github.com/facebookresearch/DiT
            modulation = None
            if modulate_feature_size is not None:
                modulation = nn.Sequential(
                    nn.LayerNorm(modulate_feature_size),
                    nn.GELU(),
                    nn.Linear(modulate_feature_size, 3 * dim, bias=True)
                )

            norms = nn.ModuleList([
                pre_branch_norm,
                post_branch_norm,
                post_main_norm,
            ])

            self.layers.append(nn.ModuleList([
                norms,
                layer,
                residual,
                modulation,
            ]))

        if deepnorm:
            init_gain = (8 * depth) ** -0.25
            deepnorm_init(self, init_gain)

    def forward(
            self,
            x,
            context=None,
            modulation=None,
            mask=None,
            context_mask=None,
    ):
        assert not (self.cross_attend ^ exists(context)), 'context must be passed in if cross_attend is set to True'

        num_layers = len(self.layer_types)
        assert num_layers % self.checkpoint_every == 0

        for start_layer_idx in range(0, num_layers, self.checkpoint_every):
            end_layer_idx = min(start_layer_idx + self.checkpoint_every, num_layers)

            def run_layers(x, context, modulation, start, end):
                for ind, (layer_type, (norm, block, residual_fn, modulation_fn), layer_dropout) in enumerate(
                        zip(self.layer_types[start: end], self.layers[start: end], self.layer_dropouts[start: end])):
                    residual = x

                    pre_branch_norm, post_branch_norm, post_main_norm = norm

                    if exists(pre_branch_norm):
                        x = pre_branch_norm(x)

                    if modulation_fn is not None:
                        shift, scale, gate = modulation_fn(modulation).chunk(3, dim=1)
                        x = modulate(x, shift, scale)

                    if layer_type == 'a':
                        out = block(x, mask=mask)
                    elif layer_type == 'c':
                        out = block(x, context=context, mask=mask, context_mask=context_mask)
                    elif layer_type == 'f':
                        out = block(x)

                    if exists(post_branch_norm):
                        out = post_branch_norm(out)

                    if modulation_fn is not None:
                        # TODO: add a option to use gate or not.
                        out = out * gate.unsqueeze(1)

                    x = residual_fn(out, residual)

                    if exists(post_main_norm):
                        x = post_main_norm(x)

                return x

            if self.checkpointing:
                # print("X checkpointing")
                x = checkpoint(run_layers, x, context, modulation, start_layer_idx, end_layer_idx)
            else:
                x = run_layers(x, context, modulation, start_layer_idx, end_layer_idx)

        return x

