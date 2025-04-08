#!/usr/bin/env python3
# Copyright (C) 2024-present Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).
#
# --------------------------------------------------------
# training executable for MASt3R
# --------------------------------------------------------
import mast3r.utils.path_to_dust3r  # noqa
from mast3r.model import AsymmetricMASt3R
from dust3r.datasets import Co3d
from dust3r.losses import Regr3D_clean
import dust3r.training
dust3r.training.AsymmetricMASt3R = AsymmetricMASt3R
import dust3r.datasets
dust3r.datasets.Co3d = Co3d
from dust3r.training import get_args_parser as dust3r_get_args_parser  # noqa
from dust3r.training import train  # noqa


def get_args_parser():
    parser = dust3r_get_args_parser()
    parser.add_argument('--stage1_pretrained', default=None, help='path of a starting checkpoint')
    parser.prog = 'MASt3R training'
    parser.set_defaults(model="AsymmetricMASt3R(patch_embed_cls='ManyAR_PatchEmbed')")
    return parser


if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    train(args)
