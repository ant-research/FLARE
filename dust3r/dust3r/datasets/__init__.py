# Copyright (C) 2024-present Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).
from .utils.transforms import *
from .base.batched_sampler import BatchedRandomSampler  # noqa
from .arkitscenes import ARKitScenes  # noqa
from .blendedmvs import BlendedMVS  # noqa
from .co3d import Co3d  # noqa
from .habitat import Habitat  # noqa
from .megadepth import MegaDepth  # noqa
from .scannetpp import ScanNetpp  # noqa
from .staticthings3d import StaticThings3D  # noqa
from .waymo import Waymo  # noqa
from .wildrgbd import WildRGBD  # noqa
from .tartanair import TartanAir  # noqa
from .blendedmvs_test import BlendedMVStest  # noqa
from .GTASFM import GTASFM  # noqa
from .GTAV import GTAV  # noqa
from .SceneNet import SceneNet  # noqa
from .DL3DV import DL3DV  # noqa
from .hypersim import Hypersim  # noqa
from .co3dtest import Co3dtest  # noqa
from .imc import IMCDataset
from .blendedmvsof import BlendedMVSof
from .ACID import ACID
from .Re10K import Re10K
from .Own import Own
from .ETH3D import ETH3D
from .dtu import DTU
from .DL3DV_test import DL3DV_test
from .Re10K_ps import Re10K_ps
from .Re10K_nopo import Re10K_nopo
from .Re10K_pose import Re10K_pose
from .DL3DV_test_eval import DL3DV_test_eval
from .dtu_num import DTU_num
from .Own_seq import Own_seq
from .DL3DV_test_eval_2views import DL3DV_test_eval_2views
from .Re10K_ps_demo import Re10K_ps_demo

def get_data_loader(dataset, batch_size, num_workers=8, shuffle=True, drop_last=True, pin_mem=True):
    import torch
    from croco.utils.misc import get_world_size, get_rank

    # pytorch dataset
    if isinstance(dataset, str):
        dataset = eval(dataset)

    world_size = get_world_size()
    rank = get_rank()

    try:
        sampler = dataset.make_sampler(batch_size, shuffle=shuffle, world_size=world_size,
                                       rank=rank, drop_last=drop_last)
    except (AttributeError, NotImplementedError):
        # not avail for this dataset
        if torch.distributed.is_initialized():
            sampler = torch.utils.data.DistributedSampler(
                dataset, num_replicas=world_size, rank=rank, shuffle=shuffle, drop_last=drop_last
            )
        elif shuffle:
            sampler = torch.utils.data.RandomSampler(dataset)
        else:
            sampler = torch.utils.data.SequentialSampler(dataset)

    data_loader = torch.utils.data.DataLoader(
        dataset,
        sampler=sampler,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_mem,
        drop_last=drop_last,
    )

    return data_loader
