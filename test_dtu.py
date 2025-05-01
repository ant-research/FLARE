#!/usr/bin/env python3
# Copyright (C) 2024-present Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).
#
# --------------------------------------------------------
# training code for DUSt3R
# --------------------------------------------------------
# References:
# MAE: https://github.com/facebookresearch/mae
# DeiT: https://github.com/facebookresearch/deit
# BEiT: https://github.com/microsoft/unilm/tree/master/beit
# --------------------------------------------------------
import argparse
import datetime
import json
import numpy as np
import os
import sys
import time
import math
from collections import defaultdict
from pathlib import Path
from typing import Sized
import mast3r.utils.path_to_dust3r  # noqa
from collections import defaultdict
from typing import Sized
from dust3r.utils.geometry import inv, geotrf, normalize_pointcloud#, matrix_to_quaternion, quaternion_to_matrix

import torch
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter
torch.backends.cuda.matmul.allow_tf32 = True  # for gpu >= Ampere and pytorch >= 1.12
from dust3r.viz import add_scene_cam, CAM_COLORS, OPENGL, pts3d_to_trimesh, cat_meshes
from mast3r.model_pose_clean_stage2_joint import AsymmetricMASt3R
from mast3r.model  import AsymmetricMASt3R as AsymmetricMASt3R_orig
from mast3r.model_pose_clean_stage2_joint_selfatten import AsymmetricMASt3R as AsymmetricMASt3R_self
from mast3r.model_pose_clean_stage2_joint_selfatten_inject import AsymmetricMASt3R as AsymmetricMASt3R_inject

from dust3r.datasets import get_data_loader  # noqa
from dust3r.losses import *  # noqa: F401, needed when loading the model
from dust3r.inference import loss_of_one_batch  # noqa
inf = float('inf')
from mast3r.losses import RotErr, ConfMatchingLoss, MatchingLoss, APLoss, InfoNCE, Regr3D_ScaleShiftInv, MeshOutput, TestCorr, TestScaling, DTUMetric, TriangulationMetric, MeshOutput_dtu
import trimesh
import dust3r.utils.path_to_croco  # noqa: F401
import croco.utils.misc as misc  # noqa
from croco.utils.misc import NativeScalerWithGradNormCount as NativeScaler  # noqa



def get_args_parser():
    parser = argparse.ArgumentParser('DUST3R training', add_help=False)
    # model and criterion
    parser.add_argument('--model', default="AsymmetricCroCo3DStereo(patch_embed_cls='ManyAR_PatchEmbed')",
                        type=str, help="string containing the model to build")
    parser.add_argument('--pretrained', default=None, help='path of a starting checkpoint')
    parser.add_argument('--test_criterion', default=None, type=str, help="test criterion")

    # dataset
    parser.add_argument('--test_dataset', default='[None]', type=str, help="testing set")

    # training
    parser.add_argument('--seed', default=0, type=int, help="Random seed")
    parser.add_argument('--batch_size', default=64, type=int,
                        help="Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus")
    parser.add_argument('--accum_iter', default=1, type=int,
                        help="Accumulate gradient iterations (for increasing the effective batch size under memory constraints)")
    parser.add_argument('--epochs', default=800, type=int, help="Maximum number of epochs for the scheduler")

    parser.add_argument('--weight_decay', type=float, default=0.05, help="weight decay (default: 0.05)")
    parser.add_argument('--lr', type=float, default=None, metavar='LR', help='learning rate (absolute lr)')
    parser.add_argument('--blr', type=float, default=1.5e-4, metavar='LR',
                        help='base learning rate: absolute_lr = base_lr * total_batch_size / 256')
    parser.add_argument('--min_lr', type=float, default=0., metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0')
    parser.add_argument('--warmup_epochs', type=int, default=40, metavar='N', help='epochs to warmup LR')

    parser.add_argument('--amp', type=int, default=0,
                        choices=[0, 1], help="Use Automatic Mixed Precision for pretraining")

    # others
    parser.add_argument('--num_workers', default=8, type=int)
    parser.add_argument('--world_size', default=1, type=int, help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')

    parser.add_argument('--eval_freq', type=int, default=1, help='Test loss evaluation frequency')
    parser.add_argument('--save_freq', default=1, type=int,
                        help='frequence (number of epochs) to save checkpoint in checkpoint-last.pth')
    parser.add_argument('--keep_freq', default=20, type=int,
                        help='frequence (number of epochs) to save checkpoint in checkpoint-%d.pth')
    parser.add_argument('--print_freq', default=20, type=int,
                        help='frequence (number of iterations) to print infos while training')
    parser.add_argument('--noise_trans', default=0.05, type=float, help='translation noise')
    parser.add_argument('--noise_rot', default=10, type=float, help='rotation noise')
    parser.add_argument('--noise_prob', default=0.5, type=float, help='rotation noise')
    parser.add_argument('--save_input_image', default=False, type=bool)

    # output dir
    parser.add_argument('--output_dir', default='./output/', type=str, help="path where to save the output")
    return parser

def main(args):
    print("output_dir: "+args.output_dir)
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    # auto resume
    last_ckpt_fname = os.path.join(args.output_dir, f'checkpoint-last.pth')
    args.resume = last_ckpt_fname if os.path.isfile(last_ckpt_fname) else None

    print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(', ', ',\n'))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)

    # training dataset and loader
    data_loader_test = {dataset.split('(')[0]: build_dataset(dataset, args.batch_size, args.num_workers, test=True)
                        for dataset in args.test_dataset.split('+')}

    # model
    print('Loading model: {:s}'.format(args.model))
    model = eval(args.model)
    test_criterion = eval(args.test_criterion or args.criterion)
    # model_orig = eval("AsymmetricMASt3R_orig(pos_embed='RoPE100', patch_embed_cls='ManyAR_PatchEmbed', img_size=(512, 512), head_type='catmlp+dpt', output_mode='pts3d+desc24', depth_mode=('exp', -inf, inf), conf_mode=('exp', 1, inf), enc_embed_dim=1024, enc_depth=24, enc_num_heads=16, dec_embed_dim=768, dec_depth=12, dec_num_heads=12, two_confs=True)")
    
    model.to(device)
    model.eval()
    model_without_ddp = model
    # model.set_noise_level(args.noise_trans, args.noise_rot, args.noise_prob)
    print("Model = %s" % str(model_without_ddp))

    if args.pretrained and not args.resume:
        print('Loading pretrained: ', args.pretrained)
        ckpt = torch.load(args.pretrained, map_location=device)
        # model_orig.load_state_dict(ckpt_orig['model'], strict=False)
        print(model.load_state_dict(ckpt['model'], strict=False))
        del ckpt  # in case it occupies memory

    eff_batch_size = args.batch_size * args.accum_iter * misc.get_world_size()
    if args.lr is None:  # only base_lr is specified
        args.lr = args.blr * eff_batch_size / 256
    print("base lr: %.2e" % (args.lr * 256 / eff_batch_size))
    print("actual lr: %.2e" % args.lr)
    print("accumulate grad iterations: %d" % args.accum_iter)
    print("effective batch size: %d" % eff_batch_size)



    global_rank = misc.get_rank()
    if global_rank == 0 and args.output_dir is not None:
        log_writer = SummaryWriter(log_dir=args.output_dir)
    else:
        log_writer = None
    param_groups = misc.get_parameter_groups(model_without_ddp, args.weight_decay)
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr, betas=(0.9, 0.95))
    print(optimizer)
    loss_scaler = NativeScaler()
    best_so_far = misc.load_model(args=args, model_without_ddp=model_without_ddp,
                                  optimizer=optimizer, loss_scaler=loss_scaler)
    if best_so_far is None:
        best_so_far = float('inf')

    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    train_stats = test_stats = {}
    
    ret_rot = []
    ret_init = []
    model_orig = None
    # scaling_unnorm_dict = {}
    # scaling_norm_dict = {}
    for test_name, testset in data_loader_test.items():
        scaling_list = []
        scaling_unnorm_list, scaling_norm_list = test_one_epoch(model, model_orig, test_criterion, testset,
                                device, 0, log_writer=log_writer, args=args, prefix=test_name)
        # scaling_unnorm_dict[test_name] = scaling_unnorm_list
        # scaling_norm_dict[test_name] = scaling_norm_list

def save_final_model(args, epoch, model_without_ddp, best_so_far=None):
    output_dir = Path(args.output_dir)
    checkpoint_path = output_dir / 'checkpoint-final.pth'
    to_save = {
        'args': args,
        'model': model_without_ddp if isinstance(model_without_ddp, dict) else model_without_ddp.cpu().state_dict(),
        'epoch': epoch
    }
    if best_so_far is not None:
        to_save['best_so_far'] = best_so_far
    print(f'>> Saving model to {checkpoint_path} ...')
    misc.save_on_master(to_save, checkpoint_path)


def build_dataset(dataset, batch_size, num_workers, test=False):
    split = ['Train', 'Test'][test]
    print(f'Building {split} Data loader for dataset: ', dataset)
    loader = get_data_loader(dataset,
                             batch_size=batch_size,
                             num_workers=num_workers,
                             pin_mem=True,
                             shuffle=not (test),
                             drop_last=not (test))

    print(f"{split} dataset length: ", len(loader))
    return loader


def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Sized, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler,
                    args,
                    log_writer=None):
    assert torch.backends.cuda.matmul.allow_tf32 == True

    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    accum_iter = args.accum_iter

    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))

    if hasattr(data_loader, 'dataset') and hasattr(data_loader.dataset, 'set_epoch'):
        data_loader.dataset.set_epoch(epoch)
    if hasattr(data_loader, 'sampler') and hasattr(data_loader.sampler, 'set_epoch'):
        data_loader.sampler.set_epoch(epoch)

    optimizer.zero_grad()
    gt_num_image = data_loader.dataset.dataset.gt_num_image
    for data_iter_step, batch in enumerate(metric_logger.log_every(data_loader, args.print_freq, header)):
        epoch_f = epoch + data_iter_step / len(data_loader)

        # we use a per iteration (instead of per epoch) lr scheduler
        if data_iter_step % accum_iter == 0:
            misc.adjust_learning_rate(optimizer, epoch_f, args)

        loss_tuple = loss_of_one_batch(batch, model, criterion, device,
                                       symmetrize_batch=True,
                                       use_amp=bool(args.amp), ret='loss')
        loss, loss_details = loss_tuple  # criterion returns two values
        loss_value = float(loss)

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value), force=True)
            sys.exit(1)

        loss /= accum_iter
        loss_scaler(loss, optimizer, parameters=model.parameters(),
                    update_grad=(data_iter_step + 1) % accum_iter == 0)
        if (data_iter_step + 1) % accum_iter == 0:
            optimizer.zero_grad()
        images_input = [gt['img_org'] for gt in batch]
        (H_org, W_org) = batch[0]['true_shape'][0]
        labels = [gt['label'] for gt in batch]
        del loss
        del batch

        lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(epoch=epoch_f)
        metric_logger.update(lr=lr)
        metric_logger.update(loss=loss_value, **loss_details)

        if (data_iter_step) % accum_iter == 0 and ((data_iter_step) % (accum_iter * args.print_freq)) == 0:
            loss_value_reduce = misc.all_reduce_mean(loss_value)  # MUST BE EXECUTED BY ALL NODES
            if log_writer is None:
                continue
            """ We use epoch_1000x as the x-axis in tensorboard.
            This calibrates different curves when batch size changes.
            """
            epoch_1000x = int(epoch_f * 1000)
            log_writer.add_scalar('train_loss', loss_value_reduce, epoch_1000x)
            log_writer.add_scalar('train_lr', lr, epoch_1000x)
            log_writer.add_scalar('train_iter', epoch_1000x, epoch_1000x)
           
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def test_one_epoch(model: torch.nn.Module, model_test, criterion: torch.nn.Module,
                   data_loader: Sized, device: torch.device, epoch: int,
                   args, log_writer=None, prefix='test'):

    model.eval()
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.meters = defaultdict(lambda: misc.SmoothedValue(window_size=9**9))
    header = 'Test Epoch: [{}]'.format(epoch)
    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))

    if hasattr(data_loader, 'dataset') and hasattr(data_loader.dataset, 'set_epoch'):
        data_loader.dataset.set_epoch(epoch)
    if hasattr(data_loader, 'sampler') and hasattr(data_loader.sampler, 'set_epoch'):
        data_loader.sampler.set_epoch(epoch)
    scaling_norm_list = []
    scaling_unnorm_list = []
    gt_num_image = data_loader.dataset.gt_num_image
    
    acc_all = []
    acc_all_med = []
    comp_all = []
    comp_all_med = []
    nc1_all = []
    nc1_all_med = []
    nc2_all = []
    nc2_all_med = []
    
    for i, batch in enumerate(metric_logger.log_every(data_loader, args.print_freq, header)):
        with torch.no_grad():
            if 'scan1' not in batch[0]['instance'][0]:
                continue
            print(batch[0]['instance'])
            ret = loss_of_one_batch(gt_num_image, batch, model, criterion, device,
                                        symmetrize_batch=True,
                                        use_amp=bool(args.amp))
                                        # (scaling_norm, scaling_unnorm)



    # import matplotlib.pyplot as plt
    # scaling_unnorm_list = np.concatenate(scaling_unnorm_list)
    # scaling_norm_list = np.concatenate(scaling_norm_list)
    # colors = ['red', 'tan', 'lime']
    # labels = ['x', 'y', 'z']
    # plt.rcParams.update({'font.size': 24})
    # plt, ((ax0, ax1)) = plt.subplots(nrows=1, ncols=2, dpi=100, figsize=(20,10))
    # ax0.hist(scaling_unnorm_list.reshape(-1,3),range=[0, 0.025], bins=100,color=colors,histtype='stepfilled',alpha=0.75, density=True,label=labels, stacked=True)
    # ax0.legend()
    # ax0.set_title('unnorm')
    # colors = ['blue', 'orange', 'purple']
    # ax1.hist(scaling_norm_list.reshape(-1,3),range=[0., 0.025], bins=100,color=colors,histtype='stepfilled',alpha=0.75, density=True, label=labels, stacked=True)
    # ax1.legend()
    # ax1.set_title('norm')
    # plt.tight_layout()
    # plt.suptitle(f'{prefix}')
    # plt.savefig(args.output_dir + f'/{prefix}.png')
    # # plt.imshow(pc_colors[0,0])
    # print(prefix)

    log_writer.add_mesh(f'{prefix}', vertices=pts.reshape(B,-1,3)[:,::4], colors=pc_colors.clone().reshape(B,-1,3)[:,::4].to(dtype=torch.int))
    from nerfvis import scene
    f = 1111.0
    import imageio
    imageio.imwrite(args.output_dir + f'/{prefix}_test.png', (pc_colors[0,0].detach().cpu().numpy()).astype(np.uint8))
    scene.set_opencv()
    scene.add_points("point_pred", pts.reshape(B,-1,3)[0].reshape(-1,3), vert_color=pc_colors.clone().reshape(B,-1,3)[0].reshape(-1,3)/255, point_size=2)
    # scene.add_points("point_gt", gt_pts.reshape(B,-1,3)[0].reshape(-1,3), vert_color=pc_colors.clone().reshape(B,-1,3)[0].reshape(-1,3)/255)
    # scene.add_images(
    #             f"images/i",
    #             pc_colors[0].clone().detach().cpu()/255, # Can be a list of paths too (requires joblib for that) 
    #             r=extrinsics[:, :3, :3].detach().cpu(),
    #             t=extrinsics[:, :3, 3].detach().cpu()/norm_factor_gt[0].detach().cpu().squeeze(),
    #             # Alternatively: from nerfvis.utils import split_mat4; **split_mat4(c2ws)
    #             focal_length=f,
    #             z=0.5,
    #             with_camera_frustum=True,
    #         )
    scene.display(port=1533)
    scene.export(args.output_dir + f'/{prefix}_test', embed_output=True)
    return scaling_unnorm_list, scaling_norm_list

if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    from hydra import compose, initialize
    from omegaconf import OmegaConf
    # cfg = omegaconf.OmegaConf.load('./vggsfm_v2/cfgs/demo.yaml')

    # print(OmegaConf.to_yaml(cfg))
    main(args)
