import argparse
import datetime
import json
import os
import sys
import time
import math
from pathlib import Path
from typing import Sized
import mast3r.utils.path_to_dust3r  # noqa
from collections import defaultdict
import copy
import torch
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter
torch.backends.cuda.matmul.allow_tf32 = True  
from mast3r.model import AsymmetricMASt3R
from dust3r.datasets import get_data_loader  # noqa
from dust3r.inference import loss_of_one_batch  # noqa
inf = float('inf')
from mast3r.losses import MeshOutput
import dust3r.utils.path_to_croco  # noqa: F401
import croco.utils.misc as misc  # noqa
import torch.nn.functional as F

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
    parser.add_argument('--batch_size', default=1, type=int,
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
    device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)
    data_loader_test = {dataset.split('(')[0]: build_dataset(dataset, args.batch_size, args.num_workers, test=True)
                        for dataset in args.test_dataset.split('+')}
    print('Loading model: {:s}'.format(args.model))
    model = eval(args.model)
    test_criterion = eval(args.test_criterion or args.criterion)
    model.to(device)
    model_without_ddp = model
    print("Model = %s" % str(model_without_ddp))
    if args.pretrained:
        print('Loading pretrained: ', args.pretrained)
        ckpt = torch.load(args.pretrained, map_location=device)
        print(model.load_state_dict(ckpt['model'], strict=False))
        del ckpt  # in case it occupies memory
    global_rank = misc.get_rank()
    if global_rank == 0 and args.output_dir is not None:
        log_writer = SummaryWriter(log_dir=args.output_dir)
    else:
        log_writer = None
    for test_name, testset in data_loader_test.items():
        test_one_epoch(model, test_criterion, testset,
                                device, 0, log_writer=log_writer, args=args, prefix=test_name)


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

def pad_to_square(reshaped_image):
    B, C, H, W = reshaped_image.shape
    max_dim = max(H, W)
    pad_height = max_dim - H
    pad_width = max_dim - W
    padding = (pad_width // 2, pad_width - pad_width // 2,
               pad_height // 2, pad_height - pad_height // 2)
    padded_image = F.pad(reshaped_image, padding, mode='constant', value=0)
    return padded_image

def generate_rank_by_dino(
    reshaped_image, backbone, query_frame_num, image_size=336
):
    # Downsample image to image_size x image_size
    # because we found it is unnecessary to use high resolution
    rgbs = pad_to_square(reshaped_image)
    rgbs = F.interpolate(
        reshaped_image,
        (image_size, image_size),
        mode="bilinear",
        align_corners=True,
    )
    rgbs = _resnet_normalize_image(rgbs.cuda())

    # Get the image features (patch level)
    frame_feat = backbone(rgbs, is_training=True)
    frame_feat = frame_feat["x_norm_patchtokens"]
    frame_feat_norm = F.normalize(frame_feat, p=2, dim=1)

    # Compute the similiarty matrix
    frame_feat_norm = frame_feat_norm.permute(1, 0, 2)
    similarity_matrix = torch.bmm(
        frame_feat_norm, frame_feat_norm.transpose(-1, -2)
    )
    similarity_matrix = similarity_matrix.mean(dim=0)
    distance_matrix = 100 - similarity_matrix.clone()

    # Ignore self-pairing
    similarity_matrix.fill_diagonal_(-100)

    similarity_sum = similarity_matrix.sum(dim=1)

    # Find the most common frame
    most_common_frame_index = torch.argmax(similarity_sum).item()
    return most_common_frame_index

_RESNET_MEAN = [0.485, 0.456, 0.406]
_RESNET_STD = [0.229, 0.224, 0.225]
_resnet_mean = torch.tensor(_RESNET_MEAN).view(1, 3, 1, 1).cuda()
_resnet_std = torch.tensor(_RESNET_STD).view(1, 3, 1, 1).cuda()
def _resnet_normalize_image(img: torch.Tensor) -> torch.Tensor:
        return (img - _resnet_mean) / _resnet_std

def calculate_index_mappings(query_index, S, device=None):
    """
    Construct an order that we can switch [query_index] and [0]
    so that the content of query_index would be placed at [0]
    """
    new_order = torch.arange(S)
    new_order[0] = query_index
    new_order[query_index] = 0
    if device is not None:
        new_order = new_order.to(device)
    return new_order

@torch.no_grad()
def test_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
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
    try:
        gt_num_image = data_loader.dataset.dataset.gt_num_image
    except:
        gt_num_image = data_loader.dataset.gt_num_image
    backbone = torch.hub.load(
        "facebookresearch/dinov2", "dinov2_vitb14_reg"
        )
    backbone = backbone.eval().cuda()
    for i, batch in enumerate(metric_logger.log_every(data_loader, args.print_freq, header)):
        images = [gt['img_org'] for gt in batch]
        images = torch.cat(images, dim=0)
        images = images / 2 + 0.5
        index = generate_rank_by_dino(images, backbone, query_frame_num=1)
        sorted_order = calculate_index_mappings(index, len(images), device=device)
        sorted_batch = []
        for i in range(len(batch)):
            sorted_batch.append(batch[sorted_order[i]])
        batch = sorted_batch
        loss_tuple = loss_of_one_batch(gt_num_image, batch, model, criterion, device,
                                    symmetrize_batch=True,
                                    use_amp=bool(args.amp))

if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    main(args)
