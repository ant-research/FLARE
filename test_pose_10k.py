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
    parser.add_argument('--stage1_pretrained', default=None, help='path of a starting checkpoint for stage1')
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
    test_criterion = eval(args.test_criterion or args.criterion).to(device)

    model.to(device)
    model_without_ddp = model
    # model.set_noise_level(args.noise_trans, args.noise_rot, args.noise_prob)
    print("Model = %s" % str(model_without_ddp))

    if args.pretrained and not args.resume:
        print('Loading pretrained: ', args.pretrained)
        ckpt = torch.load(args.pretrained, map_location=device)
        print(model.load_state_dict(ckpt['model'], strict=False))
        del ckpt  # in case it occupies memory
        if 'stage1_pretrained' in args and args.stage1_pretrained:
            print('Loading stage1 pretrained: ', args.stage1_pretrained)
            ckpt = torch.load(args.stage1_pretrained, map_location=device)
            print(model.load_state_dict_stage1(ckpt['model'], strict=False))
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
    for test_name, testset in data_loader_test.items():
        pose, rot_init_per = test_one_epoch(model, test_criterion, testset,
                                device, 0, log_writer=log_writer, args=args, prefix=test_name)
        ret_rot.append(pose)
        ret_init.append(rot_init_per)
        # test_stats[test_name] = stats

        # Save best of all
        # if stats['loss_med'] < best_so_far:
        #     best_so_far = stats['loss_med']
        #     new_best = True
    ret_rot = torch.cat(ret_rot)
    ret_init = torch.cat(ret_init)
    print("Averaged all dataset loss:", torch.mean(ret_rot).item())
    print("Averaged all dataset rot_init:", torch.mean(ret_init).item())

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

def pad_to_square(reshaped_image):
    B, C, H, W = reshaped_image.shape
    max_dim = max(H, W)
    
    # 计算高度和宽度上需要的 padding
    pad_height = max_dim - H
    pad_width = max_dim - W
    
    # 在高度和宽度的两侧均匀地添加 padding
    padding = (pad_width // 2, pad_width - pad_width // 2,
               pad_height // 2, pad_height - pad_height // 2)
    
    # 使用 F.pad 进行填充
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
    loss_value_list = []
    rot_init_list = []
    trans_init_list = []
    imaga_path_list = []
    error_dict = {"rError": [], "tError": []}
    gt_num_image = data_loader.dataset.gt_num_image
    backbone = torch.hub.load(
                "facebookresearch/dinov2", "dinov2_vitb14_reg"
            )
    backbone = backbone.eval().cuda()
    
    for iter_num, batch in enumerate(metric_logger.log_every(data_loader, args.print_freq, header)):
        with torch.no_grad():
            import time
            torch.cuda.synchronize()
            start = time.time()
            images = [gt['img_org'] for gt in batch]
            images = torch.cat(images, dim=0)
            images = images / 2 + 0.5
            index = generate_rank_by_dino(images, backbone, query_frame_num=1)
            new_order = calculate_index_mappings(index, len(images), device=device)
            new_batch = []
            for i in range(len(batch)):
                new_batch.append(batch[new_order[i]])
            batch = new_batch
            loss_tuple = loss_of_one_batch(gt_num_image, batch, model, criterion, device,
                                        symmetrize_batch=True,
                                        use_amp=bool(args.amp), ret='loss')
            end = time.time()
            print(f"Time: {end - start}")
            rel_rangle_deg, rel_tangle_deg = loss_tuple[0], loss_tuple[1]
            print(f"    --  Mean Rot   Error (Deg) for this scene: {rel_rangle_deg.mean():10.2f}")
            print(f"    --  Mean Trans Error (Deg) for this scene: {rel_tangle_deg.mean():10.2f}")
            error_dict["rError"].extend(rel_rangle_deg.cpu().numpy())
            error_dict["tError"].extend(rel_tangle_deg.cpu().numpy())
            if iter_num % 50 == 0:
                rError_temp = np.array(error_dict["rError"])
                tError_temp = np.array(error_dict["tError"])[:,0]
                Auc_30, normalized_histogram = calculate_auc_np(rError_temp, tError_temp, max_threshold=30)
                print(f"Testing Done")
                rError_temp = (rError_temp < 5).astype(np.float32)
                tError_temp = (tError_temp < 5).astype(np.float32)
                print(f"rError_temp: {rError_temp.mean():10.2f}")
                print(f"tError_temp: {tError_temp.mean():10.2f}")
                print(f"Auc_30 (%): {Auc_30 * 100}")
    rError_temp = np.array(error_dict["rError"])
    tError_temp = np.array(error_dict["tError"])[:,0]
    Auc_30, normalized_histogram = calculate_auc_np(rError_temp, tError_temp, max_threshold=30)
    print(f"Testing Done")
    rError_temp = (rError_temp < 5).astype(np.float32)
    tError_temp = (tError_temp < 5).astype(np.float32)
    print(f"rError_temp: {rError_temp.mean():10.2f}")
    print(f"tError_temp: {tError_temp.mean():10.2f}")
    print(f"Auc_30 (%): {Auc_30 * 100}")
    # gather the stats from all processes
    # metric_logger.synchronize_between_processes()
    # print("Averaged stats:", metric_logger)
    print(len(imaga_path_list))
    rot_init_list = torch.cat(rot_init_list)
    print("Averaged rot loss:", torch.mean(rot_init_list).item())
    trans_init_list = torch.cat(trans_init_list)
    print("Averaged trans loss:", torch.mean(trans_init_list).item())
    return loss_value_list, rot_init_list

if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    from hydra import compose, initialize
    from omegaconf import OmegaConf
    # cfg = omegaconf.OmegaConf.load('./vggsfm_v2/cfgs/demo.yaml')

    # print(OmegaConf.to_yaml(cfg))
    main(args)
