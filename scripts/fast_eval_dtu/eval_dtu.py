import argparse,sys,os
import multiprocessing as mp
mp.set_start_method("spawn", force=True)  # Prevent processes being suspended
import torch
import numpy as np

from tqdm import tqdm
from pathlib import Path
from functools import partial
from multiprocessing import Pool
from eval_utils import comput_one_scan_cuda, compute_scans, sava_result, limit_cpu_threads_used


scans = [1,4,9,10,11,12,13,15,23,24,29,32,33,34,48,49,62,75,77,110,114,118]
# scans = [114]
parser = argparse.ArgumentParser()
parser.add_argument('--scans', type=int, nargs='+', default=scans, help="scans to be evalutation")
parser.add_argument('--method', type=str, default='ours', help="method name, such as mvsnet,casmvsnet")
parser.add_argument('--pred_dir', type=str, default='./Predict/mvsnet', help="predict result ply file path")
parser.add_argument('--gt_dir', type=str, default='./SampleSet/MVS Data',help="groud truth ply file path")
parser.add_argument('--voxel_factor', type=float, default=1.28, help="voxel factor for alignment")
parser.add_argument('--down_dense', type=float, default=0.2, help="downsample density, Min dist between points when reducing")
parser.add_argument('--patch', type=float, default=60, help="patch size")
parser.add_argument('--max_dist', type=float, default=20, help="outlier thresshold of 20 mm")
parser.add_argument('--vis', type=bool, default=False, help="visualization")
parser.add_argument('--vis_thresh', type=float, default=10, help="visualization distance threshold of 10mm")
parser.add_argument('--out_dir', type=str, default="./outputs", help="result save dir")
parser.add_argument('--save', action="store_true", default=False,  help="save eval results to txt")
parser.add_argument('--num_workers', type=int, default=2, help="number of thread")
parser.add_argument('--device', type=int, default=0, help="cuda device id")
parser.add_argument('--model_name', type=str, default="", help="model name")
parser.add_argument('--cpu_percentage', type=float, default=0.1, help='percentage of cpu threads used')
args = parser.parse_args()
# torch.cuda.set_device(args.device)



def eval_worker(opts, args):
    scanid, shared_list = opts
    pred_ply    = os.path.join(args.pred_dir, f"{args.model_name}{args.method}{scanid:03}_l3.ply")   
    gt_ply      = os.path.join(args.gt_dir, f"Points/stl/stl{scanid:03}_total.ply")
    mask_file   = os.path.join(args.gt_dir, f'ObsMask/ObsMask{scanid}_10.mat')
    plane_file  = os.path.join(args.gt_dir, f'ObsMask/Plane{scanid}.mat')
    if os.path.exists(pred_ply) == False:
        msg = f"scan{scanid} not exist, skip!"
        return msg, [0,0,0], scanid
    result = comput_one_scan_cuda(scanid, pred_ply, gt_ply, mask_file, plane_file, args=args, shared_list=shared_list)
    msg = "scan{}\t   acc = {:.4f}   comp = {:.4f}   overall = {:.4f}".format(scanid, result[0], result[1], result[2])
    return msg, result, scanid

def eval(testlist, args):
    gpu_used = min(args.num_workers, torch.cuda.device_count())
    if gpu_used > 1: args.num_workers = gpu_used
    print(f"Lets use {gpu_used} GPU with {args.num_workers} process for evaluation!")
    manager = mp.Manager()
    lock = manager.Lock()
    share_list = manager.list([0] * args.num_workers)
    partial_func = partial(eval_worker, args=args)
    opts = [(scene, share_list) for scene in testlist]
    results = []
    res_dict = {}
    with Pool(args.num_workers, initializer=tqdm.set_lock, initargs=(lock,)) as p:
        for msg, result, scanid in tqdm(p.imap_unordered(partial_func, opts), total=len(testlist), position=0, leave=True, desc="Processed scene"):
            tqdm.write(msg)
            result = np.array(result).tolist()
            results.append(result)
            res_dict[str(scanid)] = result
    mean_res = np.array(results)[np.array(results)>0].mean(axis=0).tolist()
    if args.save:
        data = list(res_dict.items())
        data.sort(key=lambda x:int(x[0]))
        data = {k:v for k,v in data}
        data["mean"] = mean_res
        save_dir = Path(args.pred_dir)
        save_path = str(save_dir/"python_eval_result.txt")
        sava_result(save_path, data, model_name=args.model_name) 
    print("Final result:   mean_acc = {:.4f}   mean_comp = {:.4f}   mean_overall = {:.4f}".format(mean_res[0], mean_res[1], mean_res[2]))

def debug():
    args.scans    = [1,118]
    args.pred_dir = "outputs/dtu/test/unification-dcnv3/ply_fused/gipuma"
    args.gt_dir   = "/home/bip/gwc/data/DTU/eval/SampleSet/MVS Data"

    scans    = args.scans
    method   = args.method
    pred_dir = args.pred_dir
    gt_dir   = args.gt_dir
    vis      = args.vis

    exclude = ["scans", "method", "pred_dir", "gt_dir"]
    args = vars(args)
    args = {key:args[key] for key in args if key not in exclude}
    acc, comp, overall = compute_scans(scans, method, pred_dir, gt_dir, **args)
    print(f"mean acc:{acc:>12.4f}\nmean comp:{comp:>11.4f}\nmean overall:{overall:>8.4f}")


if __name__ == "__main__":
    # Limit the number of processes used in a server shared by multiple people to 
    # prevent excessive consumption of CPU resources. If the server is exclusively
    # owned by one person, this line can be commented out
    limit_cpu_threads_used(args.cpu_percentage) 
    if args.num_workers > 1:
        eval(args.scans, args)
    else:
        exclude = ["scans", "method", "pred_dir", "gt_dir"]
        scans    = args.scans
        method   = args.method
        pred_dir = args.pred_dir
        gt_dir   = args.gt_dir
        args = vars(args)
        args = {key:args[key] for key in args if key not in exclude}
        acc, comp, overall = compute_scans(scans, method, pred_dir, gt_dir, **args)
        print(f"mean acc:{acc:>12.4f}\nmean comp:{comp:>11.4f}\nmean overall:{overall:>8.4f}")


