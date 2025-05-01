import os
import time
import torch
import numpy as np
import open3d as o3d
from pathlib import Path

from tqdm import tqdm
from scipy.io import loadmat
from sklearn import neighbors as skln
from plyfile import PlyData, PlyElement
from multiprocessing import cpu_count
from chamfer3D.dist_chamfer_3D import chamfer_3DDist as chamLoss


def read_ply(file):
    data = PlyData.read(file)
    vertex = data['vertex']
    data_pcd = np.stack([vertex['x'], vertex['y'], vertex['z']], axis=-1)
    return data_pcd

def write_vis_pcd(file, points, colors):
    points = np.array([tuple(v) for v in points], dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4')])
    colors = np.array([tuple(v) for v in colors], dtype=[('red', 'u1'), ('green', 'u1'), ('blue', 'u1')])

    vertex_all = np.empty(len(points), points.dtype.descr + colors.dtype.descr)
    for prop in points.dtype.names:
        vertex_all[prop] = points[prop]
    for prop in colors.dtype.names:
        vertex_all[prop] = colors[prop]

    el = PlyElement.describe(vertex_all, 'vertex')
    PlyData([el]).write(file)

def voxel_downsample(point_cloud, voxel_size=0.2):
    device = point_cloud.device
    point_cloud_np = point_cloud.cpu().numpy()
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(point_cloud_np)

    downsampled_pcd = pcd.voxel_down_sample(voxel_size)
    downsampled_points = np.asarray(downsampled_pcd.points,dtype=np.float32)
    downsampled_points = torch.from_numpy(downsampled_points).to(device)
    downsampled_scale = downsampled_points.size(0) / point_cloud.size(0)
    return downsampled_points, downsampled_scale

def reduce_pts(data_pcd, thresh):
    n_points = data_pcd.shape[0]
    is_tensor = False
    if isinstance(data_pcd, torch.Tensor):
        is_tensor = True
        device = data_pcd.device
        if "cuda" in str(data_pcd.device):
            data_pcd = data_pcd.cpu()
        data_pcd = data_pcd.numpy()
    nn_engine = skln.NearestNeighbors(n_neighbors=1, radius=thresh, algorithm='kd_tree', n_jobs=-1)
    nn_engine.fit(data_pcd)
    rnn_idxs = nn_engine.radius_neighbors(data_pcd, radius=thresh, return_distance=False)
    mask = np.ones(data_pcd.shape[0], dtype=np.bool_)
    for curr, idxs in enumerate(rnn_idxs):
        if mask[curr]:
            mask[idxs] = 0
            mask[curr] = 1
    data_down = data_pcd[mask]
    if is_tensor:
        data_down = torch.from_numpy(data_down).to(device)
    down_scale = data_down.shape[0] / n_points
    return data_down, down_scale


def limit_cpu_threads_used(cpu_percentage=0.1):
    cpu_num = cpu_count() // 2 
    cpu_num = int(cpu_count() * cpu_percentage) 
    os.environ ['OMP_NUM_THREADS']        = str(cpu_num) 
    os.environ ['MKL_NUM_THREADS']        = str(cpu_num) 
    os.environ ['NUMEXPR_NUM_THREADS']    = str(cpu_num) 
    os.environ ['OPENBLAS_NUM_THREADS']   = str(cpu_num) 
    os.environ ['VECLIB_MAXIMUM_THREADS'] = str(cpu_num) 
    torch.set_num_threads(cpu_num)
    print(f"set the max number of processes used to {cpu_num}")


def get_pgb_pos(shared_list, num_works):
    # Acquire lock and get a progress bar slot
    for i in range(num_works):
        if shared_list[i] == 0:
            shared_list[i] = 1
            return i

def release_pgb_pos(shared_list, slot):
    shared_list[slot] = 0


def sava_result(save_path, data, model_name=""):
    f = open(save_path, 'a', encoding='utf-8')
    if len(model_name) > 1:
        if model_name.endswith("_"):
            model_name = model_name[:-1]
        f.write(f"{model_name}\n")
    for k,v in data.items():
        if "mean" not in k:
            scan = f"scan{k}" 
            f.write(f"{scan}\t acc: {v[0]:.4f},  comp: {v[1]:.4f},  overall: {v[2]:.4f}\n")
        else:
            mean_acc, mean_comp, mean_overall = v[0], v[1], v[2]
            f.write(f"mean acc:{mean_acc:>12.4f}\nmean comp:{mean_comp:>11.4f}\nmean overall:{mean_overall:>8.4f}\n")
    f.write("\n")


def comput_one_scan(scanid,             # the scan id to be computed 
                    pred_ply,           # predict points cloud file path, such as "./mvsnet001_l3.ply"
                    gt_ply,             # ground truth points cloud file path, such as "./stl001_total.ply"
                    mask_file,          # obsmask file path, decide which parts of 3D space should be used for evaluation
                    plane_file,         # plane file path, used to destinguise which Stl points are 'used'
                    down_dense  = 0.2,  # downsample density, Min dist between points when reducing
                    patch       = 60,   # patch size
                    max_dist    = 20,   # outlier thresshold of 20 mm
                    vis         = True,# whether save distance visualization result 
                    vis_thresh  = 10,   # visualization distance threshold of 10mm
                    out_dir = "outputs",# outputs directory
                    **kargs):
    '''Compute accuracy(mm), completeness(mm), overall(mm) for one scan 

        scanid:         the scan id to be computed 
        pred_ply:       predict points cloud file path, such as "./mvsnet001_l3.ply"
        gt_ply:         ground truth points cloud file path, such as "./stl001_total.ply"
        mask_file:      obsmask file path, decide which parts of 3D space should be used for evaluation
        plane_file:     plane file path, used to destinguise which Stl points are 'used'
        down_dense:     downsample density, Min dist between points when reducing
        patch:          patch size
        max_dist:       outlier thresshold of 20 mm
        vis:            whether save distance visualization result 
        vis_thresh:     visualization distance threshold of 10mm
        vis_out_dir:    visualization result save directory
    '''
    
    thresh = down_dense
    pbar = tqdm(total=8)
    pbar.set_description(f'[scan{scanid}] read data pcd')
    data_pcd = read_ply(pred_ply)

    pbar.update(1)
    pbar.set_description(f'[scan{scanid}] random shuffle pcd index')
    shuffle_rng = np.random.default_rng()
    shuffle_rng.shuffle(data_pcd, axis=0)

    pbar.update(1)
    pbar.set_description(f'[scan{scanid}] downsample pcd')
    nn_engine = skln.NearestNeighbors(n_neighbors=1, radius=thresh, algorithm='kd_tree', n_jobs=-1)
    nn_engine.fit(data_pcd)
    rnn_idxs = nn_engine.radius_neighbors(data_pcd, radius=thresh, return_distance=False)
    mask = np.ones(data_pcd.shape[0], dtype=np.bool_)
    for curr, idxs in enumerate(rnn_idxs):
        if mask[curr]:
            mask[idxs] = 0
            mask[curr] = 1
    data_down = data_pcd[mask]

    pbar.update(1)
    pbar.set_description(f'[scan{scanid}] masking data pcd')
    obs_mask_file = loadmat(mask_file)
    ObsMask, BB, Res = [obs_mask_file[attr] for attr in ['ObsMask', 'BB', 'Res']]
    BB = BB.astype(np.float32)

    inbound = ((data_down >= BB[:1]-patch) & (data_down < BB[1:]+patch*2)).sum(axis=-1) ==3
    data_in = data_down[inbound]

    data_grid = np.around((data_in - BB[:1]) / Res).astype(np.int32)
    grid_inbound = ((data_grid >= 0) & (data_grid < np.expand_dims(ObsMask.shape, 0))).sum(axis=-1) ==3
    data_grid_in = data_grid[grid_inbound]
    in_obs = ObsMask[data_grid_in[:,0], data_grid_in[:,1], data_grid_in[:,2]].astype(np.bool_)
    data_in_obs = data_in[grid_inbound][in_obs]

    pbar.update(1)
    pbar.set_description(f'[scan{scanid}] read STL pcd')
    stl = read_ply(gt_ply)

    pbar.update(1)
    pbar.set_description(f'[scan{scanid}] compute data2stl')
    nn_engine.fit(stl)
    dist_d2s, idx_d2s = nn_engine.kneighbors(data_in_obs, n_neighbors=1, return_distance=True)
    max_dist = max_dist
    mean_d2s = dist_d2s[dist_d2s < max_dist].mean()

    pbar.update(1)
    pbar.set_description(f'[scan{scanid}] compute stl2data')
    ground_plane = loadmat(plane_file)['P']

    stl_hom = np.concatenate([stl, np.ones_like(stl[:,:1])], -1)
    above = (ground_plane.reshape((1,4)) * stl_hom).sum(-1) > 0
    stl_above = stl[above]

    nn_engine.fit(data_in)
    dist_s2d, idx_s2d = nn_engine.kneighbors(stl_above, n_neighbors=1, return_distance=True)
    mean_s2d = dist_s2d[dist_s2d < max_dist].mean()

    pbar.update(1)
    pbar.set_description(f'[scan{scanid}] visualize error')
    if vis:
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        vis_dist = vis_thresh
        R = np.array([[255,0,0]], dtype=np.float64)
        G = np.array([[0,255,0]], dtype=np.float64)
        B = np.array([[0,0,255]], dtype=np.float64)
        W = np.array([[255,255,255]], dtype=np.float64)
        data_color = np.tile(B, (data_down.shape[0], 1))
        data_alpha = dist_d2s.clip(max=vis_dist) / vis_dist
        data_color[ np.where(inbound)[0][grid_inbound][in_obs] ] = R * data_alpha + W * (1-data_alpha)
        data_color[ np.where(inbound)[0][grid_inbound][in_obs][dist_d2s[:,0] >= max_dist] ] = G
        write_vis_pcd(f'{out_dir}/vis_{scanid:03}_d2s.ply', data_down, data_color)
        stl_color = np.tile(B, (stl.shape[0], 1))
        stl_alpha = dist_s2d.clip(max=vis_dist) / vis_dist
        stl_color[ np.where(above)[0] ] = R * stl_alpha + W * (1-stl_alpha)
        stl_color[ np.where(above)[0][dist_s2d[:,0] >= max_dist] ] = G
        write_vis_pcd(f'{out_dir}/vis_{scanid:03}_s2d.ply', stl, stl_color)

    pbar.update(1)
    pbar.set_description(f'[scan{scanid}] done')
    pbar.close()
    over_all = (mean_d2s + mean_s2d) / 2
    # print(f"\t\t\tacc.(mm):{mean_d2s:.4f}, comp.(mm):{mean_s2d:.4f}, overall(mm):{over_all:.4f}")
    return mean_d2s, mean_s2d, over_all


### ICP 
def align_point_clouds(gt_point_cloud, predict_point_cloud):
    initial_transform = np.eye(4)

    threshold = 0.2
    icp_result = o3d.pipelines.registration.registration_icp(
        predict_point_cloud, gt_point_cloud, threshold, initial_transform,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(with_scaling=True),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=2000))
    print(icp_result)
    return icp_result.transformation


@torch.no_grad()
def comput_one_scan_cuda(scanid,            # the scan id to be computed 
                        pred_ply,           # predict points cloud file path, such as "./mvsnet001_l3.ply"
                        gt_ply,             # ground truth points cloud file path, such as "./stl001_total.ply"
                        mask_file,          # obsmask file path, decide which parts of 3D space should be used for evaluation
                        plane_file,         # plane file path, used to destinguise which Stl points are 'used'
                        down_dense  = 0.2,  # downsample density, Min dist between points when reducing
                        patch       = 60,   # patch size
                        max_dist    = 20,   # outlier thresshold of 20 mm
                        vis         = True,# whether save distance visualization result 
                        vis_thresh  = 10,   # visualization distance threshold of 10mm
                        out_dir = "outputs",# outputs directory
                        **kargs):
    '''Compute accuracy(mm), completeness(mm), overall(mm) for one scan 

        scanid:         the scan id to be computed 
        pred_ply:       predict points cloud file path, such as "./mvsnet001_l3.ply"
        gt_ply:         ground truth points cloud file path, such as "./stl001_total.ply"
        mask_file:      obsmask file path, decide which parts of 3D space should be used for evaluation
        plane_file:     plane file path, used to destinguise which Stl points are 'used'
        down_dense:     downsample density, Min dist between points when reducing
        patch:          patch size
        max_dist:       outlier thresshold of 20 mm
        vis:            whether save distance visualization result 
        vis_thresh:     visualization distance threshold of 10mm
        vis_out_dir:    visualization result save directory
    '''
    voxel_factor = kargs.get("voxel_factor", 1.28) # Voxelization factor that aligns its results with Matlab
    shared_list  = kargs.get("shared_list", None)
    args = kargs.get("args", None)
    device = torch.device("cuda") 
    thresh = down_dense
    if shared_list is not None:
        pgb_pos = get_pgb_pos(shared_list, args.num_workers)
        device = torch.device(f"cuda:{pgb_pos}") 
        pbar = tqdm(total=8, position=pgb_pos+1, leave=False)
    else:
        pbar = tqdm(total=8)
    pbar.set_description(f'[scan{scanid}] read data pcd and stl')
    pcd_np = read_ply(pred_ply)     # (N,3)
    stl_np = read_ply(gt_ply)       # (M,3)

    data_pcd_o3d = o3d.geometry.PointCloud()
    data_pcd_o3d.points = o3d.utility.Vector3dVector(pcd_np)
    gt_pcd_o3d = o3d.geometry.PointCloud()
    gt_pcd_o3d.points = o3d.utility.Vector3dVector(stl_np)
    # transformation_matrix = align_point_clouds(gt_pcd_o3d, data_pcd_o3d)
    # data_pcd_o3d = data_pcd_o3d.transform(transformation_matrix)
    pcd_np = np.asarray(data_pcd_o3d.points)
    
    pcd = torch.from_numpy(pcd_np).to(device)
    stl = torch.from_numpy(stl_np).to(device)
    
    pbar.update(1)
    pbar.set_description(f'[scan{scanid}] random shuffle pcd index')
    pcd = pcd[torch.randperm(pcd.size(0))]

    pbar.update(1)
    pbar.set_description(f'[scan{scanid}] downsample pcd')
    # using open3d voxel down sample, more fast, but may be different with kdtree reduce
    pcd, down_scale = voxel_downsample(pcd, voxel_factor*thresh) # (N1,3) 
    # pcd, down_scale = reduce_pts(pcd, thresh)


    pbar.update(1)
    pbar.set_description(f'[scan{scanid}] masking data pcd')
    obs_mask_file = loadmat(mask_file)
    ObsMask, BB, Res = [obs_mask_file[attr] for attr in ['ObsMask', 'BB', 'Res']]
    ObsMask = torch.from_numpy(ObsMask).to(device)      # (A1, A2, A3)
    BB      = torch.from_numpy(BB).to(device).float()
    Res     = torch.from_numpy(Res).to(device)
    inbound = ((pcd >= BB[:1]-patch) & (pcd < BB[1:]+patch*2)).sum(dim=-1) == 3  # (N2)
    num = pcd.size(0)
    from copy import deepcopy
    pcd_org = deepcopy(pcd)  # (N2,3)
    pcd = pcd[inbound]  # (N2,3)

    pbar.update(1)
    pbar.set_description(f'[scan{scanid}] compute above plane mask and inbox and inobs mask')
    ground_plane = loadmat(plane_file)['P']
    ground_plane = torch.from_numpy(ground_plane).to(device)
    data_grid    = torch.round((pcd - BB[:1]) / Res).to(torch.int32)
    mask_shape   = torch.tensor(ObsMask.shape).unsqueeze(0).to(device)
    grid_inbound = ((data_grid >= 0) & (data_grid < mask_shape)).sum(dim=-1) == 3
    data_grid_in = data_grid[grid_inbound]
    in_obs = ObsMask[data_grid_in[:,0].long(), data_grid_in[:,1].long(), data_grid_in[:,2].long()].bool()

    stl_hom = torch.cat([stl, torch.ones_like(stl[:,:1])], -1)
    above   = (ground_plane.reshape((1,4)) * stl_hom).sum(-1) > 0

    pbar.update(1)
    pbar.set_description(f'[scan{scanid}] compute pcd2stl and stl2pcd')
    with torch.no_grad():
        pc_t = pcd.unsqueeze(dim=0)
        gt_t = stl.unsqueeze(dim=0)
        dist_d2s, dist_s2d, _, _ = chamLoss()(pc_t, gt_t)
        dist_d2s, dist_s2d = torch.sqrt(dist_d2s), torch.sqrt(dist_s2d)
        dist_d2s, dist_s2d = dist_d2s.squeeze(0), dist_s2d.squeeze(0)
        # torch.cuda.synchronize()

    pbar.update(1)
    pbar.set_description(f'[scan{scanid}] compute acc and comp')
    dist_d2s = dist_d2s[grid_inbound][in_obs]
    dist_s2d = dist_s2d[above]
    mean_d2s = dist_d2s[dist_d2s < max_dist].mean()
    mean_s2d = dist_s2d[dist_s2d < max_dist].mean()

    pbar.update(1)
    pbar.set_description(f'[scan{scanid}] visualize error')
    if vis:
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        vis_dist = vis_thresh
        R = np.array([[255,0,0]], dtype=np.float64)
        G = np.array([[0,255,0]], dtype=np.float64)
        B = np.array([[0,0,255]], dtype=np.float64)
        W = np.array([[255,255,255]], dtype=np.float64)
        data_color = np.tile(B, (num, 1))
        data_alpha = dist_d2s.clip(max=vis_dist) / vis_dist
        data_alpha = data_alpha.cpu().numpy()
        grid_inbound = grid_inbound.cpu().numpy()
        in_obs = in_obs.cpu().numpy()
        inbound = inbound.cpu().numpy()
        data_color[ np.where(inbound)[0][grid_inbound][in_obs] ] = R * data_alpha[:,None] + W * (1-data_alpha)[:,None]
        data_color[ np.where(inbound)[0][grid_inbound][in_obs][dist_d2s.cpu().numpy() >= max_dist] ] = G
        write_vis_pcd(f'{out_dir}/vis_{scanid:03}_d2s.ply', pcd_org, data_color)
        # stl_color = np.tile(B, (stl.shape[0], 1))
        # stl_alpha = dist_s2d.clip(max=vis_dist) / vis_dist
        # stl_color[ np.where(above)[0] ] = R * stl_alpha + W * (1-stl_alpha)
        # stl_color[ np.where(above)[0][dist_s2d[:,0] >= max_dist] ] = G
        # write_vis_pcd(f'{out_dir}/vis_{scanid:03}_s2d.ply', stl, stl_color)

    pbar.update(1)
    pbar.set_description(f'[scan{scanid}] done')
    pbar.close()
    if shared_list is not None:
        release_pgb_pos(shared_list, pgb_pos)
    over_all = (mean_d2s + mean_s2d) / 2
    # print(f"\t\t\tacc.(mm):{mean_d2s:.4f}, comp.(mm):{mean_s2d:.4f}, overall(mm):{over_all:.4f}")
    return mean_d2s.cpu().numpy(), mean_s2d.cpu().numpy(), over_all.cpu().numpy()

def compute_scans(scans, method, pred_dir, gt_dir, **kargs):
    t1 = time.time()
    save = kargs.get("save", False)
    prefix = kargs.get("prefix", "")
    if save: 
        # save_dir = Path(kargs.get("out_dir", None) or "outputs")
        # save_dir.mkdir(parents=True, exist_ok=True)
        save_dir = Path(pred_dir)
        f = open(str(save_dir/"result.txt"), 'a', encoding='utf-8')
        if len(prefix) > 1:
            f.write(prefix[:-1] + "\n")

    acc ,comp ,overall = [], [], []
    for scanid in scans:
        pred_ply    = os.path.join(pred_dir, f"{prefix}{method}{scanid:03}_l3.ply")   
        gt_ply      = os.path.join(gt_dir, f"Points/stl/stl{scanid:03}_total.ply")
        mask_file   = os.path.join(gt_dir, f'ObsMask/ObsMask{scanid}_10.mat')
        plane_file  = os.path.join(gt_dir, f'ObsMask/Plane{scanid}.mat')
        if not os.path.exists(pred_ply):
            print(f"scan{scanid} not exist, skip!")
            continue
        assert os.path.exists(pred_ply),   f"File '{pred_ply}' not found"
        assert os.path.exists(gt_ply),     f"File '{gt_ply}' not found"
        assert os.path.exists(mask_file),  f"File '{mask_file}' not found"
        assert os.path.exists(plane_file), f"File '{plane_file}' not found"
        if torch.cuda.is_available():
            result = comput_one_scan_cuda(scanid, pred_ply, gt_ply, mask_file, plane_file, **kargs)
        else:
            result = comput_one_scan(scanid, pred_ply, gt_ply, mask_file, plane_file, **kargs)
        acc.append(result[0])
        comp.append(result[1])
        overall.append(result[2])
        print(f"\t\t\tacc.(mm):{result[0]:.4f}, comp.(mm):{result[1]:.4f}, overall(mm):{result[2]:.4f}")
        if save: f.write(f"scan{scanid}\tacc: {result[0]:.4f}  comp: {result[1]:.4f}  overall: {result[2]:.4f}\n")
    mean_acc = np.mean(acc)
    mean_comp = np.mean(comp)
    mean_overall = np.mean(overall)
    t2 = time.time()
    if save: f.write(f"mean acc:{mean_acc:>12.4f}\nmean comp:{mean_comp:>11.4f}\nmean overall:{mean_overall:>8.4f}\n\n")
    print(f"mean acc:{mean_acc:>12.4f}\nmean comp:{mean_comp:>11.4f}\nmean overall:{mean_overall:>8.4f}\n\n")
    print(f"Finished, total time cost: {t2-t1:.2f}s")
    return mean_acc, mean_comp, mean_overall


if __name__ == "__main__":
    torch.cuda.set_device(1)
    limit_cpu_threads_used(0.1)
    method      = "mvsnet"
    pred_dir    = "outputs/dtu/test/ply_fused/gipuma"
    gt_dir      = "data/DTU/eval/SampleSet/MVS Data"
    scanid      = 1
    MaxDist     = 20
    pred_ply    = os.path.join(pred_dir, f"{method}{scanid:03}_l3.ply")   
    gt_ply      = os.path.join(gt_dir, f"Points/stl/stl{scanid:03}_total.ply")
    mask_file   = os.path.join(gt_dir, f'ObsMask/ObsMask{scanid}_10.mat')
    plane_file  = os.path.join(gt_dir, f'ObsMask/Plane{scanid}.mat')

    result = comput_one_scan_cuda(scanid, pred_ply, gt_ply, mask_file, plane_file)
    print(f"python\t\t\tacc.(mm):{result[0]:.4f}, comp.(mm):{result[1]:.4f}, overall(mm):{result[2]:.4f}")

    mat_data = loadmat(f"{pred_dir}/eval_out/{method}_Eval_{scanid}.mat")
    py_data = dict(zip(mat_data["data"].dtype.names, mat_data["data"][0][0].T))
    Dstl = py_data["Dstl"][py_data["StlAbovePlane"].astype(bool)]
    Dstl = Dstl[Dstl<MaxDist]
    Ddata = py_data["Ddata"][py_data["DataInMask"].astype(bool)]; 
    Ddata = Ddata[Ddata<MaxDist]
    acc = np.mean(Ddata)
    cmp = np.mean(Dstl)
    overall = (acc+cmp) / 2
    print(f"matlab\t\t\tacc.(mm):{acc:.4f}, comp.(mm):{cmp:.4f}, overall(mm):{overall:.4f}")
