import torch
import torch.nn as nn
import numpy as np
import mast3r.utils.path_to_dust3r  # noqa
from dust3r.utils.geometry import normalize_pointcloud, xy_grid, inv, matrix_to_quaternion, geotrf
import os
import torch.nn.functional as F
import cv2
from pytorch3d.ops import knn_points
from dust3r.renderers.gaussian_renderer import GaussianRenderer
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp
import imageio
from lpips import LPIPS
from pytorch3d.transforms.rotation_conversions import matrix_to_quaternion, quaternion_to_matrix
import roma
from dust3r.losses import BaseCriterion, Criterion, MultiLoss, Sum, L21



def sample_points_from_mask_torch(mask, N):
    valid_indices = torch.nonzero(mask, as_tuple=False)
    
    # Random sampling with replacement if needed
    if valid_indices.size(0) < N:
        # Allow replacement when there are not enough valid points
        sampled_indices = valid_indices[torch.randint(0, valid_indices.size(0), (N,))]
    else:
        # No replacement needed when there are enough valid points
        sampled_indices = valid_indices[torch.randperm(valid_indices.size(0))[:N]]
    
    return sampled_indices

def triangulate_one_scene(pts, intrinsics, w2cs, conf, rgb=None, num_points_3d=2048, conf_threshold=3, gt_pts=None, res2 = None):
    # pts point map:  S, H, W, 3 
    # intrinsics:  S, 3, 3
    # w2cs:  S, 4, 4
    # conf: reference view confidence map :  S, H, W
    # rgb:  S, 3, H, W


    S, H, W, _ = pts.shape
    conf_reference = conf[0]
    conf_reference_valid = conf_reference > torch.quantile(conf_reference, 0.4)
    # if conf_reference_valid.sum() < num_points_3d:
    #     conf_reference_valid = conf_reference > torch.quantile(conf_reference, 0.5)
    sampled_indices = torch.nonzero(conf_reference_valid, as_tuple=False)
    #sample_points_from_mask_torch(conf_reference_valid, num_points_3d)
    chunk_size = 32 * 32 * 2  # 每个块的最大大小
    num_chunks = (len(sampled_indices) + chunk_size - 1) // chunk_size
    sampled_indices_chunks = torch.split(sampled_indices, chunk_size, dim=0)
    triangulated_points_all = []
    track_matches_all = []
    for sampled_indices in sampled_indices_chunks:
        pts_reference_sampled = pts[0,sampled_indices[:,0], sampled_indices[:,1]]            
        rgb_reference_sampled = rgb.permute(0,2,3,1)[0, sampled_indices[:,0], sampled_indices[:,1]]
        gt_pts_reference_sampled = gt_pts[0,sampled_indices[:,0], sampled_indices[:,1]]
        # Reshape for computation: Sx(H*W)x3
        pts_reshaped = pts.view(S, H*W, 3)
        # # res2_reshaped = res2.reshape(S, H*W, 128)
        # res2_reference_sampled = res2[0, sampled_indices[:,0], sampled_indices[:,1]]
        
        track_matches = []
        track_matches.append(sampled_indices)
        
        track_confs = []
        track_confs.append(conf_reference[sampled_indices[:,0], sampled_indices[:,1]])

        original_pts = []
        original_pts.append(pts_reference_sampled)
        
        
        for idxS in range(1, S):
            with autocast(dtype=torch.double):
                dist = pts_reshaped[idxS].unsqueeze(0).to(torch.double) - pts_reference_sampled.unsqueeze(1).to(torch.double)
                dist = (dist**2).sum(dim=-1)
                # smallest_values, indices = torch.topk(-dist, 1, dim=1)
                min_dist_indices = dist.argmin(dim=1)
                # fine_feature_dist = F.cosine_similarity(res2_reshaped[idxS][indices], res2_reference_sampled.unsqueeze(1).repeat(1,16,1), dim=-1)
                #res2_reshaped[idxS][indices] - res2_reference_sampled.unsqueeze(1)
                # fine_feature_dist = (fine_feature_dist**2).sum(dim=-1)
                # min_dist_indices = fine_feature_dist.argmax(dim=1)
                # min_dist_indices_expanded = min_dist_indices.unsqueeze(1)  # 转换为 [2048, 1] 的形状
                # min_dist_indices = torch.gather(indices, 1, min_dist_indices_expanded)
                # min_dist_indices = min_dist_indices.squeeze(1)
                # min_dist_indices = indices.index_select(0, mi n_dist_indices[:,None])
            y_indices = min_dist_indices // W  
            x_indices = min_dist_indices % W   

            # Stack indices to get the final Nx2 tensor
            matches = torch.stack((y_indices, x_indices), dim=1)
            
            track_matches.append(matches)
            track_confs.append(conf[idxS, y_indices, x_indices])
            original_pts.append(pts[idxS, y_indices, x_indices])
        track_matches= torch.stack(track_matches, dim=0)
        track_confs= torch.stack(track_confs, dim=0)
        original_pts = torch.stack(original_pts, dim=0)
        
        # track_matches: SxNx2
        # track_confs: SxN
            
        assert (intrinsics[:, 0, 1] <= 1e-6).all(), "intrinsics should not have skew"
        assert (intrinsics[:, 1, 0] <= 1e-6).all(), "intrinsics should not have skew"
        
        
        principal_point = intrinsics[:, [0, 1], [2, 2]].unsqueeze(-2)
        focal_length = intrinsics[:, [0, 1], [0, 1]].unsqueeze(-2)

        # NOTE
        # Very careful here!
        # switch from yx to xy to fit the codebase of vggsfm
        track_matches = track_matches[..., [1,0]]
        # original_pts = original_pts[..., [1,0]]
        
        tracks_normalized = (track_matches - principal_point) / focal_length

        track_confs_vggsfm = track_confs.clone()
        track_confs_vggsfm = track_confs_vggsfm - conf_threshold
        track_confs_vggsfm = track_confs_vggsfm + 0.5
        
        
        
        with autocast(dtype=torch.double):
            triangulated_points, inlier_num, inlier_mask = triangulate_tracks(w2cs[:, :3], tracks_normalized, 
                            max_ransac_iters=512, lo_num=300, 
                            track_vis=torch.ones_like(tracks_normalized[..., 0]),
                            track_score=track_confs_vggsfm)
            
        camera_intrinsics = intrinsics[:1]
        B,N = w2cs.shape[:2]
        R = w2cs[0,:3,:3]
        T = w2cs[0,:3,3:]
        X_cam = torch.einsum("ik, pk -> pi", R, triangulated_points) + T.T
        proj_depth = X_cam[...,2:3].reshape(-1,1)
        uv_map = X_cam/X_cam[...,2:3]
        pos = torch.einsum("bik, bpk -> bpi", camera_intrinsics.reshape(-1,3,3), uv_map.reshape(1,-1,3)) # torch.matmul(camera_intrinsics.reshape(-1,3,3), X_cam)
        pos = pos[...,:2]
        mask = ((pos - track_matches[0]).norm(dim=-1) < 1.5) & (inlier_num>S//2)
        if mask.sum() < mask.flatten().shape[0] / 5:
            mask = (pos - track_matches[0]).norm(dim=-1) < torch.quantile((pos - track_matches[0]).norm(dim=-1), 0.05)
        triangulated_points_all.append(triangulated_points[mask[0]])
        track_matches_all.append(track_matches[0][mask[0]])
    triangulated_points_all = torch.cat(triangulated_points_all, dim=0)
    track_matches_all = torch.cat(track_matches_all, dim=0)
    # triangulated_points = triangulated_points[..., [1,0,2]]
    # triangulated_points: num_points_3d, 3
    # rgb_reference_sampled: num_points_3d, 3
    # inlier_num: num_points_3d
    # inlier_mask: num_points_3d, S
    # original_pts: S, num_points_3d, 3
    # track_matches: S, num_points_3d, 2
    # track_confs: S,num_points_3d
    
    return triangulated_points_all, rgb_reference_sampled, inlier_num, inlier_mask, original_pts, track_matches_all, track_confs, gt_pts_reference_sampled


class TriangulationMetric(nn.Module):
    def __init__(self, criterion, norm_mode='avg_dis', gt_scale=False):
        super().__init__()
        self.norm_mode = norm_mode
        self.gt_scale = gt_scale
        
    def batched_triangulate(self, pts, intrinsics, w2cs, conf, rgb=None, num_points_3d=4096, conf_threshold=3, visualize=True, min_tri_inlier_num=4, gt_pts3d=None, res2=None):
        # pts point map: B, N, H, W, 3 第0帧是reference view
        # intrinsics: B, N, 3, 3
        # w2cs: B, N, 4, 4
        # conf: reference view confidence map : B, N, H, W
        # rgb: B, N, 3, H, W用于可视化
        # 对于每个view都出相同数量的三角化出来的点云, 这个点云带有confidence, 方便求scale
        # output: triangluation_points: B, P, 3
        # output: conf: B, P
        # output: corresponding pts1 and pts2: B, P, 3 每一个三角化的point对应输入的点云的点
        B, S, H, W, _ = pts.shape
        all_triangulated_points = []
        all_triangulated_points_rgb = []
        all_ransac_inlier_num = []
        all_ransac_inlier_mask = []
        all_original_pts = []
        all_track_matches = []
        all_track_confs = []
        all_points_conf = []
        if gt_pts3d is None:
            gt_pts3d = pts

        with torch.no_grad():
            for b in range(B):
                # triangulated_points: num_points_3d, 3
                # triangulated_points_rgb: num_points_3d, 3
                # ransac_inlier_num: num_points_3d
                # ransac_inlier_mask: num_points_3d, S
                # original_pts: S, num_points_3d, 3
                # track_matches: S, num_points_3d, 2
                # track_confs: S,num_points_3d
                triangulated_points, triangulated_points_rgb, ransac_inlier_num, ransac_inlier_mask, original_pts, track_matches, track_confs, gt_pts_reference_sampled = triangulate_one_scene(pts[b], intrinsics[b], w2cs[b], conf[b], rgb[b], gt_pts = gt_pts3d[b], res2=res2[b])
                # triangulated_points_all, rgb_reference_sampled, inlier_num, inlier_mask, original_pts, track_matches_all, track_confs, gt_pts_reference_sampled
                # triangulated_points_all, rgb_reference_sampled, inlier_num, inlier_mask, original_pts, track_matches_all, track_confs, gt_pts_reference_sampled
                valid_triangulated_mask = ransac_inlier_num >= min_tri_inlier_num
                
                # Filter those points have fewer than min_tri_inlier_num by setting their conf to 1
                track_confs[:, ~valid_triangulated_mask] = 1
                
                ransac_inlier_float = ransac_inlier_mask.permute(1,0).float()
                points_conf = (track_confs * ransac_inlier_float).sum(dim=0) / ransac_inlier_num.clamp(min=1)
                
                # consistent with dust3r, whose min conf is 1 instead of 0
                points_conf[~valid_triangulated_mask] = 1
                
                
                # NOTE
                # The returned points need to be filtered by points_conf


                # visualizer = from vggsfm.utils.visualizer import Visualizer
                # if visualize:
                #     from dust3r.utils.visualizer import Visualizer
                #     vis = Visualizer(save_dir="visual", linewidth=1)
                #     # vis.visualize(rgb * 255, track_matches[None][:, ::16,:2], torch.ones_like(track_matches[:, ::16, :1])[None], filename=f"track_{b}")
                #     # import ipdb; ipdb.set_trace()
                #     # visual_mask = points_conf> conf_threshold
                #     import nerfvis.scene as scene_vis
                #     scene_vis.set_title(f"Scene {b}")
                #     scene_vis.set_opencv() 
                #     scene_vis.add_points("points", triangulated_points, vert_color=triangulated_points_rgb, point_size=3)
                #     # scene_vis.add_points("original", gt_pts_reference_sampled, vert_color=triangulated_points_rgb, point_size=3)
                #     scene_vis.display(port=1342)
                all_triangulated_points.append(triangulated_points)
                all_triangulated_points_rgb.append(triangulated_points_rgb)
                all_ransac_inlier_num.append(ransac_inlier_num)
                all_ransac_inlier_mask.append(ransac_inlier_mask)
                all_original_pts.append(original_pts)
                all_track_matches.append(track_matches)
                all_track_confs.append(track_confs)
                all_points_conf.append(points_conf)

        all_triangulated_points = torch.stack(all_triangulated_points, dim=0)   # B, num_points_3d, 3
        all_triangulated_points_rgb = torch.stack(all_triangulated_points_rgb, dim=0)   # B, num_points_3d, 3
        all_ransac_inlier_num = torch.stack(all_ransac_inlier_num, dim=0)   # B, num_points_3d
        all_ransac_inlier_mask = torch.stack(all_ransac_inlier_mask, dim=0)   # B, num_points_3d, S
        all_original_pts = torch.stack(all_original_pts, dim=0)   # B, S, num_points_3d, 3
        all_track_matches = torch.stack(all_track_matches, dim=0)   # B, S, num_points_3d, 2
        all_track_confs = torch.stack(all_track_confs, dim=0)   # B, S, num_points_3d
        all_points_conf = torch.stack(all_points_conf, dim=0)   # B, num_points_3d
        return all_triangulated_points, all_points_conf, all_original_pts, all_track_matches


    def forward(self, gt1, gt2, pred1, pred2, trajectory_pred, dist_clip=None, render_gt=None):
        view1 = gt1
        view2 = gt2
        nviews = len(view2)
        name = gt1[0]["instance"]
        scan = name[0]
        vid = gt1[0]['vid'][0]
        # frame_name = name[0].split('/')[-1]
        # if os.path.exists(f'./data/dtu/{scan}_depth/{frame_name}.npy'):
        #     return
        # from torchvision import transforms
        # from mast3r.s2dnet.s2dnet import S2DNet
        # S2DNet = S2DNet(pred2['pts3d'].device, checkpoint_path='/data0/zsz/mast3recon/checkpoints/s2dnet_weights.pth')
        # 假设 image_tensor 已经是一个大小为 [C, H, W] 的图像张量
        # 定义归一化操作
        # S2DNet = S2DNet.cuda()
        # normalize = transforms.Normalize(mean=[0.48235, 0.45882, 0.40784], std=[0.00392156862745098, 0.00392156862745098, 0.00392156862745098])

        pr_stage2 = pred2['pts3d'].to(torch.double)
        # res2 = pred2['feat_vgg_detail'].to(torch.double)
        B, num_views, H, W, _ = pr_stage2.shape
        pts3d = torch.cat([pr_stage2[:,:1].reshape(B,1,H,W,3), pr_stage2[:,1:].reshape(B,-1,H,W,3)], dim=1)
        pts3d_out = pr_stage2[:,:1].reshape(H,W,3)
        conf = pred2['conf'].to(torch.double)
        # conf = torch.cat([conf[:,:1].reshape(B,1,H,W), conf[:,:1].reshape(B,-1,H,W)], dim=1)
        intrinsics = torch.stack([view['camera_intrinsics'] for view in gt1+gt2],1).to(torch.double)
        extrinsics = torch.stack([view['camera_pose'] for view in gt1+gt2],1).to(torch.double)
        camera_pose = torch.stack([gt1_per['camera_pose'] for gt1_per in gt1], dim=1).to(torch.double)
        in_camera1 = inv(camera_pose)
        # extrinsics = torch.einsum('bnjk,bnkl->bnjl', in_camera1.repeat(1, extrinsics.shape[1],1,1), extrinsics)
        w2cs = extrinsics.inverse()
        images_gt = torch.stack([gt['img_org'] for gt in gt1+gt2], dim=1)
        images_gt = images_gt / 2 + 0.5

        # import nerfvis.scene as scene_vis 
        # scene_vis.set_opencv()
        # images_gt = images_gt.permute(0,1,3,4,2)
        # mask = conf > 3
        # mask = mask.flatten()
        # scene_vis.add_points('pred', (pts3d.reshape(-1,3))[mask], vert_color=images_gt.reshape(-1,3)[mask], size = 1)
        # scene_vis.display(port=1342)
        # localf_list = []
        # for image in images_gt[0]:
        #     image = normalize(image)
        #     feature = S2DNet(image[None].cuda())
        #     localf = feature[0]
        #     localf_list.append(localf)
        # localf = torch.cat(localf_list, dim=0)
        # localf = localf.permute(0,2,3,1)
        with torch.no_grad():
            triangluation_points, pts_conf, corresponding_pts, all_track_matches = self.batched_triangulate(pts3d, intrinsics, w2cs, conf, images_gt, gt_pts3d=pts3d, res2=pr_stage2)
            camera_intrinsics = intrinsics[:, :1]
            B,N = w2cs.shape[:2]
            R = w2cs[0,0,:3,:3]
            T = w2cs[0,0,:3,3:]
            X_cam = torch.einsum("ik, pk -> pi", R, triangluation_points[0]) + T.T
            proj_depth = X_cam[...,2:3].reshape(-1,1)
            uv_map = X_cam/X_cam[...,2:3]
            pos = torch.einsum("ik, pk -> pi", camera_intrinsics.reshape(3,3), uv_map) # torch.matmul(camera_intrinsics.reshape(-1,3,3), X_cam)
            # pos = pos[...,:2].long
            # pos = geotrf(camera_intrinsics.reshape(-1,3,3), uv_map, norm=1, ncol=2)
            # X_cam = torch.einsum("bnik, bnpk -> bnpi", camera_intrinsics, uv_map)
            warped_depth = torch.zeros(H, W).to(proj_depth)
            proj_2d = pos
            mask =  torch.where(proj_2d[:,0] < H, 1, 0) * \
                torch.where(proj_2d[:,0] >= 0, 1, 0) * \
                torch.where(proj_2d[:,1] < W, 1, 0) * \
                torch.where(proj_2d[:,1] >= 0, 1, 0)
            
            # inds = torch.where(mask)[0]
            # proj_2d = torch.index_select(pos[0].reshape(-1,2), dim=0, index=inds).to(torch.long)
            # proj_2d = (proj_2d[...,0] * W) + proj_2d[...,1]
            # proj_depth = torch.index_select(proj_depth.reshape(-1), dim=0, index=inds).squeeze()
            # warped_depth[proj_2d] = proj_depth
            pts3d_orig = pts3d_out[all_track_matches[0, :, 1], all_track_matches[0, :, 0]]
            # pts3d_orig = pts3d_out
            warped_depth[all_track_matches[0, :, 1], all_track_matches[0, :, 0]] = proj_depth.squeeze()
            
            name = gt1[0]["instance"]
            scan = name[0]
            # frame_name = name[0].split('/')[-1]
            os.makedirs(f'./data/dtu/{scan}_depth/', exist_ok=True)
            # np.save(f'./data/dtu/{scan}_depth/{frame_name}.npy', warped_depth.detach().cpu().numpy())
            warped_depth = (warped_depth - warped_depth[warped_depth>0].min()) / (warped_depth.max() - warped_depth[warped_depth>0].min()) * 255.0
            warped_depth = warped_depth.reshape(H, W).detach().cpu().numpy()
            
            warped_depth = warped_depth.astype(np.uint8)
            import matplotlib
            import cv2
            cmap = matplotlib.colormaps.get_cmap('Spectral_r')
            warped_depth = (cmap(warped_depth)[:, :, :3] * 255)[:, :, ::-1].astype(np.uint8)
            # warped_depth[mask.reshape(H, W).detach().cpu().numpy() == 0] = 0

            import roma 
            R_pts2, T, s = roma.rigid_points_registration(pts3d_orig[None], triangluation_points, compute_scaling=True)
            pts3d_orig = s[:,None, None] * torch.einsum('bik,bhk->bhi', R_pts2, pts3d_orig.reshape(-1,3)[mask].reshape(1, -1, 3)) + T[:,None, :]
            os.makedirs(f'./data/dtu/direct_{vid}/', exist_ok=True)
            np.save(f'./data/dtu/direct_{vid}/{scan}.npy', pts3d_orig.detach().cpu().numpy())

def apply_log_to_norm(xyz):
    d = xyz.norm(dim=-1, keepdim=True)
    xyz = xyz / d.clip(min=1e-8)
    xyz = xyz * torch.log1p(d)
    return xyz

def compare_translation_by_angle(t_gt, t, eps=1e-15, default_err=1e6):
    """Normalize the translation vectors and compute the angle between them."""
    t_norm = torch.norm(t, dim=1, keepdim=True)
    t = t / (t_norm + eps)

    t_gt_norm = torch.norm(t_gt, dim=1, keepdim=True)
    t_gt = t_gt / (t_gt_norm + eps)

    loss_t = torch.clamp_min(1.0 - torch.sum(t * t_gt, dim=1) ** 2, eps)
    err_t = torch.acos(torch.sqrt(1 - loss_t))

    err_t[torch.isnan(err_t) | torch.isinf(err_t)] = default_err
    return err_t

def translation_angle(tvec_gt, tvec_pred, batch_size=None, ambiguity=True):
    # tvec_gt, tvec_pred (B, 3,)
    rel_tangle_deg = compare_translation_by_angle(tvec_gt, tvec_pred)
    rel_tangle_deg = rel_tangle_deg * 180.0 / np.pi

    if ambiguity:
        rel_tangle_deg = torch.min(rel_tangle_deg, (180 - rel_tangle_deg).abs())

    if batch_size is not None:
        rel_tangle_deg = rel_tangle_deg.reshape(batch_size, -1)

    return rel_tangle_deg


def rotation_angle(rot_gt, rot_pred, batch_size=None, eps=1e-15):
    #########
    q_pred = matrix_to_quaternion(rot_pred)
    q_gt = matrix_to_quaternion(rot_gt)

    loss_q = (1 - (q_pred * q_gt).sum(dim=1) ** 2).clamp(min=eps)
    err_q = torch.arccos(1 - 2 * loss_q)

    rel_rangle_deg = err_q * 180 / np.pi

    if batch_size is not None:
        rel_rangle_deg = rel_rangle_deg.reshape(batch_size, -1)

    return rel_rangle_deg

def camera_to_rel_deg(pred_cameras, gt_cameras, device, batch_size):
    """
    Calculate relative rotation and translation angles between predicted and ground truth cameras.

    Args:
    - pred_cameras: Predicted camera.
    - gt_cameras: Ground truth camera.
    - accelerator: The device for moving tensors to GPU or others.
    - batch_size: Number of data samples in one batch.

    Returns:
    - rel_rotation_angle_deg, rel_translation_angle_deg: Relative rotation and translation angles in degrees.
    """

    with torch.no_grad():
        # Convert cameras to 4x4 SE3 transformation matrices
        gt_se3 = gt_cameras
        pred_se3 = pred_cameras

        # Generate pairwise indices to compute relative poses
        pair_idx_i1, pair_idx_i2 = batched_all_pairs(batch_size, gt_se3.shape[0] // batch_size)
        pair_idx_i1 = pair_idx_i1.to(device)

        # Compute relative camera poses between pairs
        # We use closed_form_inverse to avoid potential numerical loss by torch.inverse()
        # This is possible because of SE3
        relative_pose_gt = closed_form_inverse(gt_se3[pair_idx_i1]).bmm(gt_se3[pair_idx_i2])
        relative_pose_pred = closed_form_inverse(pred_se3[pair_idx_i1]).bmm(pred_se3[pair_idx_i2])
        # Compute the difference in rotation and translation
        # between the ground truth and predicted relative camera poses
        rel_rangle_deg = rotation_angle(relative_pose_gt[:, :3, :3], relative_pose_pred[:, :3, :3])
        rel_tangle_deg = translation_angle(relative_pose_gt[:, :3, 3:], relative_pose_pred[:, :3, 3:])
    return rel_rangle_deg, rel_tangle_deg


class DTUMetric(nn.Module):
    def __init__(self, criterion, norm_mode='avg_dis', gt_scale=False):
        super().__init__()
        self.norm_mode = norm_mode
        self.gt_scale = gt_scale
        self.mean_d2s_list = []
        self.mean_s2d_list = []
        
    def forward(self, gt1, gt2, pred1, pred2, trajectory_pred, dist_clip=None, render_gt=None):
        # data_pcd = torch.cat((pred1['pts3d'].reshape(-1,3), pred2['pts3d'].reshape(-1,3)), dim=1).reshape(-1,3).detach().cpu().numpy()
        import roma
        import sklearn.neighbors as skln
        import open3d as o3d
        pr_stage2 = pred2['pts3d']
        name = gt1[0]["instance"]

        print(f'-------------------{name}-------------------')
        B, num_views, H, W, _ = pr_stage2.shape
        valid1 = torch.stack([gt1_per['valid_mask'] for gt1_per in gt1], dim=1).view(B,-1,H,W).clone()
        valid2 = torch.stack([gt2_per['valid_mask'] for gt2_per in gt2], dim=1).view(B,-1,H,W).clone()
        trajectory = torch.stack([gt1_per['camera_pose'] for gt1_per in gt1 + gt2], dim=1)
        in_camera1 = inv(trajectory[:, :1])
        gt_pts3d = torch.stack([gt1_per['pts3d'] for gt1_per in gt1 + gt2], dim=1)
        gt_pts3d = geotrf(in_camera1.repeat(1,num_views,1,1).view(-1,4,4), gt_pts3d.view(-1,H,W,3))  # B,H,W,3
        gt_pts3d = gt_pts3d.view(B,-1,H,W,3)
        pr_pts1, pr_pts2, norm_factor_pr = normalize_pointcloud(pr_stage2[:,:1].reshape(B, -1,W,3), pr_stage2[:,1:].reshape(B, -1,W,3), self.norm_mode, valid1.reshape(B, -1,W), valid2.reshape(B, -1,W), ret_factor=True)
        pr_pts = torch.cat((pr_pts1, pr_pts2), dim=1).reshape(B, -1,3)
        gt_pts1, gt_pts2, norm_factor_gt = normalize_pointcloud(gt_pts3d[:,:1].reshape(B, -1,W,3), gt_pts3d[:,1:].reshape(B, -1,W,3), self.norm_mode, valid1.reshape(B, -1,W), valid2.reshape(B, -1,W), ret_factor=True)
        gt_pts3d = torch.cat((gt_pts1, gt_pts2), dim=1).reshape(B, -1,3)
        valid = torch.cat((valid1.reshape(B, -1,W), valid2.reshape(B, -1,W)), dim=1).reshape(B, -1)
        R_pts2, T, s = roma.rigid_points_registration(pr_pts1[valid1[0]].reshape(1,-1,3), gt_pts1[valid1[0]].reshape(1,-1,3), compute_scaling=True)
        pr_pts = s[:,None, None] * torch.einsum('bik,bhk->bhi', R_pts2, pr_pts.reshape(B, -1, 3)) + T[:,None, :]
        pr_pts = pr_pts.reshape(-1, 3).detach().cpu().numpy()
        gt_pts3d = gt_pts3d.reshape(-1, 3).detach().cpu().numpy()
        file_name = os.path.basename(name[0])
        valid = valid.detach().cpu().numpy()
        pcd_gt = o3d.geometry.PointCloud()
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pr_pts.reshape(-1, 3))
        pcd_gt.points = o3d.utility.Vector3dVector(gt_pts3d.reshape(-1, 3))
        dist1 = np.array(pcd.compute_point_cloud_distance(pcd_gt))
        dist2 = np.array(pcd_gt.compute_point_cloud_distance(pcd))
        thresh = 0.2

        mean_d2s = dist1[dist1 < thresh].mean()
        mean_s2d = dist2[dist2 < thresh].mean()
        over_all = (mean_d2s + mean_s2d) / 2
        self.mean_d2s_list.append(mean_d2s)
        self.mean_s2d_list.append(mean_s2d)
        temp_mean_d2s = np.mean(self.mean_d2s_list)
        temp_mean_s2d = np.mean(self.mean_s2d_list)
        
        print("Mean d2s: {:.4f}, Mean s2d: {:.4f}, Over all: {:.4f}".format(temp_mean_d2s, temp_mean_s2d, (temp_mean_d2s+temp_mean_s2d)/2))
        return mean_d2s, mean_s2d, over_all 


class IMC_10k(Criterion, MultiLoss):
    def __init__(self, criterion, norm_mode='avg_dis', gt_scale=False):
        super().__init__(criterion)
        self.norm_mode = norm_mode
        self.gt_scale = gt_scale
        
    def compute_loss(self, gt1, gt2, pred1, pred2, trajectory_pred, dist_clip=None, render_gt=None):
        trajectory = torch.stack([view['camera_pose'] for view in gt1+gt2], dim=1).reshape(-1,4,4)
        camera_pose = torch.cat([gt1_per['camera_pose'] for gt1_per in gt1], dim=0)
        in_camera1 = inv(camera_pose)
        trajectory = torch.einsum('bjk,bkl->bjl', in_camera1.repeat(trajectory.shape[0],1,1), trajectory)
        pred_cameras_R = trajectory_pred[-1]['R'].reshape(-1,3,3)
        pred_cameras_T = trajectory_pred[-1]['T'].reshape(-1,3)
        trajectory_pred = trajectory.clone()
        trajectory_pred[:,:3,:3] = pred_cameras_R
        pred_cameras_T = pred_cameras_T / (pred_cameras_T.norm(dim=-1, keepdim=True).mean(dim=0))
        trajectory_pred[:,:3,3] = pred_cameras_T
        trajectory_T = trajectory[:,:3,3]
        trajectory_T = trajectory_T / (trajectory_T.norm(dim=-1, keepdim=True).mean(dim=0))
        trajectory[:,:3,3] = trajectory_T
        num = -1
        rel_rangle_deg, rel_tangle_deg = camera_to_rel_deg(trajectory_pred[:num], trajectory[:num], trajectory.device, 1)
        return rel_rangle_deg, rel_tangle_deg

    

def transpose_to_landscape_render(head, activate=True):
    """ Predict in the correct aspect-ratio,
        then transpose the result in landscape 
        and stack everything back together.
    """
    def wrapper_no(decout, true_shape):
        B = len(true_shape)
        assert true_shape[0:1].allclose(true_shape), 'true_shape must be all identical'
        H, W = true_shape[0].cpu().tolist()
        res = head(decout, (H, W))
        return res

    def wrapper_yes(decout, true_shape):
        B = len(true_shape)
        # by definition, the batch is in landscape mode so W >= H
        H, W = int(true_shape.min()), int(true_shape.max())

        height, width = true_shape.T
        is_landscape = (width >= height)
        is_portrait = ~is_landscape

        # true_shape = true_shape.cpu()
        if is_landscape.all():
            return head(decout, (H, W))
        if is_portrait.all():
            return transposed(head(decout, (W, H)))
        # batch is a mix of both portraint & landscape
        def selout(ar):  
            ret = []
            for d in decout:
                if type(d) == dict:
                    ret.append(d)
                else:
                    ret.append(d[ar])
            return ret
        l_result = head(selout(is_landscape), (H, W))
        p_result = transposed(head(selout(is_portrait), (W, H)))

        # allocate full result
        result = {}
        for k in l_result | p_result:
            x = l_result[k].new(B, *l_result[k].shape[1:])
            x[is_landscape] = l_result[k]
            x[is_portrait] = p_result[k]
            result[k] = x

        return result

    return wrapper_yes if activate else wrapper_no

def convert_to_buffer(module: nn.Module, persistent: bool = True):
    # Recurse over child modules.
    for name, child in list(module.named_children()):
        convert_to_buffer(child, persistent)

    # Also re-save buffers to change persistence.
    for name, parameter_or_buffer in (
        *module.named_parameters(recurse=False),
        *module.named_buffers(recurse=False),
    ):
        value = parameter_or_buffer.detach().clone()
        delattr(module, name)
        module.register_buffer(name, value, persistent=persistent)


def rotation_6d_to_matrix(d6):
    """
    Converts 6D rotation representation by Zhou et al. [1] to rotation matrix
    using Gram--Schmidt orthogonalization per Section B of [1]. Adapted from pytorch3d.
    Args:
        d6: 6D rotation representation, of size (*, 6)

    Returns:
        batch of rotation matrices of size (*, 3, 3)

    [1] Zhou, Y., Barnes, C., Lu, J., Yang, J., & Li, H.
    On the Continuity of Rotation Representations in Neural Networks.
    IEEE Conference on Computer Vision and Pattern Recognition, 2019.
    Retrieved from http://arxiv.org/abs/1812.07035
    """

    a1, a2 = d6[..., :3], d6[..., 3:]
    b1 = F.normalize(a1, dim=-1)
    b2 = a2 - (b1 * a2).sum(-1, keepdim=True) * b1
    b2 = F.normalize(b2, dim=-1)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack((b1, b2, b3), dim=-2)


class Eval_NVS(nn.Module):
    """ Ensure that all 3D points are correct.
        Asymmetric loss: view1 is supposed to be the anchor.

        P1 = RT1 @ D1
        P2 = RT2 @ D2
        loss1 = (I @ pred_D1) - (RT1^-1 @ RT1 @ D1)
        loss2 = (RT21 @ pred_D2) - (RT1^-1 @ P2)
              = (RT21 @ pred_D2) - (RT1^-1 @ RT2 @ D2)
    """
    def __init__(self, alpha, scaling_mode='interp_5e-4_0.1_3'):
        super().__init__()
        self.scaling_mode = {'type':scaling_mode.split('_')[0], 'min_scaling': eval(scaling_mode.split('_')[1]), 'max_scaling': eval(scaling_mode.split('_')[2]), 'shift': eval(scaling_mode.split('_')[3])}
        self.render_landscape = transpose_to_landscape_render(self.render)
        self.alpha = alpha
        self.lpips = LPIPS(net="vgg")
        convert_to_buffer(self.lpips, persistent=False)
        self.lpips = self.lpips.cuda()

    def render(self, render_params, image_size):
        latent, output_fxfycxcy, output_c2ws = render_params
        (H_org, W_org) = image_size
        H = H_org
        W = W_org
        new_latent = {}
        if self.scaling_mode['type'] == 'precomp': 
            scaling_factor = latent['scaling_factor']
            x = torch.clip(latent['pre_scaling'] - self.scaling_mode['shift'], max=np.log(0.3))
            new_latent['scaling'] = torch.exp(x).clamp(min= self.scaling_mode['min_scaling'], max=self.scaling_mode['max_scaling']) / scaling_factor
            skip = ['pre_scaling', 'scaling', 'scaling_factor']
        else:
            skip = ['pre_scaling', 'scaling_factor']
        for key in latent.keys():
            if key not in skip:
                new_latent[key] = latent[key]
        gs_render = GaussianRenderer(H_org, W_org, gs_kwargs={'type':self.scaling_mode['type'], 'min_scaling': self.scaling_mode['min_scaling'], 'max_scaling': self.scaling_mode['max_scaling'], 'scaling_factor': scaling_factor})
        results = gs_render(new_latent, output_fxfycxcy.reshape(1,-1,4), output_c2ws.reshape(1,-1,4,4))
        images = results['image']
        image_mask = results['alpha']
        depth = results['depth']
        depth = depth.reshape(-1,1,H,W).permute(0,2,3,1)
        image_mask = image_mask.reshape(-1,1,H,W).permute(0,2,3,1)
        images = images.reshape(-1,3,H,W).permute(0,2,3,1)
        return {'images': images, 'image_mask': image_mask, 'depth':depth}
    
    def get_all_pts3d(self, name, gt1, gt2, pred1, pred2, trajectory_pred, dist_clip=None, render_gt=None):
        from mast3r.metrics import compute_pose_error, compute_lpips, compute_psnr, compute_ssim
        pr_pts1 = pred1['pts3d']
        pr_pts2 = pred2['pts3d']
        # GT pts3d
        B, H, W, _ = pr_pts1.shape
        # camera trajectory
        Rs = []
        Ts = []
        # valid mask
        sh_dim = pred1['feature'].shape[-1]
        feature1 = pred1['feature'].reshape(B,H,W,sh_dim)
        feature2 = pred2['feature'].reshape(B,-1,W,sh_dim)
        feature = torch.cat((feature1, feature2), dim=1).float()
        opacity1 = pred1['opacity'].reshape(B,H,W,1)
        opacity2 = pred2['opacity'].reshape(B,-1,W,1)
        opacity = torch.cat((opacity1, opacity2), dim=1).float()
        scaling1 = pred1['scaling'].reshape(B,H,W,3)
        scaling2 = pred2['scaling'].reshape(B,-1,W,3)
        scaling = torch.cat((scaling1, scaling2), dim=1).float()
        rotation1 = pred1['rotation'].reshape(B,H,W,4)
        rotation2 = pred2['rotation'].reshape(B,-1,W,4)
        rotation = torch.cat((rotation1, rotation2), dim=1).float()
        output_fxfycxcy = torch.stack([gt['fxfycxcy'] for gt in render_gt], dim=1)
        shape = torch.stack([gt['true_shape'] for gt in render_gt], dim=1)
        xyz = torch.cat((pr_pts1, pr_pts2), dim=1)
        images_gt = torch.stack([gt['img_org'] for gt in render_gt], dim=1)
        images_gt = images_gt / 2 + 0.5
        norm_factor_gt = 1
        output_c2ws = torch.stack([gt['camera_pose'] for gt in render_gt], dim=1)
        camera_pose = torch.stack([gt1_['camera_pose'] for gt1_ in gt1], dim=1)
        in_camera1 = inv(camera_pose)
        output_c2ws = torch.einsum('bnjk,bnkl->bnjl', in_camera1.repeat(1,output_c2ws.shape[1],1,1), output_c2ws)
        output_c2ws[..., :3, 3:] = output_c2ws[..., :3, 3:]
        with torch.set_grad_enabled(True):
            B = output_c2ws.shape[0]
            v = output_c2ws.shape[1]
            cam_rot_delta = nn.Parameter(torch.zeros([B, v, 6], requires_grad=True, device=output_c2ws.device))
            cam_trans_delta = nn.Parameter(torch.zeros([B, v, 3], requires_grad=True, device=output_c2ws.device))
            opt_params = []
            self.register_buffer("identity", torch.tensor([1.0, 0.0, 0.0, 0.0, 1.0, 0.0]).to(output_c2ws))
            opt_params.append(
                {
                    "params": [cam_rot_delta],
                    "lr": 0.005,
                }
            )
            opt_params.append(
                {
                    "params": [cam_trans_delta],
                    "lr": 0.005,
                }
            )
            pose_optimizer = torch.optim.Adam(opt_params)
            i = 0
            latent = {'xyz': xyz.reshape(B, -1, 3), 'feature': feature.reshape(B, -1, sh_dim), 'opacity': opacity.reshape(B, -1, 1), 'pre_scaling': scaling.reshape(B, -1, 3).clone(), 'rotation': rotation.reshape(B, -1, 4), 'scaling_factor': 1}
            extrinsics = output_c2ws
            output_c2ws_org = output_c2ws.clone()
            for i in range(100):
                pose_optimizer.zero_grad()
                dx, drot = cam_trans_delta, cam_rot_delta
                rot = rotation_6d_to_matrix(
                    drot + self.identity.expand(B, v, -1)
                )  # (..., 3, 3)
                transform = torch.eye(4, device=extrinsics.device).repeat((B, v, 1, 1))
                transform[..., :3, :3] = rot
                transform[..., :3, 3] = dx
                new_extrinsics = torch.matmul(extrinsics, transform)
                ret = self.render_landscape([latent, output_fxfycxcy[:1].reshape(-1,4), new_extrinsics.reshape(-1,4,4)], shape[:1].reshape(-1,2))
                color = ret['images']
                color = color.permute(0,3,1,2)
                loss_vgg = self.lpips.forward(
                    color,
                    images_gt[0],
                    normalize=True,
                )
                total_loss = loss_vgg.mean() * 0.1 + ((color - images_gt[0])**2).mean()
                total_loss.backward()
                pose_optimizer.step()
            i = 0
            ret = self.render_landscape([latent, output_fxfycxcy.reshape(-1,4), new_extrinsics.reshape(-1,4,4)], shape.reshape(-1,2))
            images = ret['images']
        all_metrics = {}
        images_gt = images_gt[0]
        images = images.permute(0,3,1,2)
        lpips_metric = compute_lpips(images_gt, images).mean()
        ssim_metric = compute_ssim(images_gt, images).mean()
        psnr_metric = compute_psnr(images_gt, images).mean()
        return psnr_metric, ssim_metric, lpips_metric, images

    def __call__(self, gt1, gt2, pred1, pred2, trajectory_pred, render_gt=None, **kw):
        name = gt1[0]['label'][0]
        psnr_metric, ssim_metric, lpips_metric, images = self.get_all_pts3d(name, gt1, gt2, pred1, pred2, trajectory_pred, render_gt=render_gt, **kw)
        loss_image_unsupervised = loss_2d_unsupervised = 0
        details = {"psnr": psnr_metric, "ssim": ssim_metric, "lpips": lpips_metric, 'images': images}
        return lpips_metric, details



class MeshOutput():
    def __init__(self, sam=False):
        self.sam = sam
    
    def __call__(self, gt1, gt2, pred1, pred2, trajectory_pred, render_gt=None, **kw):
        pts3d = pred2['pts3d']
        conf = pred2['conf']
        pts3d = pts3d.detach().cpu()
        B, N, H, W, _ = pts3d.shape
        thres = torch.quantile(conf.flatten(2,3), 0.1, dim=-1)[0]
        masks_conf = conf > thres[None, :, None, None]
        masks_conf = masks_conf.cpu()
        
        images = [view['img_org'] for view in gt1+gt2]
        shape = torch.stack([view['true_shape'] for view in gt1+gt2], dim=1)
        images = torch.stack(images,1).permute(0,1,3,4,2).detach().cpu().numpy()
        images = images / 2 + 0.5
        images = images.reshape(B, N, H, W, 3)
        outfile = os.path.join('./output/mesh', gt1[0]['instance'][0].split('/')[-1])
        outfile = outfile
        os.makedirs(outfile, exist_ok=True)
        
        # estimate focal length
        images = images[0]
        pts3d = pts3d[0]
        masks_conf = masks_conf[0]
        xy_over_z = (pts3d[..., :2] / pts3d[..., 2:3]).nan_to_num(posinf=0, neginf=0)  # homogeneous (x,y,1)
        pp = torch.tensor((W/2, H/2)).to(xy_over_z)
        pixels = xy_grid(W, H, device=xy_over_z.device).view(1, -1, 2) - pp.view(-1, 1, 2)  # B,HW,2
        u, v = pixels[:1].unbind(dim=-1)
        x, y, z = pts3d[:1].reshape(-1,3).unbind(dim=-1)
        fx_votes = (u * z) / x
        fy_votes = (v * z) / y
        # assume square pixels, hence same focal for X and Y
        f_votes = torch.cat((fx_votes.view(B, -1), fy_votes.view(B, -1)), dim=-1)
        focal = torch.nanmedian(f_votes, dim=-1).values
        focal = focal.item()
        pts3d = pts3d.numpy()
        # use PNP to estimate camera poses
        pred_poses = []
        for i in range(pts3d.shape[0]):
            shape_input_each = shape[:, i]
            mesh_grid = xy_grid(shape_input_each[0,1], shape_input_each[0,0])
            cur_inlier = conf[0,i] > torch.quantile(conf[0,i], 0.6)
            cur_inlier = cur_inlier.detach().cpu().numpy()
            ransac_thres = 0.5
            confidence = 0.9999
            iterationsCount = 10_000
            cur_pts3d = pts3d[i]
            K = np.float32([(focal, 0, W/2), (0, focal, H/2), (0, 0, 1)])
            success, r_pose, t_pose, _ = cv2.solvePnPRansac(cur_pts3d[cur_inlier].astype(np.float64), mesh_grid[cur_inlier].astype(np.float64), K, None,
                                                            flags=cv2.SOLVEPNP_SQPNP,
                                                            iterationsCount=iterationsCount,
                                                            reprojectionError=1,
                                                            confidence=confidence)
            r_pose = cv2.Rodrigues(r_pose)[0]  
            RT = np.r_[np.c_[r_pose, t_pose], [(0,0,0,1)]]
            cam2world = np.linalg.inv(RT)
            pred_poses.append(cam2world)
        pred_poses = np.stack(pred_poses, axis=0)
        pred_poses = torch.tensor(pred_poses)


        # use sam to segment the sky region
        if self.sam:
            from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
            from transformers import SegformerFeatureExtractor, SegformerForSemanticSegmentation
            from sam.scripts.segformer import segformer_segmentation as segformer_func
            from sam.scripts.configs.ade20k_id2label import CONFIG as CONFIG_ADE20K_ID2LABEL
            import pycocotools.mask as maskUtils

            os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
            sam = sam_model_registry["vit_h"](checkpoint='/data0/zsz/mast3recon/checkpoints/sam_vit_h_4b8939.pth').cuda()
            semantic_branch_processor = SegformerFeatureExtractor.from_pretrained(
            "nvidia/segformer-b5-finetuned-ade-640-640")
            semantic_branch_model = SegformerForSemanticSegmentation.from_pretrained(
            "nvidia/segformer-b5-finetuned-ade-640-640").to(pts2)
            id2label = CONFIG_ADE20K_ID2LABEL
            mask_branch_model = SamAutomaticMaskGenerator(
                model=sam,
                points_per_side=64,
                # Foggy driving (zero-shot evaluate) is more challenging than other dataset, so we use a larger points_per_side
                pred_iou_thresh=0.86,
                stability_score_thresh=0.92,
                crop_n_layers=1,
                crop_n_points_downscale_factor=2,
                min_mask_region_area=100,  # Requires open-cv to run post-processing
                output_mode='coco_rle',
            )
            class_masks = []
            for img in images:
                anns = {'annotations': mask_branch_model.generate(img)}
                class_ids = segformer_func((img * 255).astype(np.uint8), semantic_branch_processor, semantic_branch_model, 'cuda')
                semantc_mask = class_ids.clone()
                anns['annotations'] = sorted(anns['annotations'], key=lambda x: x['area'], reverse=True)
                bitmasks, class_names = [], []

                for ann in anns['annotations']:
                    valid_mask = torch.tensor(maskUtils.decode(ann['segmentation'])).bool()
                    # get the class ids of the valid pixels
                    propose_classes_ids = class_ids[valid_mask]
                    num_class_proposals = len(torch.unique(propose_classes_ids))
                    if num_class_proposals == 1:
                        semantc_mask[valid_mask] = propose_classes_ids[0]
                        ann['class_name'] = id2label['id2label'][str(propose_classes_ids[0].item())]
                        ann['class_proposals'] = id2label['id2label'][str(propose_classes_ids[0].item())]
                        class_names.append(ann['class_name'])
                        # bitmasks.append(maskUtils.decode(ann['segmentation']))
                        continue
                    top_1_propose_class_ids = torch.bincount(propose_classes_ids.flatten()).topk(1).indices
                    top_1_propose_class_names = [id2label['id2label'][str(class_id.item())] for class_id in top_1_propose_class_ids]

                    semantc_mask[valid_mask] = top_1_propose_class_ids
                    ann['class_name'] = top_1_propose_class_names[0]
                    ann['class_proposals'] = top_1_propose_class_names[0]
                    class_names.append(ann['class_name'])
                    # bitmasks.append(maskUtils.decode(ann['segmentation']))

                    del valid_mask
                    del propose_classes_ids
                    del num_class_proposals
                    del top_1_propose_class_ids
                    del top_1_propose_class_names
                
                sematic_class_in_img = torch.unique(semantc_mask)
                semantic_bitmasks, semantic_class_names = [], []

                # semantic prediction
                anns['semantic_mask'] = {}
                flag = False
                for i in range(len(sematic_class_in_img)):
                    class_name = id2label['id2label'][str(sematic_class_in_img[i].item())]
                    if class_name != 'sky':
                        continue
                    flag = True             
                    class_mask = semantc_mask == sematic_class_in_img[i]
                    # class_mask = class_mask.cpu().numpy().astype(np.uint8)
                    class_masks.append(class_mask)
                if flag == False:
                    class_mask = torch.zeros_like(semantc_mask) > 0 
                    class_masks.append(class_mask)
            class_masks = torch.stack(class_masks, 0)
            class_masks = ~class_masks
        else:
            class_masks = torch.ones_like(masks_conf) > 0

        # use knn to clean the point cloud
        K = 10
        points = torch.tensor(pts3d.reshape(1,-1,3)).cuda()
        knn = knn_points(points, points, K=K)
        dists = knn.dists  
        mean_dists = dists.mean(dim=-1)
        masks_dist = mean_dists < torch.quantile(mean_dists.reshape(-1), 0.95)
        masks_dist = masks_dist.detach().cpu().numpy()
        
        import nerfvis.scene as scene_vis
        scene_vis.set_opencv()
        masks_conf = (masks_conf > 0) & masks_dist.reshape(-1,H,W) & class_masks.reshape(-1,H,W)
        masks_conf = masks_conf > 0
        filtered_points = pts3d[masks_conf].reshape(-1, 3)
        colors = images[masks_conf].reshape(-1, 3)
        scene_vis.add_points('points', filtered_points.reshape(-1,3), vert_color=colors.reshape(-1,3), size = 1)
        scene_vis.add_images(
            f"images/i",
            images, # Can be a list of paths too (requires joblib for that) 
            r=pred_poses[:, :3, :3],
            t=pred_poses[:, :3, 3],
            focal_length=focal,
            z=0.1,
            with_camera_frustum=True,
        )
        np.savez(outfile + '/pred.npz', pts3d=pts3d, vert_color=images, poses=pred_poses.detach().cpu(), intrinsic=focal, images=images, mask = masks_conf)
        print(f"save {outfile}")
        save_content = 'CUDA_VISIBLE_DEVICES=1 python visualizer/run_vis.py --result_npz {} --results_folder {}'.format(outfile + '/pred.npz', outfile)
        file_path = outfile + '/run_vis.sh'
        os.system(save_content)
        print(f"run {file_path} to visualize geometry and poses")
        with open(file_path, 'a') as file:
            file.write(save_content + '\n') 
        scene_vis.display(port=8828)
        return None, None
    

def interpolate_pose(pose1, pose2, t):
    """
    Interpolate between two camera poses (4x4 matrices).

    :param pose1: First pose (4x4 matrix)
    :param pose2: Second pose (4x4 matrix)
    :param t: Interpolation factor, t in [0, 1]
    :return: Interpolated pose (4x4 matrix)
    """

    # Extract translation and rotation from both poses
    translation1 = pose1[:3, 3].detach().cpu().numpy()
    translation2 = pose2[:3, 3].detach().cpu().numpy()
    rotation1 = pose1[:3, :3].detach().cpu().numpy()
    rotation2 = pose2[:3, :3].detach().cpu().numpy()

    # Interpolate the translation (linear interpolation)
    interpolated_translation = (1 - t) * translation1 + t * translation2
    
    # Convert rotation matrices to quaternions
    quat1 = R.from_matrix(rotation1).as_quat()
    quat2 = R.from_matrix(rotation2).as_quat()

    # Slerp for rotation interpolation
    slerp = Slerp([0, 1], R.from_quat([quat1, quat2]))

    interpolated_rotation = slerp(t).as_matrix()
    # Combine the interpolated rotation and translation
    interpolated_pose = np.eye(4)
    interpolated_pose[:3, :3] = interpolated_rotation
    interpolated_pose[:3, 3] = interpolated_translation
    return interpolated_pose

def colorize(value, vmin=None, vmax=None, cmap='magma_r', invalid_val=-99, invalid_mask=None, background_color=(128, 128, 128, 255), gamma_corrected=False, value_transform=None):
    """Converts a depth map to a color image.

    Args:
        value (torch.Tensor, numpy.ndarry): Input depth map. Shape: (H, W) or (1, H, W) or (1, 1, H, W). All singular dimensions are squeezed
        vmin (float, optional): vmin-valued entries are mapped to start color of cmap. If None, value.min() is used. Defaults to None.
        vmax (float, optional):  vmax-valued entries are mapped to end color of cmap. If None, value.max() is used. Defaults to None.
        cmap (str, optional): matplotlib colormap to use. Defaults to 'magma_r'.
        invalid_val (int, optional): Specifies value of invalid pixels that should be colored as 'background_color'. Defaults to -99.
        invalid_mask (numpy.ndarray, optional): Boolean mask for invalid regions. Defaults to None.
        background_color (tuple[int], optional): 4-tuple RGB color to give to invalid pixels. Defaults to (128, 128, 128, 255).
        gamma_corrected (bool, optional): Apply gamma correction to colored image. Defaults to False.
        value_transform (Callable, optional): Apply transform function to valid pixels before coloring. Defaults to None.

    Returns:
        numpy.ndarray, dtype - uint8: Colored depth map. Shape: (H, W, 4)
    """
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()

    value = value.squeeze()
    if invalid_mask is None:
        invalid_mask = value == invalid_val
    mask = np.logical_not(invalid_mask)

    # normalize
    vmin = np.percentile(value[mask],2) if vmin is None else vmin
    vmax = np.percentile(value[mask],85) if vmax is None else vmax
    if vmin != vmax:
        value = (value - vmin) / (vmax - vmin)  # vmin..vmax
    else:
        # Avoid 0-division
        value = value * 0.

    # squeeze last dim if it exists
    # grey out the invalid values

    value[invalid_mask] = np.nan
    import matplotlib
    cmapper = matplotlib.cm.get_cmap(cmap)
    if value_transform:
        value = value_transform(value)
        # value = value / value.max()
    value = cmapper(value, bytes=True)  # (nxmx4)

    # img = value[:, :, :]
    img = value[...]
    img[invalid_mask] = background_color

    #     return img.transpose((2, 0, 1))
    if gamma_corrected:
        # gamma correction
        img = img / 255
        img = np.power(img, 2.2)
        img = img * 255
        img = img.astype(np.uint8)
    img = torch.from_numpy(img)/255.
    return img.permute(2,0,1)[:3]