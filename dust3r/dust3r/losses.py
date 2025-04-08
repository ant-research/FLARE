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
from copy import copy, deepcopy

class MultiLoss (nn.Module):
    """ Easily combinable losses (also keep track of individual loss values):
        loss = MyLoss1() + 0.1*MyLoss2()
    Usage:
        Inherit from this class and override get_name() and compute_loss()
    """

    def __init__(self):
        super().__init__()
        self._alpha = 1
        self._loss2 = None

    def compute_loss(self, *args, **kwargs):
        raise NotImplementedError()

    def get_name(self):
        raise NotImplementedError()

    def __mul__(self, alpha):
        assert isinstance(alpha, (int, float))
        res = copy(self)
        res._alpha = alpha
        return res
    __rmul__ = __mul__  # same

    def __add__(self, loss2):
        assert isinstance(loss2, MultiLoss)
        res = cur = copy(self)
        # find the end of the chain
        while cur._loss2 is not None:
            cur = cur._loss2
        cur._loss2 = loss2
        return res

    def __repr__(self):
        name = self.get_name()
        if self._alpha != 1:
            name = f'{self._alpha:g}*{name}'
        if self._loss2:
            name = f'{name} + {self._loss2}'
        return name

    def forward(self, *args, **kwargs):
        loss = self.compute_loss(*args, **kwargs)
        if isinstance(loss, tuple) and len(loss) == 2:
            loss, details = loss
            monitoring = None
        elif isinstance(loss, tuple) and len(loss) == 3:
            loss, details, monitoring = loss
        elif loss.ndim == 0:
            details = {self.get_name(): float(loss)}
        else:
            details = {}
        loss = loss * self._alpha

        if self._loss2:
            if isinstance(loss, tuple) and len(loss) == 2:
                loss2, details2 = self._loss2(*args, **kwargs)
                loss = loss + loss2
                details |= details2
                monitoring = None
            elif isinstance(loss, tuple) and len(loss) == 3:
                loss2, details2, monitoring2 = self._loss2(*args, **kwargs)
                loss = loss + loss2
                details |= details2
                monitoring |= monitoring2

        return loss, details, monitoring


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

def closed_form_inverse(se3, R=None, T=None):
    """
    Computes the inverse of each 4x4 SE3 matrix in the batch.

    Args:
    - se3 (Tensor): Nx4x4 tensor of SE3 matrices.

    Returns:
    - Tensor: Nx4x4 tensor of inverted SE3 matrices.


    | R t |
    | 0 1 |
    -->
    | R^T  -R^T t|
    | 0       1  |
    """
    if R is None:
        R = se3[:, :3, :3]

    if T is None:
        T = se3[:, :3, 3:]

    # Compute the transpose of the rotation
    R_transposed = R.transpose(1, 2)

    # -R^T t
    top_right = -R_transposed.bmm(T)

    inverted_matrix = torch.eye(4, 4)[None].repeat(len(se3), 1, 1)
    inverted_matrix = inverted_matrix.to(R.dtype).to(R.device)

    inverted_matrix[:, :3, :3] = R_transposed
    inverted_matrix[:, :3, 3:] = top_right

    return inverted_matrix

def Sum(*losses_and_masks):
    loss, mask = losses_and_masks[0]
    if loss.ndim > 0:
        # we are actually returning the loss for every pixels
        return losses_and_masks
    else:
        # we are returning the global loss
        for loss2, mask2 in losses_and_masks[1:]:
            loss = loss + loss2
        return loss

class BaseCriterion(nn.Module):
    def __init__(self, reduction='mean'):
        super().__init__()
        self.reduction = reduction
        
class LLoss (BaseCriterion):
    """ L-norm loss
    """

    def forward(self, a, b):
        assert a.shape == b.shape and a.ndim >= 2 and 1 <= a.shape[-1] <= 3, f'Bad shape = {a.shape}'
        dist = self.distance(a, b)
        assert dist.ndim == a.ndim - 1  # one dimension less
        if self.reduction == 'none':
            return dist
        if self.reduction == 'sum':
            return dist.sum()
        if self.reduction == 'mean':
            return dist.mean() if dist.numel() > 0 else dist.new_zeros(())
        raise ValueError(f'bad {self.reduction=} mode')

    def distance(self, a, b):
        raise NotImplementedError()

class L21Loss (LLoss):
    """ Euclidean distance between 3d points  """

    def distance(self, a, b):
        return torch.norm(a - b, dim=-1)  # normalized L2 distance


L21 = L21Loss()


def rotation_distance(R1,R2,eps=1e-7):
    # http://www.boris-belousov.net/2016/12/01/quat-dist/
    R_diff = R1@R2.transpose(-2,-1)
    trace = R_diff[...,0,0]+R_diff[...,1,1]+R_diff[...,2,2]
    angle = ((trace-1)/2).clamp(-1+eps,1-eps).acos_() # numerical stability near -1/+1
    return angle

class Criterion (nn.Module):
    def __init__(self, criterion=None):
        super().__init__()
        assert isinstance(criterion, BaseCriterion), f'{criterion} is not a proper criterion!'
        self.criterion = copy(criterion)

    def get_name(self):
        return f'{type(self).__name__}({self.criterion})'

    def with_reduction(self, mode='none'):
        res = loss = deepcopy(self)
        while loss is not None:
            assert isinstance(loss, Criterion)
            loss.criterion.reduction = mode  # make it return the loss for each sample
            loss = loss._loss2  # we assume loss is a Multiloss
        return res

def batched_all_pairs(B, N):
    # B, N = se3.shape[:2]
    i1_, i2_ = torch.combinations(torch.arange(N), 2, with_replacement=False).unbind(-1)
    i1, i2 = [(i[None] + torch.arange(B)[:, None] * N).reshape(-1) for i in [i1_, i2_]]

    return i1, i2

class Regr3D_clean(Criterion, MultiLoss):
    """ Ensure that all 3D points are correct.
        Asymmetric loss: view1 is supposed to be the anchor.

        P1 = RT1 @ D1
        P2 = RT2 @ D2
        loss1 = (I @ pred_D1) - (RT1^-1 @ RT1 @ D1)
        loss2 = (RT21 @ pred_D2) - (RT1^-1 @ P2)
              = (RT21 @ pred_D2) - (RT1^-1 @ RT2 @ D2)
    """

    def __init__(self, criterion, norm_mode='avg_dis', disable_rigid=True, gt_scale=False, test=False):
        super().__init__(criterion)
        self.norm_mode = norm_mode
        self.gt_scale = gt_scale
        self.epoch = 0
        self.test = test
        self.gt_num_image = 0
        self.disable_rigid = disable_rigid

    def get_all_pts3d(self, gt1, gt2, pred1, pred2, trajectory_pred, dist_clip=None, render_gt=None):
        pr_ref = pred1['pts3d']
        pr_stage2 = pred2['pts3d']
        B, num_views, H, W, _ = pr_ref.shape
        num_views_src = num_views - 1
        gt_pts3d = torch.stack([gt1_per['pts3d'] for gt1_per in gt1 + gt2], dim=1)
        B, _, H, W, _ = gt_pts3d.shape
        trajectory = torch.stack([gt1_per['camera_pose'] for gt1_per in gt1 + gt2], dim=1)
        in_camera1 = inv(trajectory[:, :1])
        trajectory = torch.einsum('bnjk,bnkl->bnjl', in_camera1.repeat(1, trajectory.shape[1],1,1), trajectory)
        trajectory_t_gt = trajectory[..., :3, 3].clone()
        trajectory_t_gt = trajectory_t_gt / (trajectory_t_gt.norm(dim=-1, keepdim=True).mean(dim=1, keepdim=True) + 1e-5)
        trajectory_normalize = trajectory.clone().detach()
        trajectory_normalize[..., :3, 3] = trajectory_t_gt
        quaternion_R = matrix_to_quaternion(trajectory[...,:3,:3])
        trajectory_gt = torch.cat([trajectory_t_gt, quaternion_R], dim=-1)[None]
        R = trajectory[...,:3,:3]
        fxfycxcy1 = torch.stack([view['fxfycxcy'] for view in gt1], dim=1).float()
        fxfycxcy2 = torch.stack([view['fxfycxcy'] for view in gt2], dim=1).float()
        fxfycxcy = torch.cat((fxfycxcy1, fxfycxcy2), dim=1).to(fxfycxcy1)
        focal_length_gt = fxfycxcy[...,:2]
        with torch.no_grad():
            trajectory_R_prior = trajectory_pred[:4][-1]['R'].reshape(B, -1, 3, 3)
            trajectory_R_post = trajectory_pred[-1]['R'].reshape(B, -1, 3, 3)
        trajectory_t_pred = torch.stack([view["T"].reshape(B, -1, 3) for view in trajectory_pred], dim=1)
        trajectory_r_pred = torch.stack([view["quaternion_R"].reshape(B, -1, 4) for view in trajectory_pred], dim=1)
        focal_length_pred = torch.stack([view["focal_length"].reshape(B, -1, 2) for view in trajectory_pred], dim=1)
        trajectory_t_pred = trajectory_t_pred / (trajectory_t_pred.norm(dim=-1, keepdim=True).mean(dim=-2, keepdim=True) + 1e-5)
        trajectory_pred = torch.cat([trajectory_t_pred, trajectory_r_pred], dim=-1)
        trajectory_pred = trajectory_pred.permute(1,0,2,3)
        focal_length_pred = focal_length_pred.permute(1,0,2,3)
        se3_gt = torch.cat((R, trajectory_t_gt[..., None]), dim=-1)
        with torch.no_grad():
            pred_R = quaternion_to_matrix(trajectory_r_pred)
            se3_pred = torch.cat((pred_R, trajectory_t_pred[..., None]), dim=-1)
        gt_pts3d = geotrf(in_camera1.repeat(1,num_views,1,1).view(-1,4,4), gt_pts3d.view(-1,H,W,3))  # B,H,W,3
        gt_pts3d = gt_pts3d.view(B,-1,H,W,3)
        pr_ref = geotrf(trajectory_normalize.view(-1,4,4), pr_ref.view(-1,H,W,3)).view(B,-1,H,W,3)
        
        # valid mask
        valid1 = torch.stack([gt1_per['valid_mask'] for gt1_per in gt1], dim=1).view(B,-1,H,W).clone()
        valid2 = torch.stack([gt2_per['valid_mask'] for gt2_per in gt2], dim=1).view(B,-1,H,W).clone()
        dist_clip = 500
        if dist_clip is not None:
            # points that are too far-away == invalid
            dis1 = pr_stage2[:,:1].reshape(B, -1,W,3).norm(dim=-1)  # (B, H, W)
            dis2 = pr_stage2[:,1:].reshape(B, -1,W,3).norm(dim=-1)  # (B, H, W)
            valid1 = valid1 & (dis1 <= dist_clip).reshape(*valid1.shape)
            valid2 = valid2 & (dis2 <= dist_clip).reshape(*valid2.shape)
        gt_pts1, gt_pts2, norm_factor_gt = normalize_pointcloud(gt_pts3d[:,:1].reshape(B, -1,W,3), gt_pts3d[:,1:].reshape(B, -1,W,3), self.norm_mode, valid1.reshape(B, -1,W), valid2.reshape(B, -1,W), ret_factor=True)
        pr_pts1_ref, pr_pts2_ref, norm_factor_pr = normalize_pointcloud(pr_ref[:,:1].reshape(B, -1,W,3), pr_ref[:,1:].reshape(B, -1,W,3), self.norm_mode, valid1.reshape(B, -1,W), valid2.reshape(B, -1,W), ret_factor=True)
        
        pr_stage2_pts1, pr_stage2_pts2, norm_factor_pr_stage2 = normalize_pointcloud(pr_stage2[:,:1].reshape(B, -1,W,3), pr_stage2[:,1:].reshape(B, -1,W,3), self.norm_mode, valid1.reshape(B, -1,W), valid2.reshape(B, -1,W), ret_factor=True)

        conf_ref = pred1['conf']
        conf_stage2 = pred2['conf']
        true_counts = valid2.view(B, num_views_src, -1).sum(dim=2)
        min_true_counts_per_B = true_counts.min().item()
        if self.disable_rigid:
            min_true_counts_per_B = 0

        if min_true_counts_per_B > 10:
            mask = valid2.view(B, num_views_src, H, W)
            mask = mask.view(B*num_views_src, H, W)
            true_coords = []
            for i in range(mask.shape[0]):
                true_indices = torch.nonzero(mask[i])  # 获取所有 True 的坐标
                if true_indices.size(0) > 0:  # 确保有 True 值
                    sampled_indices = true_indices[torch.randint(0, true_indices.size(0), (min_true_counts_per_B,))]
                    true_coords.append(sampled_indices)
            true_coords = torch.stack(true_coords, dim=0)
            sampled_pts = []
            pr_pts2_reshaped = pr_stage2_pts2.reshape(B*num_views_src, H, W, 3)
            for i in range(len(true_coords)):
                coords = true_coords[i]
                sampled_points = pr_pts2_reshaped[i][coords[:, 0], coords[:, 1], :]
                sampled_pts.append(sampled_points)
            sampled_pts = torch.stack(sampled_pts, dim=0)
            gt_pts2_reshaped = gt_pts2.reshape(B*num_views_src, H, W, 3)
            sampled_gt_pts = []
            for i in range(len(true_coords)):
                coords = true_coords[i]
                sampled_points = gt_pts2_reshaped[i][coords[:, 0], coords[:, 1], :]
                sampled_gt_pts.append(sampled_points)
            sampled_gt_pts = torch.stack(sampled_gt_pts, dim=0)
           
            with torch.no_grad():
                R_pts2, T, s = roma.rigid_points_registration(sampled_pts, sampled_gt_pts, compute_scaling=True)
            if torch.isnan(s).any() or torch.isnan(R_pts2).any() or torch.isnan(T).any():
                pr_pts2_transform = gt_pts2.reshape(B, num_views_src * H, W, 3)
            else:
                pr_pts2_transform = s[:,None, None, None] * torch.einsum('bik,bhwk->bhwi', R_pts2, pr_stage2_pts2.reshape(B * num_views_src, H, W, 3)) + T[:,None,None, :]
                pr_pts2_transform = pr_pts2_transform.reshape(B, num_views_src * H, W, 3)
        else:
            pr_pts2_transform = gt_pts2.reshape(B, num_views_src * H, W, 3)
        return gt_pts1, gt_pts2, pr_pts1_ref, pr_pts2_ref, pr_stage2_pts1, pr_stage2_pts2, valid1, valid2, conf_ref, conf_stage2, trajectory_gt, trajectory_pred, R, trajectory_R_prior, trajectory_R_post, focal_length_gt, focal_length_pred, se3_gt, se3_pred, pr_pts2_transform, [norm_factor_gt, norm_factor_pr_stage2]

    def compute_loss(self, gt1, gt2, pred1, pred2, trajectory_pred, render_gt=None, **kw):
        gt_pts1, gt_pts2, pr_pts1_ref, pr_pts2_ref, pr_stage2_pts1, pr_stage2_pts2, valid1, valid2, conf_ref, conf_stage2, trajectory_gt, trajectory_pred, R, trajectory_R_prior, trajectory_R_post, focal_length_gt, focal_length_pred, se3_gt, se3_pred, pr_pts2_transform, monitoring = self.get_all_pts3d(gt1, gt2, pred1, pred2, trajectory_pred, render_gt=render_gt, **kw)
        valid1 = valid1.flatten(1,2)    
        valid2 = valid2.flatten(1,2)   
        l1_ref = self.criterion(pr_pts1_ref[valid1], gt_pts1[valid1])
        l2_ref = self.criterion(pr_pts2_ref[valid2], gt_pts2[valid2])
        l1_stage2 = self.criterion(pr_stage2_pts1[valid1], gt_pts1[valid1])
        l2_stage2 = self.criterion(pr_stage2_pts2[valid2], gt_pts2[valid2])
        norm_factor_gt, norm_factor_pr_stage2 = monitoring
        Reg_1_ref = l1_ref.mean() if l1_ref.numel() > 0 else 0
        Reg_2_ref = l2_ref.mean() if l2_ref.numel() > 0 else 0
        Reg_1_stage2 = l1_stage2.mean() if l1_stage2.numel() > 0 else 0
        Reg_2_stage2 = l2_stage2.mean() if l2_stage2.numel() > 0 else 0

        n_predictions = len(trajectory_pred)
        pose_r_loss = 0.0
        pose_t_loss = 0.0
        pose_f_loss = 0.0
        gamma = 0.8
        l2_rigid = self.criterion(pr_pts2_transform[valid2], gt_pts2[valid2])
        Reg_2_rigid = l2_rigid.mean() if l2_rigid.numel() > 0 else 0
        if Reg_2_rigid < 1e-5:
            Reg_2_rigid = 0
            l2_rigid = 0
        for i in range(n_predictions):
            i_weight = gamma ** (n_predictions - i  - 1)
            trajectory_pred_iter = trajectory_pred[i]
            trajectory_gt_iter = trajectory_gt[0]
            focal_length_gt_iter = focal_length_pred[i]
            fxfy_gt = focal_length_gt
            trajectory_pred_r = trajectory_pred_iter[..., 3:]
            trajectory_pred_t = trajectory_pred_iter[..., :3]
            trajectory_gt_r = trajectory_gt_iter[..., 3:]
            trajectory_gt_t = trajectory_gt_iter[..., :3]
            pose_f_loss += 0. #i_weight * (focal_length_gt_iter - fxfy_gt).abs().mean()
            pose_r_loss += i_weight * (trajectory_gt_r - trajectory_pred_r).abs().mean() * 5
            pose_t_loss += i_weight * (trajectory_gt_t - trajectory_pred_t).abs().mean() * 5

        pose_t_loss = pose_t_loss 
        pose_r_loss = pose_r_loss
        with torch.no_grad():
            batch_size, num_views = trajectory_pred.shape[1:3]
            rot_err_prior = torch.rad2deg(rotation_distance(R.reshape(*trajectory_R_prior.shape).float(), trajectory_R_prior.float())).mean()
            rot_err_post = torch.rad2deg(rotation_distance(R.reshape(*trajectory_R_post.shape).float(), trajectory_R_post.float())).mean()
            pair_idx_i1, pair_idx_i2 = batched_all_pairs(batch_size, num_views)
            se3_gt = torch.cat((R, trajectory_gt[0,..., :3, None]), dim=-1).reshape(-1 , 3, 4)
            se3_pred_prior = torch.cat((trajectory_R_prior, trajectory_pred[3][..., :3][..., None]), dim=-1).reshape(-1 , 3, 4)
            se3_pred_post = torch.cat((trajectory_R_post, trajectory_pred[-1][..., :3][..., None]), dim=-1).reshape(-1 , 3, 4)
            bottom_ = torch.tensor([[[0,0,0,1]]]).to(se3_gt.device)
            se3_gt = torch.cat((se3_gt, bottom_.repeat(se3_gt.shape[0],1,1)), dim=1)
            se3_pred_prior = torch.cat((se3_pred_prior, bottom_.repeat(se3_pred_prior.shape[0],1,1)), dim=1)
            se3_pred_post = torch.cat((se3_pred_post, bottom_.repeat(se3_pred_post.shape[0],1,1)), dim=1)
            relative_pose_gt = closed_form_inverse(se3_gt[pair_idx_i1]).bmm(se3_gt[pair_idx_i2])
            relative_pose_pred_prior = closed_form_inverse(se3_pred_prior[pair_idx_i1]).bmm(se3_pred_prior[pair_idx_i2])
            relative_pose_pred_post = closed_form_inverse(se3_pred_post[pair_idx_i1]).bmm(se3_pred_post[pair_idx_i2])
            rel_tangle_deg_prior = translation_angle(relative_pose_gt[:, :3, 3], relative_pose_pred_prior[:, :3, 3]).mean()
            rel_tangle_deg_post = translation_angle(relative_pose_gt[:, :3, 3], relative_pose_pred_post[:, :3, 3]).mean()

        self_name = type(self).__name__
        details = {'pr_stage2_pts1': float(pr_stage2_pts1.mean()), 'pr_stage2_pts2': float(pr_stage2_pts2.mean()), 'norm_factor_pr_stage2': float(norm_factor_pr_stage2.mean()), 'translation_error_prior': float(rel_tangle_deg_prior), 'translation_error': float(rel_tangle_deg_post), 'rot_err_post': float(rot_err_post), 'rot_err_prior': float(rot_err_prior), self_name+'_2_rigid': float(Reg_2_rigid),
        self_name+'_f_pose': float(pose_f_loss), self_name+'_t_pose': float(pose_t_loss), self_name+'_r_pose': float(pose_r_loss), 'trajectory_gt_t_first': float(trajectory_gt_t[:,0].abs().mean()), 'trajectory_pred_t_first': float(trajectory_pred_t[:,0].abs().mean()), self_name+'_1_ref': float(Reg_1_ref), self_name+'_2_ref': float(Reg_2_ref), self_name+'_1_stage2': float(Reg_1_stage2), self_name+'_2_stage2': float(Reg_2_stage2)}
        return Sum((l1_ref, valid1), (l2_ref, valid2), (l1_stage2, valid1), (l2_stage2, valid2), (pose_r_loss, None), (pose_t_loss, None), (pose_f_loss, None),  (0, None), (0, None), (0, None), (0, None), (l2_rigid, valid2)), details, monitoring




class ConfLoss(MultiLoss):
    """ Weighted regression by learned confidence.
        Assuming the input pixel_loss is a pixel-level regression loss.

    Principle:
        high-confidence means high conf = 0.1 ==> conf_loss = x / 10 + alpha*log(10)
        low  confidence means low  conf = 10  ==> conf_loss = x * 10 - alpha*log(10) 

        alpha: hyperparameter
    """

    def __init__(self, pixel_loss, alpha=1):
        super().__init__()
        assert alpha > 0
        self.alpha = alpha
        self.pixel_loss = pixel_loss.with_reduction('none')

    def get_name(self):
        return f'ConfLoss({self.pixel_loss})'

    def get_conf_log(self, x):
        return x, torch.log(x)

    def compute_loss(self, gt1, gt2, pred1, pred2, trajectory_pred, **kw):
        # compute per-pixel loss
        ((l1_ref, msk1), (l2_ref, msk2), (l1_stage2, msk1), (l2_stage2, msk2), (pose_r_loss, _), (pose_t_loss, _), (pose_f_loss, _), (loss_image, _), (loss_2d, _), (loss1_coarse, msk1_coarse), (loss2_coarse, msk2_coarse), (l2_rigid, _)), details, monitoring = self.pixel_loss(gt1, gt2, pred1, pred2, trajectory_pred, **kw)

        # weight by confidence
        conf_ref = pred1['conf']
        conf_ref1 = conf_ref[:,0]
        conf_ref2 = conf_ref[:,1:]
        conf_stage2 = pred2['conf']
        conf_stage2_1 = conf_stage2[:,0]
        conf_stage2_2 = conf_stage2[:,1:]
        conf1_ref, log_conf1_ref = self.get_conf_log(conf_ref1[msk1])
        conf2_ref, log_conf2_ref = self.get_conf_log(conf_ref2[msk2.view(*conf_ref2.shape)])
        conf1_stage2, log_conf1_stage2 = self.get_conf_log(conf_stage2_1[msk1])
        conf2_stage2, log_conf2_stage2 = self.get_conf_log(conf_stage2_2[msk2.view(*conf_stage2_2.shape)])
        conf_loss1_ref = l1_ref * conf1_ref - self.alpha * log_conf1_ref
        conf_loss2_ref = l2_ref * conf2_ref - self.alpha * log_conf2_ref
        conf_loss1_stage2 = l1_stage2 * conf1_stage2 - self.alpha * log_conf1_stage2
        conf_loss2_stage2 = l2_stage2 * conf2_stage2 - self.alpha * log_conf2_stage2

        if type(l2_rigid) != int:
            conf_loss2_rigid = l2_rigid * conf2_stage2 - self.alpha * log_conf2_stage2
            conf_loss2_rigid = conf_loss2_rigid.mean() if conf_loss2_rigid.numel() > 0 else 0
        else:
            conf_loss2_rigid = 0 
        conf_loss1_ref = conf_loss1_ref.mean() if conf_loss1_ref.numel() > 0 else 0
        conf_loss2_ref = conf_loss2_ref.mean() if conf_loss2_ref.numel() > 0 else 0
        conf_loss1_stage2 = conf_loss1_stage2.mean() if conf_loss1_stage2.numel() > 0 else 0
        conf_loss2_stage2 = conf_loss2_stage2.mean() if conf_loss2_stage2.numel() > 0 else 0
        if loss_image is None:
            loss_image = 0
        pose_f_loss = 0
        return  conf_loss1_ref * 0.5 + conf_loss2_ref * 0.5 + conf_loss1_stage2 + conf_loss2_stage2 + pose_f_loss + pose_r_loss + pose_t_loss  + loss_image + loss_2d + conf_loss2_rigid * 0.1, dict(conf_loss2_rigid = float(conf_loss2_rigid), conf_loss_1=float(conf_loss1_ref), conf_loss_2=float(conf_loss2_ref), conf_loss_1_stage2=float(conf_loss1_stage2), conf_loss_2_stage2=float(conf_loss2_stage2), pose_r_loss=float(pose_r_loss), pose_t_loss=float(pose_t_loss), pose_f_loss=float(pose_f_loss), image_loss=float(loss_image), **details)

