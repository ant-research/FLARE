"""
Depth Loss
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

#MiDaS Loss borror from https://gist.github.com/dvdhfnr/732c26b61a0e63a0abc8a5d769dbebd0


def compute_scale_and_shift(prediction, target, mask):
    # system matrix: A = [[a_00, a_01], [a_10, a_11]]
    a_00 = torch.sum(mask * prediction * prediction, (1, 2))
    a_01 = torch.sum(mask * prediction, (1, 2))
    a_11 = torch.sum(mask, (1, 2))

    # right hand side: b = [b_0, b_1]
    b_0 = torch.sum(mask * prediction * target, (1, 2))
    b_1 = torch.sum(mask * target, (1, 2))

    # solution: x = A^-1 . b = [[a_11, -a_01], [-a_10, a_00]] / (a_00 * a_11 - a_01 * a_10) . b
    x_0 = torch.zeros_like(b_0)
    x_1 = torch.zeros_like(b_1)

    det = a_00 * a_11 - a_01 * a_01
    valid = det.nonzero()

    x_0[valid] = (a_11[valid] * b_0[valid] - a_01[valid] * b_1[valid]) / det[valid]
    x_1[valid] = (-a_01[valid] * b_0[valid] + a_00[valid] * b_1[valid]) / det[valid]

    return x_0, x_1


def reduction_batch_based(image_loss, M):
    # average of all valid pixels of the batch

    # avoid division by 0 (if sum(M) = sum(sum(mask)) = 0: sum(image_loss) = 0)
    divisor = torch.sum(M)

    if divisor == 0:
        return 0
    else:
        return torch.sum(image_loss) / divisor


def reduction_image_based(image_loss, M):
    # mean of average of valid pixels of an image

    # avoid division by 0 (if M = sum(mask) = 0: image_loss = 0)
    valid = M.nonzero()

    image_loss[valid] = image_loss[valid] / M[valid]

    return torch.mean(image_loss)


def mse_loss(prediction, target, mask, reduction=reduction_batch_based):

    M = torch.sum(mask, (1, 2))
    res = prediction - target
    image_loss = torch.sum(mask * res * res, (1, 2))

    return reduction(image_loss, 2 * M)


def gradient_loss(prediction, target, mask, reduction=reduction_batch_based):

    M = torch.sum(mask, (1, 2))

    diff = prediction - target
    diff = torch.mul(mask, diff)

    grad_x = torch.abs(diff[:, :, 1:] - diff[:, :, :-1])
    mask_x = torch.mul(mask[:, :, 1:], mask[:, :, :-1])
    grad_x = torch.mul(mask_x, grad_x)

    grad_y = torch.abs(diff[:, 1:, :] - diff[:, :-1, :])
    mask_y = torch.mul(mask[:, 1:, :], mask[:, :-1, :])
    grad_y = torch.mul(mask_y, grad_y)

    image_loss = torch.sum(grad_x, (1, 2)) + torch.sum(grad_y, (1, 2))

    return reduction(image_loss, M)


class MSELoss(nn.Module):
    def __init__(self, reduction='batch-based'):
        super().__init__()

        if reduction == 'batch-based':
            self.__reduction = reduction_batch_based
        else:
            self.__reduction = reduction_image_based

    def forward(self, prediction, target, mask):
        return mse_loss(prediction, target, mask, reduction=self.__reduction)


class GradientLoss(nn.Module):
    def __init__(self, scales=4, reduction='batch-based'):
        super().__init__()

        if reduction == 'batch-based':
            self.__reduction = reduction_batch_based
        else:
            self.__reduction = reduction_image_based

        self.__scales = scales

    def forward(self, prediction, target, mask):
        total = 0

        for scale in range(self.__scales):
            step = pow(2, scale)

            total += gradient_loss(prediction[:, ::step, ::step], target[:, ::step, ::step],
                                   mask[:, ::step, ::step], reduction=self.__reduction)

        return total


class ScaleAndShiftInvariantLoss(nn.Module):
    def __init__(self, alpha=0.25, scales=4, reduction='batch-based'):
        super().__init__()

        self.__data_loss = MSELoss(reduction=reduction)
        self.__regularization_loss = GradientLoss(scales=scales, reduction=reduction)
        self.__alpha = alpha

        self.__prediction_ssi = None

    def forward(self, prediction, target, mask):

        scale, shift = compute_scale_and_shift(prediction, target, mask)
        self.__prediction_ssi = scale.view(-1, 1, 1) * prediction + shift.view(-1, 1, 1)
        total = self.__data_loss(self.__prediction_ssi, target, mask)
        if self.__alpha > 0:
            total += self.__alpha * self.__regularization_loss(self.__prediction_ssi, target, mask)

        return total

    def __get_prediction_ssi(self):
        return self.__prediction_ssi

    prediction_ssi = property(__get_prediction_ssi)


#-------------------YYB's Baby Depth loss---------------------------
def depth_loss(gt, pred, mask = None, depth_normalize_type = '01'):
    """XX"""
    assert len(gt.shape) == 3
    assert pred.shape == gt.shape
    B, _, _ = gt.shape
    if not mask:
        mask = torch.ones_like(gt, dtype=torch.bool)
    
    if depth_normalize_type == 'normal':
            gt_depth = (gt_depth - gt_depth.mean([0,1,2,3], keepdim = True)) / gt_depth.std([0,1,2,3], keepdim = True)
            gan_depth = (gan_depth - gan_depth.mean([0,1,2,3], keepdim = True)) /gan_depth.std([0,1,2,3], keepdim = True)
    elif depth_normalize_type == '01':
            gt_min, gt_max = pred.reshape(B, -1).min(), pred.reshape(B, -1).max()
            pred_min, pred_max = pred.reshape(B, -1).min(), pred.reshape(B, -1).max()
            gt_depth = (gt - gt_min) / (gt_max - gt_min + 1e-7)
            pred  = (pred - pred_min) / (pred_max - pred_min + 1e-7)
    else:
        raise ValueError("Unsupport depth_normalize_type")

    depth_loss = F.mse_loss(gt[mask],pred[mask], reduction='mean')
    return depth_loss
        
# @torch.amp.autocast("cuda", enabled=False)
def disp_to_depth(disp, normalize = False, return_mask = False, quantile = [0.02, 0.98]):
    """Convert function between depth and dispatiry"""
    B, C, H, W = disp.shape
    # set zero as mask
    mask = torch.ones_like(disp)
    min_ = torch.quantile(disp[disp != 0].to(torch.float), quantile[0])
    max_ = torch.quantile(disp[disp != 0].to(torch.float), quantile[1])
    mask[disp < min_]
    depth = 1 / disp.clip(min_, max_)
    # normalize
    dmin, dmax =  depth.min(), depth.max()
    if normalize:
        depth = (depth - dmin) / (dmax - dmin)
        depth = depth.clip(0,1)
    if return_mask:
        depth = torch.cat((depth, mask), dim = 1)
    return depth

def depth_to_disp(depth, normalize = False, return_mask = False, z_min = 0.):
    B, H, W = depth.shape
    # set zero as mask
    mask = torch.ones_like(depth)
    mask[depth <= z_min] = 0
    depth = 1 / depth.clip(z_min, depth.max())
    # normalize
    dmin, dmax =  depth.min(), depth.max()
    if normalize:
        depth = (depth - dmin) / (dmax - dmin)
        depth = depth.clip(0,1)
    if return_mask:
        return depth, mask