from functools import cache
import torch
from einops import reduce
from lpips import LPIPS
from torch import Tensor
from skimage.metrics import structural_similarity
from jaxtyping import Float, UInt8


@torch.no_grad()
def compute_psnr(
    ground_truth: Float[Tensor, "batch channel height width"],
    predicted: Float[Tensor, "batch channel height width"],
) -> Float[Tensor, " batch"]:
    ground_truth = ground_truth.clip(min=0, max=1)
    predicted = predicted.clip(min=0, max=1)
    mse = reduce((ground_truth - predicted) ** 2, "b c h w -> b", "mean")
    return -10 * mse.log10()


@cache
def get_lpips(device: torch.device) -> LPIPS:
    return LPIPS(net="vgg").to(device)


@torch.no_grad()
def compute_lpips(
    ground_truth: Float[Tensor, "batch channel height width"],
    predicted: Float[Tensor, "batch channel height width"],
) -> Float[Tensor, " batch"]:
    value = get_lpips(predicted.device).forward(ground_truth, predicted, normalize=True)
    return value[:, 0, 0, 0]


@torch.no_grad()
def compute_ssim(
    ground_truth: Float[Tensor, "batch channel height width"],
    predicted: Float[Tensor, "batch channel height width"],
) -> Float[Tensor, " batch"]:
    ssim = [
        structural_similarity(
            gt.detach().cpu().numpy(),
            hat.detach().cpu().numpy(),
            win_size=11,
            gaussian_weights=True,
            channel_axis=0,
            data_range=1.0,
        )
        for gt, hat in zip(ground_truth, predicted)
    ]
    return torch.tensor(ssim, dtype=predicted.dtype, device=predicted.device)


def compute_geodesic_distance_from_two_matrices(m1, m2):
    batch = m1.shape[0]
    m = torch.bmm(m1, m2.transpose(1, 2))  # batch*3*3

    cos = (m[:, 0, 0] + m[:, 1, 1] + m[:, 2, 2] - 1) / 2
    cos = torch.min(cos, torch.autograd.Variable(torch.ones(batch).to(m1.device)))
    cos = torch.max(cos, torch.autograd.Variable(torch.ones(batch).to(m1.device)) * -1)

    theta = torch.acos(cos)

    # theta = torch.min(theta, 2*np.pi - theta)

    return theta


def angle_error_mat(R1, R2):
    cos = (torch.trace(torch.mm(R1.T, R2)) - 1) / 2
    cos = torch.clamp(cos, -1.0, 1.0)  # numerical errors can make it out of bounds
    return torch.rad2deg(torch.abs(torch.acos(cos)))


def angle_error_vec(v1, v2):
    n = torch.norm(v1) * torch.norm(v2)
    cos_theta = torch.dot(v1, v2) / n
    cos_theta = torch.clamp(cos_theta, -1.0, 1.0)  # numerical errors can make it out of bounds
    return torch.rad2deg(torch.acos(cos_theta))


def compute_translation_error(t1, t2):
    return torch.norm(t1 - t2)


@torch.no_grad()
def compute_pose_error(pose_gt, pose_pred):
    R_gt = pose_gt[:3, :3]
    t_gt = pose_gt[:3, 3]

    R = pose_pred[:3, :3]
    t = pose_pred[:3, 3]

    error_t = angle_error_vec(t, t_gt)
    error_t = torch.minimum(error_t, 180 - error_t)  # ambiguity of E estimation
    error_t_scale = compute_translation_error(t, t_gt)
    error_R = angle_error_mat(R, R_gt)
    return error_t, error_t_scale, error_R