# Copyright © Niantic, Inc. 2022.

import numpy as np
import torch
# import pytorch3d
# from pytorch3d.ops import knn_points


def get_pixel_grid(subsampling_factor):
    """
    Generate target pixel positions according to a subsampling factor, assuming prediction at center pixel.
    """
    pix_range = torch.arange(np.ceil(5000 / subsampling_factor), dtype=torch.float32)
    yy, xx = torch.meshgrid(pix_range, pix_range, indexing='ij')
    return subsampling_factor * (torch.stack([xx, yy]) + 0.5)


def to_homogeneous(input_tensor, dim=1):
    """
    Converts tensor to homogeneous coordinates by adding ones to the specified dimension
    """
    ones = torch.ones_like(input_tensor.select(dim, 0).unsqueeze(dim))
    output = torch.cat([input_tensor, ones], dim=dim)
    return output

def load_npz_file(npz_file):
    npz_data = np.load(npz_file)
    input_data = {}
    input_data['pts3d'] = npz_data['pts3d'].copy()
    # import pdb; pdb.set_trace()
    input_data['cam_poses'] = npz_data['poses'].copy() if 'poses' in npz_data.keys() else npz_data['cam_poses'].copy()
   
    if np.ndim(npz_data['intrinsic']) == 0:
        focal_lenth = npz_data['intrinsic']
        h, w = 384, 512
        input_data['intrinsic'] = np.float32([(focal_lenth, 0, w/2), (0, focal_lenth, h/2), (0, 0, 1)])
        ba = input_data['pts3d'].shape[0]
        input_data['intrinsic'] = np.tile(input_data['intrinsic'], (ba, 1, 1))
    else:
        input_data['intrinsic'] = npz_data['intrinsic'].copy()
    input_data['images_gt'] = npz_data['images_gt'].copy() if 'images_gt' in npz_data.keys() else npz_data['images'].copy()
    input_data['images_gt'] = ((input_data['images_gt'] - input_data['images_gt'].min()) / (input_data['images_gt'].max() - input_data['images_gt'].min()) * 2) - 1
    if input_data['images_gt'].shape[-1] == 3:
        input_data['images_gt'] = input_data['images_gt'].transpose(0, 3, 1, 2)
    
    if 'mask' in npz_data.keys():
        input_data['pts_mask'] = npz_data['mask'].copy()
    # import pdb; pdb.set_trace()
    return input_data

def compute_knn_mask(points, k=10, percentile=98, device='cuda'):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    points_tensor = torch.tensor(points, dtype=torch.float32, device=device)
    batch_size, h, c, _ = points_tensor.shape
    points_flat = points_tensor.view(batch_size * h * c, 3)
    # import pdb; pdb.set_trace()
    distances, indices, knn = knn_points(points_flat.unsqueeze(0), points_flat.unsqueeze(0), K=k)
    avg_distances = distances.mean(dim=-1).view(batch_size, h, c)
    threshold = torch.quantile(avg_distances, percentile / 100.0, dim=1, keepdim=True)
    mask = avg_distances <= threshold
    return mask.cpu().numpy()