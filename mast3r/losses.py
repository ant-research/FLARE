import torch
import bisect
import torch.nn as nn
import numpy as np
# from sklearn.metrics import average_precision_score
import imageio
from torch.cuda.amp import autocast
import mast3r.utils.path_to_dust3r  # noqa
from dust3r.utils.geometry import (geotrf, inv, normalize_pointcloud)
from dust3r.inference import get_pred_pts3d
from dust3r.utils.geometry import get_joint_pointcloud_depth, get_joint_pointcloud_center_scale
import trimesh
from dust3r.viz import add_scene_cam, CAM_COLORS, OPENGL, pts3d_to_trimesh, cat_meshes
import os
from pytorch3d.transforms.rotation_conversions import matrix_to_quaternion, quaternion_to_matrix
import torch.nn.functional as F
from PIL import Image


class MeshOutput():
    def __init__(self,sam=False):
        self.sam = sam
        pass
    
    def __call__(self, gt1, gt2, pred1, pred2, trajectory_pred, render_gt=None, **kw):
        view1 = gt1
        view2 = gt2
        pts2 = pred2['pts3d']
        pts3d = pts2.detach().cpu().numpy()
        conf2 = pred2['conf'].float()
        conf = conf2
        mask = conf > torch.quantile(conf2, 0.1)
        image_list = [view['img_org'] for view in gt1+gt2]
        image_list = torch.stack(image_list,1).permute(0,1, 3,4,2).detach().cpu().numpy()
        imgs = image_list / 2 + 0.5
        B, N, H, W, _ = pts3d.shape
        imgs = imgs.reshape(B, N, H, W, 3)
        scene = trimesh.Scene()
        meshes = []
        outfile = os.path.join('./data/mesh', view1[0]['instance'][0].split('/')[-1])
        outfile = outfile
        os.makedirs(outfile, exist_ok=True)
        imgs = imgs[0]
        pts3d = pts3d[0]
        mask = mask[0]
        pred_poses = []

        xy_over_z = (pts2[..., :2] / pts2[..., 2:3]).nan_to_num(posinf=0, neginf=0)  # homogeneous (x,y,1)
        from dust3r.utils.geometry import xy_grid
        pp = torch.tensor((W/2, H/2)).to(xy_over_z)
        pixels = xy_grid(W, H, device=xy_over_z.device).view(1, -1, 2) - pp.view(-1, 1, 2)  # B,HW,2
        H, W = pts3d.shape[1:3]
        with torch.no_grad():
            # direct estimation of focal
            u, v = pixels[:1].unbind(dim=-1)
            x, y, z = pts2[0][:1].reshape(-1,3).unbind(dim=-1)
            fx_votes = (u * z) / x
            fy_votes = (v * z) / y

            # assume square pixels, hence same focal for X and Y
            f_votes = torch.cat((fx_votes.view(B, -1), fy_votes.view(B, -1)), dim=-1)
            focal = torch.nanmedian(f_votes, dim=-1).values
 
        shape = torch.stack([view['true_shape'] for view in gt1+gt2], dim=1)
        colmap_mask = torch.stack([view['valid_mask'] for view in gt1+gt2], dim=1)
        focal = focal.item()
        for i in range(pts3d.shape[0]):
            shape_input_each = shape[:, i]
            mesh_grid = xy_grid(shape_input_each[0,1], shape_input_each[0,0])
            cur_inlier = conf2[0,i] > torch.quantile(conf2[0,i], 0.6)
            cur_inlier = cur_inlier.detach().cpu().numpy()
            ransac_thres = 0.5
            confidence = 0.9999
            iterationsCount = 10_000
            cur_pts3d = pts3d[i]
            import cv2
            K = np.float32([(focal, 0, W/2), (0, focal, H/2), (0, 0, 1)])
            success, r_pose, t_pose, _ = cv2.solvePnPRansac(cur_pts3d[cur_inlier].astype(np.float64), mesh_grid[cur_inlier].astype(np.float64), K, None,
                                                            flags=cv2.SOLVEPNP_SQPNP,
                                                            iterationsCount=iterationsCount,
                                                            reprojectionError=1,
                                                            confidence=confidence)
            r_pose = cv2.Rodrigues(r_pose)[0]  # world2cam == world2cam2
            RT = np.r_[np.c_[r_pose, t_pose], [(0,0,0,1)]] # world2cam2
            cam2world = np.linalg.inv(RT)
            pred_poses.append(cam2world)
        pred_poses = np.stack(pred_poses, axis=0)
        pred_poses = torch.tensor(pred_poses)

        mask = torch.quantile(conf2.flatten(2,3), 0.1, dim=-1)[0]
        mask_conf = conf2.cpu() > mask[None, :, None, None].cpu()
        mask_conf = mask_conf[0]
        from pytorch3d.ops import knn_points
        K = 10
        points = torch.tensor(pts3d.reshape(1,-1,3)).cuda()
        knn = knn_points(points, points, K=K)
        dists = knn.dists  
        mean_dists = dists.mean(dim=-1) 
        if self.sam:
            from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
            from transformers import SegformerFeatureExtractor, SegformerForSemanticSegmentation
            from sam.scripts.segformer import segformer_segmentation as segformer_func
            from sam.scripts.configs.ade20k_id2label import CONFIG as CONFIG_ADE20K_ID2LABEL
            import pycocotools.mask as maskUtils

            os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
            sam = sam_model_registry["vit_h"](checkpoint='/data0/zsz/mast3recon/checkpoints/sam_vit_h_4b8939.pth').to(pts2)
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
            for img in imgs:
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
        
        import nerfvis.scene as scene_vis
        scene_vis.set_opencv()
        mask = mean_dists < torch.quantile(mean_dists.reshape(-1), 0.95)
        mask = mask.detach().cpu().numpy()
        mask_conf = (mask_conf > 0) & mask.reshape(-1,H,W)
        mask_conf = mask_conf > 0
        filtered_points1 = pts3d[mask_conf].reshape(-1, 3)
        colors = imgs[mask_conf].reshape(-1, 3)
        scene_vis.add_points('points', filtered_points1.reshape(-1,3), vert_color=colors.reshape(-1,3), size = 1)
        scene_vis.add_images(
            f"images/i",
            imgs, # Can be a list of paths too (requires joblib for that) 
            r=pred_poses[:, :3, :3],
            t=pred_poses[:, :3, 3],
            focal_length=focal,
            z=0.1,
            with_camera_frustum=True,
        )
        scene_vis.display(port=8828)
        np.savez(outfile + '/pred.npz', pts3d=pts3d, vert_color=imgs, poses=pred_poses.detach().cpu(), intrinsic=focal, images=imgs, mask = mask_conf)
        print(f"save {outfile}")
        save_content = 'CUDA_VISIBLE_DEVICES=1 python visualizer/ace_zero_ours_2.py --result_npz {} --results_folder {}'.format(outfile + '/pred.npz', outfile)
        file_path = outfile + '/run_vis.sh'
        with open(file_path, 'a') as file:
            # 将内容写入文件
            file.write(save_content + '\n')  # 添加换行符以便于区分不同内容

        return imgs.mean(), {}
    

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


def solve_optimal_focal_shift(uv: np.ndarray, xyz: np.ndarray):
    "Solve `min |focal * xy / (z + shift) - uv|` with respect to shift and focal"
    from scipy.optimize import least_squares
    uv, xy, z = uv.reshape(-1, 2), xyz[..., :2].reshape(-1, 2), xyz[..., 2].reshape(-1)

    def fn(uv: np.ndarray, xy: np.ndarray, z: np.ndarray, shift: np.ndarray):
        xy_proj = xy / (z + shift)[: , None]
        f = (xy_proj * uv).sum() / np.square(xy_proj).sum()
        err = (f * xy_proj - uv).ravel()
        return err

    solution = least_squares(partial(fn, uv, xy, z), x0=0, ftol=1e-3, method='lm')
    optim_shift = solution['x'].squeeze().astype(np.float32)

    xy_proj = xy / (z)[: , None]
    optim_focal = (xy_proj * uv).sum() / np.square(xy_proj).sum()

    return optim_shift, optim_focal

def recover_focal_shift(points, mask, focal, downsample_size = (64, 64)):
    """
    Recover the depth map and FoV from a point map with unknown z shift and focal.

    Note that it assumes:
    - the optical center is at the center of the map
    - the map is undistorted
    - the map is isometric in the x and y directions

    ### Parameters:
    - `points: torch.Tensor` of shape (..., H, W, 3)
    - `mask: torch.Tensor` of shape (..., H, W). Optional.
    - `focal: torch.Tensor` of shape (...). Optional.
    - `downsample_size: Tuple[int, int]` in (height, width), the size of the downsampled map. Downsampling produces approximate solution and is efficient for large maps.

    ### Returns:
    - `focal`: torch.Tensor of shape (...) the estimated focal length, relative to the half diagonal of the map
    - `shift`: torch.Tensor of shape (...) Z-axis shift to translate the point map to camera space
    """
    shape = points.shape
    height, width = points.shape[-3], points.shape[-2]
    diagonal = (height ** 2 + width ** 2) ** 0.5

    points = points.reshape(-1, *shape[-3:])
    mask = None if mask is None else mask.reshape(-1, *shape[-3:-1])
    focal = focal.reshape(-1) if focal is not None else None
    uv = normalized_view_plane_uv(width, height, dtype=points.dtype, device=points.device)  # (H, W, 2)
    points_lr = F.interpolate(points.permute(0, 3, 1, 2), downsample_size, mode='nearest').permute(0, 2, 3, 1)
    uv_lr = F.interpolate(uv.unsqueeze(0).permute(0, 3, 1, 2), downsample_size, mode='nearest').squeeze(0).permute(1, 2, 0)
    mask_lr = None if mask is None else F.interpolate(mask.to(torch.float32).unsqueeze(1), downsample_size, mode='nearest').squeeze(1) > 0
    
    uv_lr_np = uv_lr.cpu().numpy()
    points_lr_np = points_lr.detach().cpu().numpy()
    focal_np = focal.cpu().numpy() if focal is not None else None
    mask_lr_np = None if mask is None else mask_lr.cpu().numpy()
    optim_shift, optim_focal = [], []
    for i in range(points.shape[0]):
        points_lr_i_np = points_lr_np[i] if mask is None else points_lr_np[i][mask_lr_np[i]]
        uv_lr_i_np = uv_lr_np if mask is None else uv_lr_np[mask_lr_np[i]]
        if focal is None:
            optim_shift_i, optim_focal_i = solve_optimal_focal_shift(uv_lr_i_np, points_lr_i_np)
            optim_focal.append(float(optim_focal_i))
        else:
            optim_shift_i = solve_optimal_shift(uv_lr_i_np, points_lr_i_np, focal_np[i])
        optim_shift.append(float(optim_shift_i))
    optim_shift = torch.tensor(optim_shift, device=points.device, dtype=points.dtype).reshape(shape[:-3])

    if focal is None:
        optim_focal = torch.tensor(optim_focal, device=points.device, dtype=points.dtype).reshape(shape[:-3])
    else:
        optim_focal = focal.reshape(shape[:-3])

    return optim_focal, optim_shift
 