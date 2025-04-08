import torch
import torch.nn as nn
import numpy as np
import mast3r.utils.path_to_dust3r  # noqa
from dust3r.utils.geometry import normalize_pointcloud, xy_grid, inv, matrix_to_quaternion
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