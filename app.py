import spaces
import mast3r.utils.path_to_dust3r  # noqa
import dust3r.utils.path_to_croco  # noqa: F401
import mast3r.utils.path_to_dust3r  # noqa
import os
import sys
import os.path as path
import torch
import tempfile
import gradio
import shutil
import math
from mast3r.model import AsymmetricMASt3R
import matplotlib.pyplot as pl
from dust3r.utils.image import load_images
import torch.nn.functional as F
from pytorch3d.ops import knn_points
from dust3r.utils.geometry import xy_grid
import numpy as np 
import cv2
from dust3r.utils.device import to_numpy
import trimesh
from dust3r.viz import add_scene_cam, CAM_COLORS, OPENGL, pts3d_to_trimesh, cat_meshes
from scipy.spatial.transform import Rotation

pl.ion()
# for gpu >= Ampere and pytorch >= 1.12
torch.backends.cuda.matmul.allow_tf32 = True
batch_size = 1
inf = float('inf')
weights_path = "checkpoints/geometry_pose.pth"
device = 'cuda' if torch.cuda.is_available() else 'cpu'
ckpt = torch.load(weights_path, map_location=device)
model = AsymmetricMASt3R(pos_embed='RoPE100', patch_embed_cls='ManyAR_PatchEmbed', img_size=(512, 512), head_type='catmlp+dpt', output_mode='pts3d+desc24', depth_mode=('exp', -inf, inf), conf_mode=('exp', 1, inf), enc_embed_dim=1024, enc_depth=24, enc_num_heads=16, dec_embed_dim=768, dec_depth=12, dec_num_heads=12, two_confs=True, desc_conf_mode=('exp', 0, inf))
model.load_state_dict(ckpt['model'], strict=False)
model = model.to(device).eval()
model = AsymmetricMASt3R.from_pretrained(weights_path).to(device)
tmpdirname = tempfile.mkdtemp(suffix='_FLARE_gradio_demo')
image_size = 512
silent = True
gradio_delete_cache = 7200
backbone = torch.hub.load(
    "facebookresearch/dinov2", "dinov2_vitb14_reg"
    )
backbone = backbone.eval().cuda()

class FileState:
    def __init__(self, outfile_name=None):
        self.outfile_name = outfile_name

    def __del__(self):
        if self.outfile_name is not None and os.path.isfile(self.outfile_name):
            os.remove(self.outfile_name)
        self.outfile_name = None

def pad_to_square(reshaped_image):
    B, C, H, W = reshaped_image.shape
    max_dim = max(H, W)
    pad_height = max_dim - H
    pad_width = max_dim - W
    padding = (pad_width // 2, pad_width - pad_width // 2,
               pad_height // 2, pad_height - pad_height // 2)
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

def _convert_scene_output_to_glb(outfile, imgs, pts3d, mask, focals, cams2world, cam_size=0.05,
                                 cam_color=None, as_pointcloud=False,
                                 transparent_cams=False, silent=False):
    assert len(pts3d) == len(mask) <= len(imgs) <= len(cams2world) == len(focals)
    pts3d = to_numpy(pts3d)
    imgs = to_numpy(imgs)
    focals = to_numpy(focals)
    mask = to_numpy(mask)
    cams2world = to_numpy(cams2world)

    scene = trimesh.Scene()
    # full pointcloud
    if as_pointcloud:
        pts = np.concatenate([p[m] for p, m in zip(pts3d, mask)]).reshape(-1, 3)
        col = np.concatenate([p[m] for p, m in zip(imgs, mask)]).reshape(-1, 3)
        valid_msk = np.isfinite(pts.sum(axis=1))
        pct = trimesh.PointCloud(pts[valid_msk], colors=col[valid_msk])
        scene.add_geometry(pct)
    else:
        meshes = []
        for i in range(len(imgs)):
            pts3d_i = pts3d[i].reshape(imgs[i].shape)
            msk_i = mask[i] & np.isfinite(pts3d_i.sum(axis=-1))
            meshes.append(pts3d_to_trimesh(imgs[i], pts3d_i, msk_i))
        mesh = trimesh.Trimesh(**cat_meshes(meshes))
        scene.add_geometry(mesh)

    # add each camera
    for i, pose_c2w in enumerate(cams2world):
        if isinstance(cam_color, list):
            camera_edge_color = cam_color[i]
        else:
            camera_edge_color = cam_color or CAM_COLORS[i % len(CAM_COLORS)]
        add_scene_cam(scene, pose_c2w, camera_edge_color,
                      None if transparent_cams else imgs[i], focals[i],
                      imsize=imgs[i].shape[1::-1], screen_width=cam_size)

    rot = np.eye(4)
    rot[:3, :3] = Rotation.from_euler('y', np.deg2rad(180)).as_matrix()
    scene.apply_transform(np.linalg.inv(cams2world[0] @ OPENGL @ rot))
    if not silent:
        print('(exporting 3D scene to', outfile, ')')

    scene.export(file_obj=outfile)
    return outfile

@spaces.GPU(duration=180)
def local_get_reconstructed_scene(inputfiles, min_conf_thr, cam_size):

    batch = load_images(inputfiles, size=image_size, verbose=not silent)
    images = [gt['img'] for gt in batch]
    images = torch.cat(images, dim=0)
    images = images / 2 + 0.5
    index = generate_rank_by_dino(images, backbone, query_frame_num=1)
    sorted_order = calculate_index_mappings(index, len(images), device=device)
    sorted_batch = []
    for i in range(len(batch)):
        sorted_batch.append(batch[sorted_order[i]])
    batch = sorted_batch
    ignore_keys = set(['depthmap', 'dataset', 'label', 'instance', 'idx', 'rng', 'vid'])
    ignore_dtype_keys = set(['true_shape', 'camera_pose', 'pts3d', 'fxfycxcy', 'img_org', 'camera_intrinsics', 'depthmap', 'depth_anything', 'fxfycxcy_unorm'])
    dtype = torch.bfloat16
    for view in batch:
        for name in view.keys():  # pseudo_focal
            if name in ignore_keys:
                continue
            if isinstance(view[name], torch.Tensor):
                view[name] = view[name].to(device, non_blocking=True)
            else:
                view[name] = torch.tensor(view[name]).to(device, non_blocking=True)
            if view[name].dtype == torch.float32 and name not in ignore_dtype_keys:
                view[name] = view[name].to(dtype)
    view1 = batch[:1]
    view2 = batch[1:]
    with torch.cuda.amp.autocast(enabled=True, dtype=dtype):
        pred1, pred2, pred_cameras = model(view1, view2, True, dtype)
    pts3d = pred2['pts3d']
    conf = pred2['conf']
    pts3d = pts3d.detach().cpu()
    B, N, H, W, _ = pts3d.shape
    thres = torch.quantile(conf.flatten(2,3), min_conf_thr, dim=-1)[0]
    masks_conf = conf > thres[None, :, None, None]
    masks_conf = masks_conf.cpu()
    
    images = [view['img'] for view in view1+view2]
    shape = torch.stack([view['true_shape'] for view in view1+view2], dim=1).detach().cpu().numpy()
    images = torch.stack(images,1).float().permute(0,1,3,4,2).detach().cpu().numpy()
    images = images / 2 + 0.5
    images = images.reshape(B, N, H, W, 3)
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
    # use knn to clean the point cloud
    K = 10
    points = torch.tensor(pts3d.reshape(1,-1,3)).cuda()
    knn = knn_points(points, points, K=K)
    dists = knn.dists  
    mean_dists = dists.mean(dim=-1)
    masks_dist = mean_dists < torch.quantile(mean_dists.reshape(-1), 0.95)
    masks_dist = masks_dist.detach().cpu().numpy()
    masks_conf = (masks_conf > 0) & masks_dist.reshape(-1,H,W)
    masks_conf = masks_conf > 0
    outdir = tempfile.mkdtemp(suffix='_FLARE_gradio_demo')
    os.makedirs(outdir, exist_ok=True)
    focals = [focal] * len(images)
    outfile_name = tempfile.mktemp(suffix='_scene.glb', dir=outdir)

    _convert_scene_output_to_glb(outfile_name, images, pts3d, masks_conf, focals, pred_poses, as_pointcloud=True,
                                        transparent_cams=False, cam_size=cam_size, silent=silent)
    return filestate, outfile_name

css = """.gradio-container {margin: 0 !important; min-width: 100%};"""
title = "FLARE Demo"
with gradio.Blocks(css=css, title=title, delete_cache=(gradio_delete_cache, gradio_delete_cache)) as demo:
    filestate = gradio.State(None)
    gradio.HTML('<h2 style="text-align: center;">3D Reconstruction with FLARE</h2>')
    with gradio.Column():
        inputfiles = gradio.File(file_count="multiple")
        snapshot = gradio.Image(None, visible=False)
        with gradio.Row():
            # adjust the confidence threshold
            min_conf_thr = gradio.Slider(label="min_conf_thr", value=0.1, minimum=0.0, maximum=1, step=0.05)
            # adjust the camera size in the output pointcloud
            cam_size = gradio.Slider(label="cam_size", value=0.2, minimum=0.001, maximum=1.0, step=0.001)
        run_btn = gradio.Button("Run")
        outmodel = gradio.Model3D()

        # events
        run_btn.click(fn=local_get_reconstructed_scene,
                      inputs=[inputfiles, min_conf_thr, cam_size],
                      outputs=[filestate, outmodel])

demo.launch(show_error=True, share=None, server_name=None, server_port=None)
shutil.rmtree(tmpdirname)