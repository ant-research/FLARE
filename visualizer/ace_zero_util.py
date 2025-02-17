import sys
import subprocess
import logging
import numpy as np
from distutils.util import strtobool
# from ace_trainer import TrainerACE
_logger = logging.getLogger(__name__)
import argparse
from pathlib import Path
from types import SimpleNamespace
# from dataset import CamLocDataset
import torch
import random
from ace_visualizer import ACEVisualizer
# from ace_network import Regressor
from torch.utils.data import DataLoader
import os
from ace_util import load_npz_file
import time
import re
import numpy as np
# import dsacstar
from collections import namedtuple
# import dataset_io
import pickle
import glob

def _strtobool(x):
    return bool(strtobool(x))

def get_seed_id(seed_idx):
    return f"iteration0_seed{seed_idx}"

def get_render_path(out_dir):
    return out_dir / "renderings"

def get_register_opt(
    rgb_files=None,
    hypotheses=64,
    hypotheses_max_tries=1000000,
    threshold=10.0,
    inlieralpha=100.0,
    maxpixelerror=100.0,
    render_visualization=False,
    render_target_path='renderings',
    render_flipped_portrait=False,
    render_pose_conf_threshold=5000,
    render_map_depth_filter=10,
    render_camera_z_offset=4,
    base_seed=1305,
    confidence_threshold=1000.0,
    max_estimates=-1,
    render_marker_size=0.03,
    result_npz=None,
    results_folder="result_folder_old_test_raw"
):
    if rgb_files is None:
        raise ValueError("rgb_files is required")
    if result_npz is None:
        raise ValueError("result_npz is required")
    
    opt = SimpleNamespace(
        rgb_files=rgb_files,
        hypotheses=hypotheses,
        hypotheses_max_tries=hypotheses_max_tries,
        threshold=threshold,
        inlieralpha=inlieralpha,
        maxpixelerror=maxpixelerror,
        render_visualization=render_visualization,
        render_target_path=Path(render_target_path),
        render_flipped_portrait=render_flipped_portrait,
        render_pose_conf_threshold=render_pose_conf_threshold,
        render_map_depth_filter=render_map_depth_filter,
        render_camera_z_offset=render_camera_z_offset,
        base_seed=base_seed,
        confidence_threshold=confidence_threshold,
        max_estimates=max_estimates,
        render_marker_size=render_marker_size,
        result_npz=result_npz,
        results_folder=Path(results_folder)
    )
    
    return opt

def regitser_visulization(opt):
    TestEstimate = namedtuple("TestEstimate", [
        "pose_est",
        "pose_gt",
        "focal_length",
        "confidence",
        "image_file"
    ])

    #set random seeds
    torch.manual_seed(opt.base_seed)
    np.random.seed(opt.base_seed)
    random.seed(opt.base_seed)
    avg_batch_time = 0
    num_batches = 0
    all_files = glob.glob(opt.rgb_files)

    target_path = opt.render_target_path
    os.makedirs(target_path, exist_ok=True)
    ace_visualizer = ACEVisualizer(target_path,
                                    opt.render_flipped_portrait,
                                    opt.render_map_depth_filter,
                                    reloc_vis_conf_threshold=opt.render_pose_conf_threshold,
                                    confidence_threshold=opt.confidence_threshold,
                                    marker_size=opt.render_marker_size,
                                    result_npz=opt.result_npz,
                                    pan_start_angle=opt.pan_start_angle,
                                    pan_radius_scale=opt.pan_radius_scale,
                                    )
    if 'state_dict' not in vars(opt).keys():
        frame_idx = None
        ace_visualizer.setup_reloc_visualisation(
            frame_count=len(all_files),
            camera_z_offset=opt.render_camera_z_offset,
            frame_idx=frame_idx,
            only_frustum=opt.only_frustum,
        )
    else:
        frame_idx = opt.state_dict['frame_idx']
        ace_visualizer.setup_reloc_visualisation(
            frame_count=len(all_files),
            camera_z_offset=opt.render_camera_z_offset,
            frame_idx=frame_idx,
            only_frustum=opt.only_frustum,
            state_dict=opt.state_dict,
        )

    estimates_list = []

    npz_data = load_npz_file(opt.result_npz)
    pts3d_all = npz_data['pts3d']
    cam_poses = npz_data['cam_poses']
    cam_intrinsics = npz_data['intrinsic']

    with torch.no_grad():
        # for image_B1HW, _, _, _, intrinsics_B33, _, _, filenames, indices in testset_loader:
        for filenames in [all_files]:
            batch_start_time = time.time()
            for frame_path in filenames:
                img_file = frame_path
                name = img_file.split('/')[-1]
                match = re.search(r'_(\d+)\.png', name)
                if match:
                    img_idx = int(match.group(1))  
                    print(f'current image file {img_file}')
                else:
                    print("No number found")
                ours_pts3d = pts3d_all[img_idx].copy()
                ours_K = cam_intrinsics[img_idx].copy()
                
                ours_pose = cam_poses[img_idx].copy()
                focal_length = ours_K[0, 0]
                ppX = ours_K[0, 2]
                ppY = ours_K[1, 2]
                out_pose = torch.from_numpy(ours_pose.copy()).float()
                scene_coordinates_3HW = torch.from_numpy(ours_pts3d.transpose(2, 0, 1)).float()

                # Compute the pose via RANSAC.
                # inlier_count = dsacstar.forward_rgb(
                #     scene_coordinates_3HW.unsqueeze(0),
                #     out_pose,
                #     opt.hypotheses,
                #     opt.threshold,
                #     focal_length,
                #     ppX,
                #     ppY,
                #     opt.inlieralpha,
                #     opt.maxpixelerror,
                #     1, 
                #     opt.base_seed,
                #     opt.hypotheses_max_tries
                # )

                estimates_list.append(TestEstimate(
                    pose_est=ours_pose,
                    pose_gt=None,
                    focal_length=focal_length,
                    confidence=10000,
                    image_file=frame_path
                ))

            avg_batch_time += time.time() - batch_start_time
            num_batches += 1

            if 0 < opt.max_estimates <= len(estimates_list):
                _logger.info(f"Stopping at {len(estimates_list)} estimates.")
                break

    # Process estimates and write them to file.
    for estimate in estimates_list:
        pose_est = estimate.pose_est
        # _logger.info(f"Frame: {estimate.image_file}, Confidence: {estimate.confidence}")
        for _ in range(10):
            ace_visualizer.render_reloc_frame(
                query_file=estimate.image_file,
                est_pose=pose_est,
                confidence=estimate.confidence,)

        out_pose = pose_est.copy()

    if opt.only_frustum:
        ace_visualizer.trajectory_buffer.clear_frustums()
        ace_visualizer.reset_position_markers(marker_color=ace_visualizer.progress_color_map[1] * 255)
        _, vis_error, mean_value, _, _ = ace_visualizer.get_mean_repreoject_error()
        vis_error[:] = mean_value
        ace_visualizer.render_growing_map()

    # Compute average time.
    avg_time = avg_batch_time / num_batches
    _logger.info(f"Avg. processing time: {avg_time * 1000:4.1f}ms")
    state_dict = {}

    state_dict['frame_idx'] = ace_visualizer.frame_idx
    state_dict['camera_buffer'] = ace_visualizer.scene_camera.get_camera_buffer()
    state_dict['pan_cameras'] = ace_visualizer.pan_cams
    state_dict['map_xyz'] = ace_visualizer.pts3d.reshape(-1, 3)
    state_dict['map_clr'] = ((ace_visualizer.image_gt.transpose(0, 2, 3, 1).reshape(-1, 3) + 1.0) / 2.0 * 255.0).astype('float64')
    return state_dict

