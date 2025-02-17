#!/usr/bin/env python3
# Copyright © Niantic, Inc. 2023.

import logging
import shutil
from pathlib import Path
import os
import numpy as np
import argparse
from distutils.util import strtobool
import time
import ace_zero_util as zutil
from ace_util import load_npz_file, compute_knn_mask
from joblib import Parallel, delayed
# from ace_trainer import TrainerACE
# import dataset_io
from PIL import Image
from ace_visualizer import ACEVisualizer
import subprocess
import matplotlib
_logger = logging.getLogger(__name__)
import imageio



def colorize(depth: np.ndarray, mask: np.ndarray = None, normalize: bool = True, cmap: str = 'Spectral') -> np.ndarray:
    if mask is None:
        depth = np.where(depth > 0, depth, np.nan)
    else:
        depth = np.where((depth > 0) & mask, depth, np.nan)
    disp = 1 / depth
    if normalize:
        min_disp, max_disp = np.nanquantile(disp, 0.001), np.nanquantile(disp, 0.999)
        disp = (disp - min_disp) / (max_disp - min_disp)
    colored = np.nan_to_num(matplotlib.colormaps[cmap](1.0 - disp), 0)
    colored = (colored.clip(0, 1) * 255).astype(np.uint8)[:, :, :3]
    return colored

def _strtobool(x):
    return bool(strtobool(x))

def npz2image(npz_file, save_path):
    data = np.load(npz_file)
    os.makedirs(save_path, exist_ok=True)
    image_data = data['images_gt'] if 'images_gt' in data else data['images']
    if image_data.shape[1] == 3:
        image_data = np.transpose(image_data, (0, 2, 3, 1))
    for idx in range(image_data.shape[0]):
        img = image_data[idx, :, :]
        if img.max() <= 1.0 and img.min() >= 0.0:
            img = (img * 255).astype(np.uint8)
        elif img.max() <= 1.0 and img.min() < 0.0:
            img = (img + 1.0) * 255 / 2.0
            img = img.astype(np.uint8)
        else:
            img = img.astype(np.uint8)
        # import pdb; pdb.set_trace()
        image = Image.fromarray(img)
        image.save(f"{save_path}/gt_image_{idx}.png")
        # image.save(os.path.join(save_path, str(idx) + '.png'))


class Options:
    def __init__(self, ** kwargs):
        self.__dict__.update(kwargs)


if __name__ == '__main__':

    # Setup logging levels.
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(
        description='Run ACE0 for a dataset or a scene.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--export_point_cloud', type=_strtobool, default=True,
                        help="Export the ACE0 point cloud after reconstruction, "
                             "for visualisation or to initialise splats")

    parser.add_argument('--dense_point_cloud', type=_strtobool, default=True,
                        help='when exporting a point cloud, do not filter points based on reprojection error, '
                             'bad for visualisation but good to initialise splats')

    parser.add_argument('--num_data_workers', type=int, default=12,
                        help='number of data loading workers, set according to the number of available CPU cores')

    #Registration parameters 
    parser.add_argument('--ransac_iterations', type=int, default=32,
                        help="Number of RANSAC hypothesis when registering mapping frames.")
    parser.add_argument('--ransac_threshold', type=float, default=10,
                        help='RANSAC inlier threshold in pixels')

    # Visualisation parameters
    parser.add_argument('--render_visualization', type=_strtobool, default=True,
                        help="Render visualisation frames of the whole reconstruction process.")
    parser.add_argument('--render_flipped_portrait', type=_strtobool, default=False,
                        help="Dataset images are 90deg flipped (like Wayspots).")
    parser.add_argument('--render_marker_size', type=float, default=0.03,
                        help="Size of the camera marker when rendering scenes.")
    # parser.add_argument('--iterations_output', type=int, default=500,
    #                     help='how often to print the loss and render a frame')

    parser.add_argument('--result_npz', type=str, help='Path to the .npz file containing the raw data (e.g., intrinsic matrices, 3D points, etc.) for visualization')
    parser.add_argument('--results_folder', type=Path, default="1217_new", help='path to output folder for result files')
    parser.add_argument('--knn_percent', type=float, default=98.0, help='percentage of points to keep in the knn mask')
    parser.add_argument('--K_neighbors', type=int, default=10, help='number of neighbors to use for knn mask')
    parser.add_argument('--pan_radius_scale', type=float, default=-1, help='factor to control the size of the pan camera')
    parser.add_argument('--pan_start_angle', type=int, default=-90, help='start angle of the pan camera')
    opt = parser.parse_args()
    opt.results_folder = '/'.join(opt.result_npz.split('/')[:-1])
    os.makedirs(opt.results_folder, exist_ok=True)
    opt.results_folder = Path(opt.results_folder)
    image_folder = f'{opt.results_folder}/gt_images'
    npz2image(opt.result_npz, image_folder)
    opt.rgb_files = f'{image_folder}/*.png'

    input_data = load_npz_file(opt.result_npz)
    
    if 'pts_mask' not in input_data.keys():
        start_time = time.time()
        input_data['pts_mask'] = compute_knn_mask(input_data['pts3d'], opt.K_neighbors, opt.knn_percent)
        end_time = time.time()
        _logger.info(f"Computing knn mask took {end_time - start_time:.2f} seconds.")
        npz_with_mask = f"{opt.results_folder}/points_with_mask.npz"
        np.savez_compressed(npz_with_mask, **input_data)
        opt.result_npz = npz_with_mask
    best_seed = 4
    iteration_id = zutil.get_seed_id(best_seed)
    # depth = input_data['pts3d'][0,...,-1]
    poses = input_data['cam_poses']
    H, W = input_data['pts3d'].shape[1:3]
    for idx, (pts3d, pose) in enumerate(zip(input_data['pts3d'], poses)):
        w2c = np.linalg.inv(pose)
        R = w2c[:3,:3]
        T = w2c[:3,-1] 
        pts3d_c2w = np.einsum('kl, Nl -> Nk', R, pts3d.reshape(-1,3)) + T[None]
        disp = pts3d_c2w[...,-1]
        disp = disp.reshape(H, W)
        disp_vis = colorize(disp)
        os.makedirs(f"{opt.results_folder}/depth", exist_ok=True)
        imageio.imwrite(f"{opt.results_folder}/depth/depth_vis_{idx}.png", disp_vis)
        print(f"Visualized depth map is saved to {opt.results_folder}/depth/depth_vis_{idx}.png.")
    register_opt = zutil.get_register_opt(rgb_files=opt.rgb_files, result_npz=opt.result_npz)
    modify_options = {
        'rgb_files': opt.rgb_files,
        'render_visualization': opt.render_visualization,
        'render_target_path': zutil.get_render_path(opt.results_folder),
        'render_marker_size': opt.render_marker_size,
        'render_flipped_portrait': opt.render_flipped_portrait,
        'session': f"{iteration_id}",
        'hypotheses': opt.ransac_iterations,
        'threshold': opt.ransac_threshold,
        'hypotheses_max_tries': 16,
        'result_npz': opt.result_npz,
        'only_frustum': True,
        'pan_radius_scale': opt.pan_radius_scale,
        'pan_start_angle': opt.pan_start_angle
    }
    for k, v in modify_options.items():
        setattr(register_opt, k, v)
    
    reg_state_dict_1 = zutil.regitser_visulization(register_opt)

    scheduled_to_stop_early = False
    prev_iteration_id = iteration_id

    register_opt_iter = zutil.get_register_opt(rgb_files=opt.rgb_files, result_npz=opt.result_npz)
    modify_options = {
        'rgb_files': opt.rgb_files,
        'render_visualization': opt.render_visualization,
        'render_target_path': zutil.get_render_path(opt.results_folder),
        'render_marker_size': opt.render_marker_size,
        'render_flipped_portrait': opt.render_flipped_portrait,
        'hypotheses': opt.ransac_iterations,
        'threshold': opt.ransac_threshold,
        'hypotheses_max_tries': 16,
        'result_npz': opt.result_npz,
        'state_dict': reg_state_dict_1,
        'only_frustum': False,
        'pan_radius_scale': opt.pan_radius_scale,
        'pan_start_angle': opt.pan_start_angle
    }
    for k, v in modify_options.items():
        setattr(register_opt_iter, k, v)

    reg_state_dict_2 = zutil.regitser_visulization(register_opt_iter)

    _logger.info("Rendering final sweep.")

    final_sweep_visualizer = ACEVisualizer(
        zutil.get_render_path(opt.results_folder), 
        flipped_portait=False, 
        map_depth_filter=100,
        marker_size=opt.render_marker_size,
        result_npz=opt.result_npz, 
        pan_start_angle=opt.pan_start_angle, 
        pan_radius_scale=opt.pan_radius_scale
    )
    poses = [final_sweep_visualizer.pose_align(pose) for pose in final_sweep_visualizer.cam_pose]
    rgb_path = f'{opt.results_folder}/gt_images'
    rgb_files = os.listdir(rgb_path)
    pose_dict = {str(rgb_file): 0 for rgb_file in rgb_files}
    pose_iterations = [0 for _ in range(len(rgb_files))]
    # import pdb; pdb.set_trace()
    final_sweep_visualizer.render_final_sweep(
        frame_count=150,
        camera_z_offset=4,
        poses=poses,
        pose_iterations=pose_iterations,
        total_poses=len(pose_dict),
        state_dict=reg_state_dict_2,)

    _logger.info("Converting to video.")

    # get ffmpeg path
    ffmpeg_path = shutil.which("ffmpeg")

    # run ffmpeg to convert the rendered images to a video
    ffmpeg_save_cmd = [ffmpeg_path,
                    "-y",
                    "-framerate", "30",
                    "-pattern_type", "glob",
                    "-i", f"{zutil.get_render_path(opt.results_folder)}/*.png",
                    "-c:v", "libx264",
                    "-pix_fmt", "yuv420p",
                    str(opt.results_folder / "reconstruction.mp4")
                    ]
    
    subprocess.run(ffmpeg_save_cmd, check=True)
    
    print(f'The render result video is saved to {opt.results_folder}/reconstruction.mp4.')
    if opt.export_point_cloud:
        # import pdb; pdb.set_trace()
        import trimesh
        pts = final_sweep_visualizer.pts3d.reshape(-1, 3) 
        image = final_sweep_visualizer.image_gt.transpose(0, 2, 3, 1).reshape(-1, 3)
        clr = (( image + 1.0 ) / 2.0 * 255.0).astype('float64')
        cloud = trimesh.PointCloud(pts, colors=clr)
        cloud.export(f"{opt.results_folder}/point_cloud.ply")
        print(f'The point cloud is saved to {opt.results_folder}/point_cloud.ply.')
