# Copyright © Niantic, Inc. 2022.

import os

os.environ['PYOPENGL_PLATFORM'] = 'egl'

import logging
import math
import numpy as np
import pyrender
from skimage.transform import rotate
from skimage import io, draw
import matplotlib.pyplot as plt
import pickle
import h5py
import os
import numpy as np
import ace_vis_util as vutil
import torch
from scipy.spatial.transform import Rotation
from ace_util import get_pixel_grid, to_homogeneous, load_npz_file
from skimage.transform import resize
_logger = logging.getLogger(__name__)


class ACEVisualizer:
    """
    Creates a representation of a scene (camera poses, scene point cloud) and write video frames.

    Supports mapping phase, relocalisation phase and final sweep. Inbetween, there are smooth transitions.
    For the mapping phase, the visualiser shows the mapping camera trajectory and the training process of the scene
    point cloud. For the relocalisation phase, the visualiser show relicalized frames progressively.
    For the final sweep, the visualiser shows the final camera trajectory and the scene point cloud.

    The visualiser has an internal state of three buffers that comprise all aspects of the next frame to be rendered.
    A call to _render_frame_from_buffers() will generate an image from these buffers.
    The three buffers are:

        self.scene_camera: To get the current rendering view.
        self.trajectory_buffer: Mesh geometry that represents camera frustums and trajectories.
        self.point_cloud_buffer: Point cloud geometry that represents the scene model.

    The class contains many helper classes to manipulate and fill these buffers. There are also function to manipulate
    the rendered frame before storing it, e.g. to add error histograms.

    The main interface for the mapping stage:
    1) setup_mapping_visualisation: Called once in the beginning, resets buffers, creates mapping camera trajectory
    2) render_mapping_frame: Called during learning iterations, shows currents snapshot of the scene point cloud
    3) finalize_mapping: Renders several frames that show the fully trained point cloud, stores buffers on disk
        so that the relocalisation script can resume smoothly

    The main interface for the relocalisation stage:
    1) setup_reloc_visualisation: Called once in the beginning, loads buffers of the mapping stage from disk
    2) render_reloc_frame: Called for each query image during relocalisation

    The main interface for the final sweep:
    1) render_final_sweep: Called once in the end, shows the final camera trajectory and the scene point cloud
    """

    def __init__(self,
                 target_path,
                 flipped_portait,
                 map_depth_filter,
                 pan_radius_scale,
                 pan_start_angle,
                 mapping_vis_error_threshold=10,
                 reloc_vis_conf_threshold=5000,
                 sweep_vis_iterations_threshold=10,
                 confidence_threshold=1000,
                #  mapping_state_file_name='mapping_state.pkl',
                 marker_size=0.03,
                 result_npz=None,
                 ):
        """
        Constructor. Sets standard values for visualisation parameters.

        @param target_path: where rendered frames will be stored
        @param flipped_portait: whether dataset images are 90 degree rotated portrait images
        @param map_depth_filter: filters scene point cloud by distance to the camera, removing far away points (meters)
        @param mapping_vis_error_threshold: threshold when mapping the reprojection error to a color map (pixels)
        @param reloc_vis_conf_threshold: threshold when mapping the pose confidence to a color map
        @param sweep_vis_iterations_threshold: threshold when mapping the registration iteration to a color map
        @param confidence_threshold: threshold for regarding a pose as successfully registered
        @param mapping_state_file_name: file name for storing and reading the visualiser state
        @param marker_size: size of the camera frustum markers
        """
        self.target_path = target_path
        # buffer file for smooth rendering across training and test script calls
        # self.state_file = os.path.join(self.target_path, mapping_state_file_name)

        # flip rendering by 90deg if dataset is stored as flipped portrait images (e.g. Wayspots)
        self.flipped_portrait = flipped_portait
        
        self.map_depth_filter = map_depth_filter

        # main visualisation parameters
        # self.render_width  = 1920 # output image resolution
        # self.render_height = 1080 # output image resolution
        self.render_width = 1280  # output image resolution
        self.render_height = 720  # output image resolution
        self.point_size = 2.0
        

        if self.flipped_portrait:
            # for flipped portrait datasets, we render sideways and rotate the final image back
            self.render_width, self.render_height = self.render_height, self.render_width

        reference_height = min(self.render_height, self.render_width)
        self.err_hist_bins = 40
        self.err_hist_x = int(0.05 * reference_height)
        self.err_hist_y = int(1.35 * reference_height)
        self.err_hist_h = int(0.4 * reference_height)
        self.err_hist_w_reloc = int(0.6 * reference_height)
        self.err_hist_w_mapping = int(0.2 * reference_height)

        # mapping vis parameters
        self.framecount_transition = 20  # frame count for growing the fully trained map at the end of mapping
        self.mapping_done_idx = -1  # frame idx when mapping was finalized, -1 when not finalized yet
        self.pan_angle_coverage = 60  # degrees, opening angle of the camera pan
           # scale factor for the camera frustum objects
        self.mapping_frustum_skip = 0.5  # place mapping frustum every X meters

        # threshold on reprojection error in px (for color mapping)
        self.mapping_vis_error_threshold = mapping_vis_error_threshold
        # dark magenta to bright cyan color map for reprojection error
        self.mapping_color_map = vutil.get_retro_colors()
        self.mapping_iteration = 0

        # color map for camera position change during refinement
        self.pose_color_map = plt.cm.get_cmap("plasma")(np.linspace(0, 1, 256))[:, :3]
        self.progress_color_map = plt.cm.get_cmap("cool")(np.linspace(0, 1, 256))[:, :3]
        # reloc vis parameters
        # scale factor for the camera frustum objects
        
        # threshold on pose confidence (for color mapping)
        self.reloc_vis_conf_threshold = reloc_vis_conf_threshold
        self.confidence_threshold = confidence_threshold

        conf_neg_steps = int(self.confidence_threshold / self.reloc_vis_conf_threshold * 256)
        conf_pos_steps = 256 - conf_neg_steps

        conf_pos_map = plt.cm.get_cmap("summer")(np.linspace(1, 0, conf_pos_steps))[:, :3]
        conf_neg_map = plt.cm.get_cmap("cool")(np.linspace(1, 0, conf_neg_steps))[:, :3]

        self.reloc_color_map = np.concatenate((conf_neg_map, conf_pos_map))

        # final sweep vis parameters
        self.sweep_vis_iterations_threshold = sweep_vis_iterations_threshold
        self.sweep_hist_bins = 10
        self.sweep_color_map = plt.cm.get_cmap("cool")(np.linspace(0, 1, 10))[:, :3]

        # remember last frame's estimate and error color to add a marker to the camera trajectory
        self.reloc_buffer_previous_est = None
        self.reloc_buffer_previous_clr = None
        self.reloc_frame_count = 0
        # remember all reloc errors so far
        self.reloc_conf_buffer = None
        # limit the number of reloc frames for dense and long sequences (like 7Scenes)
        self.reloc_duration = 60
        self.reloc_frame_counter = 0
        self.reloc_success_counter = 0

        # camera views for rendering the scene during mapping
        self.pan_cams = None

        # buffer for observing camera
        self.scene_camera = None

        # camera trajectory to render
        self.trajectory_buffer = None

        # buffer holding the map point cloud
        self.point_cloud_buffer = None

        # index of current frame, rendered frame
        self.frame_idx = 0
        
        # self.npz_file = r'/home/xt/3d/acezero-main/new_code/gt_new.npz'
        self.result_npz = result_npz
        self.npz_data = load_npz_file(self.result_npz)
        # self.npz_data = np.load(self.npz_file)
        self.pts3d = self.npz_data['pts3d'].copy()
        self.cam_pose = self.npz_data['cam_poses'].copy()
        self.cam_intrinsic = self.npz_data['intrinsic'].copy()
        self.image_gt = self.npz_data['images_gt'].copy()
        self.pts_mask = self.npz_data['pts_mask'].copy()


        translations = self.cam_pose[:, :3, 3]
        mean_dis = np.mean(np.linalg.norm(translations, axis=1))
        # import pdb; pdb.set_trace()
        if mean_dis > 100:
            self.marker_scale = mean_dis 
            self.frustum_scale_factor = mean_dis
        else:
            self.marker_scale = 3
            self.frustum_scale_factor = 2
            # import pdb; pdb.set_trace()

        self.marker_size = marker_size * self.marker_scale

        # self.frustum_scale_factor = mean_dis * 0.09 if mean_dis>3 else 1
        self.frustum_scale_mapping = 0.3 * self.frustum_scale_factor
        self.frustum_scale_reloc = 0.3 * self.frustum_scale_factor
        self.only_frustum = False
        self.pan_radius_scale = pan_radius_scale
        self.pan_start_angle = pan_start_angle

        # self.pan_radius_scale = pan_radius_scale
        # import pdb; pdb.set_trace()

    def _generate_camera_pan(self,
                             pan_number_cams,
                             mapping_poses,
                             pan_angle_coverage,
                             anchor_camera=None):
        """
        Generate a list of camera views that smoothly pan around the scene.

        @param pan_number_cams: Number of views to be generated.
        @param mapping_poses: Mapping camera poses that the pan should enclose.
        @param pan_angle_coverage: Opening angle of the pan (degrees).
        @param anchor_camera: Optional camera pose to be used as the center of the pan.
        @return: List of 4x4 camera poses.
        """
        pan_cams = []

        # select anchor camera to be used for the mapping camera pan
        if anchor_camera is None:
            pan_center_pose = mapping_poses[len(mapping_poses) // 2].copy()
        else:
            pose_distances = [np.linalg.norm(pose[:3, 3] - anchor_camera[:3, 3]) for pose in mapping_poses]
            pan_center_pose = mapping_poses[pose_distances.index(min(pose_distances))]

        # move pan center to the average of all pose positions
        poses_pos = [pose[:3, 3] for pose in mapping_poses]
        poses_pos = np.stack(poses_pos, axis=-1)
        pan_center_pose[:3, 3] = poses_pos.mean(axis=1)

        # get approximate extent of mapping cameras
        poses_pos_extent = poses_pos.max(axis=1) - poses_pos.min(axis=1)
        poses_extent = [poses_pos_extent[c] for c in range(3)]
        # hack to support different coordinate conventions
        # find the two axis of maximum extent and average those
        poses_extent.sort(reverse=True)
        poses_extent = 0.5 * (poses_extent[0] + poses_extent[1])

        # create a camera pan around the scene
        # pan_radius = 4.5 * poses_extent
        # import pdb; pdb.set_trace()
        if self.pan_radius_scale == -1:
            tanstlations = np.array(mapping_poses)[:, :3, 3]
            mean_dis = np.mean(np.linalg.norm(tanstlations, axis=1))
            # import pdb; pdb.set_trace()
            if mean_dis > 100:
                pan_radius_scale = 4.5
            elif mean_dis > 0.5 and mean_dis < 1:
                pan_radius_scale = 0.5
            elif mean_dis > 0 and mean_dis < 0.5:
                pan_radius_scale = 5.5
            else:
                pan_radius_scale =  0.5 + (mean_dis / 100) * 4
        else:
            pan_radius_scale = self.pan_radius_scale

        pan_radius = pan_radius_scale * poses_extent

        pan_angle_start = self.pan_start_angle - pan_angle_coverage / 2
        pan_angle_increment = pan_angle_coverage / pan_number_cams

        for i in range(pan_number_cams):
            pan_pose = np.eye(4)

            pan_angle = math.radians(pan_angle_start + pan_angle_increment * i)
            pan_x = pan_radius * math.cos(pan_angle)
            pan_z = -pan_radius * math.sin(pan_angle)

            x_axis_index = 0
            if self.flipped_portrait:
                x_axis_index = 1

            pan_pose[x_axis_index, 3] = pan_x
            pan_pose[2, 3] = pan_z

            if self.flipped_portrait:
                # rotation around x
                pan_rotation_angle = math.radians(pan_angle_coverage / 2 - pan_angle_increment * i)

                pan_pose[1, 1] = math.cos(pan_rotation_angle)
                pan_pose[1, 2] = -math.sin(pan_rotation_angle)
                pan_pose[2, 1] = math.sin(pan_rotation_angle)
                pan_pose[2, 2] = math.cos(pan_rotation_angle)
            else:
                # rotation around y
                pan_rotation_angle = math.radians(-pan_angle_coverage / 2 + pan_angle_increment * i)

                pan_pose[0, 0] = math.cos(pan_rotation_angle)
                pan_pose[0, 2] = math.sin(pan_rotation_angle)
                pan_pose[2, 0] = -math.sin(pan_rotation_angle)
                pan_pose[2, 2] = math.cos(pan_rotation_angle)

            pan_pose = pan_center_pose @ pan_pose

            pan_cams.append(pan_pose)

        return pan_cams

    def _get_pan_camera(self):
        """
        Get the pan camera from the current frame index. The camera will pan back and forth indefinitely.

        @return: 4x4 camera pose from the pan camera list.
        """

        # get correct pan cam - if index out of range, go backwards through the list
        num_pan_cams = len(self.pan_cams)
        pan_cam_cycle = self.frame_idx // num_pan_cams
        pan_cam_index = self.frame_idx % num_pan_cams

        if pan_cam_cycle % 2 == 1:
            # go backward through the list
            pan_cam_index = num_pan_cams - pan_cam_index - 1

        return self.pan_cams[pan_cam_index]
    
    def add_jitter_to_pose(self, pose, translation_jitter=0.01, rotation_jitter=0.01):
        # 添加平移抖动
        jitter_translation = np.random.uniform(-translation_jitter, translation_jitter, size=(3,))
        pose[:3, 3] += jitter_translation

        # 添加旋转抖动（通过生成一个随机的小角度旋转矩阵）
        angle = np.random.uniform(-rotation_jitter, rotation_jitter)  # 随机角度（弧度）
        axis = np.random.randn(3)  # 随机旋转轴
        axis /= np.linalg.norm(axis)  # 归一化旋转轴

        # 计算旋转矩阵（使用 Rodrigues 公式）
        cos_theta = np.cos(angle)
        sin_theta = np.sin(angle)
        ux, uy, uz = axis

        rotation_matrix = np.array([
            [cos_theta + ux**2 * (1 - cos_theta), ux * uy * (1 - cos_theta) - uz * sin_theta, ux * uz * (1 - cos_theta) + uy * sin_theta],
            [uy * ux * (1 - cos_theta) + uz * sin_theta, cos_theta + uy**2 * (1 - cos_theta), uy * uz * (1 - cos_theta) - ux * sin_theta],
            [uz * ux * (1 - cos_theta) - uy * sin_theta, uz * uy * (1 - cos_theta) + ux * sin_theta, cos_theta + uz**2 * (1 - cos_theta)]
        ])

        # 将旋转抖动添加到原始旋转部分
        pose[:3, :3] = np.dot(pose[:3, :3], rotation_matrix)

        # translation_jitter *= 0.5
        return pose
    
    def _generate_camera_trajectory(self, mapping_poses):
        """
        Add all mapping cameras (original positions) to the trajectory buffer.

        @param mapping_poses: List of camera poses (4x4)
        """
        for frame_idx in range(len(mapping_poses)):
            # get pose of mapping camera
            frustum_pose = mapping_poses[frame_idx].copy()

            self.trajectory_buffer.add_position_marker(
                marker_pose=frustum_pose,
                marker_color=(125, 125, 125))

    def pose_align(self, pose):
        matrix = np.identity(4)
        matrix[1, 1] = -1
        matrix[2, 2] = -1
        opengl_conversion_matrix = matrix
        align_rotation = np.eye(4)
        align_rotation[:3, :3] = Rotation.from_euler(
            "x", 180, degrees=True
        ).as_matrix()
        pose_aligned = (pose @ opengl_conversion_matrix)
        # pose_aligned = (pose @ opengl_conversion_matrix @ align_rotation)
        return pose_aligned
    
    @staticmethod
    def _convert_cv_to_gl(pose):
        """
        Convert a pose from OpenCV to OpenGL convention (and vice versa).

        @param pose: 4x4 camera pose.
        @return: 4x4 camera pose.
        """
        gl_to_cv = np.array([[1, -1, -1, 1], [-1, 1, 1, -1], [-1, 1, 1, -1], [1, 1, 1, 1]])
        return gl_to_cv * pose

    @staticmethod
    def _render_pc(r, pc, camera, camera_pose, only_frustum):
        """
        Render a point cloud on a black background.

        @param r: PyRender Renderer.
        @param pc: PyRender point cloud object.
        @param camera: PyRender camera object.
        @param camera_pose: 4x4 camera pose.
        @return: Rendered frame (RGB).
        """
        scene = pyrender.Scene(bg_color=(0, 0, 0), ambient_light=(1, 1, 1))
        if not only_frustum:
            scene.add(pc)
        scene.add(camera, pose=camera_pose)
        color, _ = r.render(scene)

        return color

    @staticmethod
    def _render_trajectory(r, trajectory, camera, camera_pose, frustum_images):
        """
        Renders the trajectory mesh with flat lighting on a transparent background.

        @param r: PyRender Renderer.
        @param trajectory: PyRender mesh object.
        @param camera: PyRender camera object.
        @param camera_pose: 4x4 camera pose.
        @param frustum_images: Textured meshes that represent image boxes.
        @return: Rendered frame (RGBA).
        """
        scene = pyrender.Scene(bg_color=(0, 0, 0, 0), ambient_light=(1, 1, 1))
        scene.add(trajectory)
        scene.add(camera, pose=camera_pose)

        for frustum_image in frustum_images:
            scene.add(frustum_image)

        color, _ = r.render(scene, flags=(pyrender.constants.RenderFlags.RGBA | pyrender.constants.RenderFlags.FLAT | pyrender.constants.RenderFlags.SKIP_CULL_FACES))
        return color

    @staticmethod
    def _blend_images(img1_RGB, img2_RGBA):
        """
        Add an RGBA image on top of an RGB image.

        @param img1_RGB: Background image.
        @param img2_RGBA: Transparent image for blending on top.
        @return: Blended image (RGB)
        """
        mask = img2_RGBA[:, :, 3].astype(float)
        mask /= 255
        mask = np.expand_dims(mask, axis=2)

        blended_rgb = img2_RGBA[:, :, :3].astype(float) * mask + img1_RGB.astype(float) * (1 - mask)
        return blended_rgb.astype('uint8')

    def _errors_to_colors(self, errors, max_error):
        """
        Map errors to error color map (self.mapping_color_map).

        @param errors: 1D array of N scalar errors
        @param max_error: Error threshold for mapping to the color map
        @return: Color array N3 and normalized error array N1
        """
        # map reprojection error up to X pixels to color map
        norm_errors = errors / max_error  # normalise
        norm_errors = 1 - norm_errors.clip(0, 1)  # reverse

        # error indices for color map
        errors_idxs = (norm_errors * 255).astype(int)

        # expand color map to size of the point cloud
        errors_clr = np.broadcast_to(self.mapping_color_map, (errors_idxs.shape[0], 256, 3))
        # for each point, pick color from color map according to error index
        errors_clr = errors_clr[np.arange(errors_idxs.shape[0]), errors_idxs] * 255

        return errors_clr, norm_errors

    def _get_mapping_progress(self):
        """
        Get percentage of mapping done.

        @return: Scalar (0,1)
        """
        if self.mapping_done_idx > 0:
            effective_frame_idx = self.mapping_done_idx
        else:
            effective_frame_idx = self.mapping_iteration

        return effective_frame_idx

    @staticmethod
    def _draw_hist(image, hist_values, hist_colors, hist_x, hist_y, hist_w, hist_h, hist_max, min_height=3):
        """
        Add a histogram to the frame.

        @param image: Input frame.
        @param hist_values: Values of histogram bars.
        @param hist_colors: RGB color for each bar.
        @param hist_x: Horizontal position in pixels.
        @param hist_y: Vertical position in pixels.
        @param hist_w: Width in pixels.
        @param hist_h: Height in pixels.
        @param hist_max: Normalising factor for hist_values.
        @param min_height: Minimum height of a bar in pixels.
        """
        hist_bins = len(hist_values)
        bar_h = int(hist_h / hist_bins)

        for hist_idx in range(hist_bins):
            bar_w = int(hist_w * (hist_values[hist_idx] / hist_max))
            bar_w = max(min_height, bar_w)
            bar_y = int(hist_y + hist_idx * bar_h)

            # draw the actual colored bars
            rr, cc = draw.rectangle((hist_x, bar_y), extent=(bar_w, bar_h))
            image[rr, cc, 0:3] = hist_colors[hist_idx]


    @staticmethod
    def _write_captions(image, captions_dict, text_color=(1, 1, 1)):
        """
        Write text onto frame.

        Using matplotlib following https://scikit-image.org/docs/stable/auto_examples/applications/plot_text.html

        @param image: Input frame.
        @param captions_dict: Dictionary specifying multiple captions, with fields x, y, text and fs (font size).
        @param text_color: RGB color of text.
        @return: Frame with text.
        """
        fig = plt.figure()
        fig.figimage(image, resize=True)

        for caption in captions_dict:
            fig.text(caption['x'], caption['y'], caption['text'], fontsize=caption['fs'], va="top", color=text_color)

        fig.canvas.draw()
        image = np.asarray(fig.canvas.renderer.buffer_rgba())
        plt.close(fig)

        return image

    def _write_mapping_captions(self, image):
        """
        Write all image captions for the mapping stage.

        @param image: Input frame.
        @return: Frame with captions.
        """
        image_h = image.shape[0]

        captions_dict = [
            {'x': 0.15, 'y': 0.13, 'fs': 0.04 * image_h,
             'text': "Point Cloud Estimation"},
        ]

        return self._write_captions(image, captions_dict)

    def _write_reloc_captions(self, image):
        """
        Write all image captions for the relocalisation stage.

        @param image: Input frame.
        @return: Frame with captions.
        """
        image_h = image.shape[0]
        if self.only_frustum:
            captions_dict = [
                {'x': 0.15, 'y': 0.13, 'fs': 0.04 * image_h,
                'text': "Camera Pose Estimation"},
            ]
        else:
            captions_dict = [
                {'x': 0.15, 'y': 0.13, 'fs': 0.04 * image_h,
                 'text': "Point Cloud Estimation"}]

        return self._write_captions(image, captions_dict)

    def _write_sweep_captions(self, image, frames_registered, frames_total):
        """
        Write all image captions for the final camera sweep.

        @param image: Input frame.
        @return: Frame with captions.
        """
        image_h = image.shape[0]

        captions_dict = [
            {'x': 0.15, 'y': 0.13, 'fs': 0.04 * image_h,
             'text': "Point Cloud Estimation"},
        ]

        return self._write_captions(image, captions_dict)

    def _render_frame_from_buffers_safe(self):
        """
        Wrapper for _render_frame_from_buffers, re-trying rendering if render lib throws error.

        We found the rendering backend to be brittle, throwing random errors now and then.
        Re-trying to render the same geometry worked always.

        @return: rendered frame or None if rendering failed after multiple tries
        """
        max_tries = 10

        while max_tries > 0:
            try:
                return self._render_frame_from_buffers()
            except:
                print("An error occurred:")
                import traceback
                traceback.print_exc()
                _logger.warning("Rendering failed, trying again!")
                max_tries -= 1

        raise RuntimeError("Re-rendering failed too often...")

    def _render_frame_from_buffers(self):
        """
        Render current frame according to state of internal buffers: scene camera, point cloud and trajectory mesh.

        @return: Rendered frame.
        """
        # get smooth observing camera
        smooth_camera_pose = self.scene_camera.get_current_view()

        # initialise pyrender pipeline
        camera = pyrender.PerspectiveCamera(yfov=np.pi / 3.0, aspectRatio=self.render_width / self.render_height)
        r = pyrender.OffscreenRenderer(self.render_width, self.render_height, point_size=self.point_size)

        # cast PC to rendering object
        frame_xyz, frame_clr, _ = self.point_cloud_buffer.get_point_cloud()
        ace_map = pyrender.Mesh.from_points(frame_xyz, colors=frame_clr)

        # get camera trajectory mesh
        trajectory_mesh, frustum_images = self.trajectory_buffer.get_mesh()

        bg_RGB = self._render_pc(r, ace_map, camera, smooth_camera_pose, self.only_frustum)
        # save_frame_data(self.frame_idx, frame_xyz, frame_clr, self.render_width, self.render_height, smooth_camera_pose)
        # render camera trajectory with flat shading and alpha transparency for blending
        cams_RGBA = self._render_trajectory(r, trajectory_mesh, camera, smooth_camera_pose, frustum_images)
        # combine the two renders
        blended_RGB = self._blend_images(bg_RGB, cams_RGBA)

        # rotate from portrait to landscape
        if self.flipped_portrait:
            blended_RGB = rotate(blended_RGB, -90, resize=True, preserve_range=True).astype('uint8')

        return blended_RGB

    def _save_frame(self, frame):
        """
        Store frame with current frame number to target folder.

        @param frame: Input image.
        """
        out_render_file = f"{self.target_path}/frame_{self.frame_idx:05d}.png"
        io.imsave(out_render_file, frame)
        _logger.info(f"Rendered and saved frame: {out_render_file}")

    def _render_mapping_frame_from_buffers(self):
        """
        Render current frame according to buffers, and draw mapping specific captions and the reprojection error histogram.
        """

        # update observing camera
        self.scene_camera.update_camera(self._get_pan_camera())

        current_frame = self._render_frame_from_buffers_safe()

        if current_frame is not None:
            current_frame = self._write_mapping_captions(current_frame)

            # write to disk
            self._save_frame(current_frame)

        # move frame index pointer for next render call
        self.frame_idx += 1

    @staticmethod
    def get_pose_from_buffer(pose_idx, buffer):
        """
        Get a single pose (camera to world) from a buffer of poses (world to camera).
        """

        pose_torch_34 = buffer[pose_idx]
        pose_numpy_44 = np.eye(4, 4)
        pose_numpy_44[:3] = pose_torch_34.detach().cpu().numpy()
        return np.linalg.inv(pose_numpy_44)
    
    @staticmethod
    def get_pose_from_buffer_orig(pose_idx, buffer):
        """
        Get a single poses (world to camera).
        """

        pose_torch_34 = buffer[pose_idx]
        pose_numpy_44 = np.eye(4, 4)
        pose_numpy_44[:3] = pose_torch_34.detach().cpu().numpy()
        return pose_numpy_44
    

    def visualize_cam_positions(self, pose_buffer, pose_buffer_orig):
        """
        Write camera positions to the trajectory buffer, color-coded by the distance to the original pose.

        @param pose_buffer: Buffer of refined camera poses, Nx3x4.
        @param pose_buffer_orig: Buffer of original camera poses, Nx3x4.
        """

        for pose_idx in range(pose_buffer.shape[0]):
            # calculate distance between refined and original pose
            pose_refined = self.get_pose_from_buffer_orig(pose_idx, pose_buffer)
            pose_orig = self.get_pose_from_buffer_orig(pose_idx, pose_buffer_orig)
            pose_t_distance = np.linalg.norm(pose_refined[:3, 3] - pose_orig[:3, 3])

            # map distance to color, clamping at 1m
            pose_color_idx = int(min(pose_t_distance / 1, 1) * 255)
            pose_color = self.pose_color_map[pose_color_idx] * 255
            print
            # print(f'camera trajectory pose {pose_refined} aliged {self.pose_align(pose_refined)} distance {pose_t_distance} color {pose_color}')
            self.trajectory_buffer.add_position_marker(
                # marker_pose=self._convert_cv_to_gl(pose_refined),
                # marker_pose = self.pose_align(pose_refined),
                marker_pose=pose_refined,
                marker_color=pose_color,
                marker_extent=self.marker_size,
                # marker_extent=10,
                frustum_maker=True
            )

    def map_to_camera_coords(self, our_xyz, our_pose):
        R = our_pose[:3, :3]  # 旋转矩阵
        t = our_pose[:3, 3]   # 平移向量
        
        R_inv = R.T

        H, W = our_xyz.shape[1], our_xyz.shape[2]
        points_world = our_xyz.reshape(3, -1).T 
        
        points_world_centered = points_world - t
        
        points_camera = np.dot(points_world_centered, R_inv.T) 
        
        return points_camera.T.reshape(3, H, W)



    def setup_reloc_visualisation(self, frame_count, camera_z_offset, only_frustum, frame_idx=None, state_dict=None):

        _logger.info("Setting up relocalisation visualisation.")
        # import pdb; pdb.set_trace()
        map_xyz = self.pts3d.reshape(-1, 3)
        # rgb = ((rgb + 1) / 2.0 * 255.0).astype('float64')
        map_clr = ((self.image_gt.transpose(0, 2, 3, 1).reshape(-1, 3) + 1.0) / 2.0 * 255.0).astype('float64')
        map_mask = self.pts_mask.reshape(-1)

        mapping_poses = [self.pose_align(our_pose) for our_pose in self.cam_pose]
        mapping_poses = [pose for pose in mapping_poses if
                         ((not np.any(np.isinf(pose))) and (not np.any(np.isnan(pose))))]
        
        self.pan_cams = self._generate_camera_pan(
            pan_number_cams= 100 + self.framecount_transition,
            mapping_poses=mapping_poses,
            pan_angle_coverage=self.pan_angle_coverage
        )
        # import ipdb; ipdb.set_trace()
        if frame_idx is None:
            self.frame_idx = 0  
            self.scene_camera = vutil.LazyCamera(backwards_offset=camera_z_offset)
        else:
            self.frame_idx = frame_idx
            self.scene_camera = vutil.LazyCamera(backwards_offset=camera_z_offset,
                                                 camera_buffer=state_dict['camera_buffer'])

        self.trajectory_buffer = vutil.CameraTrajectoryBuffer(frustum_skip=0, frustum_scale=self.frustum_scale_reloc)
        self.reloc_conf_buffer = []

        self.reloc_frame_count = frame_count
        self.reloc_frame_counter = 0

        self.only_frustum = only_frustum
        self.point_cloud_buffer = vutil.PointCloudBuffer()
        self.point_cloud_buffer.update_buffer(map_xyz, map_clr, pc_mask=map_mask)
        self.reset_flag = True

    def render_reloc_frame(self, query_file, est_pose, confidence,):
        """
        Update query trajectory with new GT pose and estimate and render frame.

        Stores rendered frame to target folder.

        @param query_file: image file of query
        @param est_pose: estimated pose, 4x4, OpenCV convention
        @param confidence: confidence of the estimate to determine whether it was successfully registered
        """
        renders_per_query = 1
        est_pose =self.pose_align(est_pose)

        # keep track of confidence statistics
        self.reloc_conf_buffer.append(confidence)

        # map error to color
        conf_color_idx = min(int(confidence / self.reloc_vis_conf_threshold * 255), 255)
        conf_color = self.reloc_color_map[conf_color_idx] * 255

        # remove previous frustums, and add just the new ones from the current frame
        self.trajectory_buffer.clear_frustums()
        if self.only_frustum:
            self.trajectory_buffer.add_camera_frustum(est_pose, image_file=query_file, sparse=False,
                                                    frustum_color=conf_color)
        else:
            if self.reset_flag:
                self.reset_position_markers(marker_color=self.progress_color_map[1] * 255)
        # keep camera if confidence is above threshold
        if confidence > self.confidence_threshold:

            self.reloc_success_counter += 1

            # add previous frame's estimate as a colored marker to the trajectory
            if self.reloc_buffer_previous_est is not None:
                self.trajectory_buffer.add_position_marker(
                    marker_pose=self.reloc_buffer_previous_est,
                    marker_color=self.reloc_buffer_previous_clr,
                    marker_extent=self.marker_size,
                    frustum_maker=True)

            # remember this frame's estimate for next render call
            self.reloc_buffer_previous_est = est_pose
            self.reloc_buffer_previous_clr = conf_color

        frame_skip = 1

        if self.reloc_frame_counter % frame_skip == 0:

            # for sparse queries we render multiple frames for a smooth transition
            for _ in range(renders_per_query):

                # update observing camera
                self.scene_camera.update_camera(self._get_pan_camera())

                # render actual frame
                current_frame = self._render_frame_from_buffers_safe()

                if current_frame is not None:
                    # finalize frame
                    # current_frame = self._draw_pose_conf_hist(current_frame, self.reloc_conf_buffer)
                    current_frame = self._write_reloc_captions(current_frame)

                    self._save_frame(current_frame)

                self.frame_idx += 1

        self.reloc_frame_counter += 1

    def reset_position_markers(self, marker_color):
        self.trajectory_buffer.trajectory = []
        for idx in range(self.cam_pose.shape[0]): 
            pose = self.pose_align(self.cam_pose[idx])
            pose_refined = pose[:3,:]
            self.trajectory_buffer.add_position_marker(
                marker_pose=pose_refined,
                marker_color=marker_color,
                marker_extent=self.marker_size,
                frustum_maker=True
            )

    def render_growing_map(self, growing_framecount_transition=60):
        self.only_frustum = False
        b, c, w, h = self.image_gt.shape
        # per_frmae_point = w * h
        self.point_cloud_buffer = vutil.PointCloudBuffer()
        batch_clr = ((self.image_gt.reshape(b, c, -1).transpose(0, 2, 1)+ 1.0) / 2.0 * 255.0).astype('float64')
        batch_xyz = self.pts3d.reshape(b, -1, c) 

        for i in range(b):
            current_pc = batch_xyz[i]
            current_pc_clr = batch_clr[i]
            current_pc_mask = self.pts_mask[i].reshape(-1)
            self.point_cloud_buffer.update_buffer(current_pc, current_pc_clr, pc_mask=current_pc_mask)
            for _ in range(growing_framecount_transition // b):
                self._render_mapping_frame_from_buffers()

    def get_mean_repreoject_error(self):
        vis_errors_list = []
        vis_scene_coords_list = []
    
            # print(f'current pose_jit_factor {pose_jit_factor}')
        for img_idx in range(self.image_gt.shape[0]):
            ours_pts3d = self.pts3d[img_idx].copy()
            ours_K = self.cam_intrinsic[img_idx].copy()
            
            ours_pose = self.cam_pose[img_idx].copy()

            ours_pts3d_b31 = torch.from_numpy(ours_pts3d.reshape(-1, 3, 1))
            ba = ours_pts3d_b31.shape[0]
            H, W = ours_pts3d.shape[:2]
            ours_K = torch.from_numpy(ours_K).unsqueeze(0).expand(ba, -1, -1)
            ours_pose = torch.from_numpy(ours_pose).unsqueeze(0).expand(ba, -1, -1)
            ours_w2c = torch.inverse(ours_pose)

            pred_scene_coords_b31 = ours_pts3d_b31
            # Make 3D points homogeneous so that we can easily matrix-multiply them.
            pred_scene_coords_b41 = to_homogeneous(pred_scene_coords_b31)
            gt_inv_poses_b34 = ours_w2c[:, :3]

            # Scene coordinates to camera coordinates.
            gt_inv_poses_b34 = gt_inv_poses_b34.float()
            pred_cam_coords_b31 = torch.bmm(gt_inv_poses_b34, pred_scene_coords_b41)

            Ks_b33 = ours_K
            pred_px_b31 = torch.bmm(Ks_b33, pred_cam_coords_b31)

            pred_px_b31[:, 2].clamp_(min=0.1)

            # Dehomogenise.
            pred_px_b21 = pred_px_b31[:, :2] / pred_px_b31[:, 2, None]

            # Measure reprojection error.
            pixel_grid_2HW = get_pixel_grid(1)

            pixel_positions_B2HW = pixel_grid_2HW[:, :H, :W].clone()
            pixel_positions_B2HW = pixel_positions_B2HW[None]  
            pixel_positions_B2HW = pixel_positions_B2HW.expand(1, 2, H, W) 
            target_px_b2 = pixel_positions_B2HW.transpose(0, 1).flatten(1).transpose(0, 1)
            reprojection_error_b2 = pred_px_b21.squeeze() - target_px_b2
            # import ipdb; ipdb.set_trace()
            reprojection_error_b1 = torch.norm(reprojection_error_b2, dim=1, keepdim=True, p=1)    

            vis_scene_coords = pred_scene_coords_b31.detach().cpu().squeeze().numpy()
            vis_scene_coords_list.append(vis_scene_coords)
            # reprojection_error_b1 += random_noise
            
            vis_errors = reprojection_error_b1.detach().cpu().squeeze().numpy()
            vis_errors_list.append(vis_errors)

            pose_scale = self.pose_align(self.cam_pose[img_idx].copy())
            self.our_current_pose = pose_scale.copy()

            poses_updated = torch.Tensor(pose_scale[:3,:]).unsqueeze(0)
            poses_original = torch.Tensor(pose_scale[:3,:]).unsqueeze(0)
                
            current_vis_scene_coords = np.concatenate(vis_scene_coords_list, axis=0)
            
            current_vis_errors = np.concatenate(vis_errors_list, axis=0)

            mean_value = current_vis_errors.mean() if current_vis_errors.mean() <  self.mapping_vis_error_threshold else self.mapping_vis_error_threshold
            # current_vis_errors[:] = mean_value
        return current_vis_scene_coords, current_vis_errors, mean_value, poses_updated, poses_original


    def render_gradient_clr(self, grdient_frmae_count = 15):
        current_vis_scene_coords, current_vis_errors, mean_value, poses_updated, poses_original = self.get_mean_repreoject_error()
        current_vis_errors[:] = mean_value
        end_scale = 0.2 / mean_value
        for jit_idx in range(grdient_frmae_count):
            # pose_jit_factor = init_pose_hit_factor * (1 - jit_idx / grdient_frmae_count)
            gradient_scale = (1.0 - end_scale) * ( grdient_frmae_count - jit_idx ) / grdient_frmae_count
            self.point_cloud_buffer.clear_buffer()
            
            # import pdb; pdb.set_trace()
            self.render_mapping_frame(current_vis_scene_coords, current_vis_errors, poses_updated, poses_original)
            next_mean_value = mean_value * gradient_scale
            current_vis_errors[:] = next_mean_value
            # print(f'mean val {next_mean_value}')

    def render_final_sweep(self, frame_count, camera_z_offset, poses, pose_iterations, total_poses, state_dict):
        """
        Render final camera sweep after relocalisation.

        @param frame_count: Number of frames the final sweep animation should take
        @param camera_z_offset: Distance from the scene for the camera
        @param poses: List of poses in OpenCV convention
        @param pose_iterations: For each pose, the iteration when it was registered
        @param total_poses: Total number of poses in the dataset (counting also non-registered)
        """

        map_xyz = state_dict['map_xyz']
        map_clr = state_dict['map_clr']

        map_mask = self.pts_mask.reshape(-1)
        # map_xyz = np.zeros_like(map_xyz)

        self.frame_idx = state_dict['frame_idx']
        self.scene_camera = vutil.LazyCamera(backwards_offset=camera_z_offset,
                                             camera_buffer=state_dict['camera_buffer'])

        pan_cams = state_dict['pan_cameras']
        anchor_camera = pan_cams[len(pan_cams) // 2]

        self.point_cloud_buffer = vutil.PointCloudBuffer()
        self.point_cloud_buffer.update_buffer(map_xyz, map_clr, pc_mask=map_mask)

        # reset all buffers
        self.trajectory_buffer = vutil.CameraTrajectoryBuffer(frustum_skip=0, frustum_scale=self.frustum_scale_reloc)
        self.reloc_conf_buffer = []

        # poses = [self._convert_cv_to_gl(pose) for pose in poses]

        # add poses
        max_iterations = 10
        progress_color_map = plt.cm.get_cmap("cool")(np.linspace(0, 1, max_iterations))[:, :3]

        for pose_idx, pose in enumerate(poses):
            progress_color_idx = min(pose_iterations[pose_idx], max_iterations - 1)
            current_color = progress_color_map[progress_color_idx] * 255

            self.trajectory_buffer.add_position_marker(pose, marker_color=current_color, frustum_maker=True,
                                                       marker_extent=self.marker_size)

            self.reloc_conf_buffer.append(min(pose_iterations[pose_idx], max_iterations))        
        pan_cameras = self._generate_camera_pan(frame_count, poses, pan_angle_coverage=90, anchor_camera=anchor_camera)

        _logger.info("Rendering final camera sweep.")

        for pan_idx, pan_camera in enumerate(pan_cameras):

            # update observing camera
            self.scene_camera.update_camera(pan_camera)

            # render actual frame
            current_frame = self._render_frame_from_buffers_safe()

            if current_frame is not None:
                current_frame = self._write_sweep_captions(current_frame, len(poses), total_poses)
                self._save_frame(current_frame)

            # move frame index pointer for next render call
            self.frame_idx += 1
