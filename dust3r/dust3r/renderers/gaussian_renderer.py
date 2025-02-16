import torch

from dust3r.renderers.gaussian_utils import GRMGaussianModel, render_image
from dust3r.renderers.deferred_backprop_grm_renderer import deferred_backprop_gaussian_renderer, deferred_backprop_gaussian_renderer_wodb


class GaussianRenderer:
    def __init__(self,
                 height=512,
                 width=512,
                 patch_size=512,
                 sh_degree=0,
                 render_type="deferred",
                 bg_color=(0., 0., 0.),
                 scaling_modifier=1.0,
                 gs_kwargs=dict(),
                 ):
        self.height = height 
        self.width = width
        self.patch_size = patch_size
        self.sh_degree = sh_degree
        self.render_type = render_type
        self.bg_color = bg_color
        self.scaling_modifier = scaling_modifier
        self.gs_kwargs = gs_kwargs

    @torch.cuda.amp.custom_fwd(cast_inputs=torch.float32)
    def __call__(self, gs_params, Ks, RTs, opatch_size=0, opatch_center=None, debug=False):
        Ks = Ks.to(torch.float32)
        RTs = RTs.to(torch.float32)
        
        if self.render_type == "vanilla":
            device = Ks.device
            B, T = Ks.shape[:2]
            images = torch.zeros(B, T, 3, self.height, self.width, dtype=torch.float32, device=device)
            alphas = torch.zeros(B, T, 1, self.height, self.width, dtype=torch.float32, device=device)
            depths = torch.zeros(B, T, 1, self.height, self.width, dtype=torch.float32, device=device)
            for bidx in range(B):
                gs_param = dict([(k, v[bidx]) for k, v in gs_params.items()])
                pc = GRMGaussianModel(sh_degree=self.sh_degree,
                                      scaling_kwargs=self.gs_kwargs, **gs_param)
                for vidx in range(T):
                    results = render_image(pc, Ks[bidx, vidx], RTs[bidx, vidx], height=self.height, width=self.width, bg_color=self.bg_color, debug=debug)
                    images[bidx, vidx] = results['image']
                    alphas[bidx, vidx] = results['alpha']
                    depths[bidx, vidx] = results['depth']

        elif self.render_type == "deferred":
            images, depths, alphas, patches = \
                deferred_backprop_gaussian_renderer(height=self.height, width=self.width, C2W=RTs, K=Ks,
                                                    patch_size=self.patch_size, opatch_size=opatch_size,
                                                    opatch_center=opatch_center, sh_degree=self.sh_degree,
                                                    scaling_kwargs=self.gs_kwargs, **gs_params)
        elif self.render_type == "render_wodb":
            images, depths, alphas, patches = \
                deferred_backprop_gaussian_renderer_wodb(height=self.height, width=self.width, C2W=RTs, K=Ks,
                                                    patch_size=self.patch_size, opatch_size=opatch_size,
                                                    opatch_center=opatch_center, sh_degree=self.sh_degree,
                                                    scaling_kwargs=self.gs_kwargs, **gs_params)
        else:
            raise ValueError(f"Only support vanilla and deferred `render_type`, but got {self.render_type}")

        ret = {'image': images, 'alpha': alphas, 'depth': depths}

        return ret




if __name__ == '__main__':
    import os
    import random
    from tqdm import tqdm

    import warnings
    import time

    warnings.filterwarnings("ignore")

    import numpy as np
    import matplotlib as mpl
    import cv2
    import imageio

    import torch


    def dump_video(image_sets, path, **kwargs):
        video_out = imageio.get_writer(path, mode='I', fps=30, codec='libx264')
        for image in image_sets:
            video_out.append_data(image)
        video_out.close()

    # def dump_video(images, output_path, fps=30):
    #     # Get the dimensions of the first image
    #     height, width, layers = images[0].shape

    #     # Define the codec and create a VideoWriter object
    #     fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 'mp4v' is the codec for .mp4 files
    #     video = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    #     for image in images:
    #         image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    #         video.write(image_bgr)
    #     video.release()

    def create_depth_vis(depths):
        mask = depths > 1e-4
        depth_min, depth_max = np.percentile(depths[mask], (0.01, 99.99))

        depth_vis = (depths - depth_min) / (depth_max - depth_min)
        depth_vis = depth_vis.clip(0.0, 1.0)
        depth_vis = mpl.colormaps["viridis"](depth_vis)[..., :3]
        depth_vis = depth_vis * mask[:, :, :, None]
        depth_vis = (depth_vis * 255.0).clip(0.0, 255.0).astype(np.uint8)

        return depth_vis

    def generate_input_camera(r, poses, device='cuda:0', fov=50):
        def normalize_vecs(vectors): return vectors / (torch.norm(vectors, dim=-1, keepdim=True))
        poses = np.deg2rad(poses)
        poses = torch.tensor(poses).float()
        pitch = poses[:, 0]
        yaw = poses[:, 1]

        z = r*torch.sin(pitch)
        x = r*torch.cos(pitch)*torch.cos(yaw)
        y = r*torch.cos(pitch)*torch.sin(yaw)
        cam_pos = torch.stack([x, y, z], dim=-1).reshape(z.shape[0], -1).to(device)

        forward_vector = normalize_vecs(-cam_pos)
        up_vector = torch.tensor([0, 0, -1], dtype=torch.float,
                                            device=device).reshape(-1).expand_as(forward_vector)
        left_vector = normalize_vecs(torch.cross(up_vector, forward_vector,
                                                            dim=-1))

        up_vector = normalize_vecs(torch.cross(forward_vector, left_vector,
                                                            dim=-1))
        rotate = torch.stack(
                        (left_vector, up_vector, forward_vector), dim=-1)

        rotation_matrix = torch.eye(4, device=device).unsqueeze(0).repeat(forward_vector.shape[0], 1, 1)
        rotation_matrix[:, :3, :3] = rotate

        translation_matrix = torch.eye(4, device=device).unsqueeze(0).repeat(forward_vector.shape[0], 1, 1)
        translation_matrix[:, :3, 3] = cam_pos
        cam2world = translation_matrix @ rotation_matrix

        fx = 0.5/np.tan(np.deg2rad(fov/2))
        K = np.zeros((3,3), dtype=np.float32)
        K[0][0] = fx
        K[1][1] = fx
        K[0][2] = 0.5
        K[1][2] = 0.5
        K[2][2] = 1
        fxfycxcy = torch.tensor(K, dtype=rotate.dtype, device=device)

        return cam2world, fxfycxcy


    print('Build gaussian model...')
    pc = GRMGaussianModel(sh_degree=0)
    pc1 = GRMGaussianModel(sh_degree=0)

    ply_paths = ['/mnt/petrelfs/liangzhengyang.d/hh_projects/grm_ckpts/sample_0_vis_masked1.ply']
    ply_paths = ['/home/yhxu/code/image2nerf/image2nerf.ply']
    ply_paths = ['/home/yhxu/code/image2nerf/results/ps_dr3ws4096_sigmoid_fixr_alpha1_res512_32gpu/epoch_00422_pth_visualize_gso/sample_0_vis_masked.ply']
    save_dir = 'work_dirs/debug'
    os.makedirs(save_dir, exist_ok=True)

    print('Generate camera...')
    c2ws, fxfycxcy = generate_input_camera(r=2.7, poses=[[20, azi] for azi in np.linspace(0, 360, 60, endpoint=False)], fov=50)

    num_samples = len(ply_paths)
    num_views = len(c2ws)


    pbar = tqdm(total=num_samples)
    print('Start rendering...')

    for idx, ply_path in tqdm(enumerate(ply_paths)):
        print(f'Load {ply_path}')
        pc.load_ply(ply_path)
        pc.set_device(device='cuda:0')

        pc1.load_ply(ply_path)
        pc1.set_device(device='cuda:0')

        pc._xyz.requires_grad = True
        pc._features_dc.requires_grad = True
        pc._scaling.requires_grad = True
        pc._rotation.requires_grad = True
        pc._opacity.requires_grad = True

        pc1._xyz.requires_grad = True
        pc1._features_dc.requires_grad = True
        pc1._scaling.requires_grad = True
        pc1._rotation.requires_grad = True
        pc1._opacity.requires_grad = True

        images = []
        depths = []
        alphas = []
        for vidx in range(num_views):
            import time
            torch.cuda.synchronize()
            begin = time.time()
            # render_results = render_image(pc, fxfycxcy, c2ws[vidx], 256, 256, (1.0,1.0,1.0))

            gs_params = dict([(x, y[None]) for (x, y) in pc.get_params.items() ])

            rimages, rdepths, ralphas, patches = \
                deferred_backprop_gaussian_renderer(height=256, width=256, C2W=c2ws[None, vidx:vidx+1], K=fxfycxcy[None, None],
                                                    patch_size=256, opatch_size=0,
                                                    opatch_center=0, sh_degree=0,
                                                    scaling_kwargs=dict(type='exp'), **gs_params)
          
            render_results = render_image(pc1, fxfycxcy, c2ws[vidx], 256, 256, (1.0,1.0,1.0))
            import ipdb;ipdb.set_trace()
            if False:
                image = (render_results['image'].permute(1, 2, 0).detach().cpu().numpy()*255).astype(np.uint8)
                alpha = render_results['alpha'].permute(1, 2, 0).detach().cpu().numpy()
                depth = render_results['depth'].permute(1, 2, 0).detach().cpu().numpy()
            else:
                image = (rimages[0][0].permute(1, 2, 0).detach().cpu().numpy()*255).astype(np.uint8)
                alpha = ralphas[0][0].permute(1, 2, 0).detach().cpu().numpy()
                depth = rdepths[0][0].permute(1, 2, 0).detach().cpu().numpy()

            images.append(image)
            depths.append(depth)
            alphas.append(alpha)
            torch.cuda.synchronize()
            end = time.time() 
            print('run time', end - begin)
            cv2.imwrite(f'{save_dir}/scene{idx:02d}_view{vidx:03d}.png', image[:,:,::-1])
        dump_video(images, f'{save_dir}/scene{idx:02d}.mp4')
        pbar.update(1)
