# conda install -y pytorch  pytorch-cuda=12.1 -c pytorch -c nvidia
# pip install Pillow plyfile opencv-python numpy git+https://github.com/graphdeco-inria/diff-gaussian-rasterization


import os
import torch
from torch import nn
import numpy as np
from diff_gaussian_rasterization import (
    GaussianRasterizationSettings,
    GaussianRasterizer,
)
from dust3r.gaussian_utils import render_opencv_cam, GaussianModel


class GaussianRenderer:
    def __init__(self, img_height, img_width):
        self.render_type = 'original'
        renderer_config={}
        renderer_config['scaling_activation_type'] = 'sigmoid'
        renderer_config['scale_min_act'] = 0.0001
        renderer_config['scale_max_act'] =  0.005
        renderer_config['scale_multi_act'] = 0.1
        renderer_config['sh_degree'] = 0
        self.gaussian_model = GaussianModel(sh_degree=renderer_config['sh_degree'], 
                                            scaling_activation_type=renderer_config['scaling_activation_type'],
                                            scale_min_act=renderer_config['scale_min_act'], 
                                            scale_max_act=renderer_config['scale_max_act'], 
                                            scale_multi_act=renderer_config['scale_multi_act'])
        self.img_height = img_height
        self.img_width = img_width

    @torch.cuda.amp.custom_fwd(cast_inputs=torch.float32)
    def render(self, latent, output_fxfycxcy, output_c2ws, opatch_size=0, opatch_center=None):
        features, opacity, scaling, rotation = latent['feature'], latent['opacity'], latent['scaling'], latent['rotation']

        # get xyz from depth and offset
        xyz = latent['xyz']
 
        bs, vs = output_fxfycxcy.shape[:2] 
        renderings = torch.zeros(bs, vs, 3, self.img_height, self.img_width, dtype=torch.float32, device=output_c2ws.device)
        alphas = torch.zeros(bs, vs, 1, self.img_height, self.img_width, dtype=torch.float32, device=output_c2ws.device)
        depths = torch.zeros(bs, vs, 1, self.img_height, self.img_width, dtype=torch.float32, device=output_c2ws.device)
        for idx in range(bs):
            pc = self.gaussian_model.set_data(xyz[idx], features[idx], scaling[idx], rotation[idx], opacity[idx])
            for vidx in range(vs):
                render_results = render_opencv_cam(pc, self.img_height, self.img_width, output_c2ws[idx, vidx], output_fxfycxcy[idx, vidx])
                image = render_results['render']
                alpha = render_results['alpha']
                depth = render_results['depth']
                renderings[idx, vidx] =  image
                alphas[idx, vidx] =  alpha
                depths[idx, vidx] =  depth
        
        results = {'image': renderings, 'alpha': alphas, 'depth': depths}
        return results

if __name__ == '__main__':
    import json
    import copy
    from tqdm import tqdm
    from .deferred_backprop_gaussian_render import deferred_backprop_gaussian_render
    pc = GaussianModel(sh_degree=0)


    # pc.load_obj_defered('/nas2/zifan/temp/000-000/ff6c2c51f7b040279200f8154a376841.obj', batch_size=4)
    pc.load_ply_new_defered('/nas2/zifan/temp/debug1.ply', batch_size=4)
    # pc.load_ply_new_defered('../../debug2.ply', batch_size=16)
    with open(f"/nas2/zifan/temp/000-000/ff6c2c51f7b040279200f8154a376841/uniform/opencv_cameras.json", "r") as f:
        params = json.load(f)

    c2w_list = []
    fxfycxcy_list = []

    for i, cam in tqdm(enumerate(params["frames"]), desc="Rendering progress"):
        w2c = np.array(cam["w2c"])
        c2w = np.linalg.inv(w2c)
        c2w = torch.from_numpy(c2w.astype(np.float32)).cuda()
        c2w_list.append(c2w)

        h = cam["h"]
        w = cam["w"]

        fx = cam["fx"]/h
        fy = cam["fy"]/w
        cx = cam["cx"]/h
        cy = cam["cy"]/w

        fxfycxcy = torch.tensor([fx, fy, cx, cy], dtype=torch.float32).cuda()
        fxfycxcy_list.append(fxfycxcy)

    c2w_all = torch.stack(c2w_list).reshape(-1,4,4,4)
    fxfycxcy_all = torch.stack(fxfycxcy_list).reshape(-1,4,4)

    pc._xyz.requires_grad = True
    pc._features.requires_grad = True
    pc._scaling.requires_grad = True
    pc._rotation.requires_grad = True
    pc._opacity.requires_grad = True
    xyz = pc._xyz
    features = pc._features
    scaling = pc._scaling 
    opacity = pc._opacity
    rotation = pc._rotation

    im_defer = deferred_backprop_gaussian_render(xyz, features, scaling, rotation, opacity, 512, 512, c2w_all, fxfycxcy_all, 512, pc)
    loss_defer = im_defer.sum()
    loss_defer.backward()


    bs, vs = fxfycxcy_all.shape[:2] 
    renderings = torch.zeros(bs, vs, 3, 512, 512, dtype=torch.float32, device=c2w_all.device)
    pc_new = GaussianModel(sh_degree=0)
    pc_new.load_ply_new_defered('/nas2/zifan/temp/debug1.ply', batch_size=4)

    pc_new._xyz.requires_grad = True
    pc_new._features.requires_grad = True
    pc_new._scaling.requires_grad = True
    pc_new._rotation.requires_grad = True
    pc_new._opacity.requires_grad = True
    new_xyz = pc_new._xyz
    new_features = pc_new._features
    new_scaling = pc_new._scaling
    new_rotation = pc_new._rotation
    new_opacity = pc_new._opacity 
    for idx in range(bs):
        pc_new = pc_new.set_data(new_xyz[idx], new_features[idx], new_scaling[idx], new_rotation[idx], new_opacity[idx])
        for vidx in range(vs):
            renderings[idx, vidx] = render_opencv_cam(pc_new, 512, 512, c2w_all[idx, vidx], fxfycxcy_all[idx, vidx])['render']
    loss_render = renderings.sum()
    loss_render.backward()
    import ipdb;ipdb.set_trace()
