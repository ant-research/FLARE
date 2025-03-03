import torch
from dust3r.renderers.gaussian_utils import GaussianModel
import math
from gsplat.rendering import rasterization

def render_image(pc, K, RT, height, width, bg_color=(0.0, 0.0, 0.0), scaling_modifier=1.0,debug=False):
    screenspace_points = (torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0)
    bg_color = torch.tensor(bg_color, dtype=torch.float32, device=K.device)
    screenspace_points.retain_grad()
    means3D = pc.get_xyz
    means2D = screenspace_points
    opacity = pc.get_opacity
    scales = pc.get_scaling
    rotations = pc.get_rotation
    colors_precomp = pc.get_features
    K[:1] = K[:1] * width
    K[1:2] = K[1:2] * height
    if colors_precomp.shape[1] == 3:
        sh_degree = None
    else:
        sh_degree = int(math.sqrt(colors_precomp.shape[1])) - 1
    render_colors, render_alphas, meta = rasterization(means = means3D, quats= rotations, scales = scales, opacities = opacity.squeeze(), colors = colors_precomp, sh_degree=sh_degree, viewmats = RT[None].inverse(), Ks=K[None][:,:3,:3], width=width, height=height, near_plane=0.00001, render_mode="RGB+D", radius_clip=0.1)#, rasterize_mode="antialiased")
    render_depths = render_colors.permute(0, 3, 1, 2).squeeze()[3:4, :, :]
    render_colors = render_colors.permute(0, 3, 1, 2).squeeze()[:3, :, :]
    render_alphas = render_alphas.permute(0, 3, 1, 2)[0]
    return {
        "image": render_colors,
        "alpha": render_alphas,
        "depth": render_depths,
    }

class GaussianRenderer:
    def __init__(self,
                 height=512,
                 width=512,
                 sh_degree=0,
                 bg_color=(0., 0., 0.),
                 scaling_modifier=1.0,
                 gs_kwargs=dict(),
                 ):
        self.height = height 
        self.width = width
        self.sh_degree = sh_degree
        self.bg_color = bg_color
        self.scaling_modifier = scaling_modifier
        self.gs_kwargs = gs_kwargs

    def __call__(self, gs_params, Ks, RTs):
        Ks = Ks.to(torch.float32)
        RTs = RTs.to(torch.float32)
        device = RTs.device
        b, v = RTs.shape[:2]
        patchs = None
        colors_list = []
        depths_list = []
        alphas_list = []
        xyz = gs_params['xyz']
        feature = gs_params['feature']
        opacity = gs_params['opacity']
        scaling = gs_params['scaling']
        rotation = gs_params['rotation']
        scaling_kwargs = self.gs_kwargs
        for i in range(b):
            pc = GaussianModel(sh_degree=self.sh_degree, xyz=xyz[i], feature=feature[i], opacity=opacity[i],
                                    scaling=scaling[i], rotation=rotation[i], scaling_kwargs=scaling_kwargs)
            for j in range(v):
                K_ij = Ks[i, j]
                fx, fy, cx, cy = K_ij[0], K_ij[1], K_ij[2], K_ij[3]
                new_K_ij = torch.eye(4).to(K_ij)
                new_K_ij[0][0], new_K_ij[1][1], new_K_ij[0][2], new_K_ij[1][2], new_K_ij[2][2] = fx, fy, cx, cy, 1
                render_results = render_image(pc, new_K_ij, RTs[i, j], self.height, self.width)
                colors = render_results["image"]
                depths = render_results["depth"]
                alphas = render_results["alpha"]
                colors_list.append(colors)
                depths_list.append(depths)
                alphas_list.append(alphas)
        colors = torch.stack(colors_list, dim=0)
        depths = torch.stack(depths_list, dim=0)
        alphas = torch.stack(alphas_list, dim=0)
        ret = {'image': colors, 'alpha': alphas, 'depth': depths}
        return ret