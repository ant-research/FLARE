import torch
from torch.utils.checkpoint import _get_autocast_kwargs
from dust3r.renderers.gaussian_utils import GRMGaussianModel, render_image


class DeferredBackpropGaussianRenderer(torch.autograd.Function):
    @staticmethod
    def forward(ctx, xyz, feature, scaling, rotation, opacity, height, width, C2W, K, patch_size, opatch_size,
                opatch_center, sh_degree, scaling_kwargs):
        """
        Forward rendering.
        """
        assert (xyz.dim() == 3) and (
            feature.dim() == 3
        ) and (scaling.dim() == 3) and (rotation.dim() == 3), f"xyz: {xyz.shape}, feature: {feature.shape}, " \
                                                              f"scaling: {scaling.shape}, rotation: {rotation.shape}," \
                                                              f" opacity: {opacity.shape}"
        assert height % patch_size == 0 and width % patch_size == 0, f'patch_size must be divided by H and W!'

        ctx.save_for_backward(xyz, feature, scaling, rotation, opacity)  # save tensors for backward
        ctx.height = height
        ctx.width = width
        ctx.C2W = C2W
        ctx.K = K
        ctx.opatch_size = opatch_size
        ctx.opatch_center = opatch_center
        ctx.sh_degree = sh_degree
        ctx.scaling_kwargs = scaling_kwargs
        if scaling_kwargs['type'] == 'precomp':
            ctx.patch_size = [height, width]
        else:
            ctx.patch_size = [height, width]

        ctx.gpu_autocast_kwargs, ctx.cpu_autocast_kwargs = _get_autocast_kwargs()
        ctx.manual_seeds = []

        with torch.no_grad(), torch.cuda.amp.autocast(
            **ctx.gpu_autocast_kwargs
        ), torch.cpu.amp.autocast(**ctx.cpu_autocast_kwargs):
            device = C2W.device
            b, v = C2W.shape[:2]
            colors = torch.zeros(b, v, 3, height, width, device=device)
            depths = torch.zeros(b, v, 1, height, width, device=device)
            alphas = torch.zeros(b, v, 1, height, width, device=device)
            patchs = None
            if opatch_size > 0:
                patchs = torch.zeros(b, v, 3, opatch_size, opatch_size, device=device)

            for i in range(b):

                ctx.manual_seeds.append([])
                pc = GRMGaussianModel(sh_degree=ctx.sh_degree, xyz=xyz[i], feature=feature[i], opacity=opacity[i],
                                      scaling=scaling[i], rotation=rotation[i], scaling_kwargs=ctx.scaling_kwargs)

                # pc = GRMGaussianModel(sh_degree=0)

                # pc.load_ply('/home/yhxu/code/image2nerf/image2nerf.ply')
                # pc.set_device(device='cuda:0')

                for j in range(v):
                    # import time
                    # torch.cuda.synchronize()
                    # begin = time.time()

                    K_ij = K[i, j]
                    fx, fy, cx, cy = K_ij[0], K_ij[1], K_ij[2], K_ij[3]
                    for m in range(0, ctx.width//ctx.patch_size[1]):
                        for n in range(0, ctx.height //ctx.patch_size[0]):
                            seed = torch.randint(0, 2**32, (1,)).long().item()
                            ctx.manual_seeds[-1].append(seed)
                            # implement how to transform intrinsic
                            center_x = (m*ctx.patch_size[1] + ctx.patch_size[1]//2) / ctx.width
                            center_y = (n*ctx.patch_size[0] + ctx.patch_size[0]//2) / ctx.height
                            
                            scale_x = ctx.width // ctx.patch_size[1]
                            scale_y = ctx.height // ctx.patch_size[0]
                            trans_x = 0.5 - scale_x * center_x 
                            trans_y = 0.5 - scale_y * center_y 
                            new_fx = scale_x * fx 
                            new_fy = scale_y * fy
                            new_cx = scale_x * cx + trans_x
                            new_cy = scale_y * cy + trans_y
                            new_K_ij = torch.eye(4).to(K_ij)
                            new_K_ij[0][0], new_K_ij[1][1], new_K_ij[0][2], new_K_ij[1][2], new_K_ij[2][2] = new_fx, new_fy, new_cx, new_cy, 1
                            render_results = render_image(pc, new_K_ij, C2W[i, j], ctx.patch_size[0], ctx.patch_size[1])
                            colors[i, j, :, n*ctx.patch_size[0]:(n+1)*ctx.patch_size[0], m*ctx.patch_size[1]:(m+1)*ctx.patch_size[1]] = render_results["image"]
                            depths[i, j, :, n*ctx.patch_size[0]:(n+1)*ctx.patch_size[0], m*ctx.patch_size[1]:(m+1)*ctx.patch_size[1]] = render_results["depth"]
                            alphas[i, j, :, n*ctx.patch_size[0]:(n+1)*ctx.patch_size[0], m*ctx.patch_size[1]:(m+1)*ctx.patch_size[1]] = render_results["alpha"]

                    # if ctx.opatch_size > 0.:
                    #     center_x = opatch_center[i, j, 0].item() / ctx.width
                    #     center_y = opatch_center[i, j, 1].item() / ctx.height

                    #     scale_x = ctx.width // ctx.opatch_size
                    #     scale_y = ctx.height // ctx.opatch_size
                    #     trans_x = 0.5 - scale_x * center_x
                    #     trans_y = 0.5 - scale_y * center_y

                    #     new_fx = scale_x * fx
                    #     new_fy = scale_y * fy
                    #     new_cx = scale_x * cx + trans_x
                    #     new_cy = scale_y * cy + trans_y

                    #     new_K_ij = torch.zeros_like(K_ij)
                    #     new_K_ij[0][0], new_K_ij[1][1], new_K_ij[0][2], new_K_ij[1][2], new_K_ij[2][2] = new_fx, new_fy, new_cx, new_cy, 1

                    #     render_results = render_image(pc, new_K_ij, C2W[i, j], ctx.patch_size, ctx.patch_size)
                    #     patchs[i, j] = render_results["image"]
        
                    # import time
                    # torch.cuda.synchronize()
                    # end = time.time()
                    # print('gs time', end - begin)
        return colors, depths, alphas, patchs

    @staticmethod
    def backward(ctx, grad_colors, grad_depths, grad_alphas, grad_patchs):
        """
        Backward process.
        """

        xyz, feature, scaling, rotation, opacity = ctx.saved_tensors

        xyz_nosync = xyz.detach().clone()
        xyz_nosync.requires_grad = True
        xyz_nosync.grad = None

        feature_nosync = feature.detach().clone()
        feature_nosync.requires_grad = True
        feature_nosync.grad = None

        scaling_nosync = scaling.detach().clone()
        scaling_nosync.requires_grad = True
        scaling_nosync.grad = None

        rotation_nosync = rotation.detach().clone()
        rotation_nosync.requires_grad = True
        rotation_nosync.grad = None

        opacity_nosync = opacity.detach().clone()
        opacity_nosync.requires_grad = True
        opacity_nosync.grad = None

        with torch.enable_grad(), torch.cuda.amp.autocast(
            **ctx.gpu_autocast_kwargs
        ), torch.cpu.amp.autocast(**ctx.cpu_autocast_kwargs):
            b, v = ctx.C2W.shape[:2]

            for i in range(b):
                ctx.manual_seeds.append([])
                pc = GRMGaussianModel(sh_degree=ctx.sh_degree, xyz=xyz_nosync[i], feature=feature_nosync[i],
                                      opacity=opacity_nosync[i], scaling=scaling_nosync[i], rotation=rotation_nosync[i],
                                      scaling_kwargs=ctx.scaling_kwargs)

                for j in range(v):
                    # K_ij = ctx.K[i, j]
                    # fx, fy, cx, cy = K_ij[0][0], K_ij[1][1], K_ij[0][2], K_ij[1][2]
                    K_ij = ctx.K[i, j]
                    fx, fy, cx, cy = K_ij[0], K_ij[1], K_ij[2], K_ij[3]
                    for m in range(0, ctx.width//ctx.patch_size[1]):
                        for n in range(0, ctx.height //ctx.patch_size[0]):
                            grad_colors_split = grad_colors[i, j, :, n*ctx.patch_size[0]:(n+1)*ctx.patch_size[0], m*ctx.patch_size[1]:(m+1)*ctx.patch_size[1]]
                            grad_depths_split = grad_depths[i, j, :, n*ctx.patch_size[0]:(n+1)*ctx.patch_size[0], m*ctx.patch_size[1]:(m+1)*ctx.patch_size[1]]
                            grad_alphas_split = grad_alphas[i, j, :, n*ctx.patch_size[0]:(n+1)*ctx.patch_size[0], m*ctx.patch_size[1]:(m+1)*ctx.patch_size[1]]

                            seed = torch.randint(0, 2**32, (1,)).long().item()
                            ctx.manual_seeds[-1].append(seed)
                            # Transform intrinsics
                            center_x = (m*ctx.patch_size[1] + ctx.patch_size[1]//2) / ctx.width
                            center_y = (n*ctx.patch_size[0] + ctx.patch_size[0]//2) / ctx.height

                            scale_x = ctx.width // ctx.patch_size[1]
                            scale_y = ctx.height // ctx.patch_size[0]
                            trans_x = 0.5 - scale_x * center_x
                            trans_y = 0.5 - scale_y * center_y

                            new_fx = scale_x * fx
                            new_fy = scale_y * fy
                            new_cx = scale_x * cx + trans_x
                            new_cy = scale_y * cy + trans_y

                            new_K_ij = torch.eye(4).to(K_ij)
                            new_K_ij[0][0], new_K_ij[1][1], new_K_ij[0][2], new_K_ij[1][2], new_K_ij[2][2] = new_fx, new_fy, new_cx, new_cy, 1

                            render_results = render_image(pc, new_K_ij, ctx.C2W[i, j], ctx.patch_size[0], ctx.patch_size[1])
                            color_split = render_results["image"]
                            depth_split = render_results["depth"]
                            alpha_split = render_results["alpha"]

                            render_split = torch.cat([color_split, depth_split, alpha_split], dim=0)
                            grad_split = torch.cat([grad_colors_split, grad_depths_split, grad_alphas_split], dim=0) 
                            render_split.backward(grad_split)

                    if ctx.opatch_size > 0.:
                        grad_patch = grad_patchs[i, j]

                        seed = torch.randint(0, 2**32, (1,)).long().item()
                        ctx.manual_seeds[-1].append(seed)
                        # Transform intrinsics
                        center_x = ctx.opatch_center[i, j, 0].item() / ctx.width
                        center_y = ctx.opatch_center[i, j, 1].item() / ctx.height

                        scale_x = ctx.width // ctx.opatch_size
                        scale_y = ctx.height // ctx.opatch_size
                        trans_x = 0.5 - scale_x * center_x
                        trans_y = 0.5 - scale_y * center_y

                        new_fx = scale_x * fx
                        new_fy = scale_y * fy
                        new_cx = scale_x * cx + trans_x
                        new_cy = scale_y * cy + trans_y

                        new_K_ij = torch.zeros_like(K_ij)
                        new_K_ij[0][0], new_K_ij[1][1], new_K_ij[0][2], new_K_ij[1][2], new_K_ij[2][2] = new_fx, new_fy, new_cx, new_cy, 1

                        render_results = render_image(pc, new_K_ij, ctx.C2W[i, j], ctx.patch_size[0], ctx.patch_size[1])
                        patch_color = render_results["image"]
                        patch_color.backward(grad_patch)

        return xyz_nosync.grad, feature_nosync.grad, scaling_nosync.grad, rotation_nosync.grad, opacity_nosync.grad, None, None, None, None, None, None, None, None, None


def deferred_backprop_gaussian_renderer(
        xyz, feature, scaling, rotation, opacity, height, width, C2W, K, patch_size, opatch_size, opatch_center, sh_degree, scaling_kwargs
):
    return DeferredBackpropGaussianRenderer.apply(
        xyz, feature, scaling, rotation, opacity, height, width, C2W, K, patch_size, opatch_size, opatch_center, sh_degree, scaling_kwargs
    )
    
    

def DeferredBackpropGaussianRenderer_wodb(xyz, feature, scaling, rotation, opacity, height, width, C2W, K, patch_size, opatch_size,
                opatch_center, sh_degree, scaling_kwargs):
    """
    Forward rendering.
    """
    assert (xyz.dim() == 3) and (
        feature.dim() == 3
    ) and (scaling.dim() == 3) and (rotation.dim() == 3), f"xyz: {xyz.shape}, feature: {feature.shape}, " \
                                                            f"scaling: {scaling.shape}, rotation: {rotation.shape}," \
                                                            f" opacity: {opacity.shape}"
    assert height % patch_size == 0 and width % patch_size == 0, f'patch_size must be divided by H and W!'

    device = C2W.device
    b, v = C2W.shape[:2]
    # colors = torch.zeros(b, v, 3, height, width, device=device)
    # depths = torch.zeros(b, v, 1, height, width, device=device)
    # alphas = torch.zeros(b, v, 1, height, width, device=device)
    patchs = None
    # if opatch_size > 0:
    #     patchs = torch.zeros(b, v, 3, opatch_size, opatch_size, device=device)
    colors_list = []
    depths_list = []
    alphas_list = []
    for i in range(b):
        pc = GRMGaussianModel(sh_degree=sh_degree, xyz=xyz[i], feature=feature[i], opacity=opacity[i],
                                scaling=scaling[i], rotation=rotation[i], scaling_kwargs=scaling_kwargs)
        for j in range(v):
            K_ij = K[i, j]
            fx, fy, cx, cy = K_ij[0], K_ij[1], K_ij[2], K_ij[3]
            new_K_ij = torch.eye(4).to(K_ij)
            new_K_ij[0][0], new_K_ij[1][1], new_K_ij[0][2], new_K_ij[1][2], new_K_ij[2][2] = fx, fy, cx, cy, 1
            render_results = render_image(pc, new_K_ij, C2W[i, j], height, width)
            colors = render_results["image"]
            depths = render_results["depth"]
            alphas = render_results["alpha"]
            colors_list.append(colors)
            depths_list.append(depths)
            alphas_list.append(alphas)
    colors = torch.stack(colors_list, dim=0)
    depths = torch.stack(depths_list, dim=0)
    alphas = torch.stack(alphas_list, dim=0)
    
    return colors, depths, alphas, None




def deferred_backprop_gaussian_renderer_wodb(
        xyz, feature, scaling, rotation, opacity, height, width, C2W, K, patch_size, opatch_size, opatch_center, sh_degree, scaling_kwargs
):
    return DeferredBackpropGaussianRenderer_wodb(
        xyz, feature, scaling, rotation, opacity, height, width, C2W, K, patch_size, opatch_size, opatch_center, sh_degree, scaling_kwargs
    )


