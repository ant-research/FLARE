# Copyright (C) 2024-present Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).
#
# --------------------------------------------------------
# base class for implementing datasets
# --------------------------------------------------------
import PIL
import numpy as np
import torch

from dust3r.datasets.base.easy_dataset import EasyDataset
from dust3r.datasets.utils.transforms import ImgNorm
from dust3r.utils.geometry import depthmap_to_absolute_camera_coordinates, geotrf
import dust3r.datasets.utils.cropping as cropping
import random
import copy
from scipy.spatial.transform import Rotation
import torchvision.transforms as transforms
from dust3r.utils.geometry import inv, geotrf
import cv2



class BaseStereoViewDataset_test (EasyDataset):
    """ Define all basic options.

    Usage:
        class MyDataset (BaseStereoViewDataset):
            def _get_views(self, idx, rng):
                # overload here
                views = []
                views.append(dict(img=, ...))
                return views
    """

    def __init__(self, *,  # only keyword arguments
                 split=None,
                 resolution=None,  # square_size or (width, height) or list of [(width,height), ...]
                 transform=ImgNorm,
                 aug_crop=False,
                 seed=None,
                 num_views=8,
                 gt_num_image=0,
                 aug_monocular=False,
                 aug_portrait_or_landscape=False,
                 aug_rot90=False,
                 aug_swap=False,
                 only_pose=False,
                sequential_input=False,
                overfit=False,
                caculate_mask=False):
        self.sequential_input = sequential_input
        self.split = split
        self.num_image = num_views
        self._set_resolutions(resolution)
        self.gt_num_image=gt_num_image
        self.aug_monocular=aug_monocular
        self.aug_portrait_or_landscape = aug_portrait_or_landscape
        self.transform = transform
        self.transform_org = transforms.Compose([transform for transform in transform.transforms if type(transform).__name__ != 'ColorJitter'])
        self.aug_rot90 = aug_rot90
        self.aug_swap = aug_swap
        self.only_pose = only_pose
        self.overfit = overfit
        self.rendering = False
        self.caculate_mask = caculate_mask
        if isinstance(transform, str):
            transform = eval(transform)

        self.aug_crop = aug_crop
        self.seed = seed
        self.kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(9, 9))

    def __len__(self):
        return len(self.scenes)

    # def sequential_sample(self, im_start, last, interal):
    #     im_list = [im_start + i * interal + random.choice(list(range(interal))) for i in range(self.num_image)]
    #     im_list += [random.choice(im_list) + random.choice(list(range(interal))) for _ in range(self.gt_num_image)]
    #     return im_list
    def sequential_sample(self, im_start, last, interal):
        im_list = [
            im_start + i * interal + random.choice(list(range(-interal//2, interal//2))) 
            for i in range(self.num_image)
        ]
        im_list += [
            random.choice(im_list) + random.choice(list(range(-interal//2, interal//2)))
            for _ in range(self.gt_num_image)
        ]
        return im_list
        
    def get_stats(self):
        return f"{len(self)} pairs"

    def __repr__(self):
        resolutions_str = '['+';'.join(f'{w}x{h}' for w, h in self._resolutions)+']'
        return f"""{type(self).__name__}({self.get_stats()},
            {self.split=},
            {self.seed=},
            resolutions={resolutions_str},
            {self.transform=})""".replace('self.', '').replace('\n', '').replace('   ', '')

    def _get_views(self, idx, resolution, rng):
        raise NotImplementedError()

    def _swap_view_aug(self, views):
        # if self._rng.random() < 0.5:
            # views.reverse()
        return random.shuffle(views)

    def __getitem__(self, idx):
        if isinstance(idx, tuple):
            # the idx is specifying the aspect-ratio
            idx, ar_idx = idx
        else:
            assert len(self._resolutions) == 1
            ar_idx = 0

        # set-up the rng
        if self.seed:  # reseed for each __getitem__
            self._rng = np.random.default_rng(seed=self.seed + idx)
        elif not hasattr(self, '_rng'):
            seed = torch.initial_seed()  # this is different for each dataloader process
            self._rng = np.random.default_rng(seed=seed)

        # over-loaded code
        resolution = self._resolutions[ar_idx]  # DO NOT CHANGE THIS (compatible with BatchedRandomSampler)
        flag = False
        i = 0
        # while flag == False and i < 100:
        #     try:
        #         views = self._get_views(idx, resolution, self._rng)
        #         flag = True
        #     except:
        #         flag = False
        #         i += 1

        views = self._get_views(idx, resolution, self._rng)

        # assert len(views) == self.num_image + self.gt_num_image
        if self.only_pose == True: 
        # check data-types
            for view in views:
                # transpose to make sure all views are the same size
                # this allows to check whether the RNG is is the same state each time
                view['rng'] = int.from_bytes(self._rng.bytes(4), 'big')
            return views
        else:
            for v, view in enumerate(views):
                assert 'pts3d' not in view, f"pts3d should not be there, they will be computed afterwards based on intrinsics+depthmap for view {view_name(view)}"
                view['idx'] = (idx, ar_idx, v)
                # encode the image
                width, height = view['img'].size
                view['true_shape'] = np.int32((height, width))
                view['img'] = self.transform_org(view['img'])
                view['img_org' ] = self.transform_org(view['img_org'])
                if 'depth_anything' not in view:
                    view['depth_anything'] = np.zeros_like(view['depthmap'])
                # if view['img_org'].shape[1] != 224:
                # print(view['img_org' ].shape)
                # print(view['img'].shape)
                assert 'camera_intrinsics' in view
                if 'camera_pose' not in view:
                    view['camera_pose'] = np.full((4, 4), np.nan, dtype=np.float32)
                else:
                    assert np.isfinite(view['camera_pose']).all(), f'NaN in camera pose for view {view_name(view)}'
                assert 'pts3d' not in view
                assert 'valid_mask' not in view
                assert np.isfinite(view['depthmap']).all(), f'NaN in depthmap for view {view_name(view)}'
                pts3d, valid_mask = depthmap_to_absolute_camera_coordinates(**view)

                view['pts3d'] = pts3d
                view['valid_mask'] = valid_mask & np.isfinite(pts3d).all(axis=-1)
                # print(view['pts3d'].shape)
                # print(view['valid_mask'].shape)

                # check all datatypes
                for key, val in view.items():
                    res, err_msg = is_good_type(key, val)
                    assert res, f"{err_msg} with {key}={val} for view {view_name(view)}"
                K = view['camera_intrinsics']
            

            for view in views:
                fxfycxcy = view['fxfycxcy'].copy()
                H, W = view['img'].shape[1:]
                fxfycxcy[0] = fxfycxcy[0] * W
                fxfycxcy[1] = fxfycxcy[1] * H
                fxfycxcy[2] = fxfycxcy[2] * W
                fxfycxcy[3] = fxfycxcy[3] * H
                view['fxfycxcy_unorm'] = fxfycxcy

            # last thing done!
            for view in views:
                view['render_mask'] = np.ones((view['img'].shape[1], view['img'].shape[2]), dtype=np.uint8) > 0.1

            for view in views:
                # transpose to make sure all views are the same size
                transpose_to_landscape(view)
                if 'sky_mask' in view:
                    view.pop('sky_mask')
                # this allows to check whether the RNG is is the same state each time
                view['rng'] = int.from_bytes(self._rng.bytes(4), 'big')
            return views

    def _set_resolutions(self, resolutions):
        assert resolutions is not None, 'undefined resolution'

        if not isinstance(resolutions, list):
            resolutions = [resolutions]

        self._resolutions = []
        for resolution in resolutions:
            if isinstance(resolution, int):
                width = height = resolution
            else:
                width, height = resolution
            assert isinstance(width, int), f'Bad type for {width=} {type(width)=}, should be int'
            assert isinstance(height, int), f'Bad type for {height=} {type(height)=}, should be int'
            assert width >= height
            self._resolutions.append((width, height))

    def _crop_resize_if_necessary(self, image, depthmap, intrinsics, resolution, rng=None, info=None, depth_anything=None):
        """ This function:
            - first downsizes the image with LANCZOS inteprolation,
                which is better than bilinear interpolation in
        """
        if not isinstance(image, PIL.Image.Image):
            image = PIL.Image.fromarray(image)

        # transpose the resolution if necessary
        W, H = image.size  # new size
        assert resolution[0] >= resolution[1]
        if H > 1.1 * W:
            # image is portrait mode
            resolution = resolution[::-1]
        elif 0.7 < H / W < 1.3 and resolution[0] != resolution[1] and self.aug_portrait_or_landscape:
            # image is square, so we chose (portrait, landscape) randomly
            if rng.integers(2):
                resolution = resolution[::-1]
        # resolution = resolution[::-1]
        # high-quality Lanczos down-scaling
        target_resolution = np.array(resolution)
        if depth_anything is not None:
            image, depthmap, intrinsics, depth_anything = cropping.rescale_image_depthmap(image, depthmap, intrinsics, target_resolution, depth_anything=depth_anything)
        else:
            image, depthmap, intrinsics = cropping.rescale_image_depthmap(image, depthmap, intrinsics, target_resolution)

        # actual cropping (if necessary) with bilinear interpolation
        offset_factor = 0.5
        intrinsics2 = cropping.camera_matrix_of_crop(intrinsics, image.size, resolution, offset_factor=offset_factor)
        crop_bbox = cropping.bbox_from_intrinsics_in_out(intrinsics, intrinsics2, resolution)
        if depth_anything is not None:
            image, depthmap, intrinsics2, depth_anything = cropping.crop_image_depthmap(image, depthmap, intrinsics, crop_bbox, depth_anything=depth_anything)
            return image, depthmap, intrinsics2, depth_anything
        else:
            image, depthmap, intrinsics2 = cropping.crop_image_depthmap(image, depthmap, intrinsics, crop_bbox)
            return image, depthmap, intrinsics2

    def _crop_resize_if_necessary_test(self, image, depthmap, intrinsics, resolution, rng=None, info=None, depth_anything=None):
        """ This function:
            - first downsizes the image with LANCZOS inteprolation,
                which is better than bilinear interpolation in
        """
        if not isinstance(image, PIL.Image.Image):
            image = PIL.Image.fromarray(image)

        # transpose the resolution if necessary
        W, H = image.size  # new size
        assert resolution[0] >= resolution[1]
        if H > 1.1 * W:
            # image is portrait mode
            resolution = resolution[::-1]

        # resolution = resolution[::-1]
        # high-quality Lanczos down-scaling
        target_resolution = np.array(resolution)
        if depth_anything is not None:
            image, depthmap, intrinsics, depth_anything = cropping.rescale_image_depthmap(image, depthmap, intrinsics, target_resolution, depth_anything=depth_anything)
        else:
            image, depthmap, intrinsics = cropping.rescale_image_depthmap(image, depthmap, intrinsics, target_resolution)

        # actual cropping (if necessary) with bilinear interpolation
        offset_factor = 0.5
        intrinsics2 = cropping.camera_matrix_of_crop(intrinsics, image.size, resolution, offset_factor=offset_factor)
        crop_bbox = cropping.bbox_from_intrinsics_in_out(intrinsics, intrinsics2, resolution)
        if depth_anything is not None:
            image, depthmap, intrinsics2, depth_anything = cropping.crop_image_depthmap(image, depthmap, intrinsics, crop_bbox, depth_anything=depth_anything)
        else:
            image, depthmap, intrinsics2 = cropping.crop_image_depthmap(image, depthmap, intrinsics, crop_bbox)

        return image, depthmap, intrinsics2

def rotate_90(views, k=1):
    # print('rotation =', k)
    RT = np.eye(4, dtype=np.float32)
    RT[:3, :3] = Rotation.from_euler('z', 90 * k, degrees=True).as_matrix()

    for view in views:
        view['img'] = torch.rot90(view['img'], k=k, dims=(-2, -1))  # WARNING!! dims=(-1,-2) != dims=(-2,-1)
        view['depthmap'] = np.rot90(view['depthmap'], k=k).copy()
        view['camera_pose'] = view['camera_pose'] @ RT

        RT2 = np.eye(3, dtype=np.float32)
        RT2[:2, :2] = RT[:2, :2] * ((1, -1), (-1, 1))
        H, W = view['depthmap'].shape
        if k % 4 == 0:
            pass
        elif k % 4 == 1:
            # top-left (0,0) pixel becomes (0,H-1)
            RT2[:2, 2] = (0, H - 1)
        elif k % 4 == 2:
            # top-left (0,0) pixel becomes (W-1,H-1)
            RT2[:2, 2] = (W - 1, H - 1)
        elif k % 4 == 3:
            # top-left (0,0) pixel becomes (W-1,0)
            RT2[:2, 2] = (W - 1, 0)
        else:
            raise ValueError(f'Bad value for {k=}')

        view['camera_intrinsics'][:2, 2] = geotrf(RT2, view['camera_intrinsics'][:2, 2])
        if k % 2 == 1:
            K = view['camera_intrinsics']
            np.fill_diagonal(K, K.diagonal()[[1, 0, 2]])

        pts3d, valid_mask = depthmap_to_absolute_camera_coordinates(**view)
        view['pts3d'] = pts3d
        view['valid_mask'] = np.rot90(view['valid_mask'], k=k).copy()
        view['true_shape'] = np.int32((H, W))
        intrinsics = view['camera_intrinsics']
        fxfycxcy = np.array([intrinsics[0, 0]/W, intrinsics[1, 1]/H, intrinsics[0,2]/W, intrinsics[1,2]/H]).astype(np.float32)
        view['fxfycxcy'] = fxfycxcy

def reciprocal_1d(corres_1_to_2, corres_2_to_1, shape1, shape2, ret_recip=False):
    is_reciprocal1 = np.abs(unravel_xy(corres_2_to_1[corres_1_to_2], shape1) - unravel_xy(np.arange(len(corres_1_to_2)), shape1)).sum(-1) < 4
    pos1 = is_reciprocal1.nonzero()[0]
    pos2 = corres_1_to_2[pos1]
    if ret_recip:
        return is_reciprocal1, pos1, pos2
    return pos1, pos2


def reproject_view(pts3d, view2):
    shape = view2['pts3d'].shape[:2]
    return reproject(pts3d, view2['camera_intrinsics'], inv(view2['camera_pose']), shape)


def reproject(pts3d, K, world2cam, shape):
    H, W, THREE = pts3d.shape
    assert THREE == 3

    # reproject in camera2 space
    with np.errstate(divide='ignore', invalid='ignore'):
        pos = geotrf(K @ world2cam[:3], pts3d, norm=1, ncol=2)

    # quantize to pixel positions
    return (H, W), ravel_xy(pos, shape)


def ravel_xy(pos, shape):
    H, W = shape
    with np.errstate(invalid='ignore'):
        qx, qy = pos.reshape(-1, 2).round().astype(np.int32).T
    quantized_pos = qx.clip(min=0, max=W - 1, out=qx) + W * qy.clip(min=0, max=H - 1, out=qy)
    return quantized_pos


def unravel_xy(pos, shape):
    # convert (x+W*y) back to 2d (x,y) coordinates
    return np.unravel_index(pos, shape)[0].base[:, ::-1].copy()


class BaseStereoViewDataset (EasyDataset):
    """ Define all basic options.

    Usage:
        class MyDataset (BaseStereoViewDataset):
            def _get_views(self, idx, rng):
                # overload here
                views = []
                views.append(dict(img=, ...))
                return views
    """

    def __init__(self, *,  # only keyword arguments
                 split=None,
                 resolution=None,  # square_size or (width, height) or list of [(width,height), ...]
                 transform=ImgNorm,
                 aug_crop=False,
                 seed=None,
                 num_views=8,
                 gt_num_image=0,
                 aug_monocular=False,
                 aug_portrait_or_landscape=True,
                 aug_rot90=False,
                 aug_swap=False,
                 only_pose=False,
                sequential_input=False,
                overfit=False,
                caculate_mask=False):
        self.sequential_input = sequential_input
        self.split = split
        self.num_image = num_views
        self._set_resolutions(resolution)
        self.gt_num_image=gt_num_image
        self.aug_monocular=aug_monocular
        self.aug_portrait_or_landscape = aug_portrait_or_landscape
        self.transform = transform
        self.transform_org = transforms.Compose([transform for transform in transform.transforms if type(transform).__name__ != 'ColorJitter'])
        self.aug_rot90 = aug_rot90
        self.aug_swap = aug_swap
        self.only_pose = only_pose
        self.overfit = overfit
        self.rendering = False
        self.caculate_mask = caculate_mask
        if isinstance(transform, str):
            transform = eval(transform)

        self.aug_crop = aug_crop
        self.seed = seed
        self.kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(9, 9))

    def __len__(self):
        return len(self.scenes)

    # def sequential_sample(self, im_start, last, interal):
    #     im_list = [im_start + i * interal + random.choice(list(range(interal))) for i in range(self.num_image)]
    #     im_list += [random.choice(im_list) + random.choice(list(range(interal))) for _ in range(self.gt_num_image)]
    #     return im_list
    def sequential_sample(self, im_start, last, interal):
        im_list = [
            im_start + i * interal + random.choice(list(range(-interal//2, interal//2))) 
            for i in range(self.num_image)
        ]
        im_list += [
            random.choice(im_list) + random.choice(list(range(-interal//2, interal//2)))
            for _ in range(self.gt_num_image)
        ]
        return im_list
        
    def get_stats(self):
        return f"{len(self)} pairs"

    def __repr__(self):
        resolutions_str = '['+';'.join(f'{w}x{h}' for w, h in self._resolutions)+']'
        return f"""{type(self).__name__}({self.get_stats()},
            {self.split=},
            {self.seed=},
            resolutions={resolutions_str},
            {self.transform=})""".replace('self.', '').replace('\n', '').replace('   ', '')

    def _get_views(self, idx, resolution, rng):
        raise NotImplementedError()

    def _swap_view_aug(self, views):
        # if self._rng.random() < 0.5:
            # views.reverse()
        return random.shuffle(views)

    def __getitem__(self, idx):
        if isinstance(idx, tuple):
            # the idx is specifying the aspect-ratio
            idx, ar_idx = idx
        else:
            assert len(self._resolutions) == 1
            ar_idx = 0

        # set-up the rng
        if self.seed:  # reseed for each __getitem__
            self._rng = np.random.default_rng(seed=self.seed + idx)
        elif not hasattr(self, '_rng'):
            seed = torch.initial_seed()  # this is different for each dataloader process
            self._rng = np.random.default_rng(seed=seed)

        # over-loaded code
        resolution = self._resolutions[ar_idx]  # DO NOT CHANGE THIS (compatible with BatchedRandomSampler)
        flag = False
        i = 0
        # views = self._get_views(idx, resolution, self._rng)
        while flag == False and i < 1000:
            try:
                views = self._get_views(idx, resolution, self._rng)
                flag = True
            except:
                flag = False
                i += 1

        # assert len(views) == self.num_image + self.gt_num_image
        if self.only_pose == True: 
        # check data-types
            for view in views:
                # transpose to make sure all views are the same size
                # this allows to check whether the RNG is is the same state each time
                view['rng'] = int.from_bytes(self._rng.bytes(4), 'big')
            return views
        else:
            for v, view in enumerate(views):
                assert 'pts3d' not in view, f"pts3d should not be there, they will be computed afterwards based on intrinsics+depthmap for view {view_name(view)}"
                view['idx'] = (idx, ar_idx, v)
                # encode the image
                width, height = view['img'].size
                view['true_shape'] = np.int32((height, width))
                view['img'] = self.transform(view['img'])
                view['img_org' ] = self.transform_org(view['img_org'])
                if 'depth_anything' not in view:
                    view['depth_anything'] = np.zeros_like(view['depthmap'])
                # if view['img_org'].shape[1] != 224:
                # print(view['img_org' ].shape)
                # print(view['img'].shape)
                assert 'camera_intrinsics' in view
                if 'camera_pose' not in view:
                    view['camera_pose'] = np.full((4, 4), np.nan, dtype=np.float32)
                else:
                    assert np.isfinite(view['camera_pose']).all(), f'NaN in camera pose for view {view_name(view)}'
                assert 'pts3d' not in view
                assert 'valid_mask' not in view
                assert np.isfinite(view['depthmap']).all(), f'NaN in depthmap for view {view_name(view)}'
                pts3d, valid_mask = depthmap_to_absolute_camera_coordinates(**view)

                view['pts3d'] = pts3d
                view['valid_mask'] = valid_mask & np.isfinite(pts3d).all(axis=-1)
                # print(view['pts3d'].shape)
                # print(view['valid_mask'].shape)

                # check all datatypes
                for key, val in view.items():
                    res, err_msg = is_good_type(key, val)
                    assert res, f"{err_msg} with {key}={val} for view {view_name(view)}"
                K = view['camera_intrinsics']
            
            # if self.aug_swap:
            #     self._swap_view_aug(views)

            if self.aug_monocular:
                if self._rng.random() < self.aug_monocular:
                    random_idxs = random.choices(list(range(len(views)-1)), k = self.num_image + self.gt_num_image-1)
                    views_copy = [views[-1]] + [copy.deepcopy(views[random_idxs[i]]) for i in range(len(views)-1)]
                    views = views_copy
                        
            # if self.aug_rot90 is False:
            #     pass
            # elif self.aug_rot90 == 'same':
            #     rotate_90(views, k=self._rng.choice(4))
            # elif self.aug_rot90 == 'diff':
            #     views_list = []
            #     for view in views:
            #         views_list += rotate_90([view], k=self._rng.choice(4))
            #     views = views_list
            # else:
            #     raise ValueError(f'Bad value for {self.aug_rot90=}')
            if self.rendering==False:
                self._rng.shuffle(views)
                
            if self.caculate_mask:
                for view1 in views[self.num_image:]:
                    render_mask = []
                    start = True
                    # images = []
                    height, width = view1['true_shape']
                    for view2 in views[:self.num_image]:
                        shape1, corres1_to_2 = reproject_view(view1['pts3d'], view2)
                        shape2, corres2_to_1 = reproject_view(view2['pts3d'], view1)
                        # compute reciprocal correspondences:
                        # pos1 == valid pixels (correspondences) in image1
                        # corres1_to_2 = unravel_xy(corres1_to_2, shape2)
                        # corres2_to_1 = unravel_xy(corres2_to_1, shape1)
                        is_reciprocal1, pos1, pos2 = reciprocal_1d(corres1_to_2, corres2_to_1, shape1, shape2, ret_recip=True)
                        render_mask.append(is_reciprocal1.reshape(shape1))
                        # is_reciprocal1 = is_reciprocal1.reshape(shape1)
                        # plt.subplot(1, 3, 1)
                        # plt.imshow(is_reciprocal1)
                        # plt.subplot(1, 3, 2)
                        # plt.imshow(view1['img'].permute(1, 2, 0) / 2 + 0.5)
                        # plt.subplot(1, 3, 3)
                        # plt.imshow(view2['img'].permute(1, 2, 0) / 2 + 0.5)
                        # plt.savefig('/data0/zsz/mast3recon/test/est.png')
                        # import ipdb; ipdb.set_trace()
                        # images.append(view2['img'])
                        if start:
                            view2['render_mask'] = np.ones((view2['img'].shape[1], view2['img'].shape[2]), dtype=np.uint8) > 0.1
                    start = False
                    render_mask = np.stack(render_mask, axis=0).sum(0) 
                    render_mask = cv2.dilate(render_mask/16, self.kernel)    
                    view1['render_mask'] = render_mask > 1e-5
                    # images = torch.concatenate(images, dim=2)
                    # import matplotlib.pyplot as plt
                    # plt.subplot(3, 4, 1)
                    # plt.imshow(render_mask)
                    # plt.subplot(3, 4, 2)
                    # plt.imshow(view1['img'].permute(1, 2, 0) / 2 + 0.5)
                    # for i, image in enumerate(images):
                    #     plt.subplot(3, 4, 3+i)
                    #     plt.imshow(image.permute(1, 2, 0) / 2 + 0.5)
                    # plt.savefig('/data0/zsz/mast3recon/test/est.png')
                    # import ipdb; ipdb.set_trace()
                    # if view1['render_mask'].shape != (height, width):
                    #     import ipdb; ipdb.set_trace()
                    assert view1['render_mask'].shape == (height, width)
            else:
                for view in views:
                    view['render_mask'] = np.ones((view['img'].shape[1], view['img'].shape[2]), dtype=np.uint8) > 0.1

            for view in views:
                fxfycxcy = view['fxfycxcy'].copy()
                H, W = view['img'].shape[1:]
                fxfycxcy[0] = fxfycxcy[0] * W
                fxfycxcy[1] = fxfycxcy[1] * H
                fxfycxcy[2] = fxfycxcy[2] * W
                fxfycxcy[3] = fxfycxcy[3] * H
                view['fxfycxcy_unorm'] = fxfycxcy

            # last thing done!
            for view in views:
                # transpose to make sure all views are the same size
                transpose_to_landscape(view)
                if 'sky_mask' in view:
                    view.pop('sky_mask')
                # this allows to check whether the RNG is is the same state each time
                view['rng'] = int.from_bytes(self._rng.bytes(4), 'big')
            return views

    def _set_resolutions(self, resolutions):
        assert resolutions is not None, 'undefined resolution'

        if not isinstance(resolutions, list):
            resolutions = [resolutions]

        self._resolutions = []
        for resolution in resolutions:
            if isinstance(resolution, int):
                width = height = resolution
            else:
                width, height = resolution
            assert isinstance(width, int), f'Bad type for {width=} {type(width)=}, should be int'
            assert isinstance(height, int), f'Bad type for {height=} {type(height)=}, should be int'
            assert width >= height
            self._resolutions.append((width, height))

    def _crop_resize_if_necessary(self, image, depthmap, intrinsics, resolution, rng=None, info=None, depth_anything=None):
        """ This function:
            - first downsizes the image with LANCZOS inteprolation,
                which is better than bilinear interpolation in
        """
        if not isinstance(image, PIL.Image.Image):
            image = PIL.Image.fromarray(image)

        # transpose the resolution if necessary
        W, H = image.size  # new size
        assert resolution[0] >= resolution[1]
        if H > 1.1 * W:
            # image is portrait mode
            resolution = resolution[::-1]
        elif 0.7 < H / W < 1.3 and resolution[0] != resolution[1] and self.aug_portrait_or_landscape:
            # image is square, so we chose (portrait, landscape) randomly
            if rng.integers(2):
                resolution = resolution[::-1]
        # resolution = resolution[::-1]
        # high-quality Lanczos down-scaling
        target_resolution = np.array(resolution)
        if depth_anything is not None:
            image, depthmap, intrinsics, depth_anything = cropping.rescale_image_depthmap(image, depthmap, intrinsics, target_resolution, depth_anything=depth_anything)
        else:
            image, depthmap, intrinsics = cropping.rescale_image_depthmap(image, depthmap, intrinsics, target_resolution)

        # actual cropping (if necessary) with bilinear interpolation
        offset_factor = 0.5
        intrinsics2 = cropping.camera_matrix_of_crop(intrinsics, image.size, resolution, offset_factor=offset_factor)
        crop_bbox = cropping.bbox_from_intrinsics_in_out(intrinsics, intrinsics2, resolution)
        if depth_anything is not None:
            image, depthmap, intrinsics2, depth_anything = cropping.crop_image_depthmap(image, depthmap, intrinsics, crop_bbox, depth_anything=depth_anything)
            return image, depthmap, intrinsics2, depth_anything
        else:
            image, depthmap, intrinsics2 = cropping.crop_image_depthmap(image, depthmap, intrinsics, crop_bbox)
            return image, depthmap, intrinsics2

    def _crop_resize_if_necessary_test(self, image, depthmap, intrinsics, resolution, rng=None, info=None, depth_anything=None):
        """ This function:
            - first downsizes the image with LANCZOS inteprolation,
                which is better than bilinear interpolation in
        """
        if not isinstance(image, PIL.Image.Image):
            image = PIL.Image.fromarray(image)

        # transpose the resolution if necessary
        W, H = image.size  # new size
        assert resolution[0] >= resolution[1]
        if H > 1.1 * W:
            # image is portrait mode
            resolution = resolution[::-1]

        # resolution = resolution[::-1]
        # high-quality Lanczos down-scaling
        target_resolution = np.array(resolution)
        if depth_anything is not None:
            image, depthmap, intrinsics, depth_anything = cropping.rescale_image_depthmap(image, depthmap, intrinsics, target_resolution, depth_anything=depth_anything)
        else:
            image, depthmap, intrinsics = cropping.rescale_image_depthmap(image, depthmap, intrinsics, target_resolution)

        # actual cropping (if necessary) with bilinear interpolation
        offset_factor = 0.5
        intrinsics2 = cropping.camera_matrix_of_crop(intrinsics, image.size, resolution, offset_factor=offset_factor)
        crop_bbox = cropping.bbox_from_intrinsics_in_out(intrinsics, intrinsics2, resolution)
        if depth_anything is not None:
            image, depthmap, intrinsics2, depth_anything = cropping.crop_image_depthmap(image, depthmap, intrinsics, crop_bbox, depth_anything=depth_anything)
        else:
            image, depthmap, intrinsics2 = cropping.crop_image_depthmap(image, depthmap, intrinsics, crop_bbox)

        return image, depthmap, intrinsics2
    
def is_good_type(key, v):
    """ returns (is_good, err_msg) 
    """
    if isinstance(v, (str, int, tuple)):
        return True, None
    if v.dtype not in (np.float32, torch.float32, bool, np.int32, np.int64, np.uint8):
        return False, f"bad {v.dtype=}"
    return True, None


def view_name(view, batch_index=None):
    def sel(x): return x[batch_index] if batch_index not in (None, slice(None)) else x
    db = sel(view['dataset'])
    label = sel(view['label'])
    instance = sel(view['instance'])
    return f"{db}/{label}/{instance}"


def transpose_to_landscape(view):
    height, width = view['true_shape']

    if width < height:
        # rectify portrait to landscape
        assert view['img'].shape == (3, height, width)
        view['img'] = view['img'].swapaxes(1, 2)
        # try:
        if 'render_mask' in view:
            assert view['render_mask'].shape == (height, width)
            # except:
            # import ipdb; ipdb.set_trace()
            view['render_mask'] = view['render_mask'].swapaxes(0, 1)

        assert view['img_org'].shape == (3, height, width)
        view['img_org'] = view['img_org'].swapaxes(1, 2)

        assert view['valid_mask'].shape == (height, width)
        view['valid_mask'] = view['valid_mask'].swapaxes(0, 1)

        assert view['depthmap'].shape == (height, width)
        view['depthmap'] = view['depthmap'].swapaxes(0, 1)

        assert view['pts3d'].shape == (height, width, 3)
        view['pts3d'] = view['pts3d'].swapaxes(0, 1)
        
        assert view['depth_anything'].shape == (height, width)
        view['depth_anything'] = view['depth_anything'].swapaxes(0, 1)

        # transpose x and y pixels
        view['camera_intrinsics'] = view['camera_intrinsics']#[[1, 0, 2]]
        # view['fxfycxcy'] = view['fxfycxcy']
        # print(view['img'].shape)
        # print(view['img_org'].shape)
        # print(view['valid_mask'].shape)
        # print(view['depthmap'].shape)
        # print(view['pts3d'].shape)
        # print(view['camera_intrinsics'].shape)
        # print(view['fxfycxcy'].shape)
    # assert view['img'].shape == (3, height, width)
    # assert view['img_org'].shape == (3, height, width)
    # assert view['valid_mask'].shape == (height, width)
    # assert view['depthmap'].shape == (height, width)
    # assert view['pts3d'].shape == (height, width, 3)
    # assert view['camera_intrinsics'].shape == (3, 3)
    # assert view['fxfycxcy'].shape == (4,)
    