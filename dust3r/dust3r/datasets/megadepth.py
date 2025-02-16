# Copyright (C) 2024-present Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).
#
# --------------------------------------------------------
# Dataloader for preprocessed MegaDepth
# dataset at https://www.cs.cornell.edu/projects/megadepth/
# See datasets_preprocess/preprocess_megadepth.py
# --------------------------------------------------------
import os.path as osp
import numpy as np
import mast3r.utils.path_to_dust3r  # noqa
from dust3r.datasets.base.base_stereo_view_dataset import BaseStereoViewDataset
from dust3r.utils.image import imread_cv2
import os
import glob
import tqdm
try:
    # from pcache_fileio import fileio
    import fsspec
    PCACHE_HOST = "vilabpcacheproxyi-pool.cz50c.alipay.com"
    PCACHE_PORT = 39999
    pcache_kwargs = {"host": PCACHE_HOST, "port": PCACHE_PORT}
    pcache_fs = fsspec.filesystem("pcache", pcache_kwargs=pcache_kwargs)
    oss_folder_path = 'oss://antsys-vilab/datasets/pcache_datasets/'
    pcache_folder_path = '/mnt/antsys-vilab_datasets_pcache_datasets/'
    flag_pcache = True
except:
    flag_pcache = False
    
class MegaDepth(BaseStereoViewDataset):
    def __init__(self, *args, split, ROOT, meta, **kwargs):
        self.ROOT = ROOT.replace(oss_folder_path, pcache_folder_path)
        super().__init__(*args, **kwargs)
        # self.all_scenes = glob.glob(os.path.join(self.ROOT, '*', '*'))
        # self.scenes = {}
        # self.meta = {}
        # self.intrinsics = {}
        # self.camera_poses = {}
        # self.meta = np.load(meta, allow_pickle=True)

        # for scene in tqdm.tqdm(self.all_scenes):
        #     self.scenes[scene] = glob.glob(os.path.join(scene, '*.jpg'))
        #     seq_path = osp.join(self.ROOT, scene)
        #     self.intrinsics[scene] = {}
        #     self.camera_poses[scene] = {}
        #     for im_id in self.scenes[scene]:
        #         img = im_id.split('/')[-1].split('.')[:2]
        #         img = '.'.join(img)
        #         camera_params = np.load(osp.join(seq_path, img + ".npz"))
        #         intrinsics = np.float32(camera_params['intrinsics'])
        #         camera_pose = np.float32(camera_params['cam2world'])
        #         self.intrinsics[scene][im_id] = intrinsics
        #         self.camera_poses[scene][im_id] = camera_pose
        # self.meta['intrinsic'] = self.intrinsics
        # self.meta['camera_poses'] = self.camera_poses
        # self.meta['scenes'] = self.scenes
        # np.savez(osp.join('/nas7/vilab/zsz/mast3recon/checkpoints', 'megadepth_metadata.npz'), **self.meta)
        self.meta = np.load(meta, allow_pickle=True)
        self.scenes = self.meta['scenes'].item()
        self.intrinsics = self.meta['intrinsic'].item()
        self.camera_poses = self.meta['camera_poses'].item()
        self.all_scenes = list(self.scenes.keys())
        self.skip_list = ['0294', '0312', '0380']
        # if self.split is None:
        #     pass
        # elif self.split == 'train':
        #     self.select_scene(('0015', '0022'), opposite=True)
        # elif self.split == 'val':
        #     self.select_scene(('0015', '0022'))
        # else:
        #     raise ValueError(f'bad {self.split=}')

    def __len__(self):
        return 684000

    def get_stats(self):
        return f'{len(self)} pairs from {len(list(self.scenes.keys()))} scenes'

    def _get_views(self, pair_idx, resolution, rng):
        retries = 0
        scene_id = rng.choice(list(self.scenes.keys()))
        while scene_id in self.skip_list:
            scene_id = rng.choice(list(self.scenes.keys()))
            retries += 1
        img_indx = self.scenes[scene_id]
        self.num_image_input = self.num_image
        img_indx = rng.choice(img_indx, self.num_image_input + self.gt_num_image)
        views = []
        seq_path = osp.join(self.ROOT, '/'.join(scene_id.split('/')[-2:]))
        for im_id in img_indx:
            img = im_id.split('/')[-1].split('.')[:2]
            img = '.'.join(img)
            image = imread_cv2(osp.join(seq_path, img + '.jpg'))
            depthmap = imread_cv2(osp.join(seq_path, img + ".exr"))
            with pcache_fs.open(osp.join(seq_path, img + ".npz"), 'rb') as f:
                camera_params = np.load(f)
            # intrinsics = np.float32(camera_params['intrinsics'])
            # camera_pose = np.float32(camera_params['cam2world'])
            intrinsics = self.intrinsics[scene_id][im_id]
            camera_pose = self.camera_poses[scene_id][im_id]
            image, depthmap, intrinsics = self._crop_resize_if_necessary(
                image, depthmap, intrinsics, resolution, rng, info=(seq_path, img))
            H, W = depthmap.shape[:2]
            fxfycxcy = np.array([intrinsics[0, 0]/W, intrinsics[1, 1]/H, intrinsics[0,2]/W, intrinsics[1,2]/H]).astype(np.float32)
            img_org = image.copy()

            views.append(dict(
                img_org=img_org,
                fxfycxcy=fxfycxcy,
                img=image,
                depthmap=depthmap,
                camera_pose=camera_pose,  # cam2world
                camera_intrinsics=intrinsics,
                dataset='MegaDepth',
                label=osp.relpath(seq_path, self.ROOT),
                instance=img))

        return views

if __name__ == "__main__":
    from dust3r.datasets.base.base_stereo_view_dataset import view_name
    from dust3r.viz import SceneViz, auto_cam_size
    from dust3r.utils.image import rgb
    import nerfvis.scene as scene_vis 

    dataset = MegaDepth(split='train', ROOT='oss://antsys-vilab/datasets/pcache_datasets/megadepth_process/', meta='/input_ssd/zsz/dust3r_dataset/megadepth_metadata.npz', resolution=[(512, 384)], aug_crop='auto', aug_monocular=0.005,  num_views=8, gt_num_image=0)
    for idx in np.random.permutation(len(dataset)):
        views = dataset[idx]
        print('+1')