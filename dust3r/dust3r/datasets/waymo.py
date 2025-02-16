# Copyright (C) 2024-present Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).
#
# --------------------------------------------------------
# Dataloader for preprocessed WayMo
# dataset at https://github.com/waymo-research/waymo-open-dataset
# See datasets_preprocess/preprocess_waymo.py
# --------------------------------------------------------
import os.path as osp
import numpy as np
import mast3r.utils.path_to_dust3r  # noqa

from dust3r.datasets.base.base_stereo_view_dataset import BaseStereoViewDataset
from dust3r.utils.image import imread_cv2
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
    
class Waymo (BaseStereoViewDataset):
    """ Dataset of outdoor street scenes, 5 images each time
    """

    def __init__(self, *args, ROOT, meta, **kwargs):
        self.ROOT = ROOT.replace(oss_folder_path, pcache_folder_path)
        self.meta = meta
        super().__init__(*args, **kwargs)
        self._load_data()

    def _load_data(self):
        # with np.load(osp.join(self.ROOT, 'waymo_pairs.npz')) as data:
        #     self.scenes = data['scenes']
        #     self.frames = data['frames']
        #     self.inv_frames = {frame: i for i, frame in enumerate(data['frames'])}
        #     self.pairs = data['pairs']  # (array of (scene_id, img1_id, img2_id)
        #     assert self.pairs[:, 0].max() == len(self.scenes) - 1
        self.meta = np.load(self.meta, allow_pickle=True)
        self.scenes_idxs = {}
        self.scenes_names = {}
        self.intrinsics = {}
        self.camera_poses = {}
        # for folder in tqdm.tqdm(folders):
        #     name = folder.split('/')[-2]
        #     if osp.exists(osp.join(self.graph_path, name+'.npz')) == False:
        #         continue
        #     np_path = np.load(osp.join(self.graph_path, name+'.npz'), allow_pickle=True)
        #     self.scenes_idxs[name] = np_path['imgs_idxs']
        #     self.scenes_names[name] = np_path['image_name'].item()
        #     seq_path = osp.join(self.ROOT, name)
        #     image_name = self.scenes_names[name]
        #     self.intrinsics[name] = {}
        #     self.camera_poses[name] = {}
        #     for key in image_name.keys():
        #         impath = image_name[key]
        #         camera_params = np.load(osp.join(seq_path, impath + ".npz"))
        #         intrinsics = np.float32(camera_params['intrinsics'])
        #         camera_pose = np.float32(camera_params['cam2world'])
        #         self.intrinsics[name][key] = intrinsics
        #         self.camera_poses[name][key] = camera_pose
        self.intrinsics = self.meta['intrinsics'].item()
        self.camera_poses = self.meta['camera_poses'].item()
        self.scenes_idxs = self.meta['scenes_idxs'].item()
        self.scenes_names = self.meta['scenes_names'].item()
        # np.savez(osp.join('/nas7/vilab/zsz/mast3recon/data', 'waymo_meta.npz'), **self.meta)

    def __len__(self):
        return 684000

    def get_stats(self):
        return f'{len(self)} pairs from { len(self.scenes_idxs.keys())} scenes'

    def _get_views(self, pair_idx, resolution, rng):
        # seq, img1, img2 = self.pairs[pair_idx]
        # seq_path = osp.join(self.ROOT, self.scenes[seq])
        scene = rng.choice(list(self.scenes_idxs.keys()))
        image_idexs = self.scenes_idxs[scene]
        image_idex = rng.choice(image_idexs)
        image_name = self.scenes_names[scene]
        self.num_image_input = self.num_image
        imgs_idx = rng.choice(image_idex, self.num_image_input + self.gt_num_image)
        views = []
        seq_path = osp.join(self.ROOT, scene)
        for view_index in imgs_idx:
            impath = image_name[view_index]
            image = imread_cv2(osp.join(seq_path, impath + ".jpg"))
            depthmap = imread_cv2(osp.join(seq_path, impath + ".exr"))
            # camera_params = np.load(osp.join(seq_path, impath + ".npz"))
            
            intrinsics = np.float32(self.intrinsics[scene][view_index])
            camera_pose = np.float32(self.camera_poses[scene][view_index])

            image, depthmap, intrinsics = self._crop_resize_if_necessary(
                image, depthmap, intrinsics, resolution, rng, info=(impath))
            img_org = image.copy()
            H, W = depthmap.shape[:2]
            fxfycxcy = np.array([intrinsics[0, 0]/W, intrinsics[1, 1]/H, intrinsics[0,2]/W, intrinsics[1,2]/H]).astype(np.float32)
            views.append(dict(
                fxfycxcy=fxfycxcy,
                img_org=img_org,
                img=image,
                depthmap=depthmap,
                camera_pose=camera_pose,  # cam2world
                camera_intrinsics=intrinsics,
                dataset='Waymo',
                label=osp.relpath(seq_path, self.ROOT),
                instance=impath))

        return views



if __name__ == '__main__':
    from dust3r.datasets.base.base_stereo_view_dataset import view_name
    from dust3r.viz import SceneViz, auto_cam_size
    from dust3r.utils.image import rgb
    import nerfvis.scene as scene_vis 

    dataset =  Waymo(split='train', ROOT='oss://antsys-vilab/datasets/pcache_datasets/waymo_processed/', meta='/input_ssd/zsz/dust3r_dataset/waymo_meta.npz', resolution=[(512, 384)], aug_crop='auto', aug_monocular=0.005,  num_views=8, gt_num_image=0) 
    for idx in np.random.permutation(len(dataset)):
        views = dataset[idx]
        print('+1')