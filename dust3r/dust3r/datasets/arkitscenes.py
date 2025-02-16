# Copyright (C) 2024-present Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).
#
# --------------------------------------------------------
# Dataloader for preprocessed arkitscenes
# dataset at https://github.com/apple/ARKitScenes - Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International Public License https://github.com/apple/ARKitScenes/tree/main?tab=readme-ov-file#license
# See datasets_preprocess/preprocess_arkitscenes.py
# --------------------------------------------------------
import os.path as osp
import cv2
import numpy as np
import random
import mast3r.utils.path_to_dust3r  # noqa
from dust3r.datasets.base.base_stereo_view_dataset import BaseStereoViewDataset
from dust3r.utils.image import imread_cv2, imread_cv2_orig
from collections import deque
import os
import json
import time
try:
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


def load_subgraphs_from_npz(filename):
    # 从 npz 文件中加载数据
    with np.load(filename, allow_pickle=True) as data:
        serialized_subgraphs = json.loads(data['subgraphs'].item())
    
    # 反序列化子图
    subgraphs = []
    for subgraph_data in serialized_subgraphs:
        subgraph = nx.Graph()
        subgraph.add_nodes_from(subgraph_data['nodes'])
        subgraph.add_edges_from(subgraph_data['edges'])
        subgraphs.append(subgraph)
    
    return subgraphs

def get_random_connected_subgraph(graph, num_nodes):
    # 随机选择一个初始节点
    start_node = random.choice(list(graph.nodes()))
    
    # 使用 BFS 或 DFS 构建连通子图
    visited = set()
    queue = [start_node]
    
    while queue and len(visited) < num_nodes:
        node = queue.pop(0)
        if node not in visited:
            visited.add(node)
            # 将相邻节点添加到队列中
            neighbors = list(graph.neighbors(node))
            random.shuffle(neighbors)  # 打乱邻居节点顺序
            for neighbor in neighbors:
                if neighbor not in visited:
                    queue.append(neighbor)
    
    # 如果连通子图的节点数量不足，填充节点
    if len(visited) < num_nodes:
        additional_nodes = set(random.sample([n for n in graph.nodes() if n not in visited], num_nodes - len(visited)))
        visited.update(additional_nodes)
    
    # 生成最终子图
    subgraph = graph.subgraph(visited).copy()
    
    return subgraph

class ARKitScenes(BaseStereoViewDataset):
    def __init__(self, *args, split, meta, ROOT, **kwargs):
        if flag_pcache:
            self.ROOT = ROOT.replace(oss_folder_path, pcache_folder_path)
        super().__init__(*args, **kwargs)
        if split == "train":
            self.split = "Training"
        elif split == "test":
            self.split = "Test"
        else:
            raise ValueError("")
        self.meta = np.load(meta, allow_pickle=True)
        self.loaded_data = self._load_data(self.split)

    def _load_data(self, split):
        # data_path = osp.join(self.ROOT, split)
        # folders = [os.path.join(data_path, f) for f in os.listdir(data_path) if os.path.isdir(os.path.join(data_path, f))]
        # self.images = {}
        # self.intrinsics = {}
        # self.trajectories = {}
        # self.subgraphs = {}
        self.images = self.meta['images'].item()
        self.intrinsics = self.meta['intrinsics'].item()
        self.trajectories = self.meta['trajectories'].item()
        self.subgraphs = self.meta['subgraphs'].item()
        # meta['images'] = self.images
        # meta['intrinsics'] = self.intrinsics
        # meta['trajectories'] = self.trajectories
        # meta['subgraphs'] = self.subgraphs
        # np.savez(osp.join('/nas7/datasets/ARKitScenes/dataset/', f'{split}_metadata.npz'), **meta)

    def __len__(self):
        return 684000

    def _get_views(self, idx, resolution, rng):
        # image_idx1, image_idx2 = self.pairs[idx]
        scene = random.choice(list(self.images.keys()))
        scene_path = osp.join(self.ROOT, self.split, scene.split('/')[-1])
        subgraphs = self.subgraphs[scene]
        imgs_idxs = random.choice(subgraphs)
        # subgraph = get_random_connected_subgraph(random_subgraph, self.num_image+self.gt_num_image)
        # imgs_idxs = list(subgraph.nodes())
        # image_num = len(self.images[scene])
        self.num_image_input = self.num_image
        # end = image_num-self.num_image_input*2
        # end = max(1, end)
        # im_start = random.choice(range(end))
        # # add a bit of randomness
        # last = image_num-1
        # # seqh, seql, img1, img2, score = self.pairs[pair_idx]
        # im_list = [im_start + i * 3 + random.choice([-2,-1,0,1,2]) for i in range(self.num_image_input)]
        # im_end = min(im_start + self.num_image_input * 3, last)
        # im_list += [random.choice(im_list) + random.choice([-2,-1,1,2]) for _ in range(self.gt_num_image)]
        # views = []
        # imgs_idxs = [max(0, min(im_idx, im_end)) for im_idx in im_list]
        random.shuffle(imgs_idxs)
        if len(imgs_idxs) < self.num_image_input+self.gt_num_image:
            imgs_idxs = random.choices(imgs_idxs, k=self.num_image_input+self.gt_num_image)
        else:
            imgs_idxs = imgs_idxs[:self.num_image_input+self.gt_num_image]
        imgs_idxs = deque(imgs_idxs)
        views = []
        while len(imgs_idxs) > 0:  # some images (few) have zero depth
            view_idx = imgs_idxs.pop()
            intrinsics = self.intrinsics[scene][view_idx]
            K = np.eye(3)
            fx, fy, hw, hh = intrinsics[2:]
            K[0, 0] = fx#[fx for _, _, fx, _, _, _ in intrinsics]
            K[1, 1] = fy#[fy for _, _, _, fy, _, _ in intrinsics]
            K[0, 2] = hw#[hw for _, _, _, _, hw, _ in intrinsics]
            K[1, 2] = hh#[hh for _, _, _, _, _, hh in intrinsics]
            intrinsics = K
            camera_pose = self.trajectories[scene][view_idx]

            basename = self.images[scene][view_idx]
            # Load RGB image
            rgb_image = imread_cv2(osp.join(scene_path, 'vga_wide', basename.replace('.png', '.jpg')))
            # Load depthmap
            depthmap = imread_cv2(osp.join(scene_path, 'lowres_depth', basename), cv2.IMREAD_UNCHANGED)
            depthmap = depthmap.astype(np.float32) / 1000
            depthmap[~np.isfinite(depthmap)] = 0  # invalid
            # print((depthmap==0).sum(), force=True)
            rgb_image, depthmap, intrinsics = self._crop_resize_if_necessary(
                rgb_image, depthmap, intrinsics, resolution, rng=rng, info=osp.join(scene_path, 'vga_wide', basename.replace('.png', '.jpg')))
            rgb_image_orig = rgb_image.copy()
            H, W = depthmap.shape[:2]
            fxfycxcy = np.array([intrinsics[0, 0]/W, intrinsics[1, 1]/H, intrinsics[0,2]/W, intrinsics[1,2]/H]).astype(np.float32)
            views.append(dict(
                img_org=rgb_image_orig,
                img=rgb_image,
                depthmap=depthmap.astype(np.float32),
                camera_pose=camera_pose.astype(np.float32),
                camera_intrinsics=intrinsics.astype(np.float32),
                fxfycxcy=fxfycxcy,
                dataset='arkitscenes',
                label=osp.join(scene, 'vga_wide', basename.replace('.png', '.jpg')),
                instance=f'{str(idx)}_{str(view_idx)}',
            ))
        
        return views

if __name__ == "__main__":
    from dust3r.datasets.base.base_stereo_view_dataset import view_name
    from dust3r.viz import SceneViz, auto_cam_size
    from dust3r.utils.image import rgb
    import nerfvis.scene as scene_vis
    # dataset = ARKitScenes(split='train', ROOT="/nas7/datasets/ARKitScenes/dataset/arkitscenes_processed", meta='/input_ssd/zsz/dust3r_dataset/Training_metadata.npz', resolution=[(512, 384)], aug_crop=16)
    dataset = ARKitScenes(split='train', ROOT="/nas7/datasets/ARKitScenes/dataset/arkitscenes_processed", meta='/nas7/datasets/ARKitScenes/dataset/Training_metadata.npz', resolution=[(512, 384)], aug_crop=16)

    for idx in np.random.permutation(len(dataset)):
        views = dataset[idx]
        # assert len(views) == 2
        # print(view_name(views[0]), view_name(views[1]))
        view_idxs = list(range(len(views)))
        poses = [views[view_idx]['camera_pose'] for view_idx in view_idxs]
        cam_size = max(auto_cam_size(poses), 0.001)
        pts3ds = []
        colors = []
        valid_masks = []
        c2ws = []
        intrinsics = []
        for view_idx in view_idxs:
            pts3d = views[view_idx]['pts3d']
            pts3ds.append(pts3d)
            valid_mask = views[view_idx]['valid_mask']
            valid_masks.append(valid_mask)
            color = rgb(views[view_idx]['img'])
            colors.append(color)
            c2ws.append(views[view_idx]['camera_pose'])

        pts3ds = np.stack(pts3ds, axis=0)
        colors = np.stack(colors, axis=0)
        valid_masks = np.stack(valid_masks, axis=0)
        c2ws = np.stack(c2ws)
        scene_vis.set_title("My Scene")
        scene_vis.set_opencv() 
        pts_mean = pts3ds.reshape(-1,3)[valid_masks.reshape(-1)].mean(0)
        scene_vis.add_points("points", pts3ds.reshape(-1,3)[valid_masks.reshape(-1)]-pts_mean, vert_color=colors.reshape(-1,3)[valid_masks.reshape(-1)], point_size=1)
        scene_vis.add_camera_frustum(
            "gt_cameras",
            r=c2ws[:, :3, :3],
            t=c2ws[:, :3, 3]-pts_mean,
            focal_length=320,
            z=1,
            connect=False,
            image_width=640,
            image_height=480,
            color=[0.0, 1.0, 0.0],
        )
        scene_vis.display()

