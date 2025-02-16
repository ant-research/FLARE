import os.path as osp
import cv2
import numpy as np
import random
import mast3r.utils.path_to_dust3r  # noqa
# check the presence of models directory in repo to be sure its cloned
from dust3r.datasets.base.base_stereo_view_dataset import BaseStereoViewDataset
from dust3r.utils.image import imread_cv2, imread_cv2_orig
from collections import deque
import os
import json
import time
import glob
import tqdm
try:
    from pcache_fileio import fileio
    flag_pcache = True
except:
    flag_pcache = False
from enum import Enum, auto
from pathlib import Path
from .imc_helper import parse_file_to_list, load_calib
import torch

class IMCDataset(BaseStereoViewDataset):
    def __init__(
        self,
        *args,
        ROOT,
        split="train",
        transform=None,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.sequences = {}
        IMC_DIR = ROOT
        if IMC_DIR == None:
            raise NotImplementedError

        print(f"IMC_DIR is {IMC_DIR}")
        
        if split == "train":
            raise ValueError("We don't want to train on IMC")
        elif split == "test":
            bag_names = glob.glob(os.path.join(IMC_DIR, "*/set_100/sub_set/*.txt"))

            if False:
                # In some settings, the scene london_bridge is removed from IMC
                bag_names = [name for name in bag_names if "london_bridge" not in name]

            for bag_name in bag_names:
                # if '5bag' in bag_name:
                #     continue
                parts = bag_name.split("/")  # Split the string into parts by '/'
                location = parts[-4]  # The location part is at index 5
                bag_info = parts[-1].split(".")[0]  # The bag info part is the last part, and remove '.txt'
                new_bag_name = f"{bag_info}_{location}"  # Format the new bag name

                img_filenames = parse_file_to_list(bag_name, "/".join(parts[:-2]))
                filtered_data = []
                for img_name in img_filenames:
                    calib_file = img_name.replace("images", "calibration").replace("jpg", "h5")
                    calib_file = "/".join(
                        calib_file.rsplit("/", 1)[:-1] + ["calibration_" + calib_file.rsplit("/", 1)[-1]]
                    )
                    calib_dict = load_calib([calib_file])

                    calib = calib_dict[os.path.basename(img_name).split(".")[0]]
                    intri = torch.from_numpy(np.copy(calib["K"]))

                    R = torch.from_numpy(np.copy(calib["R"]))

                    tvec = torch.from_numpy(np.copy(calib["T"]).reshape((3,)))

                    fl = torch.from_numpy(np.stack([intri[0, 0], intri[1, 1]], axis=0))
                    pp = torch.from_numpy(np.stack([intri[0, 2], intri[1, 2]], axis=0))

                    filtered_data.append(
                        {
                            "filepath": img_name,
                            "R": R,
                            "T": tvec,
                            "focal_length": fl,
                            "principal_point": pp,
                            "calib": calib,
                            'K': intri
                        }
                    )
                self.sequences[new_bag_name] = filtered_data
        else:
            raise ValueError("please specify correct set")

        self.IMC_DIR = IMC_DIR
        self.crop_longest = True

        self.sequence_list = sorted(self.sequences.keys())
        # self.sequence_list = ['10bag_021_piazza_san_marco']
        self.split = split
        self.sort_by_filename = True
        self.index = 0


        print(f"Data size of IMC: {len(self)}")

    def __len__(self):
        return len(self.sequence_list)

    def _get_views(self, idx, resolution, rng):
        index = self.index
        sequence_name = self.sequence_list[index]
        # sequence_name = random.choice(self.sequence_list)
        print(sequence_name)
        self.index += 1
        metadata = self.sequences[sequence_name]

        # if ids is None:
        ids = np.arange(len(metadata))

        annos = [metadata[i] for i in ids]

        if self.sort_by_filename:
            annos = sorted(annos, key=lambda x: x["filepath"])

        views = []
        num = len(annos)
        # if num <= 8:
        #     annos = annos + annos[:8 - num]
        for anno in annos:
            filepath = anno["filepath"]
            image_path = os.path.join(self.IMC_DIR, filepath)
            rgb_image = imread_cv2(image_path)
            camera_pose = np.eye(4)
            camera_pose[:3, :3] = anno["R"].detach().cpu().numpy()
            camera_pose[:3, 3] = anno["T"].detach().cpu().numpy()
            depthmap = np.zeros_like(rgb_image)[...,0]
            intrinsics = anno["K"].detach().cpu().numpy()
            rgb_image, depthmap, intrinsics = self._crop_resize_if_necessary(
                rgb_image, depthmap, intrinsics, resolution, rng=rng, info=filepath)
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
                dataset='tartantair',
                label=filepath,
                instance=filepath.split('/')[-3],
                num=num
            ))
        return views


def calculate_crop_parameters(image, bbox_jitter, crop_dim, img_size):
    crop_center = (bbox_jitter[:2] + bbox_jitter[2:]) / 2
    # convert crop center to correspond to a "square" image
    width, height = image.size
    length = max(width, height)
    s = length / min(width, height)
    crop_center = crop_center + (length - np.array([width, height])) / 2
    # convert to NDC
    cc = s - 2 * s * crop_center / length
    crop_width = 2 * s * (bbox_jitter[2] - bbox_jitter[0]) / length
    bbox_after = bbox_jitter / crop_dim * img_size
    crop_parameters = torch.tensor(
        [-cc[0], -cc[1], crop_width, s, bbox_after[0], bbox_after[1], bbox_after[2], bbox_after[3]]
    ).float()
    return crop_parameters
