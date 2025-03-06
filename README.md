# FLARE: Feed-forward Geometry, Appearance and Camera Estimation from Uncalibrated Sparse Views
[![Website](https://img.shields.io/website-up-down-green-red/http/shields.io.svg)](https://zhanghe3z.github.io/FLARE/)
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97-Hugging%20Face-yellow)](https://huggingface.co/AntResearch/FLARE)
[![Video](https://img.shields.io/badge/Video-Demo-red)](https://zhanghe3z.github.io/FLARE/videos/teaser_video.mp4)

Official implementation of **FLARE** (CVPR 2025) - a feed-forward model for joint camera pose estimation, 3D reconstruction and novel view synthesis from sparse uncalibrated views.

![Teaser Video](./assets/teaser.jpg)


<!-- TOC start (generated with https://github.com/derlin/bitdowntoc) -->

- [📖 Overview](#-overview)
- [🛠️ TODO List](#-todo-list)
- [🌍 Installation](#-installation)
- [💿 Checkpoints](#-checkpoints)
- [🎯 Run a Demo (Point Cloud and Camera Pose Estimation) ](#-run-a-demo-point-cloud-and-camera-pose-estimation)
- [📽️ Evaluating Novel View Synthesis](#-evaluating-novel-view-synthesis)
- [👀 Visualization ](#-visualization)
- [📜 Citation ](#-citation)

<!-- TOC end -->

## 📖 Overview
We present FLARE, a feed-forward model that simultaneously estimates high-quality camera poses, 3D geometry, and appearance from as few as 2-8 uncalibrated images. Our cascaded learning paradigm:

1. **Camera Pose Estimation**: Directly regress camera poses without bundle adjustment
2. **Geometry Reconstruction**: Decompose geometry reconstruction into two simpler sub-problems
3. **Appearance Modeling**: Enable photorealistic novel view synthesis via 3D Gaussians

Achieves SOTA performance with inference times <0.5 seconds!

## 🛠️ TODO List
- [x] Release point cloud and camera pose estimation code.
- [x] Updated Gradio demo (app.py).
- [x] Release novel view synthesis code.
- [x] Release evaluation code.
- [ ] Release training code.
- [ ] Release data processing code.

## 🌍 Installation

```
conda create -n flare python=3.8
conda activate flare 
conda install pytorch torchvision pytorch-cuda=12.1 -c pytorch -c nvidia  # use the correct version of cuda for your system
pip install -r requirements.txt
conda uninstall ffmpeg  
conda install -c conda-forge ffmpeg
```


## 💿 Checkpoints
Download the checkpoint from [huggingface_geometry](https://huggingface.co/AntResearch/FLARE/blob/main/geometry_pose.pth) [huggingface_nvs](https://huggingface.co/zhang3z/FLARE_NVS/blob/main/NVS.pth) and place it in the checkpoints/geometry_pose.pth and checkpoints/NVS.pth directory.

## 🎯 Run a Demo (Point Cloud and Camera Pose Estimation) 


```bash
sh scripts/run_pose_pointcloud.sh
```


```bash
torchrun --nproc_per_node=1 run_pose_pointcloud.py \
    --test_dataset "1 @ CustomDataset(split='train', ROOT='Your/Data/Path', resolution=(512,384), seed=1, num_views=7, gt_num_image=0, aug_portrait_or_landscape=False, sequential_input=False)" \
    --model "AsymmetricMASt3R(pos_embed='RoPE100', patch_embed_cls='ManyAR_PatchEmbed', img_size=(512, 512), head_type='catmlp+dpt', output_mode='pts3d+desc24', depth_mode=('exp', -inf, inf), conf_mode=('exp', 1, inf), enc_embed_dim=1024, enc_depth=24, enc_num_heads=16, dec_embed_dim=768, dec_depth=12, dec_num_heads=12, two_confs=True, desc_conf_mode=('exp', 0, inf))" \
    --pretrained "Your/Checkpoint/Path" \
    --test_criterion "MeshOutput(sam=False)" --output_dir "log/" --amp 1 --seed 1 --num_workers 0
```

**To run the demo using ground truth camera poses:**
Enable the wpose=True flag in both the CustomDataset and AsymmetricMASt3R. An example script demonstrating this setup is provided in run_pose_pointcloud_wpose.sh.

```bash
sh scripts/run_pose_pointcloud_wpose.sh
```




## 👀 Visualization 

```
sh ./visualizer/vis.sh
```
 

```
CUDA_VISIBLE_DEVICES=0 python visualizer/run_vis.py --result_npz data/mesh/IMG_1511.HEIC.JPG.JPG/pred.npz --results_folder data/mesh/IMG_1511.HEIC.JPG.JPG/
``` 

## 📽️ Evaluating Novel View Synthesis

You can evaluate the novel view synthesis on RealEstate10K dataset using the following command:

```
sh scripts/run_eval_nvs.sh
```

### RealEstate10K Dataset

Our model uses the same training and test datasets as pixelSplat. Below we quote pixelSplat's [detailed instructions](https://github.com/dcharatan/pixelsplat?tab=readme-ov-file#acquiring-datasets) on getting datasets.

> pixelSplat was trained using versions of the RealEstate10k and ACID datasets that were split into ~100 MB chunks for use on server cluster file systems. Small subsets of the Real Estate 10k and ACID datasets in this format can be found [here](https://drive.google.com/drive/folders/1joiezNCyQK2BvWMnfwHJpm2V77c7iYGe?usp=sharing). To use them, simply unzip them into a newly created `datasets` folder in the project root directory.

> If you would like to convert downloaded versions of the Real Estate 10k and ACID datasets to our format, you can use the [scripts here](https://github.com/dcharatan/real_estate_10k_tools). Reach out to us (pixelSplat) if you want the full versions of our processed datasets, which are about 500 GB and 160 GB for Real Estate 10k and ACID respectively.


## 📜 Citation 
```bibtex
@misc{zhang2025flarefeedforwardgeometryappearance,
      title={FLARE: Feed-forward Geometry, Appearance and Camera Estimation from Uncalibrated Sparse Views}, 
      author={Shangzhan Zhang and Jianyuan Wang and Yinghao Xu and Nan Xue and Christian Rupprecht and Xiaowei Zhou and Yujun Shen and Gordon Wetzstein},
      year={2025},
      eprint={2502.12138},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2502.12138}, 
}
```
## 🙏 Acknowledgement
FLARE is constructed on the outstanding open-source projects. We are extremely grateful for the contributions of these projects and their communities, whose hard work has greatly propelled the development of the field and enabled our work to be realized.

- [DUSt3R](https://dust3r.europe.naverlabs.com/)
- [VGGSfM](https://github.com/facebookresearch/vggsfm)
- [MASt3R](https://github.com/naver/mast3r)
  



