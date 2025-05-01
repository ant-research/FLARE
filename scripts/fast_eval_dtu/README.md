English | [简体中文](README_zh.md)
# Fast DTU Evaluation Using GPU with Python

This repository implements the utilization of GPUs to expedite the point cloud evaluation process on DTU, utilizing Python and CUDA.  

## Problem of MATLAB Evaluation
When conducting point cloud evaluation on DTU, it is necessary to install MATLAB and use the MATLAB code provided by DTU's official. This process is exceedingly time-consuming, often taking hours or even half a day. To enhance evaluation speed, it is common to employ multiprocessing (`parfor` in MATLAB) for evaluation. This can consume significant CPU resources on linux matchine shared by multiple people, leading to CPU stalls when others are utilizing it.  
The evaluation process on DTU actually calculates the [chamfer distance](https://www.youtube.com/watch?v=P4IyrsWicfs) between the predicted point cloud and the ground truth point cloud, which is defined as `accuracy` and `completeness` in DTU. The primary reason why utilizing MATLAB for this process is time-consuming is that the calculation process is completed entirely on the CPU, without leveraging GPU acceleration. To address this issue, referencing the repositories [ChamferDistancePytorch](https://github.com/ThibaultGROUEIX/ChamferDistancePytorch) and [MVS_Evaluation_Benchmark](https://github.com/ToughStoneX/MVS_Evaluation_Benchmark), we utilizes Python and CUDA to implement GPU acceleration for DTU's point cloud evaluation.



## Usage
1. Install dependency 
```bash
pip install -r requirements.txt
cd chamfer3D && python setup.py install # build and install chamfer3D package
```
2. Prepare Dataset
- Download the STL [Point clouds](http://roboimagedata2.compute.dtu.dk/data/MVS/Points.zip) and [Sample Set](http://roboimagedata2.compute.dtu.dk/data/MVS/SampleSet.zip) and unzip them, then copy the `Points/Points/stl/stlxxx_total.ply` file in to the `SampleSet/MVS Data/Points/stl` folder.
- Get prediction results of your mvs algorithm, the naming format of predicted ply file is `{method}{scanid}_l3.ply` such as `mvsnet001_l3.ply`
After the two steps above, your data directory will like bellow:
```
./SampleSet/MVS Data/
|--Points
|   |--stl
|       |--stlxxx_total.ply
|--ObsMask
|   |--ObsMaskxxx_10.mat
|   |--Planexxx.mat

./Predict/
|--mvsnet
|   |--mvsnetxxx_l3.ply
```

3. evaluation
- Set `--pred_dir` and `--gt_dir` in `eval_dtu.sh`
- Use following command to evaluation:
```bash
bash eval_dtu.sh
# or 
# CUDA_VISIBLE_DEVICES=0 python eval_dtu.py --method mvsnet --pred_dir "./Preidct/mvsnet/" --gt_dir "./SampleSet/MVS Data" --save --num_workers 1
```

## Main time-consuming analysis:
In the evaluation code of matlab, the main time consumption comes from the downsampling of point clouds (`reducePts_haa` function) and the calculation of chamfer distance (`MaxDistCP` function). Both the downsampling of point clouds and the calculation of chamfer distance use the KD tree structure and nearest neighbor algorithm. Because KD trees cannot be accelerated by GPUs, we replace it with voxel downsampling in the open3d library to accelerate. At the same time, we use the chamfer distance implemented by CUDA to leverage GPUs for acceleration. **It should be noted that due to the use of voxel downsampling in open3d, the downsampled point clouds will be different from the downsampling results in matlab. In order to align with matlab overall results as much as possible, we add a hyperparameter `voxel_factor` in the code for adjustment. Adjusting this parameter will change the downsampling size of open3d's voxel downsampling**. In our experiment, setting it to **1.28** basically approximates matlab's result.


## Compare with MATLAB Evalution result
We compare the evaluation results of matlab and python in [DMVSNet](https://github.com/DIVE128/DMVSNet) and [TransMVSNet](https://github.com/megvii-research/TransMVSNet). The point clouds are download from their offitial repositories. The overall results obtained from this implementation are basically the same with MATLAB code but more fast and memory saving, can be used during experiments. Results are shown bellow:

|Method|acc.(mm)|comp.(mm)|overall(mm)|num_workers|time|CPU%|
|------|--------|---------|-----------|---- |---- |----|
|TransMVSNet(matlab)|0.3210|0.2890|0.3050|12|1h17m|1200%|
|TransMVSNet(python)|0.3206|0.2894|0.3050|2|45m|200%|
|DMVSNet(matlab)|0.3495|0.2767|0.3131|12|1h17m|1200%|
|DMVSNet(python)|0.3466|0.2789|0.3128|2|46m  |200%|

A larger `voxel_factor`(>1.28) may make the evaluation results worse than matlab's evaluation results, but the evaluation speed will be faster and can be used to quickly evaluate whether your innovations improve the final overall results.