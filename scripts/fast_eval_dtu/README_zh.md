简体中文 | [English](README.md)

# Fast DTU Evaluation Using GPU with Python
这个仓库实现了利用GPU来加速DTU上的点云评估过程，使用Python和CUDA实现

## 用MATLAB评估存在的问题
在DTU上进行点云数据评估时，一般会安装matlab并且通过DTU官方提供的matlab代码来评估点云结果，这一过程非常耗时，动辄需要几个小时甚至半天的时间，要想加快评估速度一般会使用多进程（matlab中的`parfor`）进行评估，这在多人使用的linux服务器上会消耗大量CPU资源导致其他人使用时CPU卡顿。DTU上的评估过程实际上是计算预测点云和真实点云之间的[倒角距离](https://www.youtube.com/watch?v=P4IyrsWicfs)，也就是DTU中定义的accuracy和completeness，而使用这一过程使用matlab耗时的主要原因是计算过程全部在CPU上完成，没有利用到GPU进行加速。为了解决这个问题，参考[ChamferDistancePytorch](https://github.com/ThibaultGROUEIX/ChamferDistancePytorch)和[MVS_Evaluation_Benchmark](https://github.com/ToughStoneX/MVS_Evaluation_Benchmark)这两个仓库，本仓库利用python和cuda实现了利用GPU来加速DTU的点云评估



## 使用方式
1. 安装环境依赖
```bash
pip install -r requirements.txt
cd chamfer3D && python setup.py install # build and install chamfer3D package
```
2. 准备评估数据
- 下载GT点云： STL [Point clouds](http://roboimagedata2.compute.dtu.dk/data/MVS/Points.zip) 和 [Sample Set](http://roboimagedata2.compute.dtu.dk/data/MVS/SampleSet.zip) 然后进行解压, 并且把 `Points/Points/stl/stlxxx_total.ply` 文件复制到 `SampleSet/MVS Data/Points/stl` 文件夹。
- 准备好要评估的算法的点云文件, 按照以下格式进行命名：`{method}{scanid}_l3.ply` ，比如 `mvsnet001_l3.ply`。
准备好数据集后，文件目录看起来像如下这样:
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

3. 进行评估
- 将`eval_dtu.sh`中的 `--pred_dir` 和 `--gt_dir` 设为你的路径， 运行如下命令:
```bash
bash eval_dtu.sh
# or 
# CUDA_VISIBLE_DEVICES=0 python eval_dtu.py --method mvsnet --pred_dir "./Preidct/mvsnet/" --gt_dir "./SampleSet/MVS Data" --save --num_workers 1
```

## 主要耗时分析：
在matlab的评估代码中，主要的耗时主要来自于点云的下采样(`reducePts_haa`函数)和倒角距离的计算（`MaxDistCP`函数），点云下采样部分和倒角距离的计算都使用了的KD树结构和最近邻算法，由于KD树无法用GPU加速，所以我们利用open3d库中体素下采样代替matlab中的点云下采样来进行加速，同时利用CUDA实现的倒角距离来利用GPU进行加速。需要特别注意的是，由于使用了open3d中的体素下采样，所以下采样后点云会和matlab中的下采样结果有所差别，为了尽量与matlab的结果进行对齐，我在代码中加了一个超参数`voxel_factor`进行调节，调节这个参数会改变open3d的体素下采样尺寸，实验中，设置为1.28基本近似matlab的结果。

## Compare with MATLAB Evalution result
下表对比了在[DMVSNet](https://github.com/DIVE128/DMVSNet) 和 [TransMVSNet](https://github.com/megvii-research/TransMVSNet)两个算法上 matlab 和 python 的评估结果， 用来评估的点云从算法对应的github下载。 可以看到，overall基本与MATLAB代码的结果相同，并且速度更快，CPU使用更少，如下所示：


|Method|acc.(mm)|comp.(mm)|overall(mm)|num_workers|time|CPU%|
|------|--------|---------|-----------|---- |---- |----|
|TransMVSNet(matlab)|0.3210|0.2890|0.3050|12|1h17m|1200%|
|TransMVSNet(python)|0.3206|0.2894|0.3050|2|45m|200%|
|DMVSNet(matlab)|0.3495|0.2767|0.3131|12|1h17m|1200%|
|DMVSNet(python)|0.3466|0.2789|0.3128|2|46m  |200%|


更大的`voxel_factor`（>1.28）可能会让评估结果比matlab的评估结果差，但评估速度会更快，所以也可以用来快速评估你的改进是否有效。