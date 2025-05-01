import numpy as np
import os
from plyfile import PlyData, PlyElement

# 设定npy文件目录和输出的ply文件名
scans = [1,]#4,9,10,11,12,13,15,23,24,29,32,33,34,48,49,62,75,77,110,114,118]
for i in scans:
    npy_dir = f"/data0/zsz/mast3recon/data/dtu/direct_25/scan{i}.npy"
    plyfilename = f"/data0/zsz/mast3recon/data/dtu_test_25/ours{i:03}_l3.ply"
    if os.path.exists(npy_dir) == False:
        print("without", plyfilename)
        continue
    os.makedirs(os.path.dirname(plyfilename), exist_ok=True)
    # 读取所有的npy文件
    vertexs = []
    # for filename in os.listdir(npy_dir):
    #     if filename.endswith(".npy"):
    points = np.load(npy_dir)#os.path.join(npy_dir, filename))
    vertexs.append(points)
    # 将点数据合并
    if len(vertexs[0].shape) == 3: 
        vertexs = np.concatenate(vertexs, axis=1)[0]
    else:
        vertexs = np.concatenate(vertexs, axis=0)
    vertexs = np.array([tuple(v) for v in vertexs], dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4')])

    # 创建用于PlyData的结构
    vertex_all = np.empty(len(vertexs), vertexs.dtype.descr)
    for prop in vertexs.dtype.names:
        vertex_all[prop] = vertexs[prop]

    # 创建PlyElement并写入到ply文件
    el = PlyElement.describe(vertex_all, 'vertex')
    PlyData([el]).write(plyfilename)
    print("Saving the final model to", plyfilename)
