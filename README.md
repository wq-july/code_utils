# Code Utils

## 介绍

这个代码库是个人做的一个小项目，实现一些基本的slam底层算法，包含视觉和激光的一些常用算法。

## 使用

- docker: 提供了dockerfile文件可以手动构建镜像，提供了shell脚本快速的创建docker镜像以及进入容器；
- 编译

``` bash
# 由于dockerhub近期网络不稳定，镜像放在了阿里云服务器，先配置阿里云服务器，然后拉取镜像；

git clone https://github.com/wq-july/code_utils.git

cd code_utils

# 首次使用需要拉取镜像
bash ./scrips/docker.sh init

# 之后直接使用脚本进入到容器中
bash ./scripts/docker.sh

# 脚本编译
bash ./scripts/build.sh

# 手动编译
mkdir build \ 
&& cd build \
&& cmake .. \
&& make -j4

```

- 模型转换

```bash
python3 convert2onnx/convert_superpoint_to_onnx.py --weight_file superpoint_pth_file_path --output_dir superpoint_onnx_file_dir
python3 convert2onnx/convert_superglue_to_onnx.py --weight_file superglue_pth_file_path --output_dir superglue_onnx_file_dir

```

## TODOs

### 基本算法

- [x] imu预积分；
- [x] kdtree搜索K邻近点；
- [x] 实现voxel版本的最近点搜索；
- [x] 实现非线性优化器；
  - [x] 实现GN优化细节；
  - [x] 实现LM优化细节；
  - [ ] 实现Dog-Leg
- [ ] 简单的EKF算法实现；

### lidar部分算法

  - [x] vicp；
  - [x] gicp；
  - [ ] vgicp；
  - [x] ndt；
  - [x] 降采样算法；


### 视觉部分算法

- 初始化相关
  - [ ] H矩阵和F矩阵求解和反解；
  - [ ] SFM算法；

- 前端跟踪算法
  - [x] 实现特征点提取和的匹配的集成算法，基于opencv版本；
  - [x] SuperPoint, SuperGlue；
  - [ ] 异常值剔除算法；
  
- 回环检测相关算法
  - [ ] dbow3，fbow基于词袋传统的算法；
  - [ ] 基于深度学习相关回环检测算法；
  
- 边缘化相关算法
  - [ ] 实现一个简单的边缘化demo算法；