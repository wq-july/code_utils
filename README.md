# Code Utils

## 介绍

这个代码库是个人做的一个小项目，实现一些基本的slam底层算法，包含视觉和激光的一些常用算法。

## 使用

- docker: 提供了dockerfile文件可以手动构建镜像，提供了shell脚本快速的创建docker镜像以及进入容器；
- 编译

``` bash
git clone https://github.com/wq-july/code_utils.git

cd code_utils

# 首次使用会创建镜像，然后进入到容器中
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
- [] vgicp；
- [x] ndt；
- [x] 降采样算法；

### 视觉部分算法

- 初始化相关
  - [ ] H矩阵和F矩阵求解和反解，基于opencv和自己实现；
  - [ ] SFM算法；

- 前端跟踪算法
  - [ ] 实现特征点提取和的匹配的集成算法，基于opencv版本；
  - [ ] 实现基于深度学习的特征提取和匹配算法；
  - [ ] 异常值剔除算法；
  
- 回环检测相关算法
  - [ ] dbow3，fbow基于词袋传统的算法；
  - [ ] 基于深度学习相关回环检测算法；
  
- 边缘化相关算法
  - [ ] 实现一个简单的边缘化demo算法；

## 其他

### 基本的配置类

- config类 （实现其他类初始化参数配置）
  - [x] 实现imu预积分类的参数配置；
  - [x] imu类相关的参数配置，imu预积分是这个类的成员变量；
  - [x] 实现logger类相关的配置；
  - [x] 实现timer类相关的参数配置；
  - [ ] 实现kdtree类相关参数配置；
  - [ ] 实现voxel_map类相关参数配置；

- time类
  - [x] 实现代码时间统计，开始，停止，记录持续时间，可以设置时间单位等；
  - [ ] 进一步实现更简单的使用方法，因为通常统计一段代码段需要至少三行代码，实现一个函数，将需要统计的代码段放到这个作用域下就可以；

- log类
  - [x] 实现基本的一个日志类，便于统计代码中的日志，支持日志输出到文件，日志不同等级，输出到控制台的等级，日志颜色，日期等；
  - [ ] 继承glog类，实现自己想要的一些基本的功能；

- util类
  - [x] 哈希函数，实现多种哈希函数，主要是用来降低哈希碰撞的概率；
  - [x] 计算点之间的距离；
  - [x] 实现根据输入函数曲线生成随机仿真数据的函数；

- math类
  - [x] 计算一个容器中的方差，均值；
