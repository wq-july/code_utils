#!/bin/zsh

# 指定构建目录
BUILD_DIR="build"

# 创建构建目录并进入
mkdir -p $BUILD_DIR
cd $BUILD_DIR || exit

# 执行 CMake 配置，将项目构建文件生成到构建目录中
cmake ..

# 使用所有可用的 CPU 核心并行编译
NUM_CORES=$(grep -c ^processor /proc/cpuinfo)
make -j$NUM_CORES

./hello_world

# 返回项目根目录
# cd ..
