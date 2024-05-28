#!/bin/zsh

# 设置 locale
export LC_ALL=C

# 指定构建目录
BUILD_DIR="build"

# 检查是否存在 build 目录，如果不存在，则创建它
if [ ! -d "$BUILD_DIR" ]; then
  echo "Build directory does not exist. Creating..."
  mkdir -p "$BUILD_DIR"
fi

# 切换到 build 目录
cd "$BUILD_DIR" || { echo "Cannot change directory to $BUILD_DIR"; exit 1; }

# 执行 CMake 配置，将项目构建文件生成到构建目录中
cmake ..

# 使用所有可用的 CPU 核心并行编译
NUM_CORES=$(grep -c ^processor /proc/cpuinfo)
make -j$NUM_CORES

./hello_world
