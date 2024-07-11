#!/bin/zsh

# 设置 locale
export LC_ALL=C

# 执行 CMake 配置，将项目构建文件生成到构建目录中
cmake -B build -G Ninja

cmake --build build

./build/hello_world
