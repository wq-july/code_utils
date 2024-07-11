#!/bin/bash

# 设置 locale
export LC_ALL=C

# 设置要查找的目录
BUILD_DIR="./build"

# 检查是否存在 build 目录，如果不存在，则创建它
if [ ! -d "$BUILD_DIR" ]; then
  echo "Build directory does not exist. Creating..."
  mkdir -p "$BUILD_DIR"
fi

# 切换到 build 目录
cd "$BUILD_DIR" || { echo "Cannot change directory to $BUILD_DIR"; exit 1; }

# 检查是否需要重新编译
if [ ! -f "CMakeCache.txt" ]; then
  echo "CMakeCache.txt not found. Running CMake..."
  cmake ..
fi

# 使用 make 编译代码
echo "Building project..."
# 使用所有可用的 CPU 核心并行编译
NUM_CORES=$(grep -c ^processor /proc/cpuinfo)
make -j$NUM_CORES

# 查找所有符合模式的可执行文件（例如，以 _test 结尾的文件）
TEST_FILES=$(find . -type f -executable -name '*_test')

if [ -z "$TEST_FILES" ]; then
  echo "No test files found."
  exit 1
fi

# 运行所有测试文件
for TEST_FILE in $TEST_FILES; 
do
    echo "Running test: $TEST_FILE"
    ./"$TEST_FILE"
    RESULT=$?
    if [ $RESULT -ne 0 ]; then
        echo "Test failed: $TEST_FILE"
        exit 1
    else
        echo "Test passed: $TEST_FILE"
    fi
done

echo "All tests passed."
