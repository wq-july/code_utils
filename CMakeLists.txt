cmake_minimum_required(VERSION 2.8.3)
project(code_utils)

set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_CXX_FLAGS "-std=c++14")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -Wall -g -fPIC -fopenmp")

#set(EIGEN_INCLUDE_DIR "/usr/local/include/eigen3")
find_package(Eigen3 REQUIRED)
find_package(GTest REQUIRED)
# 寻找 yaml-cpp 库
find_package(yaml-cpp REQUIRED)

include_directories(
  ${EIGEN3_INCLUDE_DIR}
  ${GTEST_INCLUDE_DIRS}  # Include Google Test headers
  ${PROJECT_SOURCE_DIR}/include
)

add_library(code_utils_libs
    include/common/sensor_data.hpp
    include/common/lru.hpp
    src/imu/imu_processor.cc
    src/imu/preintegration.cc
    src/util/logger.cc
    src/util/time.cc
    src/util/config.cc
    src/common/state.cc
    src/common/so3.cc
)

target_link_libraries(code_utils_libs ${OpenCV_LIBS} ${CERES_LIBRARIES} yaml-cpp)

option(BUILD_TESTS "Build tests" ON)
if(BUILD_TESTS)
add_executable(imu_test test/imu_test.cc)
target_link_libraries(imu_test code_utils_libs ${GTEST_LIBRARIES})
endif()

