cmake_minimum_required(VERSION 2.8.3)
project(code_utils)

set(CMAKE_BUILD_TYPE "Debug")
set(CMAKE_CXX_FLAGS "-std=c++17")
set(CMAKE_CXX_FLAGS_DEBUG "-O3 -Wall -g -fPIC -fopenmp")

# 添加这行来包含您的 FindGLog.cmake 文件
list(APPEND CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake")

find_package(Eigen3 REQUIRED)
find_package(Ceres REQUIRED)
find_package(GTest REQUIRED)
find_package(yaml-cpp REQUIRED)
find_package(OpenCV REQUIRED)
find_package(PCL REQUIRED)
#sophus need fmt
find_package(fmt REQUIRED)
find_package(Sophus REQUIRED)
find_package(gflags REQUIRED)
find_package(Glog REQUIRED)

include_directories(
  ${EIGEN3_INCLUDE_DIR}
  ${CERES_INCLUDE_DIR}
  ${GTEST_INCLUDE_DIRS}
  ${PCL_INCLUDE_DIRS}
  ${GFLAGS_INCLUDE_DIRS}
  ${GLOG_INCLUDE_DIRS}
  ${SOPHUS_INCLUDE_DIRS}
  ${PROJECT_SOURCE_DIR}/include
  "usr/local/include/tsl"
)

add_library(code_utils_libs
    include/common/sensor_data.hpp
    include/optimizer/slam_data_struct.hpp
    include/optimizer/slam_model.hpp
    src/imu/imu_processor.cc
    src/imu/preintegration.cc
    src/util/logger.cc
    src/util/time.cc
    src/util/config.cc
    src/util/math.cc
    src/util/utils.cc
    src/common/state.cc
    src/common/kdtree.cc
    src/common/voxel.cc
    src/common/voxel_map.cc
    src/common/lru.tpp
    src/optimizer/optimizer.cc
    src/optimizer/curve_fitter.cc
)

target_link_libraries(code_utils_libs
  ${OpenCV_LIBS}
  ${CERES_LIBRARIES}
  ${PCL_LIBRARIES}
  ${SOPHUS_LIBRARIES}
  glog
  gflags
  yaml-cpp
  tbb
  fmt
)

option(BUILD_TESTS "Build tests" ON)
if(BUILD_TESTS)
  add_executable(imu_test bin/test/imu_test.cc)
  target_link_libraries(imu_test code_utils_libs ${GTEST_LIBRARIES})

  add_executable(search_test bin/test/search_test.cc)
  target_link_libraries(search_test code_utils_libs ${GTEST_LIBRARIES})

  add_executable(utils_test bin/test/utils_test.cc)
  target_link_libraries(utils_test code_utils_libs ${GTEST_LIBRARIES})

  add_executable(optimizer_test bin/test/optimizer_test.cc)
  target_link_libraries(optimizer_test code_utils_libs ${GTEST_LIBRARIES})

  add_executable(ceres_test bin/test/ceres_test.cc)
  target_link_libraries(ceres_test code_utils_libs ${GTEST_LIBRARIES})
endif()

add_executable(hello_world bin/hello_world.cc)
target_link_libraries(hello_world code_utils_libs)