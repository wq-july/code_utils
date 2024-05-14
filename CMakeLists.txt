cmake_minimum_required(VERSION 2.8.3)
project(code_utils)

set(CMAKE_BUILD_TYPE "Debug")
set(CMAKE_CXX_FLAGS "-std=c++17")
set(CMAKE_CXX_FLAGS_DEBUG "-O3 -Wall -g -fPIC -fopenmp")

# 添加这行来包含您的 FindGLog.cmake 文件
list(APPEND CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake")

find_package(Eigen3 REQUIRED)
find_package(GTest REQUIRED)
find_package(yaml-cpp REQUIRED)
find_package(PCL REQUIRED)
find_package(Sophus REQUIRED)
find_package(gflags REQUIRED)
find_package(Glog REQUIRED)

add_subdirectory(${PROJECT_SOURCE_DIR}/thirdparty/robin-map)

include_directories(
  ${EIGEN3_INCLUDE_DIR}
  ${GTEST_INCLUDE_DIRS}
  ${PCL_INCLUDE_DIRS}
  ${GFLAGS_INCLUDE_DIRS}
  ${GLOG_INCLUDE_DIRS}
  ${SOPHUS_INCLUDE_DIRS}
  ${PROJECT_SOURCE_DIR}/include
  ${PROJECT_SOURCE_DIR}/thirdparty/robin-map/include
)

add_library(code_utils_libs
    include/common/hpp/sensor_data.hpp
    include/common/hpp/lru.hpp
    src/imu/imu_processor.cc
    src/imu/preintegration.cc
    src/util/logger.cc
    src/util/time.cc
    src/util/config.cc
    src/util/math.cc
    src/common/state.cc
    src/common/so3.cc
    src/common/search/kdtree.cc
    src/common/search/voxel.cc
    src/common/search/voxel_map.cc
)


target_link_libraries(code_utils_libs
  PRIVATE tsl::robin_map
  ${OpenCV_LIBS}
  ${CERES_LIBRARIES}
  ${PCL_LIBRARIES}
  ${SOPHUS_LIBRARIES}
  glog
  gflags
  yaml-cpp
  tbb
)

option(BUILD_TESTS "Build tests" ON)
if(BUILD_TESTS)
  add_executable(imu_test test/imu_test.cc)
  target_link_libraries(imu_test code_utils_libs ${GTEST_LIBRARIES})

  add_executable(search_test test/search_test.cc)
  target_link_libraries(search_test code_utils_libs ${GTEST_LIBRARIES})

  add_executable(utils_test test/utils_test.cc)
  target_link_libraries(utils_test code_utils_libs ${GTEST_LIBRARIES})
endif()
