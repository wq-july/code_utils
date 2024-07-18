cmake_minimum_required(VERSION 2.8.3)
project(code_utils)

set(CMAKE_BUILD_TYPE "Debug")
set(CMAKE_CXX_FLAGS "-std=c++17")
set(CMAKE_CXX_FLAGS_DEBUG "-O3 -Wall -g -fPIC -fopenmp")
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
cmake_policy(SET CMP0028 NEW)

# 添加这行来包含您的 FindGLog.cmake 文件
list(APPEND CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake")

find_package(Eigen3 REQUIRED)
find_package(Ceres REQUIRED)
find_package(GTest REQUIRED)
find_package(yaml-cpp REQUIRED)
find_package(OpenCV 4.10.0 REQUIRED)
find_package(PCL REQUIRED)
#sophus need fmt
find_package(fmt REQUIRED)
find_package(Sophus REQUIRED)
find_package(gflags REQUIRED)
find_package(Glog REQUIRED)
find_package(Epoxy REQUIRED)
find_package(OpenGL REQUIRED COMPONENTS EGL)
find_package(Pangolin REQUIRED)
find_package(CUDA REQUIRED)
find_package(Protobuf REQUIRED)

include_directories(
  ${CERES_INCLUDE_DIR}
  ${CUDA_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIR}
  ${EPOXY_INCLUDE_DIRS}
  ${GTEST_INCLUDE_DIRS}
  ${GFLAGS_INCLUDE_DIRS}
  ${GLOG_INCLUDE_DIRS}
  ${OPENGL_INCLUDE_DIRS}
  ${Pangolin_INCLUDE_DIRS}
  ${PCL_INCLUDE_DIRS}
  ${Protobuf_INCLUDE_DIRS}
  ${SOPHUS_INCLUDE_DIRS}
  ${PROJECT_SOURCE_DIR}/include
  ${PROJECT_SOURCE_DIR}/src
  "usr/local/include/tsl"
)

# Specify proto files
set(PROTO_FILES
    ${CMAKE_SOURCE_DIR}/protos/all_config.proto
    ${CMAKE_SOURCE_DIR}/protos/camera.proto
    ${CMAKE_SOURCE_DIR}/protos/imu.proto
    ${CMAKE_SOURCE_DIR}/protos/tensorRT.proto
    ${CMAKE_SOURCE_DIR}/protos/utils.proto
)

# Output directory for generated files
set(GENERATED_DIR ${CMAKE_SOURCE_DIR}/protos/pb)
file(MAKE_DIRECTORY ${GENERATED_DIR})

# Generate C++ files from proto files
foreach(PROTO_FILE ${PROTO_FILES})
    get_filename_component(PROTO_NAME ${PROTO_FILE} NAME_WE)
    add_custom_command(
        OUTPUT ${GENERATED_DIR}/${PROTO_NAME}.pb.cc ${GENERATED_DIR}/${PROTO_NAME}.pb.h
        COMMAND ${Protobuf_PROTOC_EXECUTABLE}
        ARGS --cpp_out=${GENERATED_DIR} -I ${CMAKE_SOURCE_DIR}/protos ${PROTO_FILE}
        DEPENDS ${PROTO_FILE}
        COMMENT "Generating C++ source files from ${PROTO_FILE}"
    )
    list(APPEND GENERATED_SRCS ${GENERATED_DIR}/${PROTO_NAME}.pb.cc)
    list(APPEND GENERATED_HDRS ${GENERATED_DIR}/${PROTO_NAME}.pb.h)
endforeach()

add_library(proto_files ${GENERATED_SRCS} ${GENERATED_HDRS})
target_include_directories(proto_files PUBLIC ${GENERATED_DIR})
target_link_libraries(proto_files ${Protobuf_LIBRARIES})


add_library(code_utils_libs
    src/imu/imu_processor.cc
    src/imu/preintegration.cc
    src/util/logger.cc
    src/util/time.cc
    src/util/math.cc
    src/util/utils.cc
    src/common/state.cc
    src/common/kdtree.cc
    src/common/voxel.cc
    src/common/voxel_map.cc
    src/optimizer/optimizer.cc
    src/optimizer/curve_fitter.cc
    src/optimizer/slam_data_struct.cc
    src/lidar/match.cc
    src/lidar/filter.cc
    src/camera/camera_model/fisheye.cc
    src/camera/camera_model/pinhole.cc
    src/camera/feature_manager.cc
    src/camera/super_point.cc
    src/camera/super_glue.cc
    src/camera/open_stereo/inference.cc
    src/camera/open_stereo/transforms.cc
    src/camera/pnp_solver.cc
    src/tensorRT/generic.cc
    src/tensorRT/logger.cc
)

target_link_libraries(code_utils_libs
  ${CERES_LIBRARIES}
  ${CUDA_LIBRARIES}
  ${EPOXY_LIBRARIES}
  ${OpenCV_LIBS}
  ${PCL_LIBRARIES}
  ${Pangolin_LIBRARIES}
  ${Protobuf_LIBRARIES}
  ${SOPHUS_LIBRARIES}
  fmt
  glog
  gflags
  nvinfer
  nvonnxparser
  OpenGL::EGL
  proto_files
  tbb
  yaml-cpp
)

option(BUILD_TESTS "Build tests" ON)
if(BUILD_TESTS)
  add_executable(imu_test bin/test/imu_test.cc)
  target_link_libraries(imu_test code_utils_libs ${GTEST_LIBRARIES})

  add_executable(search_test bin/test/search_test.cc)
  target_link_libraries(search_test code_utils_libs ${GTEST_LIBRARIES})

  add_executable(search_gui_test bin/test/search_gui_test.cc)
  target_link_libraries(search_gui_test code_utils_libs ${GTEST_LIBRARIES})

  add_executable(utils_test bin/test/utils_test.cc)
  target_link_libraries(utils_test code_utils_libs ${GTEST_LIBRARIES})

  add_executable(optimizer_test bin/test/optimizer_test.cc)
  target_link_libraries(optimizer_test code_utils_libs ${GTEST_LIBRARIES})

  add_executable(ceres_test bin/test/ceres_test.cc)
  target_link_libraries(ceres_test code_utils_libs ${GTEST_LIBRARIES})

  add_executable(point_test bin/test/point_test.cc)
  target_link_libraries(point_test code_utils_libs ${GTEST_LIBRARIES})

  add_executable(point_cloud_test bin/test/point_cloud_test.cc)
  target_link_libraries(point_cloud_test code_utils_libs ${GTEST_LIBRARIES})

  add_executable(pangolin_test bin/test/pangolin_test.cc)
  target_link_libraries(pangolin_test code_utils_libs ${GTEST_LIBRARIES})

  add_executable(match_gui_test bin/test/match_gui_test.cc)
  target_link_libraries(match_gui_test code_utils_libs ${GTEST_LIBRARIES})

  add_executable(cv_test bin/test/cv_test.cc)
  target_link_libraries(cv_test code_utils_libs ${GTEST_LIBRARIES})
endif()

add_executable(hello_world bin/hello_world.cc)
target_link_libraries(hello_world code_utils_libs)

add_executable(open_stereo bin/open_stereo.cc)
target_link_libraries(open_stereo code_utils_libs)