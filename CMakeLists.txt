cmake_minimum_required(VERSION 2.8.3)
project(code_utils)

set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_CXX_FLAGS "-std=c++14")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -Wall -g -fPIC -fopenmp")

#set(EIGEN_INCLUDE_DIR "/usr/local/include/eigen3")
find_package(Eigen3 REQUIRED)

include_directories(
  ${EIGEN3_INCLUDE_DIR}
  ${PROJECT_SOURCE_DIR}/include
)

add_library(code_utils_libs
    src/sensor/imu.cc
    src/sensor/imu.cc
    src/util/logger.cc
    src/util/time.cc)


target_link_libraries(code_utils_libs ${OpenCV_LIBS} ${CERES_LIBRARIES})


add_executable(imu_imp bin/imu_imp.cc)
target_link_libraries(imu_imp code_utils_libs)