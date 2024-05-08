cmake_minimum_required(VERSION 2.8.3)
project(code_utils)

set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_CXX_FLAGS "-std=c++14")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -Wall -g -fPIC -fopenmp")

#set(EIGEN_INCLUDE_DIR "/usr/local/include/eigen3")
find_package(Eigen3 REQUIRED)
find_package(GTest REQUIRED)

include_directories(
  ${EIGEN3_INCLUDE_DIR}
  ${GTEST_INCLUDE_DIRS}  # Include Google Test headers
  ${PROJECT_SOURCE_DIR}/include
)

add_library(code_utils_libs
    src/imu/imu.cc
    src/util/logger.cc
    src/util/time.cc
    src/util/math.cc
)

target_link_libraries(code_utils_libs ${OpenCV_LIBS} ${CERES_LIBRARIES})


option(BUILD_TESTS "Build tests" ON)
if(BUILD_TESTS)
add_executable(imu_test test/imu_test.cc)
target_link_libraries(imu_test code_utils_libs ${GTEST_LIBRARIES})
endif()

