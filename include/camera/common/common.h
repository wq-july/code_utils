#pragma once

#include "Eigen/Dense"
#include "glog/logging.h"
#include "opencv2/core/core.hpp"

namespace Camera {

namespace {
constexpr double Epsilon = 1e-3;
}

template <int rows, int cols>
bool CHECK_EQUAL_MAT(const cv::Mat& mat, const Eigen::Matrix<double, rows, cols>& matrix) {
  // LOG(INFO) << mat.cols - cols << mat.rows - rows << std::endl;
  if (rows != mat.rows || cols != mat.cols) {
    LOG(ERROR) << "wrong mat size\n";
    return false;
  }
  for (int i = 0; i < rows; i++)
    for (int j = 0; j < cols; j++)
      if (std::fabs(mat.at<double>(i, j) - (matrix(i, j)) > Epsilon)) {
        LOG(ERROR) << "cv mat:\n" << mat << std::endl;
        LOG(ERROR) << "eig mat:\n" << matrix << std::endl;
        return false;
      }
  return true;
}

template <typename T, int rows, int cols>
bool CHECK_EQUAL_MAT(const Eigen::Matrix<T, rows, cols>& matrix1,
                     const Eigen::Matrix<T, rows, cols>& matrix2) {
  for (int i = 0; i < rows; i++)
    for (int j = 0; j < cols; j++)
      if (std::fabs(matrix1(i, j) - (matrix2(i, j)) > Epsilon)) {
        LOG(ERROR) << "eig mat 1:\n" << matrix1 << std::endl;
        LOG(ERROR) << "eig mat 2:\n" << matrix2 << std::endl;
        return false;
      }
  return true;
}

}  // namespace Camera
