#pragma once

#include <math.h>

#include <cassert>
#include <cmath>
#include <cstdint>
#include <vector>

#include "Eigen/Core"
#include "Eigen/Dense"

#include "ceres/ceres.h"
#include "glog/logging.h"

namespace Utils {

namespace Math {

namespace ConstMath {
constexpr double kGravity = -9.81;                   // 重力
constexpr const double kDegreeToRad = M_PI / 180.0;  // 角度转为弧度
constexpr double kRadToDegree = 180.0 / M_PI;        // 弧度转角度
// 非法定义
constexpr uint32_t kINVALID = std::numeric_limits<uint32_t>::max();
};  // namespace ConstMath


template <typename ContainerType>
void ComputeMeanAndVariance(const std::vector<uint32_t>& indices,
                            const ContainerType& data,
                            Eigen::Vector3d* const mean,
                            Eigen::Vector3d* const variance) {
  assert(!indices.empty() && "Indices vector cannot be empty!");
  assert(!data.empty() && "data vector cannot be empty!");

  // 计算指定索引处元素的总和
  Eigen::Vector3d sum = Eigen::Vector3d::Zero();
  for (uint32_t index : indices) {
    assert(index < data.size() && "Index out of range");
    sum += data[index];
  }

  // 计算均值
  *mean = sum / indices.size();

  // 计算方差
  Eigen::Vector3d squared_diff_sum = Eigen::Vector3d::Zero();
  for (uint32_t index : indices) {
    Eigen::Vector3d diff = data[index] - *mean;
    squared_diff_sum += diff.cwiseProduct(diff);
  }
  *variance = squared_diff_sum / indices.size();
}

// Normalizes the angle in radians between [-pi and pi).
template <typename T>
inline T NormalizeAngle(const T& angle_radians) {
  // Use ceres::floor because it is specialized for double and Jet types.
  T two_pi(2.0 * ceres::constants::pi);
  return angle_radians - two_pi * ceres::floor((angle_radians + T(ceres::constants::pi)) / two_pi);
}

}  // namespace Math

}  // namespace Utils