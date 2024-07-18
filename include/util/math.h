#pragma once

#include <math.h>

#include <cassert>
#include <cmath>
#include <cstdint>
#include <opencv2/core/types.hpp>
#include <vector>

#include "Eigen/Dense"
#include "ceres/ceres.h"

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

// SO(3)中的hat函数
inline Eigen::Matrix3d So3Symmetric(const Eigen::Vector3d& v) {
  Eigen::Matrix3d m;
  m << 0, -v.z(), v.y(), v.z(), 0, -v.x(), -v.y(), v.x(), 0;
  return m;
}

// SE(3)中的hat函数
inline Eigen::Matrix4d Se3Symmetric(const Eigen::Matrix<double, 6, 1>& v) {
  Eigen::Matrix4d m;
  m.setZero();
  m.block<3, 3>(0, 0) = So3Symmetric(v.head<3>());
  m.block<3, 1>(0, 3) = v.tail<3>();
  return m;
}

inline Eigen::Matrix3d Exp(const Eigen::Vector3d& omega) {
  double theta = omega.norm();
  Eigen::Matrix3d Omega_hat = So3Symmetric(omega);
  if (theta < 1e-10) {
    // 当theta接近0时，使用近似展开
    return Eigen::Matrix3d::Identity() + Omega_hat;
  } else {
    // 使用罗德里格斯公式
    Eigen::Matrix3d Omega_hat_squared = Omega_hat * Omega_hat;
    return Eigen::Matrix3d::Identity() + (std::sin(theta) / theta) * Omega_hat +
           ((1 - std::cos(theta)) / (theta * theta)) * Omega_hat_squared;
  }
}

Eigen::Vector3d ComputeCentroid(const std::vector<Eigen::Vector3d>& points);

bool PlaneFit(const std::vector<Eigen::Vector3d>& points,
              Eigen::Vector4d* const plane_coeffs,
              double eps = 1e-3);

bool Line3DFit(std::vector<Eigen::Vector3d>& points,
               Eigen::Vector3d* const origin,
               Eigen::Vector3d* const dir,
               double eps = 0.1);

// 计算一组2D点的方差
double CalculateVariance(const std::vector<Eigen::Vector2d>& points);

}  // namespace Math

}  // namespace Utils