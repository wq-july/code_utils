#pragma once

#include <Eigen/Core>
#include <iostream>
#include <sophus/se3.hpp>
#include <sophus/so3.hpp>

namespace Common {

struct SimpleState {
  SimpleState();
  // 防止隐式类型转换，因为其中sophus库有点特殊
  explicit SimpleState(double time, const Sophus::SO3d& rot = Sophus::SO3d(),
                       const Eigen::Vector3d& trans = Eigen::Vector3d::Zero(),
                       const Eigen::Vector3d& vel = Eigen::Vector3d::Zero(),
                       const Eigen::Vector3d& bg = Eigen::Vector3d::Zero(),
                       const Eigen::Vector3d& ba = Eigen::Vector3d::Zero());
  SimpleState(double time, const Sophus::SE3d& pose,
              const Eigen::Vector3d& vel = Eigen::Vector3d::Zero());

  Sophus::SE3d GetSE3() const;

  friend std::ostream& operator<<(std::ostream& os, const SimpleState& s);

  double timestamp_ = 0;                             // 时间
  Sophus::SO3d rot_;                                 // 旋转
  Eigen::Vector3d trans_ = Eigen::Vector3d::Zero();  // 平移
  Eigen::Vector3d vel_ = Eigen::Vector3d::Zero();    // 速度
  Eigen::Vector3d bg_ = Eigen::Vector3d::Zero();     // gyro 零偏
  Eigen::Vector3d ba_ = Eigen::Vector3d::Zero();     // acce 零偏
};

}  // namespace Common
