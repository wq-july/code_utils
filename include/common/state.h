#pragma once

#include <Eigen/Core>
#include <iostream>

#include "sophus/so3.hpp"

namespace Common {

struct State {
  State();
  // 防止隐式类型转换
  explicit State(double time, const Sophus::SO3d& rot = Sophus::SO3d(),
                       const Eigen::Vector3d& trans = Eigen::Vector3d::Zero(),
                       const Eigen::Vector3d& vel = Eigen::Vector3d::Zero(),
                       const Eigen::Vector3d& bg = Eigen::Vector3d::Zero(),
                       const Eigen::Vector3d& ba = Eigen::Vector3d::Zero());

  State(double time, const Eigen::Isometry3d& pose,
              const Eigen::Vector3d& vel = Eigen::Vector3d::Zero());

  State(double time, const Eigen::Matrix3d& rot, const Eigen::Vector3d& pos,
              const Eigen::Vector3d& vel = Eigen::Vector3d::Zero());

  Eigen::Isometry3d GetSE3() const;

  friend std::ostream& operator<<(std::ostream& os, const State& s);

  double timestamp_ = 0;                             // 时间
  Sophus::SO3d rot_;                                 // 旋转
  Eigen::Vector3d trans_ = Eigen::Vector3d::Zero();  // 平移
  Eigen::Vector3d vel_ = Eigen::Vector3d::Zero();    // 速度
  Eigen::Vector3d bg_ = Eigen::Vector3d::Zero();     // gyro 零偏
  Eigen::Vector3d ba_ = Eigen::Vector3d::Zero();     // acce 零偏
};

}  // namespace Common
