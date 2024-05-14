#pragma once

#include <Eigen/Core>
#include <iostream>

#include "sophus/so3.hpp"

namespace Common {

static constexpr double Gravity = -9.81;

struct DIMS {
  // static error state index
  static constexpr uint32_t DIM_STATE = 27u;
  static constexpr uint32_t DIM_SimpleState = 15u;

  static constexpr uint32_t POS = 0u;
  static constexpr uint32_t VEL = 3u;
  static constexpr uint32_t ROT = 6u;
  static constexpr uint32_t BG = 9u;
  static constexpr uint32_t BA = 12u;
  static constexpr uint32_t GRA = 15u;
  static constexpr uint32_t R_L_I = 18u;
  static constexpr uint32_t T_L_I = 21u;
  static constexpr uint32_t R_B_I = 24u;
  // useless
  static constexpr uint32_t T_B_I = 27u;

  // static noise index
  static constexpr uint32_t DIM_NOISE = 12u;
  static constexpr uint32_t N_G = 0u;
  static constexpr uint32_t N_A = 3u;
  static constexpr uint32_t N_BG = 6u;
  static constexpr uint32_t N_BA = 9u;
};

struct SimpleState {
  SimpleState();
  // 防止隐式类型转换
  explicit SimpleState(double time, const Sophus::SO3d& rot = Sophus::SO3d(),
                       const Eigen::Vector3d& trans = Eigen::Vector3d::Zero(),
                       const Eigen::Vector3d& vel = Eigen::Vector3d::Zero(),
                       const Eigen::Vector3d& bg = Eigen::Vector3d::Zero(),
                       const Eigen::Vector3d& ba = Eigen::Vector3d::Zero());

  SimpleState(double time, const Eigen::Isometry3d& pose,
              const Eigen::Vector3d& vel = Eigen::Vector3d::Zero());

  SimpleState(double time, const Eigen::Matrix3d& rot, const Eigen::Vector3d& pos,
              const Eigen::Vector3d& vel = Eigen::Vector3d::Zero());

  Eigen::Isometry3d GetSE3() const;

  friend std::ostream& operator<<(std::ostream& os, const SimpleState& s);

  double timestamp_ = 0;                             // 时间
  Sophus::SO3d rot_;                                 // 旋转
  Eigen::Vector3d trans_ = Eigen::Vector3d::Zero();  // 平移
  Eigen::Vector3d vel_ = Eigen::Vector3d::Zero();    // 速度
  Eigen::Vector3d bg_ = Eigen::Vector3d::Zero();     // gyro 零偏
  Eigen::Vector3d ba_ = Eigen::Vector3d::Zero();     // acce 零偏
};

}  // namespace Common
