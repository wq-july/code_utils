#pragma once

#include <vector>

#include "Eigen/Core"
#include "Eigen/Dense"

#include "common/data/imu.h"
#include "common/state.h"
#include "util/config.h"

using namespace Common;
using namespace Common::Data;
using namespace Utils;

namespace IMU {

// TODO, 将下面的几个维度控制开关集成到Common下的Constant中

// 预积分量的噪声，也就是误差项
static constexpr uint32_t PreIntegrationDims = 9u;
// 白噪声，直接来源于加速计和陀螺仪的白噪声
static constexpr uint32_t WhiteNoiseDims = 6u;

static constexpr uint32_t ROT = 0u;
static constexpr uint32_t VEL = 3u;
static constexpr uint32_t POS = 6u;

static constexpr uint32_t NG = 0u;
static constexpr uint32_t NA = 3u;

class IMUPreIntegration {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  IMUPreIntegration(const ImuPreIntegrationConfig& config);

  void Update(const IMUData& imu);

  // 重力可能是一个待优化的变量，如果参与优化的话，会变化
  State Predict(const State& state, const Eigen::Vector3d& gravity) const;

  // 获取修正之后的观测量，bias可以与预积分时期的不同，会有一阶修正
  Sophus::SO3d GetDeltaRotation(const Eigen::Vector3d& bg);

  Eigen::Vector3d GetDeltaVelocity(const Eigen::Vector3d& bg, const Eigen::Vector3d& ba);

  Eigen::Vector3d GetDeltaPosition(const Eigen::Vector3d& bg, const Eigen::Vector3d& ba);

  // TODO
  void Reset();

 private:
  double latest_time_ = -1.0;
  // 预积分的时间量，表示一共往前积分了多久
  double dt_ = 0.0;

  // 累计噪声矩阵，p,v,q
  Eigen::Matrix<double, 9, 9> cov_ = Eigen::Matrix<double, 9, 9>::Zero();
  // 测量噪声矩阵, na, ng
  Eigen::Matrix<double, 6, 6> gyr_acc_noise_ = Eigen::Matrix<double, 6, 6>::Zero();

  // 零偏
  Eigen::Vector3d bg_ = Eigen::Vector3d::Zero();
  Eigen::Vector3d ba_ = Eigen::Vector3d::Zero();

  // 预积分观测量
  Eigen::Vector3d dv_ = Eigen::Vector3d::Zero();
  Eigen::Vector3d dp_ = Eigen::Vector3d::Zero();
  Sophus::SO3d dq_ = Sophus::SO3d();

  // Jacobian
  Eigen::Matrix3d dr_dbg_ = Eigen::Matrix3d::Zero();
  Eigen::Matrix3d dv_dbg_ = Eigen::Matrix3d::Zero();
  Eigen::Matrix3d dv_dba_ = Eigen::Matrix3d::Zero();
  Eigen::Matrix3d dp_dbg_ = Eigen::Matrix3d::Zero();
  Eigen::Matrix3d dp_dba_ = Eigen::Matrix3d::Zero();
};

}  // namespace IMU