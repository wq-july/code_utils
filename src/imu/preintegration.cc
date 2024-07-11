#include "imu/preintegration.h"

namespace IMU {
IMUPreIntegration::IMUPreIntegration(const IMUConfig::PreIntegration& config) {
  bg_ = Eigen::Vector3d(config.init_bg().x(), config.init_bg().y(), config.init_bg().z());
  ba_ = Eigen::Vector3d(config.init_ba().x(), config.init_ba().y(), config.init_ba().z());
  const float ng_square = config.noise_gyr() * config.noise_gyr();
  const float na_square = config.noise_acc() * config.noise_acc();
  gyr_acc_noise_.diagonal() << ng_square, ng_square, ng_square, na_square, na_square, na_square;
}

void IMUPreIntegration::Update(const IMUData& imu) {
  if (latest_time_ < 0.0) {
    latest_time_ = imu.timestamp_;
    return;
  }

  const double dt = imu.timestamp_ - latest_time_;
  latest_time_ = imu.timestamp_;

  // 首先根据传感器测量模型，将偏差量去掉
  Eigen::Vector3d acc = imu.acc_ - ba_;
  Eigen::Vector3d gyr = imu.gyr_ - bg_;

  // 然后更新dv和dp的观测量，也就是忽略二阶小量的预积分量，也就是带有误差的积分量
  dp_ = dp_ + dv_ * dt + 0.5 * dq_.matrix() * acc * dt * dt;
  dv_ = dv_ + dq_.matrix() * acc * dt;

  // 计算相关中间量
  Eigen::Matrix3d acc_hat = Sophus::SO3d::hat(acc);
  Eigen::Vector3d omega = gyr * dt;
  Eigen::Matrix3d right_jacobian = Sophus::SO3d::leftJacobian(-omega);
  Sophus::SO3d delta_rot = Sophus::SO3d::exp(omega);

  // 计算零偏噪声传递的系数
  Eigen::Matrix<double, PreIntegrationDims, PreIntegrationDims> A =
      Eigen::Matrix<double, PreIntegrationDims, PreIntegrationDims>::Identity();
  Eigen::Matrix<double, PreIntegrationDims, WhiteNoiseDims> B =
      Eigen::Matrix<double, PreIntegrationDims, WhiteNoiseDims>::Zero();

  A.block<3, 3>(ROT, ROT) = delta_rot.matrix().transpose();
  A.block<3, 3>(VEL, ROT) = -dq_.matrix() * acc_hat * dt;
  A.block<3, 3>(POS, ROT) = -0.5 * dq_.matrix() * acc_hat * dt * dt;
  A.block<3, 3>(POS, VEL) = Eigen::Matrix3d::Identity() * dt;

  B.block<3, 3>(ROT, NG) = right_jacobian * dt;
  B.block<3, 3>(VEL, NA) = dq_.matrix() * dt;
  B.block<3, 3>(POS, NA) = 0.5 * dq_.matrix() * dt * dt;

  // 更新各雅可比，见式(4.39)
  dp_dba_ = dp_dba_ + dv_dba_ * dt - 0.5 * dq_.matrix() * dt * dt;
  dp_dbg_ = dp_dbg_ + dv_dbg_ * dt - 0.5 * dq_.matrix() * dt * dt * acc_hat * dr_dbg_;
  dv_dba_ = dv_dba_ - dq_.matrix() * dt;
  dv_dbg_ = dv_dbg_ - dq_.matrix() * dt * acc_hat * dr_dbg_;

  cov_ = A * cov_ * A.transpose() + B * gyr_acc_noise_ * B.transpose();
  // 更新旋转量相关的雅可比矩阵和旋转的预积分量
  dr_dbg_ = delta_rot.matrix().transpose() * dr_dbg_ - right_jacobian * dt;

  dq_ = dq_ * delta_rot;

  // 增量积分时间
  dt_ += dt;
}

Sophus::SO3d IMUPreIntegration::GetDeltaRotation(const Eigen::Vector3d& bg) {
  return dq_ * Sophus::SO3d::exp(dr_dbg_ * (bg - bg_));
}

Eigen::Vector3d IMUPreIntegration::GetDeltaVelocity(const Eigen::Vector3d& bg,
                                                    const Eigen::Vector3d& ba) {
  return dv_ + dv_dbg_ * (bg - bg_) + dv_dba_ * (ba - ba_);
}

Eigen::Vector3d IMUPreIntegration::GetDeltaPosition(const Eigen::Vector3d& bg,
                                                    const Eigen::Vector3d& ba) {
  return dp_ + dp_dbg_ * (bg - bg_) + dp_dba_ * (ba - ba_);
}

State IMUPreIntegration::Predict(const State& start, const Eigen::Vector3d& gravity) const {
  Sophus::SO3d rot_j = start.rot_ * dq_;
  Eigen::Vector3d vel_j = start.rot_ * dv_ + start.vel_ + gravity * dt_;
  Eigen::Vector3d pos_j =
      start.rot_ * dp_ + start.trans_ + start.vel_ * dt_ + 0.5f * gravity * dt_ * dt_;

  auto state = State(start.timestamp_ + dt_, rot_j, pos_j, vel_j);
  state.bg_ = bg_;
  state.ba_ = ba_;
  return state;
}

void IMUPreIntegration::Reset() {
  dp_.setZero();
  dv_.setZero();
  dq_ = Sophus::SO3d();

  // 零偏
  bg_.setZero();
  ba_.setZero();

  // Jacobian
  dr_dbg_.setZero();
  dv_dbg_.setZero();
  dv_dba_.setZero();
  dp_dbg_.setZero();
  dp_dba_.setZero();

  cov_.setZero();

  dt_ = 0.0;
  latest_time_ = -1.0;
}

}  // namespace IMU