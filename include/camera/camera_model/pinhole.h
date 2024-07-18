#pragma once
#include "camera/camera_model/camera_model.h"

namespace Camera {
// 相机模型类，为了去畸变等必要的操作，其他SFM相关的操作不会放在这个类中
class Pinhole : public CameraBase {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
 public:
  Pinhole(const CameraConfig::PinholeConfig& config);
  ~Pinhole() = default;

  std::vector<Eigen::Vector3d> DistPixToCameraP3d(
      const std::vector<Eigen::Vector2d>& points) const override;

 private:
  // 计算公式 \Delta x 和 \Delta x; d_u=[\Delta x,\Delta y]
  void Distortion(const Eigen::Vector2d& p_u, Eigen::Vector2d* const d_u) const;

 private:
  CameraConfig::PinholeConfig config_;
};
}  // namespace Camera
