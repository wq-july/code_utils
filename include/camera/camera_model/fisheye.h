#pragma once

#include "camera/camera_model/camera_model.h"

namespace Camera {

class FishEyeKB : public CameraBase {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
 public:
  FishEyeKB(const CameraConfig::FishEyeConfig& config);
  ~FishEyeKB() = default;

  /**
   * @brief 相机坐标系下的三维点投影到像素平面, 根据鱼眼模型进行特征点投影
   * @param p3D 三维点
   * @return 像素坐标
   */
  Eigen::Vector2d CameraP3dToPix(const Eigen::Vector3d& p3d) const override;

  /**
   * @brief 反投影, 根据鱼眼模型反投影
   * @param p2d 特征点
   * @return 归一化坐标
   */
  Eigen::Vector3d PixToCameraP3d(const Eigen::Vector2d& p2d) const override;

  std::vector<Eigen::Vector3d> DistPixToCameraP3d(
      const std::vector<Eigen::Vector2d>& points) const override;

 private:
  CameraConfig::FishEyeConfig config_;
};
}  // namespace Camera
