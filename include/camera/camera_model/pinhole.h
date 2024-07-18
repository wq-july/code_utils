#pragma once
#include "../protos/pb/camera.pb.h"
#include "camera/camera_model/camera_model.h"

namespace Camera {
// 相机模型类，为了去畸变等必要的操作，其他SFM相关的操作不会放在这个类中
class Pinhole : public CameraBase {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
 public:
  Pinhole(const CameraConfig::PinholeConfig& config);
  ~Pinhole() = default;

  /**
   * @brief 相机坐标系下的三维点投影到像素平面(直接乘上K矩阵)
   * @param p3D 三维点
   * @return 像素坐标
   */
  Eigen::Vector2d CameraP3dToPix(const Eigen::Vector3d& p3d) const override;

  /**
   * @brief 反投影, 理想模型下无畸变的点直接反乘上K矩阵获得归一化3D点坐标
   * @param p2d 特征点
   * @return 归一化坐标
   */
  Eigen::Vector3d PixToCameraP3d(const Eigen::Vector2d& p2d) const override;

  std::vector<Eigen::Vector3d> DistPixToCameraP3d(
      const std::vector<Eigen::Vector2d>& points) const override;

 private:
  // 计算公式 \Delta x 和 \Delta x; d_u=[\Delta x,\Delta y]
  void Distortion(const Eigen::Vector2d& p_u, Eigen::Vector2d* const d_u) const;

 private:
  CameraConfig::PinholeConfig config_;
};
}  // namespace Camera
