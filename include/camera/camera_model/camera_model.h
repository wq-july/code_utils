// From ORB-SLAM3
#pragma once

#include "Eigen/Core"
#include "opencv2/core/core.hpp"
#include "opencv2/core/eigen.hpp"

namespace Camera {

// 相机模型类，为了去畸变等必要的操作，其他SFM相关的操作不会放在这个类中
class CameraBase {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
 public:
  CameraBase() {}
  ~CameraBase() = default;

  /**
   * @brief 相机坐标系下的三维点投影到像素平面(直接乘上K矩阵)
   * @param p3D 三维点
   * @return 像素坐标
   */
  virtual Eigen::Vector2d CameraP3dToPix(const Eigen::Vector3d& p3d) const;

  /**
   * @brief 反投影, 理想模型下无畸变的点直接反乘上K矩阵获得归一化3D点坐标
   * @param p2d 特征点
   * @return 归一化坐标
   */
  virtual Eigen::Vector3d PixToCameraP3d(const Eigen::Vector2d& p2d) const;
  /**
   * @brief
   * 反投影，将畸变模型融合在这个函数中，也就是知道图像上的像素点直接得到相机系下的3D点归一化坐标
   * 这个需要重载，根据不同的相机模型来
   * @param p2D 特征点
   * @return 归一化坐标
   */
  virtual std::vector<Eigen::Vector3d> DistPixToCameraP3d(
      const std::vector<Eigen::Vector2d>& points) const;

  /**
   * @brief 计算去畸变图像的边界，也需要重载
   *
   * @param[in] img            需要计算边界的图像
   */
  void ComputeUndistortedImageBounds(const cv::Mat& img);

 public:
  double inv_fx_ = 0.0;
  double inv_fy_ = 0.0;
  double focal_length_ = 0.0;
  Eigen::Matrix3d K_ = Eigen::Matrix3d::Zero();
  cv::Mat K_mat_;
  Eigen::Vector4d dist_coef_ = Eigen::Vector4d::Zero();
  cv::Mat dist_coef_mat_;
  Eigen::Vector4d undist_bounds_ = Eigen::Vector4d::Zero();
};

}  // namespace Camera
