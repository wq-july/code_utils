// From ORB-SLAM3
#pragma once

// clang-format off
#include "camera/common/common.h"
// clang-format on

#include "opencv2/calib3d.hpp"
#include "opencv2/core/core.hpp"
#include "opencv2/core/eigen.hpp"

#include "../protos/pb/camera.pb.h"

namespace Camera {

// 相机模型类，为了去畸变等必要的操作，其他SFM相关的操作不会放在这个类中
class CameraBase {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
 public:
  CameraBase() = default;
  ~CameraBase() = default;

  /**
   * @brief 相机坐标系下的三维点投影到像素平面(直接乘上K矩阵)
   * @param p3D 三维点
   * @return 像素坐标
   */
  virtual Eigen::Vector2d CameraP3dToPix(const Eigen::Vector3d& p3d) const {
    Eigen::Vector2d res;
    res.x() = K_(0, 0) * p3d[0] / p3d[2] + K_(0, 2);
    res.y() = K_(1, 1) * p3d[1] / p3d[2] + K_(1, 2);
    return res;
  }

  /**
   * @brief 反投影, 理想模型下无畸变的点直接反乘上K矩阵获得归一化3D点坐标
   * @param p2d 特征点
   * @return 归一化坐标
   */
  virtual Eigen::Vector3d PixToCameraP3d(const Eigen::Vector2d& p2d) const {
    Eigen::Vector3d res;
    res.x() = (p2d.x() - K_(0, 2)) * inv_fx_;
    res.y() = (p2d.y() - K_(1, 2)) * inv_fy_;
    res.z() = 1.0;
    return res;
  }

  /**
   * @brief 计算去畸变图像的边界，也需要重载
   *
   * @param[in] img            需要计算边界的图像
   */
  void ComputeUndistortedImageBounds(const cv::Mat& img) {
    // 如果畸变参数不为0，用OpenCV函数进行畸变矫正
    if (std::fabs(dist_coef_(0, 0)) < 1.0e-6) {
      // 保存矫正前的图像四个边界点坐标： (0,0) (cols,0) (0,rows) (cols,rows)
      std::vector<Eigen::Vector2d> corners_coordinate = {Eigen::Vector2d(0, 0),
                                                         Eigen::Vector2d(img.cols, 0),
                                                         Eigen::Vector2d(0, img.rows),
                                                         Eigen::Vector2d(img.cols, img.rows)};
      std::vector<Eigen::Vector3d> undist_points = DistPixToCameraP3d(corners_coordinate);
      // Undistort corners
      //  [xmin, xmax, ymin, ymax]^T
      // 校正后的四个边界点已经不能够围成一个严格的矩形，因此在这个四边形的外侧加边框作为坐标的边界
      for (const auto& pt : undist_points) {
        auto p2t = CameraP3dToPix(pt);
        if (undist_bounds_(0) > p2t.x()) {
          undist_bounds_(0) = p2t.x();
        }
        if (undist_bounds_(1) < p2t.x()) {
          undist_bounds_(0) = p2t.x();
        }
        if (undist_bounds_(2) > p2t.y()) {
          undist_bounds_(0) = p2t.y();
        }
        if (undist_bounds_(3) < p2t.y()) {
          undist_bounds_(0) = p2t.y();
        }
      }
    } else {
      // 如果畸变参数为0，就直接获得图像边界
      undist_bounds_(0) = 0.0f;
      undist_bounds_(1) = img.cols;
      undist_bounds_(2) = 0.0f;
      undist_bounds_(3) = img.rows;
    }
  }

  /**
   * @brief
   * 反投影，将畸变模型融合在这个函数中，也就是知道图像上的像素点直接得到相机系下的3D点归一化坐标
   * 这个需要重载，根据不同的相机模型来
   * @param p2D 特征点
   * @return 归一化坐标
   */
  virtual std::vector<Eigen::Vector3d> DistPixToCameraP3d(
      const std::vector<Eigen::Vector2d>& points) const;

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
