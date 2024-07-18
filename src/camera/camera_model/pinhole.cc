#include "camera/camera_model/pinhole.h"

#include "Eigen/Dense"
#include "glog/logging.h"
#include "opencv2/calib3d.hpp"
namespace Camera {

Pinhole::Pinhole(const CameraConfig::PinholeConfig& config) : config_(config) {
  K_(0, 0) = config_.fx();
  K_(1, 1) = config_.fy();
  K_(0, 2) = config_.cx();
  K_(1, 2) = config_.cy();
  focal_length_ = config_.focal_length();
  if (std::fabs(K_.determinant()) < 1.0e-6) {
    LOG(WARNING) << "Maybe Empty intrinsic matrix, please check !";
  }

  inv_fx_ = 1.0 / K_(0, 0);
  inv_fy_ = 1.0 / K_(1, 1);
  dist_coef_ = Eigen::Vector4d(config_.k1(), config_.k2(), config_.p1(), config_.p2());
  cv::eigen2cv(K_, K_mat_);
  cv::eigen2cv(dist_coef_, dist_coef_mat_);
}

/**
 * @brief 相机坐标系下的三维点投影到像素平面(直接乘上K矩阵)
 * @param p3D 三维点
 * @return 像素坐标
 */
Eigen::Vector2d Pinhole::CameraP3dToPix(const Eigen::Vector3d& p3d) const {
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
Eigen::Vector3d Pinhole::PixToCameraP3d(const Eigen::Vector2d& p2d) const {
  Eigen::Vector3d res;
  res.x() = (p2d.x() - K_(0, 2)) * inv_fx_;
  res.y() = (p2d.y() - K_(1, 2)) * inv_fy_;
  res.z() = 1.0;
  return res;
}

std::vector<Eigen::Vector3d> Pinhole::DistPixToCameraP3d(
    const std::vector<Eigen::Vector2d>& points) const {
  CHECK(!points.empty()) << "Empty points !";
  CHECK(points.size() < std::numeric_limits<int32_t>::max()) << "Too many feature points !";
  const int32_t points_nums = static_cast<int32_t>(points.size());
  std::vector<Eigen::Vector3d> dist_points;
  dist_points.reserve(points_nums);

  if (std::fabs(dist_coef_(0)) < 1.0e-6) {
    for (const auto& pt : points) {
      dist_points.emplace_back(PixToCameraP3d(pt));
    }
    return dist_points;
  }

  if (config_.enable_cv_undistort()) {
    // 函数reshape(int cn,int rows=0) 其中cn为更改后的通道数，rows=0表示这个行将保持原来的参数不变
    // 为了能够直接调用opencv的函数来去畸变，需要先将矩阵调整为2通道（对应坐标x,y）
    // cv::undistortPoints最后一个矩阵为空矩阵时，得到的点为归一化坐标点
    cv::Mat points_mat(points_nums, 2, CV_64F);
    for (int32_t i = 0; i < points_nums; ++i) {
      points_mat.at<double>(i, 0) = points[i].x();
      points_mat.at<double>(i, 1) = points[i].y();
    }
    points_mat = points_mat.reshape(2);
    cv::undistortPoints(points_mat, points_mat, K_mat_, dist_coef_mat_, cv::Mat(), K_mat_);
    points_mat = points_mat.reshape(1);

    for (int i = 0; i < points_nums; ++i) {
      Eigen::Vector2d pt = Eigen::Vector2d::Zero();
      pt.x() = points_mat.at<double>(i, 0);
      pt.y() = points_mat.at<double>(i, 1);
      dist_points.emplace_back(PixToCameraP3d(pt));
    }

  } else {
    // VINS中更为高效的迭代方式去畸变， 将原始图像中的点投影到没有畸变的3D点
    int iter_times = config_.iter_times();
    Eigen::Vector2d camera_p3d = Eigen::Vector2d::Zero();
    Eigen::Vector2d d_u = Eigen::Vector2d::Zero();
    for (const auto& pt : points) {
      camera_p3d = PixToCameraP3d(pt).head<2>();
      d_u = Eigen::Vector2d::Zero();
      for (int i = 0; i < iter_times; ++i) {
        // 此时的camera_p3d - d_u相当于被更新了一次的点A_1
        Distortion(camera_p3d - d_u, &d_u);
      }
      auto p2d = camera_p3d - d_u;
      dist_points.emplace_back(Eigen::Vector3d(p2d.x(), p2d.y(), 1.0));
    }
  }

  return dist_points;
}

// 计算公式 \Delta x 和 \Delta x; d_u=[\Delta x,\Delta y]
void Pinhole::Distortion(const Eigen::Vector2d& p_u, Eigen::Vector2d* const d_u) const {
  // 畸变参数
  double k1 = dist_coef_(0);
  double k2 = dist_coef_(1);
  double p1 = dist_coef_(2);
  double p2 = dist_coef_(3);
  double mx2_u, my2_u, mxy_u, rho2_u, rad_dist_u;
  // x^2
  mx2_u = p_u(0) * p_u(0);
  // y^2
  my2_u = p_u(1) * p_u(1);
  // xy
  mxy_u = p_u(0) * p_u(1);
  // r^2=x^2+y^2
  rho2_u = mx2_u + my2_u;
  // k1 * r^2 + k2 * r^4
  rad_dist_u = k1 * rho2_u + k2 * rho2_u * rho2_u;
  // \Delta x = x(k1 * r^2 + k2 * r^4)+2 * p1 * xy +p2(r^2+2x^2)
  // \Delta y = y(k1 * r^2 + k2 * r^4)+2 * p2 * xy +p1(r^2+2y^2)
  (*d_u) << p_u(0) * rad_dist_u + 2.0 * p1 * mxy_u + p2 * (rho2_u + 2.0 * mx2_u),
      p_u(1) * rad_dist_u + 2.0 * p2 * mxy_u + p1 * (rho2_u + 2.0 * my2_u);
}

}  // namespace Camera
