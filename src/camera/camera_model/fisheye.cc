#include "camera/camera_model/fisheye.h"

namespace Camera {

FishEyeKB::FishEyeKB(const CameraConfig::FishEyeConfig& config) : config_(config) {
  K_(0, 0) = config_.fx();
  K_(1, 1) = config_.fy();
  K_(0, 2) = config_.cx();
  K_(1, 2) = config_.cy();
  CHECK(K_.determinant() != 0.0) << "Maybe Empty intrinsic matrix, please check !";

  inv_fx_ = 1.0 / K_(0, 0);
  inv_fy_ = 1.0 / K_(1, 1);

  dist_coef_ = Eigen::Vector4d(config_.k1(), config_.k2(), config_.k3(), config_.k4());

  // K_mat_ =
  //     cv::Mat_<double>(3, 3) << (K_(0, 0), 0.0, K_(1, 1), 0.0, K_(0, 2), K_(1, 2), 0.0,
  //     0.0, 1.0);

  // dist_coef_mat_.at<double>(0) = dist_coef_(0);
  // dist_coef_mat_.at<double>(1) = dist_coef_(1);
  // dist_coef_mat_.at<double>(2) = dist_coef_(2);
  // dist_coef_mat_.at<double>(3) = dist_coef_(3);
}

/**
 * @brief 相机系下的3D点投影到像素平面上的像素点
 * xc​ = Xc/Zc, yc = Yc/Zc
 * r^2 = xc^2 + yc^2
 * θ = arctan(r)
 * θd = k0*θ + k1*θ^3 + k2*θ^5 + k3*θ^7 + k4*θ^9
 * xd = θd/r * xc   yd = θd/r * yc
 * u = fx*xd + cx  v = fy*yd + cy
 * @param p3D 三维点
 * @return 像素坐标
 */
Eigen::Vector2d FishEyeKB::CameraP3dToPix(const Eigen::Vector3d& p3d) const {
  const double x2_plus_y2 = p3d[0] * p3d[0] + p3d[1] * p3d[1];
  const double theta = atan2f(sqrtf(x2_plus_y2), p3d[2]);
  const double psi = atan2f(p3d[1], p3d[0]);

  const double theta2 = theta * theta;
  const double theta3 = theta * theta2;
  const double theta5 = theta3 * theta2;
  const double theta7 = theta5 * theta2;
  const double theta9 = theta7 * theta2;
  const double r = theta + dist_coef_(0) * theta3 + dist_coef_(1) * theta5 +
                   dist_coef_(2) * theta7 + dist_coef_(3) * theta9;

  Eigen::Vector2d res;
  res.x() = K_(0, 0) * r * cos(psi) + K_(0, 2);
  res.y() = K_(1, 1) * r * sin(psi) + K_(1, 2);

  return res;
}

/**
 * @brief 反投影， 像素点反投影得到归一化的3D点坐标
 * 投影过程
 * xc​ = Xc/Zc, yc = Yc/Zc
 * r^2 = xc^2 + yc^2
 * θ = arctan(r)
 * θd = k0*θ + k1*θ^3 + k2*θ^5 + k3*θ^7 + k4*θ^9
 * xd = θd/r * xc   yd = θd/r * yc
 * u = fx*xd + cx  v = fy*yd + cy
 *
 *
 * 已知u与v 未矫正的特征点像素坐标
 * xd = (u - cx) / fx    yd = (v - cy) / fy
 * 待求的 xc = xd * r / θd  yc = yd * r / θd
 * 待求的 xc = xd * tan(θ) / θd  yc = yd * tan(θ) / θd
 * 其中 θd的算法如下：
 *     xd^2 + yd^2 = θd^2/r^2 * (xc^2 + yc^2)
 *     θd = sqrt(xd^2 + yd^2) / sqrt(xc^2 + yc^2) * r
 *     其中r = sqrt(xc^2 + yc^2)
 *     所以 θd = sqrt(xd^2 + yd^2)  已知
 * 所以待求的只有tan(θ),也就是θ
 * 这里θd = θ + k1*θ^3 + k2*θ^5 + k3*θ^7 + k4*θ^9
 * 直接求解θ比较麻烦，这里用迭代的方式通过误差的一阶导数求θ
 * θ的初始值定为了θd，设θ + k1*θ^3 + k2*θ^5 + k3*θ^7 + k4*θ^9 = f(θ)
 * e(θ) = f(θ) - θd 目标是求解一个θ值另e(θ) = 0
 * 泰勒展开e(θ+δθ) = e(θ) + e`(θ) * δθ = 0
 * e`(θ) = 1 + 3*k1*θ^*2 + 5*k2*θ^4 + 7*k3*θ^6 + 8*k4*θ^8
 * δθ = -e(θ)/e`(θ) 经过多次迭代可求得θ
 * 最后xc = xd * tan(θ) / θd  yc = yd * tan(θ) / θd
 * @param p2d 特征点
 * @return
 */
Eigen::Vector3d FishEyeKB::PixToCameraP3d(const Eigen::Vector2d& p2d) const {
  // Use Newthon method to solve for theta with good precision (err ~ e-6)
  Eigen::Vector2d pw((p2d.x() - K_(0, 2)) / K_(0, 0), (p2d.y() - K_(1, 2)) / K_(1, 1));
  double scale = 1.0;
  double theta_d = sqrtf(pw.x() * pw.x() + pw.y() * pw.y());       // sin(psi) = yc / r
  theta_d = std::min(std::max(-M_PI / 2.0, theta_d), M_PI / 2.0);  // 不能超过180度

  if (theta_d > 1e-8) {
    // Compensate distortion iteratively
    // θ的初始值定为了θd
    double theta = theta_d;

    // 开始迭代
    for (int j = 0; j < config_.iter_times(); j++) {
      double theta2 = theta * theta, theta4 = theta2 * theta2, theta6 = theta4 * theta2,
             theta8 = theta4 * theta4;
      double k0_theta2 = dist_coef_(0) * theta2, k1_theta4 = dist_coef_(1) * theta4;
      double k2_theta6 = dist_coef_(2) * theta6, k3_theta8 = dist_coef_(3) * theta8;
      double theta_fix = (theta * (1 + k0_theta2 + k1_theta4 + k2_theta6 + k3_theta8) - theta_d) /
                         (1 + 3 * k0_theta2 + 5 * k1_theta4 + 7 * k2_theta6 + 9 * k3_theta8);
      theta = theta - theta_fix;
      if (fabsf(theta_fix) < 1.0e-6)  // 如果更新量变得很小，表示接近最终值
        break;
    }
    // scale = theta - theta_d;
    // 求得tan(θ) / θd
    scale = std::tan(theta) / theta_d;
  }

  return Eigen::Vector3d(pw.x() * scale, pw.y() * scale, 1.0);
}

std::vector<Eigen::Vector3d> FishEyeKB::DistPixToCameraP3d(
    const std::vector<Eigen::Vector2d>& points) const {
  CHECK(!points.empty()) << "Empty points !";
  CHECK(points.size() < std::numeric_limits<int32_t>::max()) << "Too many feature points !";
  const int32_t points_nums = static_cast<int32_t>(points.size());
  std::vector<Eigen::Vector3d> dist_points;
  dist_points.reserve(points_nums);
  for (const auto& pt : points) {
    dist_points.emplace_back(PixToCameraP3d(pt));
  }
  return dist_points;
}

}  // namespace Camera
