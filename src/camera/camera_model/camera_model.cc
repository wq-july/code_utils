#include "camera/camera_model/camera_model.h"

namespace Camera {

/**
 * @brief 相机坐标系下的三维点投影到像素平面(直接乘上K矩阵)
 * @param p3D 三维点
 * @return 像素坐标
 */
Eigen::Vector2d CameraBase::CameraP3dToPix(const Eigen::Vector3d& p3d) const {
  Eigen::Vector2d res = Eigen::Vector2d::Zero();
  return res;
}

/**
 * @brief 反投影, 理想模型下无畸变的点直接反乘上K矩阵获得归一化3D点坐标
 * @param p2d 特征点
 * @return 归一化坐标
 */
Eigen::Vector3d CameraBase::PixToCameraP3d(const Eigen::Vector2d& p2d) const {
  Eigen::Vector3d res = Eigen::Vector3d::Zero();
  return res;
}

std::vector<Eigen::Vector3d> CameraBase::DistPixToCameraP3d(
    const std::vector<Eigen::Vector2d>& points) const {
  std::vector<Eigen::Vector3d> res;
  return res;
}

/**
 * @brief 计算去畸变图像的边界，也需要重载
 *
 * @param[in] img            需要计算边界的图像
 */
void CameraBase::ComputeUndistortedImageBounds(const cv::Mat& img) {
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

}  // namespace Camera
