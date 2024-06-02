#include "util/math.h"

namespace Utils {

namespace Math {

// 模板函数必须放到头文件中，非模板函数可以写在这个文件中

bool Line3DFit(std::vector<Eigen::Vector3d>& points,
               Eigen::Vector3d* const origin,
               Eigen::Vector3d* const dir,
               double eps) {
  if (points.size() < 2) {
    return false;
  }
  *origin =
      std::accumulate(points.begin(), points.end(), Eigen::Vector3d::Zero().eval()) / points.size();
  Eigen::MatrixXd Y(points.size(), 3);
  for (uint32_t i = 0u; i < points.size(); ++i) {
    Y.row(i) = (points[i] - *origin).transpose();
  }
  Eigen::JacobiSVD<Eigen::MatrixXd> svd(Y, Eigen::ComputeFullV);
  *dir = svd.matrixV().col(0);
  // check eps
  for (const auto& d : points) {
    if (dir->cross(d - *origin).squaredNorm() > eps) {
      return false;
    }
  }
  return true;
}

bool PlaneFit(const std::vector<Eigen::Vector3d>& points,
              Eigen::Vector4d* const plane_coeffs,
              double eps) {
  if (points.size() < 3) {
    return false;
  }
  Eigen::MatrixXd A(points.size(), 4);
  for (uint32_t i = 0u; i < points.size(); ++i) {
    A.row(i).head<3>() = points[i].transpose();
    A.row(i)[3] = 1.0;
  }
  Eigen::JacobiSVD<Eigen::MatrixXd> svd(A, Eigen::ComputeThinV);
  *plane_coeffs = svd.matrixV().col(3);
  // check error eps
  for (uint32_t i = 0u; i < points.size(); ++i) {
    double err = plane_coeffs->head<3>().dot(points[i]) + (*plane_coeffs)[3];
    if (err * err > eps) {
      return false;
    }
  }
  return true;
}

Eigen::Vector3d ComputeCentroid(const std::vector<Eigen::Vector3d>& points) {
  if (points.empty()) {
    LOG(FATAL) << "Point cloud is empty";
    return Eigen::Vector3d::Zero();  // 返回一个零向量以避免编译错误
  }
  Eigen::Vector3d centroid(0.0, 0.0, 0.0);
  for (const auto& point : points) {
    centroid += point;
  }
  centroid /= static_cast<double>(points.size());
  return centroid;
}

}  // namespace Math

}  // namespace Utils