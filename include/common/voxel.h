#pragma once

#include <vector>

#include "Eigen/Core"
#include "Eigen/Geometry"

namespace Common {

// 每个最小voxel单元，其中存储着点
struct Voxel {
  Voxel(const Eigen::Vector3d& point, const int32_t max_num) : max_nums_(max_num) {
    AddPoint(point);
  };
  Voxel(const Voxel& other) : points_(other.points_), max_nums_(other.max_nums_) {}
  Voxel(const int32_t max_num) : max_nums_(max_num) {}
  Voxel() = default;
  ~Voxel() = default;

  virtual void AddPoint(const Eigen::Vector3d& point) {
    int32_t size = points_.size();
    if (size < max_nums_) {
      points_.emplace_back(point);
    }
  }

  // buffer of points with a max limit of n_points
  std::vector<Eigen::Vector3d> points_;
  int32_t max_nums_ = std::numeric_limits<int32_t>::max();
};

struct GaussianVoxel : public Voxel {
 public:
  GaussianVoxel() = default;
  GaussianVoxel(const Eigen::Vector3d& point, const int32_t max_num) : Voxel(point, max_num){};
  GaussianVoxel(const int32_t max_num) : Voxel(max_num) {}
  GaussianVoxel(const GaussianVoxel& other)
      : Voxel(other),
        size_(other.size_),
        mean_(other.mean_),
        cov_(other.cov_),
        inv_cov_(other.inv_cov_) {}

  ~GaussianVoxel() = default;

  void AddPoint(const Eigen::Vector3d& point) override {
    int32_t size = points_.size();
    if (size < max_nums_) {
      points_.emplace_back(point);
      ++size_;
      mean_ = mean_ + (point - mean_) / static_cast<double>(size_);
      if (size_ > 1) {
        cov_ = (cov_ * (size_ - 1) + (point - mean_) * (point - mean_).transpose()) /
               static_cast<double>(size_ - 1);
      }
      // SVD 检查最大与最小奇异值，限制最小奇异值
      Eigen::JacobiSVD<Eigen::Matrix3d> svd(cov_, Eigen::ComputeFullU | Eigen::ComputeFullV);
      Eigen::Vector3d lambda = svd.singularValues();
      if (lambda[1] < lambda[0] * 1e-3) {
        lambda[1] = lambda[0] * 1e-3;
      }
      if (lambda[2] < lambda[0] * 1e-3) {
        lambda[2] = lambda[0] * 1e-3;
      }
      Eigen::Matrix3d inv_lambda =
          Eigen::Vector3d(1.0 / lambda[0], 1.0 / lambda[1], 1.0 / lambda[2]).asDiagonal();
      // inv_cov_ = (cov_ + Mat3d::Identity() * 1e-3).inverse();  // 避免出nan
      inv_cov_ = svd.matrixV() * inv_lambda * svd.matrixU().transpose();
    }
  }

  int32_t size_ = 0;

  Eigen::Vector3d mean_ = Eigen::Vector3d::Zero();
  Eigen::Matrix3d cov_ = Eigen::Matrix3d::Zero();
  Eigen::Matrix3d inv_cov_ = Eigen::Matrix3d::Zero();
};

}  // namespace Common
