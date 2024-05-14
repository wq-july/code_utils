#include "common/search/voxel.h"

namespace Common {

void Voxel::AddPoint(const Eigen::Vector3d& point) {
  if (points_.size() < static_cast<size_t>(max_nums_)) {
    points_.push_back(point);
  }
}

void GaussianVoxel::AddPoint(const Eigen::Vector3d& points, const Eigen::Matrix3d& cov,
                             const Eigen::Isometry3d& transformation) {
  if (finalized_) {
    this->finalized_ = false;
    this->mean_ *= num_points_;
    this->cov_ *= num_points_;
  }
  ++num_points_;
  this->mean_ += transformation * points;
  Eigen::Matrix3d rot = transformation.matrix().block<3, 3>(0, 0);
  this->cov_ += rot * cov * rot.transpose();
}

void GaussianVoxel::Finalize() {
  if (finalized_) {
    return;
  }
  if (num_points_) {
    mean_ /= num_points_;
    cov_ /= num_points_;
  }
  finalized_ = true;
}


Eigen::Vector3d GaussianVoxel::GetMean() const { return mean_; }

Eigen::Matrix3d GaussianVoxel::GetCov() const { return cov_; }

uint32_t GaussianVoxel::GetPointsNum() const { return num_points_; }

uint32_t GaussianVoxel::GetIndex() const { return index_; }

}  // namespace Common
