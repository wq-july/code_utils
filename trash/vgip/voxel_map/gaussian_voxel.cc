/****************************************************************************
 *
 * Copyright (c) 2021 ZelosTech.com, Inc. All Rights Reserved
 *
 ***************************************************************************/
/**
 * @file gaussian_voxel.cc
 **/

#include "localization/matching/registration/vg_icp/voxel_map/gaussian_voxel.h"

namespace zelos {
namespace zoe {
namespace localization {

void GaussianVoxel::AddPoints(const Eigen::Vector3d& transformed_pt, const GaussianPointCloud& input_cloud,
    const int32_t index, const Eigen::Isometry3d& transformation) {
  if (finalized_) {
    this->finalized_ = false;
    this->mean_ *= num_points_;
    this->cov_ *= num_points_;
  }
  ++num_points_;
  this->mean_ += transformed_pt;
  Eigen::Matrix3d rot = transformation.matrix().block<3, 3>(0, 0);
  this->cov_ += rot * input_cloud.cov(index) * rot.transpose();
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

Eigen::Vector3d GaussianVoxel::GetMean() const {
  return mean_;
}

Eigen::Matrix3d GaussianVoxel::GetCov() const {
  return cov_;
}

int32_t GaussianVoxel::GetPointsNum() const {
  return num_points_;
}

int32_t GaussianVoxel::GetIndex() const {
  return index_;
}

void GaussianVoxel::PrintData() const {
  std::cout << "points num is " << num_points_ << ", mean is " << mean_.transpose() << "\n";
}

void GaussianVoxel::SetIndex(const int32_t index) {
  index_ = index;
}

}  // namespace localization
}  // namespace zoe
}  // namespace zelos

/* vim: set expandtab ts=2 sw=2 sts=2 tw=100: */
