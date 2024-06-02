/****************************************************************************
 *
 * Copyright (c) 2021 ZelosTech.com, Inc. All Rights Reserved
 *
 ***************************************************************************/
/**
 * @file gaussian_voxel.h
 **/

#pragma once

#include "Eigen/Core"
#include "Eigen/Geometry"

#include "localization/matching/registration/vg_icp/common/util.h"
#include "localization/matching/registration/vg_icp/common/gaussian_point_cloud.h"

namespace zelos {
namespace zoe {
namespace localization {

namespace {
using zelos::zoe::localization::GaussianPointCloud;
} // namespace name

struct GaussianVoxel {
 public:
  GaussianVoxel() = default;
  ~GaussianVoxel() = default;
  void AddPoints(const Eigen::Vector3d& transformed_pt, const GaussianPointCloud& input_cloud,
      const int32_t index, const Eigen::Isometry3d& transformation);
  void Finalize();
  Eigen::Vector3d GetMean() const;
  Eigen::Matrix3d GetCov() const;
  int32_t GetPointsNum() const;
  int32_t GetIndex() const;

  void PrintData() const;
  void SetIndex(const int32_t index);
  // void NearestNeighborSearch(const Eigen::Vector3d& pt, int32_t* const k_index,
  //     double* const k_sq_dist) const;

  // void KNNSearch(const Eigen::Vector3d& pt, int32_t k, std::vector<int32_t>* const k_index,
  //     std::vector<double>* const  k_sq_dist) const;

 private:
  bool finalized_ = false;
  int32_t index_ = 0;
  int32_t num_points_ = 0;
  Eigen::Vector3d mean_ = Eigen::Vector3d::Zero();
  Eigen::Matrix3d cov_ = Eigen::Matrix3d::Zero();
};

}  // namespace localization
}  // namespace zoe
}  // namespace zelos

/* vim: set expandtab ts=2 sw=2 sts=2 tw=100: */
