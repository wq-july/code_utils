/****************************************************************************
 *
 * Copyright (c) 2021 ZelosTech.com, Inc. All Rights Reserved
 *
 ***************************************************************************/
/**
 * @file gaussian_voxel_map.h
 **/

#pragma once
#include <memory>

#include "Eigen/Core"
#include "Eigen/Geometry"

#include "localization/matching/registration/vg_icp/common/lru.h"
#include "localization/matching/registration/vg_icp/common/util.h"
#include "localization/matching/registration/vg_icp/common/gaussian_point_cloud.h"
#include "localization/matching/registration/vg_icp/voxel_map/gaussian_voxel.h"
#include "localization/matching/registration/proto/vgicp_config.pb.h"

namespace zelos {
namespace zoe {
namespace localization {

namespace {
using zelos::zoe::localization::GaussianPointCloud;
using zelos::zoe::localization::LRUCache;
using zelos::zoe::localization::XORVector3iHash;
using zelos::zoe::localization::NearbyType;
} // namespace name

class GaussianVoxelMap {
 public:
  GaussianVoxelMap(const ::zelos::zoe::localization::proto::VGICPMapConfig& config);
  ~GaussianVoxelMap() = default;
  int32_t Size() const;

  void Insert(const GaussianPointCloud::Ptr& input_cloud,
      const Eigen::Isometry3d& transformation = Eigen::Isometry3d::Identity());

  void InsertTest(const GaussianPointCloud::Ptr& input_cloud);

  bool NeaestGridNeighborSearch(const Eigen::Vector3d& pt, std::pair<Eigen::Vector3i, double>* const res);
  bool NeaestNeighborSearch(const Eigen::Vector3d& pt, std::pair<Eigen::Vector3i, double>* const res);

  void GenerateNearbyGrids();

  Eigen::Vector3d GetMean(const Eigen::Vector3i& key) const;
  Eigen::Matrix3d GetCov(const Eigen::Vector3i& key) const;

  // void KNNSearch(const Eigen::Vector3d& pt, const int32_t k, std::vector<std::pair<int32_t, double>>* const res);

 private:
  double inv_leaf_size_ = 10.0;
  std::vector<Eigen::Vector3i> nearby_grids_;
  NearbyType nearby_type_ = NearbyType::NEARBY6;
  ::zelos::zoe::localization::proto::VGICPMapConfig config_;
  std::unique_ptr<LRUCache<Eigen::Vector3i, GaussianVoxel, XORVector3iHash>> lru_voxel_map_ = nullptr;
};

}  // namespace localization
}  // namespace zoe
}  // namespace zelos

/* vim: set expandtab ts=2 sw=2 sts=2 tw=100: */
