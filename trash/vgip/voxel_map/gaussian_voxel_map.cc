/****************************************************************************
 *
 * Copyright (c) 2021 ZelosTech.com, Inc. All Rights Reserved
 *
 ***************************************************************************/
/**
 * @file gaussian_voxel_map.cc
 **/

#include "localization/matching/registration/vg_icp/voxel_map/gaussian_voxel_map.h"


namespace zelos {
namespace zoe {
namespace localization {

static int32_t index_count = 0;

GaussianVoxelMap::GaussianVoxelMap(const ::zelos::zoe::localization::proto::VGICPMapConfig& config) {
  config_ = config;
  inv_leaf_size_ = 1.0 / config_.leaf_size();
  lru_voxel_map_ = std::make_unique<LRUCache<Eigen::Vector3i, GaussianVoxel, XORVector3iHash>>(config_.capacity());
}

void GaussianVoxelMap::Insert(const GaussianPointCloud::Ptr& input_cloud, const Eigen::Isometry3d& transformation) {
  for (int32_t i = 0; i < input_cloud->size(); ++i) {
    Eigen::Vector3d pt_3d(input_cloud->at(i).x(), input_cloud->at(i).y(), input_cloud->at(i).z());    
    const Eigen::Vector3d pt = transformation * pt_3d;
    const Eigen::Vector3i coord = FastFloor(pt * inv_leaf_size_);
    auto found = lru_voxel_map_->GetData(coord);
    if (!found) {
      auto voxel = GaussianVoxel();
      ++index_count;
      voxel.SetIndex(index_count);
      voxel.AddPoints(pt, *input_cloud, i, transformation);
      lru_voxel_map_->Put(coord, voxel);
    } else {
      GaussianVoxel& value = *found;
      value.AddPoints(pt, *input_cloud, i, transformation);
    }
  }

  // Finalize voxel means and covs
  lru_voxel_map_->Traverse([](const Eigen::Vector3i& key, GaussianVoxel* const value) {
    value->Finalize();
  });
  // lru_voxel_map_->PrintCacheValues();
}

bool GaussianVoxelMap::NeaestGridNeighborSearch(const Eigen::Vector3d& pt, std::pair<Eigen::Vector3i, double>* const res) {
  const Eigen::Vector3i coord = FastFloor(pt * inv_leaf_size_);
  int32_t count = 0;
  for (uint32_t i = 0; i < nearby_grids_.size(); ++i) {
    auto dkey = coord.matrix() + nearby_grids_[i];
    const auto voxel = lru_voxel_map_->Get(dkey);
    if (!voxel) {
      continue;
    } else {
      double dis = (voxel->GetMean() - pt).squaredNorm();
      if (dis < res->second) {
        res->first = dkey;
        res->second = dis;
      }
      ++count;
    }
    if (count == 0) {
      return false;
    }
  }
  return true;
}

bool GaussianVoxelMap::NeaestNeighborSearch(const Eigen::Vector3d& pt, std::pair<Eigen::Vector3i, double>* const res) {
  const Eigen::Vector3i coord = FastFloor(pt * inv_leaf_size_);
  const auto voxel = lru_voxel_map_->Get(coord);
  if (!voxel) {
    return false;
  } else {
    double dis = (voxel->GetMean() - pt).squaredNorm();
    res->first = coord;
    res->second = dis;
  }
  return true;
}

void GaussianVoxelMap::GenerateNearbyGrids() {
  if (nearby_type_ == NearbyType::CENTER) {
    nearby_grids_.emplace_back(Eigen::Vector3i::Zero());
  } else if (nearby_type_ == NearbyType::NEARBY6) {
    nearby_grids_ = {
        Eigen::Vector3i(0, 0, 0), Eigen::Vector3i(-1, 0, 0), Eigen::Vector3i(1, 0, 0),
        Eigen::Vector3i(0, 1, 0), Eigen::Vector3i(0, -1, 0), Eigen::Vector3i(0, 0, -1), 
        Eigen::Vector3i(0, 0, 1)};
  }
}

int32_t GaussianVoxelMap::Size() const {
  return lru_voxel_map_->Size();
}

Eigen::Vector3d GaussianVoxelMap::GetMean(const Eigen::Vector3i& coord) const {
  return lru_voxel_map_->Get(coord)->GetMean();
}

Eigen::Matrix3d GaussianVoxelMap::GetCov(const Eigen::Vector3i& coord) const {
  return lru_voxel_map_->Get(coord)->GetCov();
}

}  // namespace localization
}  // namespace zoe
}  // namespace zelos

/* vim: set expandtab ts=2 sw=2 sts=2 tw=100: */
