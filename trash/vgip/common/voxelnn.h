/****************************************************************************
 *
 * Copyright (c) 2021 ZelosTech.com, Inc. All Rights Reserved
 *
 ***************************************************************************/
/**
 * @file VoxelNN.h
 **/

#pragma once

#include <omp.h>
#include <vector>
#include <map>
#include <unordered_map>

#include "localization/matching/registration/vg_icp/common/gaussian_point_cloud.h"
#include "localization/matching/registration/vg_icp/common/util.h"
#include "localization/matching/filter/pc_filter_interface.h"
#include "localization/matching/common/boundary.h"

namespace zelos {
namespace zoe {
namespace localization {

namespace {
using ::zelos::zoe::localization::GaussianPointCloud;
}  // namespace

class VoxelNN {
 public:
  /**
   * 构造函数
   * @param resolution 分辨率
   * @param nearby_type 近邻判定方法
   */
  explicit VoxelNN(const double resolution, const NearbyType& nearby_type)
      : resolution_(resolution), nearby_type_(nearby_type) {
      inv_resolution_ = 1.0 / resolution_;
      GenerateNearbyGrids();
  }

  std::shared_ptr<GaussianPointCloud> VoxelGridSample(const GaussianPointCloud& input_cloud, double leaf_size);

  /// 设置点云，建立栅格
  bool SetPointCloud(const GaussianPointCloud::Ptr& cloud);

  bool SetPointCloud(const pcl::PointCloud<pcl::PointXYZI>::Ptr& input_cloud);
  /// 获取最近邻
  bool KnnSearch(const int32_t k, const Eigen::Vector3d& pt, std::vector<std::pair<int32_t, Eigen::Vector3d>>* const res);

  std::vector<std::pair<int32_t, Eigen::Vector3d>> BFNN(
      const GaussianPointCloud::Ptr& cloud, const Eigen::Vector3d& point, const int32_t k);

  void EstimateNormalsCovariancesOMP(const int32_t num_neighbors, const int32_t num_threads);

  GaussianPointCloud::Ptr GetCloud() const;

  int32_t Size() const;

 private:
  /// 根据最近邻的类型，生成附近网格
  void GenerateNearbyGrids();

 private:
  double resolution_ = 0.25;       // 分辨率
  double inv_resolution_ = 4.0;  // 分辨率倒数

  Boundary lidar_roi_;
  std::shared_ptr<GaussianPointCloud> cloud_ = nullptr;
  std::vector<Eigen::Vector3i> nearby_grids_;  // 附近的栅格
  NearbyType nearby_type_ = NearbyType::NEARBY6;
  std::unordered_map<Eigen::Vector3i, std::vector<int32_t>, XORVector3iHash> grids_;  //  栅格数据
};

}  // namespace localization
}  // namespace zoe
}  // namespace zelos

/* vim: set expandtab ts=2 sw=2 sts=2 tw=100: */
