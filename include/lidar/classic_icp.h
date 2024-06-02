#pragma once

#include <execution>  // for execution policies
#include <memory>
#include <numeric>  // std::accumulate， std::iota

#include "Eigen/Geometry"

#include "common/data/point_cloud.h"
#include "common/kdtree.h"
#include "common/voxel_map.h"
#include "optimizer/cloud_match_optimize.h"

#include "lidar/filter.h"
#include "sophus/se3.hpp"

using namespace Common::Data;

namespace Lidar {

enum class AlignMethod {
  PCL_ICP,
  SVD_ICP,
  POINT_TO_POINT_ICP,
  POINT_TO_LINE_ICP,
  POINT_TO_PLANE_ICP,
  GENERAL_ICP
};

enum class SearchMethod { KDTREE, VOXEL_MAP };

class ClassicICP{
 public:
  ClassicICP(const double voxel_size,
      const double max_distance,
      const int32_t max_num_per_voxel,
      const AlignMethod method,
      const SearchMethod search_method,
      const int32_t max_iters,
      const double break_dx,
      const int32_t min_effect_points,
      const bool use_downsample = false);
  ~ClassicICP() = default;
  // 核心问题，通过最近邻方法找到对应关系，然后构建最小二乘问题，反复交替迭代
  bool Align(const Eigen::Isometry3d& pred_pose,
             const PointCloudPtr& source_cloud,
             const PointCloudPtr& target_cloud,
             Eigen::Isometry3d* const final_pose);

 public:
  friend Optimizer::CloudMatchOptimizer;

 private:
  bool SetSourceCloud(const PointCloudPtr& source);

  bool SetTargetCloud(const PointCloudPtr& target);

  bool SvdIcp();

  bool PointToPointICP();

  bool PointToLineICP();

  bool PointToPlaneICP();

  bool GeneralICP();

  bool PclICP();

 private:
  AlignMethod align_method_;
  SearchMethod search_method_;

  Common::Data::PointCloudPtr source_cloud_ = nullptr;
  Common::Data::PointCloudPtr target_cloud_ = nullptr;

  Eigen::Vector3d source_center_ = Eigen::Vector3d::Zero();
  Eigen::Vector3d target_center_ = Eigen::Vector3d::Zero();

  // 对应两种最近临搜索方式
  std::shared_ptr<Common::KdTree> kdtree_ = nullptr;
  std::shared_ptr<Common::VoxelMap> voxel_map_ = nullptr;

  std::shared_ptr<Lidar::PointCloudFilter> filter_ = nullptr;

  int32_t max_iters_ = 30;
  double max_point_point_distance_ = 1.0;
  double max_point_line_distance_ = 0.5;
  double max_point_plane_distance_ = 0.5;
  double break_dx_ = 1.0e-6;
  int32_t min_effect_points_ = 50;
  bool use_downsample_ = true;

  Sophus::SE3d pose_;

  Utils::Timer timer_;
};

}  // namespace Lidar
