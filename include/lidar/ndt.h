#pragma once

#include <memory>
#include <vector>

#include "Eigen/Core"
#include "Eigen/Dense"

#include "common/data/point_cloud.h"
#include "common/voxel_map.h"
#include "lidar/filter.h"
#include "optimizer/cloud_match_optimize.h"
#include "sophus/se3.hpp"
#include "sophus/so3.hpp"
#include "util/time.h"
#include "util/utils.h"

using namespace Common;
namespace Lidar {

class NDT {
 public:
  NDT() = default;
  NDT(const double voxel_size,
      const int32_t max_iters,
      const double break_dx,
      const int32_t min_effect_points,
      const bool use_downsample,
      const double outlier_th,
      const int32_t min_effective_points);
  ~NDT() = default;

  // 核心问题，通过最近邻方法找到对应关系，然后构建最小二乘问题，反复交替迭代
  bool Align(const Eigen::Isometry3d& pred_pose,
             const PointCloudPtr& source_cloud,
             const PointCloudPtr& target_cloud,
             Eigen::Isometry3d* const final_pose);

 public:
  // TODO，友元类，使用这个玩意代替手写的高斯牛顿方法，提升代码复用率，拓展其他优化方法，之后ndt和icp的优化更新用自己写的优化器实现
  friend Optimizer::CloudMatchOptimizer;

 private:
  bool SetSourceCloud(const PointCloudPtr& source);

  bool SetTargetCloud(const PointCloudPtr& target);

  bool NDTAlign();

 private:
  Common::Data::PointCloudPtr source_cloud_ = nullptr;
  Common::Data::PointCloudPtr target_cloud_ = nullptr;

  Eigen::Vector3d source_center_ = Eigen::Vector3d::Zero();
  Eigen::Vector3d target_center_ = Eigen::Vector3d::Zero();

  std::shared_ptr<Common::VoxelMap> voxel_map_ = nullptr;
  std::shared_ptr<Lidar::PointCloudFilter> filter_ = nullptr;

  double voxel_size_ = 0.1;
  int32_t max_iters_ = 30;
  double break_dx_ = 1.0e-6;
  int32_t min_effect_points_ = 50;
  bool use_downsample_ = true;
  // err.transpose() * cov_inv_ * err
  double outlier_th_ = 20.0;
  int32_t min_effective_points_ = 50;

  // 用于存储计算得到的位姿，在过程中使用李代数更新
  Sophus::SE3d pose_;

  Utils::Timer timer_;
};

}  // namespace Lidar