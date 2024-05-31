#pragma once

#include <memory>
#include <numeric>  // std::accumulate

#include "Eigen/Geometry"

#include "common/data/point_cloud.h"
#include "common/kdtree.h"
#include "common/voxel_map.h"
#include "optimizer/cloud_match_optimize.h"

using namespace Common::Data;

namespace Lidar {

enum class AlignMethod {
  POINT_TO_POINT_ICP,
  POINT_TO_LINE_ICP,
  POINT_TO_PLANE_ICP,
  NDT,
};

class VICP {
 public:
  VICP(const double voxel_size, const double max_distance, const int32_t max_num_per_voxel) {
    kdtree_ = std::make_shared<Common::KdTree>();
    voxel_map_ = std::make_shared<Common::VoxelMap>(voxel_size, max_distance, max_num_per_voxel);
  }
  ~VICP() = default;

  void Align(const Eigen::Isometry3d& pred_pose,
             const PointCloudPtr& source_cloud,
             const PointCloudPtr& target_cloud,
             Eigen::Isometry3d* const final_pose) {
    CHECK_NOTNULL(source_cloud);
    CHECK_NOTNULL(target_cloud);
    CHECK_NOTNULL(final_pose);
    SetSourceCloud(source_cloud);
    SetTargetCloud(target_cloud);

    // build kdtree or voxel map
    kdtree_->BuildTree(target_cloud);

    // final_pose->translate(pred_pose.translation());
    final_pose->translate(target_center_ - source_center_);
    final_pose->rotate(pred_pose.rotation());
  }

 public:
  friend Optimizer::CloudMatchOptimizer;

 private:
  bool SetSourceCloud(const PointCloudPtr& source) {
    if (!source || source->empty()) {
      LOG(FATAL) << "Set Target Point Cloud Failed, Beacuse of NULL !";
      return false;
    }
    source_cloud_ = std::make_shared<PointCloud>(source);
    source_center_ =
        std::accumulate(
            source->points().begin(), source->points().end(), Eigen::Vector3d::Zero().eval()) /
        source->size();
    LOG(INFO) << "Target center is " << target_center_.transpose();
    return true;
  }

  bool SetTargetCloud(const PointCloudPtr& target) {
    if (!target || target->empty()) {
      LOG(FATAL) << "Set Target Point Cloud Failed, Beacuse of NULL !";
      return false;
    }
    target_cloud_ = std::make_shared<PointCloud>(target);
    target_center_ =
        std::accumulate(
            target->points().begin(), target->points().end(), Eigen::Vector3d::Zero().eval()) /
        target->size();
    LOG(INFO) << "Target center is " << target_center_.transpose();
    return true;
  }

 private:
  Common::Data::PointCloudPtr source_cloud_ = nullptr;
  Common::Data::PointCloudPtr target_cloud_ = nullptr;

  Eigen::Vector3d source_center_ = Eigen::Vector3d::Zero();
  Eigen::Vector3d target_center_ = Eigen::Vector3d::Zero();

  // 对应两种最近临搜索方式
  std::shared_ptr<Common::KdTree> kdtree_ = nullptr;
  std::shared_ptr<Common::VoxelMap> voxel_map_ = nullptr;
};

}  // namespace Lidar
