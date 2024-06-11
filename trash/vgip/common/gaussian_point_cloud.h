/****************************************************************************
 *
 * Copyright (c) 2021 ZelosTech.com, Inc. All Rights Reserved
 *
 ***************************************************************************/
/**
 * @file util.h
 **/

#pragma once

#include <cstdint>
#include <memory>
#include <vector>

#include "Eigen/Core"

#include "pcl/point_types.h"
#include "pcl/point_cloud.h"

#include "zlog/logger.h"

namespace zelos {
namespace zoe {
namespace localization {

struct GaussianPointCloud {
public:
  using Ptr = std::shared_ptr<GaussianPointCloud>;
  using ConstPtr = std::shared_ptr<const GaussianPointCloud>;

  GaussianPointCloud() {}

  GaussianPointCloud(const GaussianPointCloud& other) {
    points_ = other.points_;
    normals_ = other.normals_;
    covs_ = other.covs_;
  }

  GaussianPointCloud(const pcl::PointCloud<pcl::PointXYZI>::Ptr& input_cloud) {
    ZCHECK_NOTNULL(input_cloud);
    this->resize(input_cloud->size());
    for (uint32_t i = 0u; i < input_cloud->size(); ++i) {
      this->at(i) =
          Eigen::Vector3d(input_cloud->at(i).x, input_cloud->at(i).y, input_cloud->at(i).z);
    }
  }

  GaussianPointCloud(const std::vector<Eigen::Vector3d>& points) {
    this->resize(points.size());
    for (uint32_t i = 0u; i < points.size(); ++i) {
      this->at(i) = points.at(i);
    }
  }

  /// @brief Destructor
  ~GaussianPointCloud() {}

  /// @brief Number of points_.
  int32_t size() const { return points_.size(); }

  /// @brief Check if the point cloud is empty.
  bool empty() const { return points_.empty(); }

  /// @brief Resize point/normal/cov buffers.
  /// @param n  Number of points_
  void resize(int32_t n) {
    points_.resize(n);
    normals_.resize(n);
    covs_.resize(n);
  }

  void emplace_back(const Eigen::Vector3d& point) {
    points_.emplace_back(point);
  }

  std::vector<Eigen::Vector3d> points() {
    return points_;
  }

  /// @brief Get i-th point.
  Eigen::Vector3d& at(int32_t i) { return points_[i]; }

  /// @brief Get i-th normal.
  Eigen::Vector3d& normal(int32_t i) { return normals_[i]; }

  /// @brief Get i-th covariance.
  Eigen::Matrix3d& cov(int32_t i) { return covs_[i]; }

  /// @brief Get i-th point (const).
  const Eigen::Vector3d& at(int32_t i) const { return points_[i]; }

  /// @brief Get i-th normal (const).
  const Eigen::Vector3d& normal(int32_t i) const { return normals_[i]; }

  /// @brief Get i-th covariance (const).
  const Eigen::Matrix3d& cov(int32_t i) const { return covs_[i]; }

 public:
  std::vector<Eigen::Vector3d> points_;   ///< Point coordinates (x, y, z, 1)
  std::vector<Eigen::Vector3d> normals_;  ///< Point normals_ (nx, ny, nz, 0)
  std::vector<Eigen::Matrix3d> covs_;     ///< Point covariances (3x3 matrix) + zero padding
};

}  // namespace localization
}  // namespace zoe
}  // namespace zelos

/* vim: set expandtab ts=2 sw=2 sts=2 tw=100: */
