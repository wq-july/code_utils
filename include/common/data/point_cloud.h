#pragma once

#include <stdexcept>
#include <vector>

#include "Eigen/Core"
#include "pcl/point_cloud.h"
#include "pcl/point_types.h"

#include "common/data/point.h"

namespace Common {
namespace Data {

class PointCloud {
 public:
  // 默认构造函数
  PointCloud() = default;

  // 复制构造函数
  PointCloud(const PointCloud& other) : points_(other.points_) {}

  // std::vector<Eigen::Vector3d> 为入参的构造函数
  PointCloud(const std::vector<Eigen::Vector3d>& eigen_points) {
    points_.reserve(eigen_points.size());
    for (const auto& pt : eigen_points) {
      points_.emplace_back(PointXYZ(pt));
    }
  }

  // 模板函数用于创建 PointCloud 对象
  template <typename PclPointType>
  void GetPointsFromPCL(const typename pcl::PointCloud<PclPointType>::Ptr& pcl_points) {
    if (!pcl_points) {
      throw std::invalid_argument("Input point cloud pointer is null");
    }
    points_.clear();
    points_.reserve(pcl_points->size());
    for (const auto& pt : pcl_points->points) {
      points_.emplace_back(PointXYZ(pt.x, pt.y, pt.z));
    }
  }

  // 插入点
  void emplace_back(float x, float y, float z) {
    points_.emplace_back(PointXYZ(x, y, z));
  }

  // 返回点的数量
  int32_t size() const {
    return points_.size();
  }

  // 检查是否为空
  bool empty() const {
    return points_.empty();
  }

  // 预留空间
  void reserve(int32_t n) {
    points_.reserve(n);
  }

  // 清空点云
  void clear() {
    points_.clear();
  }

  // 通过索引获取点
  PointXYZ& at(int32_t index) {
    if (index >= static_cast<int32_t>(points_.size())) {
      throw std::out_of_range("Index out of range");
    }
    return points_[index];
  }

  const PointXYZ& at(int32_t index) const {
    if (index >= static_cast<int32_t>(points_.size())) {
      throw std::out_of_range("Index out of range");
    }
    return points_[index];
  }

  // 返回所有点
  const std::vector<PointXYZ>& points() const {
    return points_;
  }

 private:
  std::vector<PointXYZ> points_;
};

// TODO 进一步提供均值和方差的函数以及成员变量

}  // namespace Data

}  // namespace Common
