#pragma once

#include <pcl/io/pcd_io.h>

#include <execution>
#include <stdexcept>

#include "Eigen/Core"
#include "pcl/point_cloud.h"
#include "pcl/point_types.h"

#include "util/utils.h"

// 后期有需要再用这个，因为这个没什么必要，其他算法中用到这个点的时候可能需要和Eigen::Vector3d进行转换，这个很耗时
// 之后的代码都基于Eigen::Vector3d来进行
// #include "common/data/point.h"
#include "glog/logging.h"

#include "common/data/point.h"

namespace Common {
namespace Data {

class PointCloud {
  using PointXYZ = Eigen::Vector3d;
 public:
  // 默认构造函数
  PointCloud() = default;

  // 复制构造函数
  PointCloud(const PointCloud& other) : points_(other.points_) {}

  // 移动构造函数
  // PointCloud(PointCloud&& other) noexcept : points_(std::move(other.points_)) {}

  // 从 shared_ptr 移动复制构造函数
  // PointCloud(const std::shared_ptr<PointCloud>& cloud) {
  //   points_ = std::move(cloud->points_);
  // }

  // 从 shared_ptr 复制构造函数
  PointCloud(const std::shared_ptr<PointCloud>& cloud) {
    points_ = cloud->points_;
  }

  // std::vector<Eigen::Vector3d> 为入参的构造函数
  PointCloud(const std::vector<Eigen::Vector3d>& eigen_points) {
    points_.reserve(eigen_points.size());
    for (const auto& pt : eigen_points) {
      points_.emplace_back(PointXYZ(pt));
    }
  }

  template <typename PclPointType>
  bool LoadPCDFile(const std::string& filename) {
    typename pcl::PointCloud<PclPointType>::Ptr pcl_cloud(new pcl::PointCloud<PclPointType>);

    if (pcl::io::loadPCDFile<PclPointType>(filename, *pcl_cloud) == -1) {
      LOG(ERROR) << "Couldn't read the PCD file: " << filename;
      return false;
    }
    GetPointsFromPCL<PclPointType>(pcl_cloud);
    return true;
  }

  // 模板函数用于创建 PointCloud 对象
  template <typename PclPointType>
  bool GetPointsFromPCL(const typename pcl::PointCloud<PclPointType>::Ptr& pcl_points) {
    if (!pcl_points || pcl_points->empty()) {
      LOG(FATAL) << "Input Point Cloud Pointer Is Null or Empty!";
      return false;
    }
    points_.clear();
    points_ = std::move(Utils::PclToVec3d<PclPointType>(pcl_points));
    return true;
  }

  // 插入点
  void emplace_back(float x, float y, float z) {
    points_.emplace_back(PointXYZ(x, y, z));
  }

  // 返回点的数量
  uint32_t size() const {
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
    return points_.at(index);
  }

  const PointXYZ& at(int32_t index) const {
    if (index >= static_cast<int32_t>(points_.size())) {
      throw std::out_of_range("Index out of range");
    }
    return points_.at(index);
  }

  // 返回所有点
  const std::vector<PointXYZ>& points() const {
    return points_;
  }

 private:
  std::vector<PointXYZ> points_;
};


// 定义共享指针类型
using PointCloudPtr = std::shared_ptr<PointCloud>;

// TODO 进一步提供均值和方差的函数以及成员变量

}  // namespace Data

}  // namespace Common
