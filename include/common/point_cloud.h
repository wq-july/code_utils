#pragma once

#include <pcl/io/pcd_io.h>

#include <execution>
#include <stdexcept>

#include "Eigen/Core"
#include "pcl/point_cloud.h"
#include "pcl/point_types.h"

#include "util/utils.h"

// TODO，后期有需要再用这个，因为这个没什么必要，其他算法中用到这个点的时候可能需要和Eigen::Vector3d进行转换，这个很耗时
// 之后的代码都基于Eigen::Vector3d来进行
// #include "common/point.h"
#include "glog/logging.h"

#include "common/point.h"

namespace Common {

// TODO, 需要将一些成员函数转移到util工具类中，点云尽量保持纯粹
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
  void emplace_back(const float x, const float y, const float z) {
    points_.emplace_back(PointXYZ(x, y, z));
  }

  void emplace_back(const PointXYZ pt) {
    points_.emplace_back(pt);
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

  // 计算点云的中心
  Eigen::Vector3d ComputeCentroid() const {
    if (points_.empty()) {
      LOG(FATAL) << "Point cloud is empty";
      return Eigen::Vector3d::Zero();  // 返回一个零向量以避免编译错误
    }
    Eigen::Vector3d centroid(0.0, 0.0, 0.0);
    for (const auto& point : points_) {
      centroid += point;
    }
    centroid /= static_cast<double>(points_.size());
    return centroid;
  }

  void DemeanPointCloud() {
    Eigen::Vector3d centroid = this->ComputeCentroid();
    for (auto& point : points_) {
      point = point - centroid;
    }
  }

  // 去中心化点云
  void DemeanPointCloud(PointCloud* const target) {
    CHECK_NOTNULL(target);
    Eigen::Vector3d centroid = this->ComputeCentroid();
    target->clear();
    target->reserve(points_.size());
    for (const auto& point : points_) {
      Eigen::Vector3d demeaned_point = point - centroid;
      target->emplace_back(demeaned_point.x(), demeaned_point.y(), demeaned_point.z());
    }
  }

  // 这个会改变自身点云点的坐标
  void TransformPointCloud(const Eigen::Matrix4d& transformation) {
    Eigen::Matrix3d rotation = transformation.block<3, 3>(0, 0);
    Eigen::Vector3d translation = transformation.block<3, 1>(0, 3);
    for (auto& point : points_) {
      point = rotation * point + translation;
    }
  }

  // 用于将变换后的点云赋值给目标点云
  void TransformPointCloud(const Eigen::Matrix4d& transformation, PointCloud* const target) {
    CHECK_NOTNULL(target);
    target->clear();
    target->reserve(points_.size());

    Eigen::Matrix3d rotation = transformation.block<3, 3>(0, 0);
    Eigen::Vector3d translation = transformation.block<3, 1>(0, 3);

    for (const auto& point : points_) {
      Eigen::Vector3d transformed_point = rotation * point + translation;
      target->emplace_back(transformed_point.x(), transformed_point.y(), transformed_point.z());
    }
  }

  // 用于将变换后的点云赋值给目标点云
  void TransformPointCloud(const Eigen::Matrix4d& transformation,
                           const std::vector<uint32_t>& indexs,
                           PointCloud* const target) {
    CHECK_NOTNULL(target);
    target->clear();
    target->reserve(points_.size());
    Eigen::Matrix3d rotation = transformation.block<3, 3>(0, 0);
    Eigen::Vector3d translation = transformation.block<3, 1>(0, 3);
    for (const auto& iter : indexs) {
      Eigen::Vector3d transformed_point = rotation * points_.at(iter) + translation;
      target->emplace_back(transformed_point.x(), transformed_point.y(), transformed_point.z());
    }
  }

  // 转换点云为 Eigen::MatrixXd
  // Eigen::MatrixXd ToMatrixXd() const {
  //   Eigen::MatrixXd matrix(points_.size(), 3);
  //   for (uint32_t i = 0; i < points_.size(); ++i) {
  //     matrix(i, 0) = points_[i].x();
  //     matrix(i, 1) = points_[i].y();
  //     matrix(i, 2) = points_[i].z();
  //   }
  //   return matrix;
  // }

  Eigen::MatrixXd ToMatrixXd() const {
    Eigen::MatrixXd matrix(points_.size(), 3);
    std::for_each(std::execution::par,
                  points_.begin(),
                  points_.end(),
                  [&matrix, idx = 0](const PointXYZ& point) mutable {
                    matrix(idx, 0) = point.x();
                    matrix(idx, 1) = point.y();
                    matrix(idx, 2) = point.z();
                    ++idx;
                  });
    return matrix;
  }

  pcl::PointCloud<pcl::PointXYZ>::Ptr ToPCLPointCloud() const {
    auto pcl_cloud = boost::make_shared<pcl::PointCloud<pcl::PointXYZ>>();
    pcl_cloud->width = points_.size();
    pcl_cloud->height = 1;  // Assuming unorganized point cloud
    pcl_cloud->is_dense = false;
    pcl_cloud->points.resize(points_.size());

#pragma omp parallel for
    for (uint32_t i = 0; i < points_.size(); ++i) {
      pcl_cloud->points[i].x = points_[i].x();
      pcl_cloud->points[i].y = points_[i].y();
      pcl_cloud->points[i].z = points_[i].z();
    }
    return pcl_cloud;
  }

 private:
  std::vector<PointXYZ> points_;
};

// 定义共享指针类型
using PointCloudPtr = std::shared_ptr<PointCloud>;

// TODO 进一步提供均值和方差的函数以及成员变量

}  // namespace Common
