#pragma once

#include <memory>
#include <unordered_map>
#include <vector>
// tsl::robin_map据说性能可以达到std::unordered_map的十倍，具体需要进行测试，我们这里写两个map来对比一下
#include <tsl/robin_map.h>

#include "Eigen/Core"
#include "sophus/se3.hpp"
#include "pcl/point_cloud.h"
#include "pcl/point_types.h"
#include "util/utils.h"

namespace Common {

class VoxelMap {
  // 每个最小voxel单元，其中存储着点
  struct VoxelData {
    // buffer of points with a max limit of n_points
    std::vector<Eigen::Vector3d> points;
    uint32_t num_points_;
    inline void AddPoint(const Eigen::Vector3d &point) {
      if (points.size() < static_cast<size_t>(num_points_)) points.push_back(point);
    }
  };

 public:
  // 我觉得这个函数之后可以改写一下，把这些参数放到配置类中去
  explicit VoxelMap(const double voxel_size, const double max_distance,
                    const int32_t max_points_per_voxel)
      : voxel_size_(voxel_size),
        max_distance_(max_distance),
        max_points_per_voxel_(max_points_per_voxel) {}

  inline void Clear() { map_.clear(); }
  inline bool Empty() { return map_.empty(); }

  inline Eigen::Vector3i PointToVoxel(const Eigen::Vector3d &point) const {
    return Utils::FastFloor(point);
  }


  void Update(const std::vector<Eigen::Vector3d> &points, const Eigen::Vector3d &origin);
  
  void Update(const std::vector<Eigen::Vector3d> &points, const Sophus::SE3d &pose);
  
  // 添加地图点
  void AddPoints(const std::vector<Eigen::Vector3d> &points);
  
  // 用来移除距离目前位置比较远的地图点
  void RemovePointsFarFromLocation(const Eigen::Vector3d &origin);

  std::vector<Eigen::Vector3d> Pointcloud() const;
  
  std::vector<Eigen::Vector3d> GetPoints(const std::vector<Eigen::Vector3i> &query_voxels) const;

 private:
  uint32_t voxel_size_ = 0u;
  double max_distance_ = 0.0;
  uint32_t max_points_per_voxel_ = 0u;
  tsl::robin_map<Eigen::Vector3i, VoxelData, Utils::KissIcpHash> map_;

  // [robin_map] vs [unordered_map]
  // std::unordered_map<Eigen::Vector3i, VoxelData, Utils::KissIcpHash> map_;
};

}  // namespace Common
