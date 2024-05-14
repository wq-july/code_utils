#pragma once

#include <memory>
#include <unordered_map>
#include <vector>

#include "common/search/voxel.h"

// tsl::robin_map据说性能可以达到std::unordered_map的十倍，具体需要进行测试，我们这里写两个map来对比一下
#include <tsl/robin_map.h>

#include "Eigen/Core"
#include "glog/logging.h"
#include "pcl/point_cloud.h"
#include "pcl/point_types.h"
#include "sophus/se3.hpp"
#include "util/utils.h"

namespace Common {

class VoxelMap {
 public:
  enum class NearbyType {
    CENTER,    // 只考虑中心（同一个Voxel）中元素
    NEARBY6,   // 上下左右前后
    NEARBY18,  // 各个角
    NEARBY26   // 立方体
  };

 public:
  // 我觉得这个函数之后可以改写一下，把这些参数放到配置类中去
  explicit VoxelMap(const double voxel_size, const double max_distance,
                    const int32_t max_points_per_voxel)
      : voxel_size_(voxel_size),
        max_distance_(max_distance),
        max_points_per_voxel_(max_points_per_voxel) {
    GenerateNearbyGrids(NearbyType::NEARBY6);
  }

  inline void Clear() { map_.clear(); }
  inline bool Empty() { return map_.empty(); }

  inline Eigen::Vector3i PointToVoxelIndex(const Eigen::Vector3d &point) const {
    return Utils::FastFloor(point / voxel_size_);
  }

  void Update(const std::vector<Eigen::Vector3d> &points, const Eigen::Vector3d &origin);

  void Update(const std::vector<Eigen::Vector3d> &points, const Sophus::SE3d &pose);

  // 添加地图点
  void AddPoints(const std::vector<Eigen::Vector3d> &points);

  // 用来移除距离目前位置比较远的地图点
  void RemovePointsFarFromLocation(const Eigen::Vector3d &origin);

  std::vector<Eigen::Vector3d> Pointcloud() const;

  std::vector<Eigen::Vector3d> GetPoints(const std::vector<Eigen::Vector3i> &query_voxels) const;

  std::pair<Eigen::Vector3d, double> GetClosestNeighbor(const Eigen::Vector3d &point);

 private:
  bool GenerateNearbyGrids(const NearbyType &type);

 private:
  std::vector<Eigen::Vector3i> nearby_grids_;
  uint32_t voxel_size_ = 0u;
  double max_distance_ = 0.0;
  uint32_t max_points_per_voxel_ = 0u;
  tsl::robin_map<Eigen::Vector3i, Voxel, Utils::KissIcpHash> map_;

  // [robin_map] vs [unordered_map]
  // std::unordered_map<Eigen::Vector3i, Voxel, Utils::KissIcpHash> map_;
};

}  // namespace Common
