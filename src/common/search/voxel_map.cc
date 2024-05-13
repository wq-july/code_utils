#include "common/search/voxel_map.h"

#include <algorithm>

namespace Common {

std::vector<Eigen::Vector3d> VoxelMap::GetPoints(
    const std::vector<Eigen::Vector3i> &query_voxels) const {
  std::vector<Eigen::Vector3d> points;
  points.reserve(query_voxels.size() * max_points_per_voxel_);
  // 采用线程池来并行的塞点，后期可以尝试omp对比一下
  std::for_each(query_voxels.cbegin(), query_voxels.cend(), [&](const auto &query) {
    auto search = map_.find(query);
    // 也就是地图中有这个体素，那么久将这个体素中所有的点塞进去
    if (search != map_.end()) {
      for (const auto &point : search->second.points) {
        points.emplace_back(point);
      }
    }
  });
  return points;
}

std::vector<Eigen::Vector3d> VoxelMap::Pointcloud() const {
  std::vector<Eigen::Vector3d> points;
  points.reserve(max_points_per_voxel_ * map_.size());
  for (const auto &[voxel, voxel_block] : map_) {
    (void)voxel;
    for (const auto &point : voxel_block.points) {
      points.push_back(point);
    }
  }
  return points;
}

void VoxelMap::Update(const std::vector<Eigen::Vector3d> &points, const Eigen::Vector3d &origin) {
  AddPoints(points);
  RemovePointsFarFromLocation(origin);
}

void VoxelMap::Update(const std::vector<Eigen::Vector3d> &points, const Sophus::SE3d &pose) {
  std::vector<Eigen::Vector3d> points_transformed(points.size());
  std::transform(points.cbegin(), points.cend(), points_transformed.begin(),
                 [&](const auto &point) { return pose * point; });
  const Eigen::Vector3d &origin = pose.translation();
  Update(points_transformed, origin);
}

void VoxelMap::AddPoints(const std::vector<Eigen::Vector3d> &points) {
  std::for_each(points.cbegin(), points.cend(), [&](const auto &point) {
    auto voxel = Eigen::Vector3i((point / voxel_size_).template cast<int>());
    auto search = map_.find(voxel);
    if (search != map_.end()) {
      auto &voxel_block = search.value();
      voxel_block.AddPoint(point);
    } else {
      map_.insert({voxel, VoxelData{{point}, max_points_per_voxel_}});
    }
  });
}

void VoxelMap::RemovePointsFarFromLocation(const Eigen::Vector3d &origin) {
  const auto max_distance2 = max_distance_ * max_distance_;
  for (auto it = map_.begin(); it != map_.end();) {
    const auto &[voxel, voxel_block] = *it;
    const auto &pt = voxel_block.points.front();
    if ((pt - origin).squaredNorm() >= (max_distance2)) {
      it = map_.erase(it);
    } else {
      ++it;
    }
  }
}

}  // namespace Common
