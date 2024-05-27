#include "common/voxel_map.h"

#include <algorithm>

namespace Common {

bool VoxelMap::GenerateNearbyGrids(const NearbyType &type) {
  switch (type) {
    case NearbyType::CENTER:
      nearby_grids_.emplace_back(Eigen::Matrix<int32_t, 3, 1>(0, 0, 0));
      break;
    case NearbyType::NEARBY6:
      nearby_grids_ = {Eigen::Vector3i(0, 0, 0),  Eigen::Vector3i(-1, 0, 0),
                       Eigen::Vector3i(1, 0, 0),  Eigen::Vector3i(0, 1, 0),
                       Eigen::Vector3i(0, -1, 0), Eigen::Vector3i(0, 0, -1),
                       Eigen::Vector3i(0, 0, 1)};
      break;
    case NearbyType::NEARBY18:
      nearby_grids_ = {
          Eigen::Vector3i(0, 0, 0),  Eigen::Vector3i(-1, 0, 0),  Eigen::Vector3i(1, 0, 0),
          Eigen::Vector3i(0, 1, 0),  Eigen::Vector3i(0, -1, 0),  Eigen::Vector3i(0, 0, -1),
          Eigen::Vector3i(0, 0, 1),  Eigen::Vector3i(1, 1, 0),   Eigen::Vector3i(-1, 1, 0),
          Eigen::Vector3i(1, -1, 0), Eigen::Vector3i(-1, -1, 0), Eigen::Vector3i(1, 0, 1),
          Eigen::Vector3i(-1, 0, 1), Eigen::Vector3i(1, 0, -1),  Eigen::Vector3i(-1, 0, -1),
          Eigen::Vector3i(0, 1, 1),  Eigen::Vector3i(0, -1, 1),  Eigen::Vector3i(0, 1, -1),
          Eigen::Vector3i(0, -1, -1)};
      break;
    case NearbyType::NEARBY26:
      nearby_grids_ = {
          Eigen::Vector3i(0, 0, 0),   Eigen::Vector3i(-1, 0, 0),  Eigen::Vector3i(1, 0, 0),
          Eigen::Vector3i(0, 1, 0),   Eigen::Vector3i(0, -1, 0),  Eigen::Vector3i(0, 0, -1),
          Eigen::Vector3i(0, 0, 1),   Eigen::Vector3i(1, 1, 0),   Eigen::Vector3i(-1, 1, 0),
          Eigen::Vector3i(1, -1, 0),  Eigen::Vector3i(-1, -1, 0), Eigen::Vector3i(1, 0, 1),
          Eigen::Vector3i(-1, 0, 1),  Eigen::Vector3i(1, 0, -1),  Eigen::Vector3i(-1, 0, -1),
          Eigen::Vector3i(0, 1, 1),   Eigen::Vector3i(0, -1, 1),  Eigen::Vector3i(0, 1, -1),
          Eigen::Vector3i(0, -1, -1), Eigen::Vector3i(1, 1, 1),   Eigen::Vector3i(-1, 1, 1),
          Eigen::Vector3i(1, -1, 1),  Eigen::Vector3i(1, 1, -1),  Eigen::Vector3i(-1, -1, 1),
          Eigen::Vector3i(-1, 1, -1), Eigen::Vector3i(1, -1, -1), Eigen::Vector3i(-1, -1, -1)};
      break;
    default:
      LOG(FATAL) << "Nearby type is invalid.";
      return false;
  }
  return true;
}

std::pair<Eigen::Vector3d, double> VoxelMap::GetClosestNeighbor(const Eigen::Vector3d &point) {
  const auto &index = PointToVoxelIndex(point);
  auto query_grids = nearby_grids_;
  for (auto &iter : query_grids) {
    iter += index;
  }

  // TODO
  // 直接在这个函数内部进行最近点查询，因为这个函数内部使用了多线程来选点，在选点的过程中可以使用有序队列
  const auto &neighbors = GetPoints(query_grids);

  // Find the nearest neighbor
  Eigen::Vector3d closest_neighbor;
  double closest_distance = std::numeric_limits<double>::max();
  std::for_each(neighbors.cbegin(), neighbors.cend(), [&](const auto &neighbor) {
    double distance = Utils::ComputeDistance(neighbor, point);
    if (distance < closest_distance) {
      closest_neighbor = neighbor;
      closest_distance = distance;
    }
  });
  return std::make_pair(closest_neighbor, closest_distance);
}

// TODO，其中points可以使用有序的队列来搞定，并且只需要最小值，最大值可以弹出
std::vector<Eigen::Vector3d> VoxelMap::GetPoints(
    const std::vector<Eigen::Vector3i> &query_voxels) const {
  std::vector<Eigen::Vector3d> points;
  points.reserve(query_voxels.size() * max_points_per_voxel_);
  // 采用线程池来并行的塞点，后期可以尝试omp对比一下
  std::for_each(query_voxels.cbegin(), query_voxels.cend(), [&](const auto &query) {
    auto search = map_.find(query);
    // 也就是地图中有这个体素，那么久将这个体素中所有的点塞进去
    if (search != map_.end()) {
      for (const auto &point : search->second.points_) {
        points.emplace_back(point);
      }
    }
  });
  return points;
}

std::vector<Eigen::Vector3d> VoxelMap::Pointcloud() const {
  std::vector<Eigen::Vector3d> points;
  points.reserve(max_points_per_voxel_ * map_.size());
  for (const auto &[index, voxel] : map_) {
    (void)index;
    for (const auto &point : voxel.points_) {
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
    auto index = PointToVoxelIndex(point);
    auto search = map_.find(index);
    if (search != map_.end()) {
      auto &voxel = search.value();
      voxel.AddPoint(point);
    } else {
      map_.insert({index, Voxel(point, max_points_per_voxel_)});
    }
  });
}

void VoxelMap::RemovePointsFarFromLocation(const Eigen::Vector3d &origin) {
  const auto max_distance2 = max_distance_ * max_distance_;
  for (auto it = map_.begin(); it != map_.end();) {
    const auto &[index, voxel] = *it;
    const auto &pt = voxel.points_.front();
    // 这里说明这些已经存进去的点的坐标都应该是被transformed过的
    if ((pt - origin).squaredNorm() >= (max_distance2)) {
      it = map_.erase(it);
    } else {
      ++it;
    }
  }
}

}  // namespace Common
