#include "common/voxel_map.h"

#include <algorithm>
#include <future>
#include <mutex>
#include <queue>

#include "util/time.h"

namespace Common {

bool VoxelMap::GenerateNearbyGrids(const NearbyType& type) {
  switch (type) {
    case NearbyType::CENTER:
      nearby_grids_.emplace_back(Eigen::Matrix<int32_t, 3, 1>(0, 0, 0));
      break;
    case NearbyType::NEARBY6:
      nearby_grids_ = {Eigen::Vector3i(0, 0, 0),
                       Eigen::Vector3i(-1, 0, 0),
                       Eigen::Vector3i(1, 0, 0),
                       Eigen::Vector3i(0, 1, 0),
                       Eigen::Vector3i(0, -1, 0),
                       Eigen::Vector3i(0, 0, -1),
                       Eigen::Vector3i(0, 0, 1)};
      break;
    case NearbyType::NEARBY18:
      nearby_grids_ = {Eigen::Vector3i(0, 0, 0),
                       Eigen::Vector3i(-1, 0, 0),
                       Eigen::Vector3i(1, 0, 0),
                       Eigen::Vector3i(0, 1, 0),
                       Eigen::Vector3i(0, -1, 0),
                       Eigen::Vector3i(0, 0, -1),
                       Eigen::Vector3i(0, 0, 1),
                       Eigen::Vector3i(1, 1, 0),
                       Eigen::Vector3i(-1, 1, 0),
                       Eigen::Vector3i(1, -1, 0),
                       Eigen::Vector3i(-1, -1, 0),
                       Eigen::Vector3i(1, 0, 1),
                       Eigen::Vector3i(-1, 0, 1),
                       Eigen::Vector3i(1, 0, -1),
                       Eigen::Vector3i(-1, 0, -1),
                       Eigen::Vector3i(0, 1, 1),
                       Eigen::Vector3i(0, -1, 1),
                       Eigen::Vector3i(0, 1, -1),
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

void VoxelMap::GetClosestNeighbor(const Eigen::Vector3d& point,
                                  std::vector<std::pair<Eigen::Vector3d, double>>* const res,
                                  const uint32_t k_nums) {
  const auto& index = PointToVoxelIndex(point);
  auto query_grids = nearby_grids_;
  for (auto& iter : query_grids) {
    iter += index;
  }
  // 定义一个最大堆来保存最近的 k 个邻居
  auto compare = [](const std::pair<Eigen::Vector3d, double>& a,
                    const std::pair<Eigen::Vector3d, double>& b) {
    return a.second < b.second;  // 最大堆：较大的距离优先级较高
  };
  std::priority_queue<std::pair<Eigen::Vector3d, double>,
                      std::vector<std::pair<Eigen::Vector3d, double>>,
                      decltype(compare)>
      ordered_queue(compare);
  std::mutex ordered_queue_mutex;
  auto process_grid = [&](const auto& query) {
    std::priority_queue<std::pair<Eigen::Vector3d, double>,
                        std::vector<std::pair<Eigen::Vector3d, double>>,
                        decltype(compare)>
        local_ordered_queue(compare);
    auto search = map_.find(query);
    if (search != map_.end()) {
      for (const auto& pt : search->second.points_) {
        double distance = (pt - point).squaredNorm();
        local_ordered_queue.emplace(pt, distance);
        if (local_ordered_queue.size() > k_nums) {
          local_ordered_queue.pop();  // 保持堆的大小不超过 k_nums
        }
      }
    }
    // 合并局部优先队列到全局优先队列
    std::lock_guard<std::mutex> lock(ordered_queue_mutex);
    while (!local_ordered_queue.empty()) {
      ordered_queue.emplace(local_ordered_queue.top());
      local_ordered_queue.pop();
      while (ordered_queue.size() > k_nums) {
        ordered_queue.pop();  // 保持全局堆的大小不超过 k_nums
      }
    }
  };
  // TODO, 这里是for循环，并不是并发处理！
  for (const auto& grid : query_grids) {
    process_grid(grid);
  }
  // 将结果从优先队列转移到 res
  res->clear();
  while (!ordered_queue.empty()) {
    res->emplace_back(ordered_queue.top());
    ordered_queue.pop();
  }
  // 因为 priority_queue 是最大堆，我们需要反转结果以得到最近邻的顺序
  std::reverse(res->begin(), res->end());
}

void VoxelMap::GetNeighborVoxels(const Eigen::Vector3d& point,
                                 std::vector<GaussianVoxel>* const nearby_voxels) {
  nearby_voxels->clear();
  const auto& key = PointToVoxelIndex(point);
  for (const auto& iter : nearby_grids_) {
    auto keyoff = iter + key;
    auto search = map_.find(keyoff);
    if (search != map_.end()) {
      nearby_voxels->emplace_back(search->second);
    }
  }
}

std::vector<Eigen::Vector3d> VoxelMap::Pointcloud() const {
  std::vector<Eigen::Vector3d> points;
  points.reserve(max_pts_per_voxel_ * map_.size());
  for (const auto& [index, voxel] : map_) {
    (void)index;
    for (const auto& point : voxel.points_) {
      points.push_back(point);
    }
  }
  return points;
}

void VoxelMap::Update(const std::vector<Eigen::Vector3d>& points, const Eigen::Vector3d& origin) {
  AddPoints(points);
  RemovePointsFarFromLocation(origin);
}

void VoxelMap::Update(const std::vector<Eigen::Vector3d>& points, const Sophus::SE3d& pose) {
  std::vector<Eigen::Vector3d> points_transformed(points.size());
  std::transform(
      points.cbegin(), points.cend(), points_transformed.begin(), [&](const auto& point) {
        return pose * point;
      });
  const Eigen::Vector3d& origin = pose.translation();
  Update(points_transformed, origin);
}

// 采用自己写的点云结构
void VoxelMap::AddPoints(const Common::PointCloud& cloud_points) {
  AddPoints(cloud_points.points());
}

void VoxelMap::AddPoints(const std::vector<Eigen::Vector3d>& points) {
  if (points.empty()) {
    return;
  }
  std::for_each(points.cbegin(), points.cend(), [&](const auto& point) {
    auto index = PointToVoxelIndex(point);
    map_[index].max_nums_ = max_pts_per_voxel_;
    map_[index].AddPoint(point);
  });
}

void VoxelMap::RemovePointsFarFromLocation(const Eigen::Vector3d& origin) {
  const auto max_distance2 = max_distance_ * max_distance_;
  for (auto it = map_.begin(); it != map_.end();) {
    const auto& [index, voxel] = *it;
    const auto& pt = voxel.points_.front();
    // 这里说明这些已经存进去的点的坐标都应该是被transformed过的
    if ((pt - origin).squaredNorm() >= max_distance2) {
      it = map_.erase(it);
    } else {
      ++it;
    }
  }
}

void VoxelMap::RemoveFewerPointsVoxel() {
  for (auto it = map_.begin(); it != map_.end();) {
    if (it->second.size_ > min_pts_per_voxel_) {
      ++it;
    } else {
      it = map_.erase(it);
    }
  }
}

}  // namespace Common
