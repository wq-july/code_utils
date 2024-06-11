/****************************************************************************
 *
 * Copyright (c) 2021 ZelosTech.com, Inc. All Rights Reserved
 *
 ***************************************************************************/
/**
 * @file voxelnn.cc
 **/

#include "localization/matching/registration/vg_icp/common/voxelnn.h"

#include "Eigen/Eigenvalues"

#include "pcl/point_cloud.h"
#include "pcl/point_types.h"

namespace zelos {
namespace zoe {
namespace localization {

bool VoxelNN::SetPointCloud(const pcl::PointCloud<pcl::PointXYZI>::Ptr& input_cloud) {
  if (input_cloud->empty()) {
    return false;
  }

  cloud_ = std::make_shared<GaussianPointCloud>(input_cloud);
  std::unordered_map<Eigen::Vector3i, std::vector<int32_t>, XORVector3iHash> empty_grids;
  grids_.swap(empty_grids);

  std::vector<int32_t> index(cloud_->size());
  // 使用 OpenMP 初始化索引向量
  #pragma omp parallel for
  for (size_t i = 0; i < index.size(); ++i) {
    index[i] = static_cast<int32_t>(i);
  }
  // 为了避免竞争条件和数据一致性问题，我们使用局部容器收集每个线程的结果
  std::vector<std::unordered_map<Eigen::Vector3i, std::vector<int32_t>, XORVector3iHash>> local_grids(omp_get_max_threads());
  #pragma omp parallel
  {
    int32_t thread_id = omp_get_thread_num();
    auto& local_grid = local_grids[thread_id];
    #pragma omp for nowait
    for (size_t i = 0; i < index.size(); ++i) {
      const int32_t& idx = index[i];
      auto pt = cloud_->at(idx);
      auto key = FastFloor(pt);
      if (local_grid.find(key) == local_grid.end()) {
        local_grid[key] = {idx};
      } else {
        local_grid[key].emplace_back(idx);
      }
    }
  }
  // 合并所有线程的结果
  for (auto& lg : local_grids) {
    for (auto& entry : lg) {
      auto& key = entry.first;
      auto& list = entry.second;
      if (grids_.find(key) == grids_.end()) {
        grids_[key] = std::move(list);
      } else {
        grids_[key].insert(grids_[key].end(), list.begin(), list.end());
      }
    }
  }

  EstimateNormalsCovariancesOMP(20, 4);
  return true;
}

void VoxelNN::GenerateNearbyGrids() {
  if (nearby_type_ == NearbyType::CENTER) {
    nearby_grids_.emplace_back(Eigen::Vector3i::Zero());
  } else if (nearby_type_ == NearbyType::NEARBY6) {
    nearby_grids_ = {
        Eigen::Vector3i(0, 0, 0), Eigen::Vector3i(-1, 0, 0), Eigen::Vector3i(1, 0, 0),
        Eigen::Vector3i(0, 1, 0), Eigen::Vector3i(0, -1, 0), Eigen::Vector3i(0, 0, -1), 
        Eigen::Vector3i(0, 0, 1)};
  }
}

bool VoxelNN::KnnSearch(const int32_t k, const Eigen::Vector3d& pt, std::vector<std::pair<int32_t, Eigen::Vector3d>>* const res) {
  // 在pt栅格周边寻找最近邻
  std::vector<int32_t> idx_to_check;
  auto key = FastFloor(pt);
  // OpenMP并行处理查找附近栅格
  #pragma omp parallel
  {
    std::vector<int32_t> private_idx_to_check; // 每个线程的私有容器
    #pragma omp for nowait // 无需在循环结束时同步线程
    for (size_t i = 0; i < nearby_grids_.size(); ++i) {
        auto dkey = key.matrix() + nearby_grids_[i];
        auto iter = grids_.find(dkey);
        if (iter != grids_.end()) {
            private_idx_to_check.insert(private_idx_to_check.end(), iter->second.begin(), iter->second.end());
        }
    }
    // 合并私有容器到共享容器
    #pragma omp critical
    {
      idx_to_check.insert(idx_to_check.end(), private_idx_to_check.begin(), private_idx_to_check.end());
    }
  }
  if (idx_to_check.empty()) {
    return false;
  }
  // 构建近邻点云
  GaussianPointCloud::Ptr nearby_cloud(new GaussianPointCloud());
  std::vector<size_t> nearby_idx;
  #pragma omp parallel for
  for (size_t i = 0; i < idx_to_check.size(); ++i) {
    #pragma omp critical // 确保线程安全地访问nearby_cloud
    {
      nearby_cloud->points_.emplace_back(cloud_->at(idx_to_check[i]));
      nearby_idx.emplace_back(idx_to_check[i]);
    }
  }
  // 计算最近邻
  *res = BFNN(nearby_cloud, pt, k);
  return true;
}


std::vector<std::pair<int32_t, Eigen::Vector3d>> VoxelNN::BFNN(
    const GaussianPointCloud::Ptr& cloud, const Eigen::Vector3d& point, const int32_t k) {
  std::vector<std::pair<int32_t, double>> distances;
  distances.reserve(cloud->points().size());

  for (int32_t i = 0; i < cloud->size(); ++i) {
      double dist = (cloud->at(i) - point).squaredNorm();  // 使用平方距离减少计算成本
      distances.emplace_back(i, dist);
  }

  // 对距离进行部分排序，只找到最小的k个元素，  // 添加检查以防k大于distances的大小
  if (k < static_cast<int32_t>(distances.size())) {
    std::nth_element(distances.begin(), distances.begin() + k, distances.end(), [](const auto& a, const auto& b) {
      return a.second < b.second;
    });
  }

  std::vector<std::pair<int32_t, Eigen::Vector3d>> indices;
  indices.reserve(k);  // 预先分配空间以提高性能
  int32_t count = std::min(k, static_cast<int32_t>(distances.size()));  // 防止访问超出范围的元素
  for (int32_t i = 0; i < count; ++i) {
      indices.emplace_back(distances[i].first, cloud->at(distances[i].first));  // 返回索引和对应的点坐标
  }
  return indices;
}


void VoxelNN::EstimateNormalsCovariancesOMP(const int32_t num_neighbors, const int32_t num_threads) {
  ZCHECK_NOTNULL(cloud_);
  #pragma omp parallel for num_threads(num_threads)
  for (int32_t i = 0; i < cloud_->size(); ++i) {
    std::vector<std::pair<int32_t, Eigen::Vector3d>> k_pts;
    k_pts.reserve(num_neighbors);
    KnnSearch(num_neighbors, cloud_->at(i), &k_pts);
    const int32_t nums = k_pts.size();
    if (nums < 5) {
      // Insufficient number of neighbors
      cloud_->normal(i) = Eigen::Vector3d::Zero();
      cloud_->cov(i) = Eigen::Matrix3d::Identity();
      continue;
    }
    Eigen::Vector3d sum_points = Eigen::Vector3d::Zero();
    Eigen::Matrix3d sum_cross = Eigen::Matrix3d::Zero();
    for (int32_t i = 0; i < nums; ++i) {
      const auto& pt = cloud_->at(k_pts[i].first);
      sum_points += pt;
      sum_cross += pt * pt.transpose();
    }
    const Eigen::Vector3d mean = sum_points / nums;
    const Eigen::Matrix3d cov = (sum_cross - mean * sum_points.transpose()) / nums;

    Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> eig;
    eig.computeDirect(cov);

    const Eigen::Vector3d normal = eig.eigenvectors().col(0).normalized();
    if (cloud_->at(i).dot(normal) > 0.0) {
      cloud_->normal(i) = -normal;
    } else {
      cloud_->normal(i) = normal;
    }

    const Eigen::Vector3d values(1e-3, 1e-2, 1e-2);
    Eigen::Matrix3d final_cov = Eigen::Matrix3d::Zero();
    final_cov = eig.eigenvectors() * values.asDiagonal() * eig.eigenvectors().transpose();
    cloud_->cov(i) = final_cov;
  }
}

GaussianPointCloud::Ptr VoxelNN::GetCloud() const {
  return cloud_;
}

int32_t VoxelNN::Size() const {
  return cloud_->size();
}

}  // namespace localization
}  // namespace zoe
}  // namespace zelos

/* vim: set expandtab ts=2 sw=2 sts=2 tw=100: */
