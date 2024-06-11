#include "lidar/filter.h"

#include "tsl/robin_map.h"
namespace Lidar {

// 构造函数，初始化叶子大小
PointCloudFilter::PointCloudFilter(float leaf_size) : leaf_size_(leaf_size) {}

// 降采样函数
void PointCloudFilter::Filter(const pcl::PointCloud<pcl::PointXYZ>::Ptr& input_cloud,
                              pcl::PointCloud<pcl::PointXYZ>* const filtered_cloud) {
  pcl::VoxelGrid<pcl::PointXYZ> sor;
  sor.setInputCloud(input_cloud);
  sor.setLeafSize(leaf_size_, leaf_size_, leaf_size_);
  sor.filter(*filtered_cloud);
}

void PointCloudFilter::Filter(const PointCloudPtr& input_cloud, PointCloud* const filtered_cloud) {
  tsl::robin_map<Eigen::Vector3i, std::vector<Eigen::Vector3d>, Utils::KissIcpHash> voxel_map;
  // 遍历点云数据并填充哈希表
  for (const auto& point : input_cloud->points()) {
    auto voxel_key = Utils::FastFloor(point / leaf_size_);
    voxel_map[voxel_key].emplace_back(point);
  }
  // 计算每个体素的平均值并存储到输出点云
  for (auto& voxel : voxel_map) {
    Eigen::Vector3d init(0, 0, 0);
    Eigen::Vector3d average_point =
        std::accumulate(voxel.second.begin(),
                        voxel.second.end(),
                        init,
                        [&](const Eigen::Vector3d& a, const Eigen::Vector3d& b) -> Eigen::Vector3d {
                          return a + b;
                        }) / static_cast<double>(voxel.second.size());
    filtered_cloud->emplace_back(average_point);
  }
}

}  // namespace Lidar