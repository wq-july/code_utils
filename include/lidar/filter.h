#pragma once

#include <memory>

#include "pcl/filters/voxel_grid.h"
#include "pcl/point_cloud.h"
#include "pcl/point_types.h"

#include "lidar/filter.h"
#include "common/data/point_cloud.h"

using namespace Common::Data;

namespace Lidar {
class PointCloudFilter {
 public:
  PointCloudFilter() = default;
  ~PointCloudFilter() = default;
  PointCloudFilter(float leaf_size);

  // 降采样函数
  void Filter(const pcl::PointCloud<pcl::PointXYZ>::Ptr& input_cloud,
              pcl::PointCloud<pcl::PointXYZ>* const filtered_cloud);

  void Filter(const PointCloudPtr& input_cloud, PointCloud* const filtered_cloud);

 private:
  float leaf_size_;
};

}  // namespace Lidar
