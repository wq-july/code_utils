#pragma once

#include <memory>

#include "common/data/point_cloud.h"
#include "common/kdtree.h"
#include "common/voxel_map.h"

using namespace Common::Data;

namespace Lidar {

class VICP {
 public:
  VICP();
  ~VICP() = default;
  

 private:
  // 对应两种最近临搜索方式
  std::shared_ptr<Common::KdTree> kdtree_ = nullptr;
  std::shared_ptr<Common::VoxelMap> voxel_map_ = nullptr;
};

}  // namespace Lidar

// Test case for IMU Preintegration
// TEST_F(SearchTest, VoxelMapKnnSearchTest) {
//   if (scan_->empty() || map_->empty()) {
//     LOG(ERROR) << "cannot load cloud";
//     FAIL();
//   }

//   // TODO，地图初始化改成配置类进行初始化
//   Common::VoxelMap voxel_map(0.5, 100, 10);
//   auto map_points = Utils::PclToEigen3d(map_);

//   timer_.StartTimer("Voxel Map Load map");
//   voxel_map.AddPoints(map_points);
//   timer_.StopTimer();
//   timer_.PrintElapsedTime();

//   timer_.StartTimer("100 VoxelMap KnnSearch");
//   for (uint32_t i = 0; i < 100; ++i) {
//     voxel_map.GetClosestNeighbor(map_points.at(i));
//   }
//   timer_.StopTimer();
//   timer_.PrintElapsedTime();

//   LOG(INFO) << "done....";

//   SUCCEED();
// }
