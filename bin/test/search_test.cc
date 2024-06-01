#include <gflags/gflags.h>
#include <gtest/gtest.h>

#include <vector>

#include "pcl/io/pcd_io.h"

#include "util/time.h"

#define private public

#include "common/kdtree.h"
#include "common/voxel_map.h"

DEFINE_string(scan_pcd_path, "../bin/data/lidar/scan.pcd", "scan点云路径");
DEFINE_string(map_pcd_path, "../bin/data/lidar/map.pcd", "地图点云路径");
DEFINE_double(ann_alpha, 1.0, "AAN的比例因子");

class SearchTest : public testing::Test {
 public:
  void SetUp() override {
    scan_ = std::make_shared<Common::Data::PointCloud>();
    map_ = std::make_shared<Common::Data::PointCloud>();
    scan_->LoadPCDFile<pcl::PointXYZ>(FLAGS_scan_pcd_path);
    map_->LoadPCDFile<pcl::PointXYZ>(FLAGS_map_pcd_path);
  }

  Common::Data::PointCloudPtr scan_ = nullptr;
  Common::Data::PointCloudPtr map_ = nullptr;
  Utils::Timer timer_;
  bool enable_test_ = true;
};

// Test case for IMU Preintegration
TEST_F(SearchTest, KdTreeKnnSearchTest) {
  if (!enable_test_) {
    return;
  }
  if (scan_->empty() || map_->empty()) {
    LOG(ERROR) << "cannot load cloud";
    FAIL();
  }

  Common::KdTree kdtree;
  int32_t size = map_->size();

  timer_.StartTimer("Build KdTree");
  // 注意kdtree成员变量map_使用了移动构造函数，这个map_将会变成空的；
  kdtree.BuildTree(map_);
  timer_.StopTimer();
  timer_.PrintElapsedTime();

  kdtree.SetEnableANN(true, FLAGS_ann_alpha);
  LOG(INFO) << "Kd tree leaves: " << kdtree.Size() << ", points: " << size
            << ", time / size: " << timer_.GetElapsedTime(Utils::Timer::Microseconds) / size;

  timer_.StartTimer("100 KnnSearch Single Thread");
  for (uint32_t i = 0; i < 100; ++i) {
    std::vector<uint32_t> cloest_index;
    kdtree.GetClosestPoint(map_->points().at(i), &cloest_index, 5);
  }
  timer_.StopTimer();
  timer_.PrintElapsedTime();

  // timer_.StartTimer("KnnSearch Multi Threads");
  // std::vector<std::pair<uint32_t, uint32_t>> matches;
  // kdtree.GetClosestPointMT(scan_, &matches, 5);
  // timer_.StopTimer();
  // timer_.PrintElapsedTime();

  LOG(INFO) << "done....";

  SUCCEED();
}

// Test case for IMU Preintegration
TEST_F(SearchTest, VoxelMapKnnSearchTest) {
  if (!enable_test_) {
    return;
  }
  if (scan_->empty() || map_->empty()) {
    LOG(ERROR) << "cannot load cloud";
    FAIL();
  }

  // TODO，地图初始化改成配置类进行初始化
  Common::VoxelMap voxel_map(0.5, 100, 10);

  timer_.StartTimer("Voxel Map Load map");
  voxel_map.AddPoints(*map_);
  timer_.StopTimer();
  timer_.PrintElapsedTime();

  timer_.StartTimer("100 VoxelMap KnnSearch");
  for (uint32_t i = 0; i < 100; ++i) {
    voxel_map.GetClosestNeighbor(map_->points().at(i));
  }
  timer_.StopTimer();
  timer_.PrintElapsedTime();

  LOG(INFO) << "done....";

  SUCCEED();
}

int main(int argc, char** argv) {
  google::InitGoogleLogging(argv[0]);
  FLAGS_stderrthreshold = google::INFO;
  FLAGS_colorlogtostderr = true;

  // Initialize Google Test framework
  ::testing::InitGoogleTest(&argc, argv);
  google::ParseCommandLineFlags(&argc, &argv, true);
  // Run tests
  return RUN_ALL_TESTS();
}
