#include <gflags/gflags.h>
#include <gtest/gtest.h>

#include <vector>

#include "pcl/io/pcd_io.h"
#include "util/time.h"

#define private public

#include "common/search/kdtree.h"

DEFINE_string(scan_pcd_path, "../test/data/lidar/scan.pcd", "scan点云路径");
DEFINE_string(map_pcd_path, "../test/data/lidar/map.pcd", "地图点云路径");
DEFINE_double(ann_alpha, 1.0, "AAN的比例因子");

class SearchTest : public testing::Test {
 protected:
  void SetUp() override {
    scan_.reset(new pcl::PointCloud<pcl::PointXYZ>());
    map_.reset(new pcl::PointCloud<pcl::PointXYZ>());
    pcl::io::loadPCDFile<pcl::PointXYZ>(FLAGS_scan_pcd_path, *scan_);
    pcl::io::loadPCDFile<pcl::PointXYZ>(FLAGS_map_pcd_path, *map_);
  }

  pcl::PointCloud<pcl::PointXYZ>::Ptr scan_ = nullptr;
  pcl::PointCloud<pcl::PointXYZ>::Ptr map_ = nullptr;
  Utils::Timer timer_;
};

// Test case for IMU Preintegration
TEST_F(SearchTest, KdTreeKnnSearchTest) {
  if (scan_->empty() || map_->empty()) {
    LOG(ERROR) << "cannot load cloud";
    FAIL();
  }

  Common::KdTree kdtree;

  timer_.StartTimer("Build KdTree");
  kdtree.BuildTree(map_);
  timer_.StopTimer();
  timer_.PrintElapsedTime();

  kdtree.SetEnableANN(true, FLAGS_ann_alpha);
  LOG(INFO) << "Kd tree leaves: " << kdtree.Size() << ", points: " << map_->size()
            << ", time / size: "
            << timer_.GetElapsedTime(Utils::Timer::Microseconds) / map_->size();

  timer_.StartTimer("KnnSearch Single Thread");
  std::vector<uint32_t> cloest_index;
  kdtree.GetClosestPoint(scan_->points.at(0), &cloest_index, 5);
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
