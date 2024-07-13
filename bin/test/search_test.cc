#include <gflags/gflags.h>
#include <gtest/gtest.h>

#include <vector>

#include "pcl/io/pcd_io.h"
#include "pcl/kdtree/kdtree_flann.h"

#include "util/time.h"

#define private public

#include "common/kdtree.h"
#include "common/voxel_map.h"

DEFINE_string(scan_pcd_path, "../bin/data/lidar/scan.pcd", "scan点云路径");
DEFINE_string(map_pcd_path, "../bin/data/lidar/map.pcd", "地图点云路径");
DEFINE_double(ann_alpha, 1.0, "AAN的比例因子");

typedef pcl::PointCloud<pcl::PointXYZ> PCLPointCloud;

class SearchTest : public testing::Test {
 public:
  void SetUp() override {
    scan_ = std::make_shared<Common::PointCloud>();
    map_ = std::make_shared<Common::PointCloud>();
    scan_->LoadPCDFile<pcl::PointXYZ>(FLAGS_scan_pcd_path);
    map_->LoadPCDFile<pcl::PointXYZ>(FLAGS_map_pcd_path);

    pcl_scan_.reset(new pcl::PointCloud<pcl::PointXYZ>());
    pcl_map_.reset(new pcl::PointCloud<pcl::PointXYZ>());
    pcl::io::loadPCDFile<pcl::PointXYZ>(FLAGS_scan_pcd_path, *pcl_scan_);
    pcl::io::loadPCDFile<pcl::PointXYZ>(FLAGS_map_pcd_path, *pcl_map_);
  }

  Common::PointCloudPtr scan_ = nullptr;
  Common::PointCloudPtr map_ = nullptr;

  PCLPointCloud::Ptr pcl_scan_ = nullptr;
  PCLPointCloud::Ptr pcl_map_ = nullptr;
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

  int32_t size = map_->size();

  // TODO，地图初始化改成配置类进行初始化
  Common::VoxelMap voxel_map(0.5, 100, 50);
  timer_.StartTimer("Voxel Map Load map");
  voxel_map.AddPoints(*map_);
  timer_.StopTimer();
  timer_.PrintElapsedTime();

  Common::KdTree kdtree;
  timer_.StartTimer("Build KdTree");
  // 注意kdtree成员变量map_如果使用了移动构造函数，这个map_将会变成空的；
  kdtree.BuildTree(map_);
  timer_.StopTimer();
  timer_.PrintElapsedTime();
  kdtree.SetEnableANN(true, FLAGS_ann_alpha);
  LOG(INFO) << "Kd tree leaves: " << kdtree.Size() << ", points: " << size
            << ", time / size: " << timer_.GetElapsedTime(Utils::Timer::Microseconds) / size;

  // ----------------------------------------------
  // 使用pcl点云库实现kdtree搜索
  pcl::KdTreeFLANN<pcl::PointXYZ> pcl_kdtree;
  pcl_kdtree.setInputCloud(pcl_map_);
  // 执行 K 近邻搜索
  int32_t k_nums = 5;  // 最近邻的数量
  // ----------------------------------------------

  timer_.StartTimer("1000 MyKdtree Search");
  for (uint32_t i = 0; i < 1000; ++i) {
    std::vector<std::pair<uint32_t, double>> cloest_index;
    kdtree.GetClosestPoint(map_->points().at(i), &cloest_index, k_nums);
  }
  timer_.StopTimer();
  timer_.PrintElapsedTime();

  std::vector<pcl::PointXYZ> pts(1000);
  for (uint32_t i = 0; i < 1000; ++i) {
    pcl::PointXYZ pt;
    pt.x = map_->points().at(i).x();
    pt.y = map_->points().at(i).y();
    pt.z = map_->points().at(i).z();
  }

  timer_.StartTimer("1000 PclKdtree Search");
  for (uint32_t i = 0; i < 1000; ++i) {
    std::vector<int32_t> point_index(k_nums);
    std::vector<float> point_distance_square(k_nums);
    pcl_kdtree.nearestKSearch(pts.at(i), k_nums, point_index, point_distance_square);
  }
  timer_.StopTimer();
  timer_.PrintElapsedTime();

  timer_.StartTimer("1000 VoxelMap Search");
  for (uint32_t i = 0; i < 1000; ++i) {
    std::vector<std::pair<Eigen::Vector3d, double>> res;
    voxel_map.GetClosestNeighbor(map_->points().at(i), &res, k_nums);
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
