#include "pcl/registration/icp.h"

#include <vector>

#include "gflags/gflags.h"
#include "gtest/gtest.h"
#include "pcl/filters/approximate_voxel_grid.h"
#include "pcl/io/pcd_io.h"
#include "pcl/point_types.h"
#include "pcl/registration/ndt.h"

#include "glog/logging.h"

#define private public

#include "lidar/classic_icp.h"
#include "lidar/ndt.h"

DEFINE_string(scan_pcd_path, "../bin/data/lidar/scan.pcd", "scan点云路径");
DEFINE_string(map_pcd_path, "../bin/data/lidar/map.pcd", "地图点云路径");

typedef pcl::PointCloud<pcl::PointXYZ> PCLPointCloud;

class LidarTest : public testing::Test {
 protected:
  void SetUp() override {
    icp_ = std::make_shared<Lidar::ClassicICP>(0.5,
                                               100.0,
                                               10,
                                               Lidar::AlignMethod::POINT_TO_PLANE_ICP,
                                               Lidar::SearchMethod::KDTREE,
                                               20,
                                               1.0e-6,
                                               20);
    scan_ = std::make_shared<PointCloud>();
    map_ = std::make_shared<PointCloud>();
    pcl_scan_.reset(new pcl::PointCloud<pcl::PointXYZ>());
    pcl_map_.reset(new pcl::PointCloud<pcl::PointXYZ>());
    scan_->LoadPCDFile<pcl::PointXYZ>(FLAGS_scan_pcd_path);
    map_->LoadPCDFile<pcl::PointXYZ>(FLAGS_map_pcd_path);
    pcl::io::loadPCDFile<pcl::PointXYZ>(FLAGS_scan_pcd_path, *pcl_scan_);
    pcl::io::loadPCDFile<pcl::PointXYZ>(FLAGS_map_pcd_path, *pcl_map_);
  }
  std::shared_ptr<Lidar::ClassicICP> icp_ = nullptr;
  PointCloudPtr scan_ = nullptr;
  PointCloudPtr map_ = nullptr;
  PCLPointCloud::Ptr pcl_scan_ = nullptr;
  PCLPointCloud::Ptr pcl_map_ = nullptr;
  Utils::Timer timer_;
  bool enable_test_ = true;
};

TEST_F(LidarTest, PCLIcpTest) {
  if (enable_test_) {
    return;
  }
  // 设置初始变换矩阵
  Eigen::Matrix4f initial_guess = Eigen::Matrix4f::Identity();
  initial_guess(0, 3) = 1094.f;
  initial_guess(1, 3) = 8678.f;
  initial_guess(2, 3) = 6.f;

  // ---------------------------------------------------
  // PCL ICP
  pcl::IterativeClosestPoint<pcl::PointXYZ, pcl::PointXYZ> icp;
  timer_.StartTimer("Build Source Tree");
  pcl::search::KdTree<pcl::PointXYZ>::Ptr source_tree(new pcl::search::KdTree<pcl::PointXYZ>);
  source_tree->setInputCloud(pcl_scan_);
  icp.setSearchMethodSource(source_tree);
  timer_.StopTimer();
  timer_.PrintElapsedTime();

  timer_.StartTimer("Build Target Tree");
  pcl::search::KdTree<pcl::PointXYZ>::Ptr target_tree(new pcl::search::KdTree<pcl::PointXYZ>);
  target_tree->setInputCloud(pcl_map_);
  icp.setSearchMethodTarget(target_tree);
  timer_.StopTimer();
  timer_.PrintElapsedTime();

  icp.setInputSource(pcl_scan_);
  icp.setInputTarget(pcl_map_);

  pcl::PointCloud<pcl::PointXYZ>::Ptr icp_output(new pcl::PointCloud<pcl::PointXYZ>);
  timer_.StartTimer("PCL ICP");
  icp.align(*icp_output, initial_guess);
  timer_.StopTimer();
  timer_.PrintElapsedTime();
  if (icp.hasConverged()) {
    std::cout << "ICP has converged, score: " << icp.getFitnessScore() << "\n";
    std::cout << "Final Transformation is \n" << icp.getFinalTransformation() << "\n";
  } else {
    LOG(FATAL) << "ICP did not converge.";
  }

  // ---------------------------------------------------
  // PCL NDT
  pcl::NormalDistributionsTransform<pcl::PointXYZ, pcl::PointXYZ> ndt;
  // 设置要配准的点云
  ndt.setInputSource(pcl_scan_);
  // 设置点云配准目标
  ndt.setInputTarget(pcl_map_);
  pcl::PointCloud<pcl::PointXYZ>::Ptr ndt_output(new pcl::PointCloud<pcl::PointXYZ>);
  timer_.StartTimer("PCL NDT");
  ndt.align(*ndt_output, initial_guess);
  timer_.StopTimer();
  timer_.PrintElapsedTime();
  if (ndt.hasConverged()) {
    std::cout << "NDT has converged, score: " << ndt.getFitnessScore() << "\n";
    std::cout << "Final Transformation is \n" << ndt.getFinalTransformation() << "\n";
  } else {
    LOG(FATAL) << "NDT did not converge.";
  }
}

TEST_F(LidarTest, MyIcpTest) {
  if (enable_test_) {
    return;
  }
  Eigen::Isometry3d initial_guess = Eigen::Isometry3d::Identity();
  initial_guess.translation() = Eigen::Vector3d(1094.0, 8678.0, 6.0);
  Eigen::Matrix3d rotation_matrix;
  rotation_matrix = Eigen::AngleAxisd(1.8, Eigen::Vector3d::UnitZ());
  initial_guess.linear() = rotation_matrix;
  Eigen::Isometry3d final_transform = Eigen::Isometry3d::Identity();
  icp_->Align(initial_guess, scan_, map_, &final_transform);
  std::cout << "trans: " << final_transform.translation().transpose() << ", rot is "
            << final_transform.rotation().eulerAngles(0, 1, 2).transpose() << "\n";
}

TEST_F(LidarTest, MyNDTTest) {
  if (!enable_test_) {
    return;
  }
  Lidar::NDT my_ndt(1.0, 30, 1.0e-6, 50, false, 20, 30);
  Eigen::Isometry3d initial_guess = Eigen::Isometry3d::Identity();
  initial_guess.translation() = Eigen::Vector3d(1094.0, 8678.0, 6.0);
  Eigen::Matrix3d rotation_matrix;
  rotation_matrix = Eigen::AngleAxisd(1.8, Eigen::Vector3d::UnitZ());
  initial_guess.linear() = rotation_matrix;
  Eigen::Isometry3d final_transform = Eigen::Isometry3d::Identity();
  my_ndt.Align(initial_guess, scan_, map_, &final_transform);
  std::cout << "trans: " << final_transform.translation().transpose() << ", rot is "
            << final_transform.rotation().eulerAngles(0, 1, 2).transpose() << "\n";
}

int main(int argc, char** argv) {
  // Initialize Google Test framework
  google::InitGoogleLogging(argv[0]);
  FLAGS_stderrthreshold = google::INFO;
  FLAGS_colorlogtostderr = true;

  ::testing::InitGoogleTest(&argc, argv);
  // Run tests
  return RUN_ALL_TESTS();
}
