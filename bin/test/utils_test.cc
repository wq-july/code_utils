#include <gflags/gflags.h>
#include <glog/logging.h>
#include <gtest/gtest.h>

#include <filesystem>
#include <vector>

#include "pcl/io/pcd_io.h"

#define private public

#include "util/math.h"
#include "util/time.h"
#include "util/utils.h"

DEFINE_string(config_path, "../conf/imu_config.yaml", "imu的配置文件");
DEFINE_string(scan_pcd_path, "../bin/data/lidar/scan.pcd", "scan点云路径");
DEFINE_string(random_data_path, "../log/random_data.txt", "随机生成的函数仿真数据");

class UtilsTest : public testing::Test {
 public:
  void SetUp() override {
    scan_.reset(new pcl::PointCloud<pcl::PointXYZ>());
    pcl::io::loadPCDFile<pcl::PointXYZ>(FLAGS_scan_pcd_path, *scan_);
  }
  pcl::PointCloud<pcl::PointXYZ>::Ptr scan_ = nullptr;

  Utils::Timer timer_;
};

TEST_F(UtilsTest, PclToEigen3dTest) {
  timer_.StartTimer("Pcl to Eigen scan");
  auto scan_points = Utils::PclToEigen3d(scan_);
  timer_.StopTimer();
  timer_.PrintElapsedTime();

  EXPECT_GT(scan_points.size(), 100);
}

// Math::ComputeMeanAndVariance
TEST_F(UtilsTest, ComputeMeanAndVariance) {
  std::vector<uint32_t> indices{0u, 1u, 2u};
  std::vector<Eigen::Vector3d> data = {Eigen::Vector3d(1.0, 2.0, 3.0),
                                       Eigen::Vector3d(4.0, 5.0, 6.0),
                                       Eigen::Vector3d(7.0, 8.0, 9.0)};
  Eigen::Vector3d mean, variance;

  // Call the function under test
  Utils::Math::ComputeMeanAndVariance(indices, data, &mean, &variance);
  EXPECT_EQ(mean, Eigen::Vector3d(4.0, 5.0, 6.0));
  EXPECT_EQ(variance, Eigen::Vector3d(6.0, 6.0, 6.0));
}

// Math::GenerateRandomCoefficientsAndData
TEST_F(UtilsTest, GenerateRandomCoefficientsAndData) {
  Eigen::VectorXd parameters;
  std::vector<std::pair<double, double>> data;

  // Define a lambda function for the polynomial exp(a0 + a1*x + a2*x^2)
  auto poly_func = [](const Eigen::VectorXd& params, double x) {
    double result = 0.0;

    for (uint32_t i = 0; i < params.size(); ++i) {
      result += params[i] * std::pow(x, i);
    }
    return std::exp(result);
  };

  Utils::GenerateRandomCoefficientsAndData(poly_func,
                                           3,            // Number of parameters
                                           1000,         // Number of data points
                                           {-1.0, 1.0},  // Parameter range
                                           {-0.1, 0.1},  // Noise range
                                           FLAGS_random_data_path, &parameters, &data);
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
