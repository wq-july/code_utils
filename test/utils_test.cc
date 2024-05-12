#include <gflags/gflags.h>
#include <gtest/gtest.h>

#include <vector>

#define private public
#include "../../../../../usr/include/glog/logging.h"
#include "util/math.h"
#include "util/time.h"

DEFINE_string(config_path, "../conf/imu_config.yaml", "imu的配置文件");

class UtilsTest : public testing::Test {
 public:
  void SetUp() override {}
  Utils::Timer timer_;
};

// Define your test cases
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
