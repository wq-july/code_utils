#include <gtest/gtest.h>

#include <vector>

#include "util/config.h"

#define private public

#include "imu/imu.h"

class IMUTest : public testing::Test {
 protected:
  void SetUp() override {
    Config config("../conf/imu_config.yaml");
    imu_processer_ = std::make_shared<Sensor::IMU>(config.imu_config_);
  }
  std::shared_ptr<Sensor::IMU> imu_processer_ = nullptr;
};

// Test case for IMUData reading
TEST_F(IMUTest, ReadDataTest) {
  std::vector<Sensor::IMUData> data_vec;
  // Replace "test_data_file.txt" with your test data file path
  imu_processer_->ReadData("../data/imu/MS1.txt", &data_vec);
  EXPECT_GT(data_vec.size(), 0);  // Expecting at least one data point
}

int main(int argc, char** argv) {
  // Initialize Google Test framework
  ::testing::InitGoogleTest(&argc, argv);
  // Run tests
  return RUN_ALL_TESTS();
}
