#include <gtest/gtest.h>
#include <vector>
#include "imu/imu.h"

class IMUTest : public testing::Test {
protected:
  void SetUp() override {
    // Setup code that runs before each test case
    // For example, you can initialize variables here
  }

  void TearDown() override {
    // Teardown code that runs after each test case
    // For example, you can release resources here
  }

  // Add any member variables or helper functions you need
};

// Test case for IMUData reading
TEST_F(IMUTest, ReadDataTest) {
  Sensor::IMU imu_processer;
  std::vector<Sensor::IMUData> data_vec;
  // Replace "test_data_file.txt" with your test data file path
  imu_processer.ReadData("../data/imu/MS1.txt", &data_vec);
  EXPECT_GT(data_vec.size(), 0);  // Expecting at least one data point
}

int main(int argc, char** argv) {
  // Initialize Google Test framework
  ::testing::InitGoogleTest(&argc, argv);
  // Run tests
  return RUN_ALL_TESTS();
}
