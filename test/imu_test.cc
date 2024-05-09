#include <gtest/gtest.h>

#include <vector>

#include "common/sensor_data.hpp"
#include "util/config.h"

#define private public

#include "imu/imu_processor.h"

class IMUTest : public testing::Test {
 protected:
  void SetUp() override {
    Config config("../conf/imu_config.yaml");
    imu_processer_ = std::make_shared<IMU::ImuProcessor>(config.imu_config_);
  }
  std::shared_ptr<IMU::ImuProcessor> imu_processer_ = nullptr;
};

// Test case for IMUData reading
TEST_F(IMUTest, ReadDataTest) {
  std::vector<Common::IMUData> data_vec;
  imu_processer_->ReadData("../test/data/imu/MS1.txt", &data_vec);
  EXPECT_GT(data_vec.size(), 0);  // Expecting at least one data point
}

// Test case for IMU Preintegration
TEST_F(IMUTest, PreIntegrationTest) {
  std::vector<Common::IMUData> data_vec;
  imu_processer_->ReadData("../test/data/imu/MG35.txt", &data_vec);

  Common::SimpleState start_status(data_vec.front().timestamp_);

  for (int i = 0; i < 100; ++i) {
    Common::IMUData imu_data = data_vec[i];
    imu_processer_->pre_integrator_->Update(imu_data);
  }

  auto this_status =
      imu_processer_->pre_integrator_->Predict(start_status, Eigen::Vector3d(0.0, 0.0, -9.81));

  std::cout << "preinteg result: \n";
  std::cout << "end rotation: \n" << this_status.rot_.GetMatrix() << "\n";
  std::cout << "end trans: \n" << this_status.trans_.transpose() << "\n";
  std::cout << "end v: \n" << this_status.vel_.transpose() << "\n";
}

int main(int argc, char** argv) {
  // Initialize Google Test framework
  ::testing::InitGoogleTest(&argc, argv);
  // Run tests
  return RUN_ALL_TESTS();
}
