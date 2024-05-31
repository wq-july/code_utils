#include "common/data/imu.h"

#include <vector>

#include "gtest/gtest.h"

#include "glog/logging.h"
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
  bool enable_test_ = true;
};

// Test case for IMUData reading
TEST_F(IMUTest, ReadDataTest) {
  if (!enable_test_) {
    return;
  }
  std::vector<Common::Data::IMUData> data_vec;
  imu_processer_->ReadData("../bin/data/imu/MS1.txt", &data_vec);
  EXPECT_GT(data_vec.size(), 0);  // Expecting at least one data point
}

// Test case for IMU Preintegration
TEST_F(IMUTest, PreIntegrationTest) {
  if (!enable_test_) {
    return;
  }
  std::vector<Common::Data::IMUData> data_vec;
  imu_processer_->ReadData("../bin/data/imu/MG35.txt", &data_vec);

  Common::State start_status(data_vec.front().timestamp_);

  for (int i = 0; i < 100; ++i) {
    Common::Data::IMUData imu_data = data_vec[i];
    imu_processer_->pre_integrator_->Update(imu_data);
  }

  auto this_status =
      imu_processer_->pre_integrator_->Predict(start_status, Eigen::Vector3d(0.0, 0.0, -9.81));

  LOG(INFO) << "preinteg result: \n";
  LOG(INFO) << "end rotation: \n" << this_status.rot_.matrix();
  LOG(INFO) << "end trans: \n" << this_status.trans_.transpose();
  LOG(INFO) << "end v: \n" << this_status.vel_.transpose();
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
