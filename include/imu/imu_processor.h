#pragma once

#include <vector>

#include "Eigen/Dense"

#include "../protos/pb/imu.pb.h"
#include "common/imu.h"
#include "imu/preintegration.h"
#include "util/logger.h"
#include "util/utils.h"

using namespace Common;

namespace IMU {

class ImuProcessor {
 public:
  ImuProcessor(const IMUConfig::Config& config);
  ~ImuProcessor() = default;

  bool ProcessImu();

 private:
  bool ReadData(const std::string& file_path, std::vector<IMUData>* const data_vec);

 private:
  IMUConfig::Config config_;
  Utils::Logger logger_;
  std::shared_ptr<IMUPreIntegration> pre_integrator_ = nullptr;
};

}  // namespace IMU
