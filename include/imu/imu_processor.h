#pragma once

#include <vector>

#include "Eigen/Dense"

#include "common/sensor_data.h"
#include "util/config.h"
#include "util/logger.h"

#include "imu/preintegration.h"

using namespace Utils;
using namespace Common;

namespace IMU {

class ImuProcessor {
 public:
  ImuProcessor() = default;
  ImuProcessor(const Utils::ImuConfig& config);
  ~ImuProcessor() = default;

  bool ProcessImu();

 private:
  // 用于读取配置文件参数
  void SetConfig(const Utils::ImuConfig& config);
  // 算法相关参数初始化
  void Initialize(const Utils::ImuConfig& config);
  bool ReadData(const std::string& file_path, std::vector<IMUData>* const data_vec);

 private:
  Utils::Logger logger_;
  std::shared_ptr<IMUPreIntegration> pre_integrator_ = nullptr;
};

}  // namespace IMU
