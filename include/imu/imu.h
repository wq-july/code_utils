#pragma once

#include <vector>

#include "Eigen/Dense"

#include "util/config.h"
#include "util/logger.h"

using namespace Utils;

namespace IMU {

struct IMUData {
  double timestamp_ = -1.0;
  Eigen::Vector3d acc_ = Eigen::Vector3d::Zero();
  Eigen::Vector3d gyr_ = Eigen::Vector3d::Zero();
  IMUData() {}
  IMUData(double ts, const Eigen::Vector3d& acc, const Eigen::Vector3d& gyr)
      : timestamp_(ts), acc_(acc), gyr_(gyr) {}

  IMUData& operator=(const IMUData& other) {
    if (this != &other) {
      timestamp_ = other.timestamp_;
      acc_ = other.acc_;
      gyr_ = other.gyr_;
    }
    return *this;
  }
};

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
  void Initialize();
  bool ReadData(const std::string& file_path,
                std::vector<IMUData>* const data_vec);

 private:
  Utils::Logger logger_;
};

}  // namespace Sensor
