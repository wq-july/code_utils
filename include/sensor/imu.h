#pragma once

#include <vector>

#include "Eigen/Dense"
#include "sensor_base.h"

using namespace Utils;

namespace Sensor {

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

class IMU : public SensorBase {
  // override
 public:
  IMU();
  ~IMU() = default;
  void Initialize() override;
  bool ReadData(const std::string& file_path,
                std::vector<IMUData>* const data_vec);
};

}  // namespace Sensor
