#include "common/sensor_data.h"

namespace Common {

IMUData::IMUData()
    : timestamp_(-1.0), acc_(Eigen::Vector3d::Zero()), gyr_(Eigen::Vector3d::Zero()) {
}

IMUData::IMUData(double ts, const Eigen::Vector3d& acc, const Eigen::Vector3d& gyr)
    : timestamp_(ts), acc_(acc), gyr_(gyr) {
}

IMUData& IMUData::operator=(const IMUData& other) {
  if (this != &other) {
    timestamp_ = other.timestamp_;
    acc_ = other.acc_;
    gyr_ = other.gyr_;
  }
  return *this;
}

}  // namespace Common
