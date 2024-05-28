#pragma once

#include "Eigen/Dense"

namespace Common {

struct IMUData {
  double timestamp_;
  Eigen::Vector3d acc_;
  Eigen::Vector3d gyr_;

  IMUData();
  IMUData(double ts, const Eigen::Vector3d& acc, const Eigen::Vector3d& gyr);

  IMUData& operator=(const IMUData& other);
};

}  // namespace Common
