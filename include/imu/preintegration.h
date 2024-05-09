#pragma once

#include <vector>

#include "Eigen/Core"
#include "Eigen/Dense"

#include "util/config.h"

namespace IMU {

class IMUPreIntegration {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  IMUPreIntegration(const Utils::ImuPreIntegrationConfig& config);










};

}  // namespace IMU