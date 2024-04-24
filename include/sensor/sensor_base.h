/**
 * 1531729238@qq.com, qiang.wang
*/
#pragma once

#include <iostream>
#include <string>
#include <fstream>
#include <iostream>

#include "util/logger.h"

namespace Sensor{

class SensorBase {
 public:
  virtual ~SensorBase() {}
  virtual void Initialize() = 0;

protected:
  Utils::Logger logger_;
};
}



