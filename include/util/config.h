#pragma once

#include <yaml-cpp/yaml.h>

#include <exception>
#include <iostream>
#include <string>
#include <vector>

#include "Eigen/Dense"
#include "Eigen/Geometry"

namespace Utils {

struct LoggerConfig {
  std::string log_file_path_ = "";
  bool enable_console_log_ = "";
  std::vector<std::string> console_log_levels_;
};

struct TimerConfig {
  std::string time_unit = "ms";
};

struct ImuConfig {
  std::string imu_file_path_ = "";
  std::string imu_topic = "";
  int32_t frequency_ = 100;
  // Transformation camera to imu
  Eigen::Isometry3d Transformation_i_c = Eigen::Isometry3d::Identity();
  // Transformation lidar to imu
  Eigen::Isometry3d Transformation_i_l = Eigen::Isometry3d::Identity();
  // Transformation body to imu
  Eigen::Isometry3d Transformation_i_b = Eigen::Isometry3d::Identity();
  LoggerConfig logger_config_;
  TimerConfig timer_config_;
};

class Config {
 public:
  explicit Config(const std::string& filename);

 public:
  LoggerConfig logger_config_;
  TimerConfig time_config_;
  ImuConfig imu_config_;

 private:
  void LoadConfigFile(const std::string& filename);
  void LoadTransformation(const YAML::Node& node,
                          Eigen::Isometry3d* const transform);
};

}  // namespace Utils
