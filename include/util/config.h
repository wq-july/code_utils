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

struct ImuPreIntegrationConfig {
  // 初始零偏
  Eigen::Vector3d init_ba_ = Eigen::Vector3d::Zero();
  Eigen::Vector3d init_bg_ = Eigen::Vector3d::Zero();
  // 噪声
  double noise_gyr_ = 0.0;
  double noise_acc_ = 0.0;
};

struct ImuConfig {
  std::string imu_file_path_ = "";
  std::string imu_topic = "";
  int32_t frequency_ = 100;
  // Transformation camera to imu
  Eigen::Isometry3d Transformation_i_c_ = Eigen::Isometry3d::Identity();
  // Transformation lidar to imu
  Eigen::Isometry3d Transformation_i_l_ = Eigen::Isometry3d::Identity();
  // Transformation body to imu
  Eigen::Isometry3d Transformation_i_b_ = Eigen::Isometry3d::Identity();
  ImuPreIntegrationConfig pre_integration_config_;
  LoggerConfig logger_config_;
  TimerConfig timer_config_;
};

struct SuperPointConfig {
  std::string model_dir_;
  int32_t max_keypoints_;
  double keypoint_threshold_;
  int32_t remove_borders_;
  int32_t dla_core_;
  std::vector<std::string> input_tensor_names_;
  std::vector<std::string> output_tensor_names_;
  std::string onnx_file_;
  std::string engine_file_;
};

struct SuperGlueConfig {
  std::string model_dir_;
  int32_t image_width_;
  int32_t image_height_;
  int32_t dla_core_;
  std::vector<std::string> input_tensor_names_;
  std::vector<std::string> output_tensor_names_;
  std::string onnx_file_;
  std::string engine_file_;
};

class Config {
 public:
  explicit Config(const std::string& filename);

 public:
  LoggerConfig logger_config_;
  TimerConfig time_config_;
  ImuConfig imu_config_;
  SuperPointConfig super_point_config_;
  SuperGlueConfig super_glue_config_;

 private:
  void LoadConfigFile(const std::string& filename);
  void LoadTransformation(const YAML::Node& node, Eigen::Isometry3d* const transform);
  void LoadTransformation(const YAML::Node& node, Eigen::Vector3d* const vector3d);
};

}  // namespace Utils
