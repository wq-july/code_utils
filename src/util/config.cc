#include "util/config.h"

namespace Utils {

Config::Config(const std::string& filename) { LoadConfigFile(filename); }

void Config::LoadConfigFile(const std::string& filename) {
  try {
    YAML::Node config = YAML::LoadFile(filename);
    // Load ImuConfig
    if (config["ImuConfig"]) {
      const YAML::Node& imu = config["ImuConfig"];
      imu_config_.imu_file_path_ = imu["imu_file_path"].as<std::string>();
      imu_config_.imu_topic = imu["imu_topic"].as<std::string>();
      imu_config_.frequency_ = imu["frequency"].as<int>();

      // Load transformations
      LoadTransformation(imu["Transformation_i_c"],
                         &imu_config_.Transformation_i_c);
      LoadTransformation(imu["Transformation_i_l"],
                         &imu_config_.Transformation_i_l);
      LoadTransformation(imu["Transformation_i_b"],
                         &imu_config_.Transformation_i_b);

      // Load IMU LoggerConfig
      if (imu["LoggerConfig"]) {
        const YAML::Node& logger = imu["LoggerConfig"];
        imu_config_.logger_config_.log_file_path_ =
            logger["log_file_path"].as<std::string>();
        imu_config_.logger_config_.enable_console_log_ =
            logger["enable_console_log"].as<bool>();
        for (const auto& level : logger["console_log_levels"]) {
          imu_config_.logger_config_.console_log_levels_.emplace_back(
              level.as<std::string>());
        }
      }

      // Load TimerConfig
      if (imu["TimerConfig"]) {
        const YAML::Node& timer = imu["TimerConfig"];
        imu_config_.timer_config_.time_unit =
            timer["time_unit"].as<std::string>();
      }
    }
  } catch (const std::exception& e) {
    std::cerr << "Failed to load config file: " << e.what() << std::endl;
    throw;
  }
}

void Config::LoadTransformation(const YAML::Node& node,
                                Eigen::Isometry3d* const transform) {
  if (!node) return;
  Eigen::Vector3d translation(node["translation"]["x"].as<double>(),
                              node["translation"]["y"].as<double>(),
                              node["translation"]["z"].as<double>());
  Eigen::Vector3d euler_angles(node["rotation_euler"]["roll"].as<double>(),
                               node["rotation_euler"]["pitch"].as<double>(),
                               node["rotation_euler"]["yaw"].as<double>());
  *transform = Eigen::Translation3d(translation) *
               Eigen::AngleAxisd(euler_angles.z(), Eigen::Vector3d::UnitZ()) *
               Eigen::AngleAxisd(euler_angles.y(), Eigen::Vector3d::UnitY()) *
               Eigen::AngleAxisd(euler_angles.x(), Eigen::Vector3d::UnitX());
}

}  // namespace Utils
