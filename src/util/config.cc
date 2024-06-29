#include "util/config.h"

#include "util/utils.h"

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
      imu_config_.frequency_ = imu["frequency"].as<int32_t>();

      // Load transformations
      LoadTransformation(imu["Transformation_i_c"], &imu_config_.Transformation_i_c_);
      LoadTransformation(imu["Transformation_i_l"], &imu_config_.Transformation_i_l_);
      LoadTransformation(imu["Transformation_i_b"], &imu_config_.Transformation_i_b_);

      // Load IMU PreIntegration conf
      if (imu["ImuPreIntegration"]) {
        const YAML::Node& preintegration = imu["ImuPreIntegration"];
        LoadTransformation(preintegration["init_ba"],
                           &imu_config_.pre_integration_config_.init_ba_);
        LoadTransformation(preintegration["init_bg"],
                           &imu_config_.pre_integration_config_.init_bg_);
        imu_config_.pre_integration_config_.noise_gyr_ = preintegration["noise_gyr"].as<double>();
        imu_config_.pre_integration_config_.noise_acc_ = preintegration["noise_acc"].as<double>();
      }

      // Load IMU LoggerConfig
      if (imu["LoggerConfig"]) {
        const YAML::Node& logger = imu["LoggerConfig"];
        imu_config_.logger_config_.log_file_path_ = logger["log_file_path"].as<std::string>();
        imu_config_.logger_config_.enable_console_log_ = logger["enable_console_log"].as<bool>();
        for (const auto& level : logger["console_log_levels"]) {
          imu_config_.logger_config_.console_log_levels_.emplace_back(level.as<std::string>());
        }
      }

      // Load TimerConfig
      if (imu["TimerConfig"]) {
        const YAML::Node& timer = imu["TimerConfig"];
        imu_config_.timer_config_.time_unit = timer["time_unit"].as<std::string>();
      }
    }

    // SuperPoint
    if (config["SuperPoint"]) {
      const YAML::Node& superpoint_node = config["SuperPoint"];
      super_point_config_.max_keypoints_ = superpoint_node["max_keypoints"].as<int>();
      super_point_config_.keypoint_threshold_ = superpoint_node["keypoint_threshold"].as<double>();
      super_point_config_.remove_borders_ = superpoint_node["remove_borders"].as<int>();
      super_point_config_.dla_core_ = superpoint_node["dla_core"].as<int>();
      super_point_config_.model_dir_ = superpoint_node["model_dir"].as<std::string>();
      YAML::Node superpoint_input_tensor_names_node = superpoint_node["input_tensor_names"];
      size_t superpoint_num_input_tensor_names = superpoint_input_tensor_names_node.size();
      for (size_t i = 0; i < superpoint_num_input_tensor_names; i++) {
        super_point_config_.input_tensor_names_.push_back(
            superpoint_input_tensor_names_node[i].as<std::string>());
      }
      YAML::Node superpoint_output_tensor_names_node = superpoint_node["output_tensor_names"];
      size_t superpoint_num_output_tensor_names = superpoint_output_tensor_names_node.size();
      for (size_t i = 0; i < superpoint_num_output_tensor_names; i++) {
        super_point_config_.output_tensor_names_.push_back(
            superpoint_output_tensor_names_node[i].as<std::string>());
      }
      std::string superpoint_onnx_file = superpoint_node["onnx_file"].as<std::string>();
      std::string superpoint_engine_file = superpoint_node["engine_file"].as<std::string>();
      super_point_config_.onnx_file_ =
          ConcatenateFolderAndFileName(super_point_config_.model_dir_, superpoint_onnx_file);
      super_point_config_.engine_file_ =
          ConcatenateFolderAndFileName(super_point_config_.model_dir_, superpoint_engine_file);
    }

    // SuperGlue
    if (config["SuperGlue"]) {
      const YAML::Node& superglue_node = config["SuperGlue"];
      super_glue_config_.image_width_ = superglue_node["image_width"].as<int>();
      super_glue_config_.image_height_ = superglue_node["image_height"].as<int>();
      super_glue_config_.dla_core_ = superglue_node["dla_core"].as<int>();
      YAML::Node superglue_input_tensor_names_node = superglue_node["input_tensor_names"];
      size_t superglue_num_input_tensor_names = superglue_input_tensor_names_node.size();
      for (size_t i = 0; i < superglue_num_input_tensor_names; i++) {
        super_glue_config_.input_tensor_names_.push_back(
            superglue_input_tensor_names_node[i].as<std::string>());
      }
      YAML::Node superglue_output_tensor_names_node = superglue_node["output_tensor_names"];
      size_t superglue_num_output_tensor_names = superglue_output_tensor_names_node.size();
      for (size_t i = 0; i < superglue_num_output_tensor_names; i++) {
        super_glue_config_.output_tensor_names_.push_back(
            superglue_output_tensor_names_node[i].as<std::string>());
      }
      std::string superglue_onnx_file = superglue_node["onnx_file"].as<std::string>();
      std::string superglue_engine_file = superglue_node["engine_file"].as<std::string>();
      super_glue_config_.onnx_file_ =
          ConcatenateFolderAndFileName(super_glue_config_.model_dir_, superglue_onnx_file);
      super_glue_config_.engine_file_ =
          ConcatenateFolderAndFileName(super_glue_config_.model_dir_, superglue_engine_file);
    }

    // other configs ...

  } catch (const std::exception& e) {
    std::cerr << "Failed to load config file: " << e.what() << std::endl;
    throw;
  }
}

void Config::LoadTransformation(const YAML::Node& node, Eigen::Vector3d* const vector3d) {
  if (!node) {
    std::cerr << "null node! \n";
    return;
  }
  *vector3d =
      Eigen::Vector3d(node["x"].as<double>(), node["y"].as<double>(), node["z"].as<double>());
}

void Config::LoadTransformation(const YAML::Node& node, Eigen::Isometry3d* const transform) {
  if (!node) {
    std::cerr << "null node! \n";
    return;
  }
  Eigen::Vector3d translation = Eigen::Vector3d::Zero();
  Eigen::Vector3d euler_angles = Eigen::Vector3d::Zero();
  LoadTransformation(node["translation"], &translation);
  LoadTransformation(node["rotation_euler"], &euler_angles);
  *transform = Eigen::Translation3d(translation) *
               Eigen::AngleAxisd(euler_angles.z(), Eigen::Vector3d::UnitZ()) *
               Eigen::AngleAxisd(euler_angles.y(), Eigen::Vector3d::UnitY()) *
               Eigen::AngleAxisd(euler_angles.x(), Eigen::Vector3d::UnitX());
}

}  // namespace Utils
