#pragma once
#include <unordered_map>

#include "Eigen/Core"
namespace Camera {

static uint64_t global_features_id = 0;

struct Features {
  Features() {
    id_ = global_features_id;
    ++global_features_id;
  }

  uint64_t id_ = 0;
  // 和图像帧绑定，键为图像帧的ID，值为对应图像帧下像素坐标和3D坐标点；
  std::unordered_map<uint64_t, std::pair<Eigen::Vector2d, Eigen::Vector3d>> pt_map_;
  // 表示全局坐标系下的坐标点
  Eigen::Vector3d global_pt_ = Eigen::Vector3d::Zero();

  // ? 我觉得这里可以再放一个描述符，用来做回环检测
  // float* des_;
};

}  // namespace Camera
