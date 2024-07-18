#pragma once

#include <stdint.h>

#include <unordered_map>
#include <vector>

#include "Eigen/Core"
#include "opencv2/core.hpp"
#include "sophus/se3.hpp"

#include "camera/common/features.h"

namespace Camera {

static uint64_t global_frame_id = 0u;

struct Frame {
  // 利用图像来直接进行frame构建，仅作为存储数据用，不应该提供多余的函数
  Frame(const cv::Mat img) : img_(img) {
    id_ = global_frame_id;
    ++global_frame_id;
  }

  double time_stamp_ = 0.0;
  bool is_keyframe_ = false;

  uint64_t id_ = 0u;

  // 地图坐标系下的位姿，也就是最终想要的东西
  Sophus::SE3d pose_ = Sophus::SE3d();

  // 当前位姿下，对应好拓扑关系的特征点，如果是双目，那么需要立体匹配之后的，如果是单目的，则为提取的特征点
  std::vector<Camera::Features> features_;

  std::vector<Frame*> connected_keyframes_;

  // 绑定的原始图像帧，仅作为临时存储用，当被tracker处理之后，需要清空掉！
  cv::Mat img_;

  // TODO 回环检测词袋相关, 或者基于深度学习的回环检测模块
  // DBoW3::BowVector mBowVec;
  // DBoW3::FeatureVector mFeatVec;

  // TODO, 对齐的其他测量值，比如IMU等，用作多传感器融合部分
  // IMUPreIntegration
};

}  // namespace Camera
