#pragma once

#include <memory>
#include <string>

#include "gflags/gflags.h"
#include "glog/logging.h"
#include "opencv2/features2d.hpp"
#include "opencv2/opencv.hpp"
#include "opencv2/xfeatures2d/nonfree.hpp"

#include "../protos/pb/camera.pb.h"
#include "camera/features.h"
#include "camera/frame.h"
#include "camera/super_glue.h"
#include "camera/super_point.h"
#include "util/time.h"
#include "util/utils.h"

namespace Camera {

class FeatureManager {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  FeatureManager(const CameraConfig::FeatureConfig& config);
  void ExtractFeatures(const cv::Mat& image,
                       std::vector<cv::KeyPoint>* const keypoints,
                       cv::Mat* const descriptors);

  void Match(const cv::Mat& descriptors0,
             const cv::Mat& descriptors1,
             std::vector<cv::DMatch>& matches);

  // 可以用做双目匹配，也可以跟踪
  bool KLOpticalFlowTrack(const cv::Mat& left_img,
                          const cv::Mat& right_img,
                          const std::vector<cv::Point2f>& left_img_keypoints,
                          std::vector<cv::Point2f>* const right_img_keypoints,
                          std::vector<uchar>* const status) const;

  // TODO, 基于深度学习的双目立体匹配，
  // 基于TensorRT，https://github.com/XiandaGuo/OpenStereo/blob/v2/deploy/cpp/main.cpp
  bool OpenStereoMatch();

  bool TriangulatePoint(const Eigen::Matrix<double, 3, 4>& Pose0,
                        const Eigen::Matrix<double, 3, 4>& Pose1,
                        const Eigen::Vector2d& point0,
                        const Eigen::Vector2d& point1,
                        Eigen::Vector3d* const point_3d) const;

 private:
  void CreateFeatureDetector();
  void CreateDescriptorExtractor();
  void CreateMatcher();

 private:
  CameraConfig::FeatureConfig config_;
  cv::Ptr<cv::Feature2D> detector_ = nullptr;
  cv::Ptr<cv::Feature2D> descriptor_ = nullptr;
  cv::Ptr<cv::DescriptorMatcher> matcher_ = nullptr;
  Utils::Timer timer_;
};
}  // namespace Camera
