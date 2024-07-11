#pragma once

#include <memory>
#include <string>

#include "gflags/gflags.h"
#include "glog/logging.h"
#include "opencv2/features2d.hpp"
#include "opencv2/opencv.hpp"
#include "opencv2/xfeatures2d/nonfree.hpp"

#include "../protos/pb/camera.pb.h"
#include "camera/super_glue.h"
#include "camera/super_point.h"
#include "util/time.h"
#include "util/utils.h"

namespace Camera {

class FeatureManger {
 public:
  FeatureManger(const CameraConfig::FeatureConfig& config);
  void ExtractFeatures(const cv::Mat& image,
                       std::vector<cv::KeyPoint>* const keypoints,
                       cv::Mat* const descriptors);

  void Match(const cv::Mat& descriptors0,
             const cv::Mat& descriptors1,
             std::vector<cv::DMatch>& matches);

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
