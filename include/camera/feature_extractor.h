#pragma once

#include <memory>
#include <string>

#include "glog/logging.h"
#include "opencv2/features2d.hpp"
#include "opencv2/opencv.hpp"
#include "opencv2/xfeatures2d/nonfree.hpp"

#include "util/config.h"

namespace Camera {

enum class FeatureType {
  SIFT,
  SURF,
  ORB,
  BRISK,
  AKAZE,
  SUPERPOINT,
  // TEBLID,
  // FAST_SIFT,
};
enum class DescriptorType {
  SIFT,
  SURF,
  ORB,
  BRISK,
  AKAZE,
  SUPERPOINT,
  // FREAK,
  // TEBLID,
};

class FeatureExtractor {
 public:
  FeatureExtractor(FeatureType featureType, DescriptorType descriptorType);
  void ExtractFeatures(const cv::Mat& image,
                       std::vector<cv::KeyPoint>* const keypoints,
                       cv::Mat* const descriptors);

 private:
  void CreateFeatureDetector();
  void CreateDescriptorExtractor();
  void ProcessSuperPoint();

 private:
  FeatureType feature_type_;
  DescriptorType descriptor_type_;
  cv::Ptr<cv::Feature2D> detector_;
  cv::Ptr<cv::Feature2D> descriptor_;
};
}  // namespace Camera