#pragma once

#include <memory>
#include <string>

#include "glog/logging.h"
#include "opencv2/features2d.hpp"
#include "opencv2/opencv.hpp"

namespace Camera {

enum class FeatureType {
  SIFT,
  SURF,
  ORB,
  BRISK,
  AKAZE,
  // TEBLID,
  // FAST_SIFT,
};
enum class DescriptorType {
  SIFT,
  SURF,
  ORB,
  BRISK,
  AKAZE,
  // FREAK,
  // TEBLID,
};

class FeatureExtractor {
 public:
  FeatureExtractor(FeatureType featureType, DescriptorType descriptorType)
      : feature_type_(featureType), descriptor_type_(descriptorType) {
    CreateFeatureDetector();
    CreateDescriptorExtractor();
  }

  void ExtractFeatures(const cv::Mat& image,
                       std::vector<cv::KeyPoint>* const keypoints,
                       cv::Mat* const descriptors) {
    if (!detector_ || !descriptor_) {
      LOG(FATAL) << "Feature detector or descriptor extractor is not initialized." << std::endl;
      return;
    }
    detector_->detect(image, *keypoints);
    descriptor_->compute(image, *keypoints, *descriptors);
  }

 private:
  void CreateFeatureDetector() {
    switch (feature_type_) {
      case FeatureType::SIFT:
        detector_ = cv::SIFT::create();
        break;
      case FeatureType::SURF:
        detector_ = cv::xfeatures2d::SURF::create();
        break;
      case FeatureType::ORB:
        detector_ = cv::ORB::create();
        break;
      case FeatureType::BRISK:
        detector_ = cv::BRISK::create();
        break;
      case FeatureType::AKAZE:
        detector_ = cv::AKAZE::create();
        break;
      default:
        std::cerr << "Unknown feature type." << std::endl;
        break;
    }
  }

  void CreateDescriptorExtractor() {
    switch (descriptor_type_) {
      case DescriptorType::SIFT:
        descriptor_ = cv::SIFT::create();
        break;
      case DescriptorType::SURF:
        descriptor_ = cv::xfeatures2d::SURF::create();
        break;
      case DescriptorType::ORB:
        descriptor_ = cv::ORB::create();
        break;
      case DescriptorType::BRISK:
        descriptor_ = cv::BRISK::create();
        break;
      case DescriptorType::AKAZE:
        descriptor_ = cv::AKAZE::create();
        break;
      default:
        std::cerr << "Unknown descriptor type." << std::endl;
        break;
    }
  }

  FeatureType feature_type_;
  DescriptorType descriptor_type_;
  cv::Ptr<cv::Feature2D> detector_;
  cv::Ptr<cv::Feature2D> descriptor_;
};
}  // namespace Camera