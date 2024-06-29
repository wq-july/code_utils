#include "camera/feature_extractor.h"

namespace Camera {

FeatureExtractor::FeatureExtractor(FeatureType featureType, DescriptorType descriptorType)
    : feature_type_(featureType), descriptor_type_(descriptorType) {
  if (feature_type_ == FeatureType::SUPERPOINT) {
    ProcessSuperPoint();
    return;
  }
  CreateFeatureDetector();
  CreateDescriptorExtractor();
}

void FeatureExtractor::ExtractFeatures(const cv::Mat& image,
                                       std::vector<cv::KeyPoint>* const keypoints,
                                       cv::Mat* const descriptors) {
  if (!detector_ || !descriptor_) {
    LOG(FATAL) << "Feature detector or descriptor extractor is not initialized." << std::endl;
    return;
  }
  detector_->detect(image, *keypoints);
  descriptor_->compute(image, *keypoints, *descriptors);
}

void FeatureExtractor::CreateFeatureDetector() {
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
    case FeatureType::SUPERPOINT:

      break;
    default:
      std::cerr << "Unknown feature type." << std::endl;
      break;
  }
}

void FeatureExtractor::CreateDescriptorExtractor() {
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

void FeatureExtractor::ProcessSuperPoint() {
}

}  // namespace Camera