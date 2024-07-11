#include "camera/feature_manager.h"

#include "opencv2/core/eigen.hpp"
namespace Camera {

FeatureManger::FeatureManger(const CameraConfig::FeatureConfig& config) : config_(config) {
  std::cout << "Initializing FeatureManager... \n";
  CreateFeatureDetector();
  CreateDescriptorExtractor();
  CreateMatcher();
}

void FeatureManger::CreateFeatureDetector() {
  switch (config_.feature_type()) {
    case CameraConfig::FeatureType::F_SIFT:
      detector_ = cv::SIFT::create();
      break;
    case CameraConfig::FeatureType::F_SURF:
      detector_ = cv::xfeatures2d::SURF::create();
      break;
    case CameraConfig::FeatureType::F_ORB:
      detector_ = cv::ORB::create();
      break;
    case CameraConfig::FeatureType::F_BRISK:
      detector_ = cv::BRISK::create();
      break;
    case CameraConfig::FeatureType::F_AKAZE:
      detector_ = cv::AKAZE::create();
      break;
    case CameraConfig::FeatureType::F_SUPERPOINT:
      detector_ = Camera::SuperPoint::create(config_.super_point());
      break;
    default:
      std::cerr << "Unknown feature type." << std::endl;
      break;
  }
}

void FeatureManger::CreateDescriptorExtractor() {
  switch (config_.descriptor_type()) {
    case CameraConfig::DescriptorType::D_SIFT:
      descriptor_ = cv::SIFT::create();
      break;
    case CameraConfig::DescriptorType::D_SURF:
      descriptor_ = cv::xfeatures2d::SURF::create();
      break;
    case CameraConfig::DescriptorType::D_ORB:
      descriptor_ = cv::ORB::create();
      break;
    case CameraConfig::DescriptorType::D_BRISK:
      descriptor_ = cv::BRISK::create();
      break;
    case CameraConfig::DescriptorType::D_AKAZE:
      descriptor_ = cv::AKAZE::create();
      break;
    case CameraConfig::DescriptorType::D_SUPERPOINT:
      break;
    default:
      std::cerr << "Unknown descriptor type." << std::endl;
      break;
  }
}

void FeatureManger::CreateMatcher() {
  switch (config_.matcher_type()) {
    case CameraConfig::MatcherType::SUPERGLUE:
      matcher_ = Camera::SuperGlue::create(config_.super_glue());
      break;
    case CameraConfig::MatcherType::HANMING:
      matcher_ = cv::BFMatcher::create();
      break;
    case CameraConfig::MatcherType::FLANN:
      matcher_ = cv::FlannBasedMatcher::create();
      break;
    default:
      std::cerr << "Unknown matcher type." << std::endl;
      break;
  }
}
void FeatureManger::ExtractFeatures(const cv::Mat& image,
                                    std::vector<cv::KeyPoint>* const keypoints,
                                    cv::Mat* const descriptors) {
  // if (!detector_ || !descriptor_) {
  //   LOG(FATAL) << "Feature detector or descriptor extractor is not initialized.";
  //   return;
  // }
  // detector_->detect(image, *keypoints);
  // descriptor_->compute(image, *keypoints, *descriptors);
  timer_.StartTimer("Extract Features");
  detector_->detectAndCompute(image, cv::noArray(), *keypoints, *descriptors, false);
  timer_.StopTimer();
  timer_.PrintElapsedTime();
}

void FeatureManger::Match(const cv::Mat& descriptors0,
                          const cv::Mat& descriptors1,
                          std::vector<cv::DMatch>& matches) {
  switch (config_.matcher_type()) {
    case CameraConfig::MatcherType::SUPERGLUE: {
      Eigen::MatrixXd des0;
      Eigen::MatrixXd des1;
      cv::cv2eigen(descriptors0, des0);
      cv::cv2eigen(descriptors1, des1);
      matcher_.dynamicCast<SuperGlue>()->SuperGlueMatch(des0, des1, matches);
      break;
    }
    case CameraConfig::MatcherType::HANMING:
      matcher_->match(descriptors0, descriptors1, matches);
      break;
    case CameraConfig::MatcherType::FLANN:
      matcher_->match(descriptors0, descriptors1, matches);
      break;
    default:
      std::cerr << "Unknown matcher type." << std::endl;
      break;
  }
}

}  // namespace Camera
