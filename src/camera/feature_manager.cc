#include "camera/feature_manager.h"

#include "opencv2/core/eigen.hpp"
namespace Camera {

FeatureManager::FeatureManager(const CameraConfig::FeatureConfig& config) : config_(config) {
  std::cout << "Initializing FeatureManager... \n";
  Initialize();
}

void FeatureManager::Initialize() {
  // 创建特征点提取器
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

  // 创建描述符提取器
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

  // 创建特征点匹配器
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

  // 初始化其他指针成员变量
  // .....
}

void FeatureManager::ExtractFeatures(const cv::Mat& image,
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

void FeatureManager::Match(const cv::Mat& descriptors0,
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

// TODO, 跟踪到的特征点应该使用自定义的feature类型，因为还需要进一步的确定拓扑关系
bool FeatureManager::KLOpticalFlowTrack(const cv::Mat& left_img,
                                        const cv::Mat& right_img,
                                        const std::vector<cv::Point2f>& left_img_keypoints,
                                        std::vector<cv::Point2f>* const right_img_keypoints,
                                        std::vector<uchar>* const status) const {
  CHECK(left_img.data) << "Left image is null !";
  CHECK(right_img.data) << "Right Image is null !";
  CHECK(!left_img_keypoints.empty()) << "Keypoints is empty !";

  const auto config = config_.tracker_config().kloptical_flow_config();

  // TODO，这里是调用的OpenCV的光流跟踪方法，以后可能会添加其他的方法：https://github.com/VladyslavUsenko/basalt-mirror/blob/master/src/optical_flow/optical_flow.cpp
  std::vector<float> err;
  // cur left ---- cur right
  cv::calcOpticalFlowPyrLK(
      left_img,
      right_img,
      left_img_keypoints,  // 注意这里只能用cv::Point2f格式！
      *right_img_keypoints,  // 注意这个得到的结果size和输入的点是一致的，status会标记是否成功跟踪
      *status,
      err,
      cv::Size(21, 21),
      3);

  // reverse check cur right ---- cur left
  std::vector<cv::Point2f> reverse_left_img_keypoints;
  std::vector<uchar> reverse_status;
  if (config.reverse_check()) {
    cv::calcOpticalFlowPyrLK(right_img,
                             left_img,
                             *right_img_keypoints,
                             reverse_left_img_keypoints,
                             reverse_status,
                             err,
                             cv::Size(21, 21),
                             3);
    for (uint32_t i = 0; i < status->size(); ++i) {
      if (status->at(i) && reverse_status.at(i) &&
          Utils::IsInBorder(right_img, right_img_keypoints->at(i)) &&
          Utils::ComputeDistance(left_img_keypoints.at(i), reverse_left_img_keypoints.at(i)) <=
              config.pt_err()) {
        status->at(i) = 1;
      } else {
        status->at(i) = 0;
      }
    }
  }
  int32_t count = std::accumulate(status->begin(), status->end(), 0);
  if (count < config.min_tracked_nums()) {
    return false;
  }
  return true;
}

}  // namespace Camera
