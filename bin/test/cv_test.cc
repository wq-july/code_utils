#include <thread>
#include <vector>

#include "gflags/gflags.h"
#include "glog/logging.h"
#include "gtest/gtest.h"
#include "opencv2/xfeatures2d.hpp"
#include "opencv2/xfeatures2d/nonfree.hpp"
#include "pangolin/display/display.h"
#include "pangolin/display/view.h"
#include "pangolin/gl/gldraw.h"
#include "pangolin/handler/handler.h"

#define private public
#include "camera/feature_extractor.h"

DEFINE_string(img1, "../bin/data/images/left/1711440249880019912.jpg", "第一张测试图像");
DEFINE_string(img2, "../bin/data/images/left/1711440251880066976.jpg", "第二张测试图像");

class CVTest : public testing::Test {
 public:
  void SetUp() override {
    img1_ = cv::imread(FLAGS_img1, cv::IMREAD_COLOR);
    img2_ = cv::imread(FLAGS_img2, cv::IMREAD_COLOR);
    if (img1_.empty() || img2_.empty()) {
      LOG(ERROR) << "Could not open or find the images!";
      enable_test_ = false;
    }
    extractor_ = std::make_shared<Camera::FeatureExtractor>(Camera::FeatureType::BRISK,
                                                            Camera::DescriptorType::BRISK);
  }

  cv::Mat DrawMatches(const cv::Mat& img1, const std::vector<cv::KeyPoint>& keypoints1,
                      const cv::Mat& img2, const std::vector<cv::KeyPoint>& keypoints2,
                      const std::vector<cv::DMatch>& matches) {
    cv::Mat img_matches;
    cv::hconcat(img1, img2, img_matches);

    for (const auto& match : matches) {
      const cv::KeyPoint& kp1 = keypoints1[match.queryIdx];
      const cv::KeyPoint& kp2 = keypoints2[match.trainIdx];
      cv::Point2f pt1 = kp1.pt;
      cv::Point2f pt2 = kp2.pt + cv::Point2f(static_cast<float>(img1.cols), 0);
      cv::line(img_matches, pt1, pt2, cv::Scalar(0, 255, 0), 1.0);
    }
    return img_matches;
  }

  cv::Mat img1_;
  cv::Mat img2_;
  std::shared_ptr<Camera::FeatureExtractor> extractor_ = nullptr;
  bool enable_test_ = true;
  std::vector<cv::KeyPoint> keypoints1_;
  std::vector<cv::KeyPoint> keypoints2_;
  std::vector<cv::DMatch> matches_;
};

TEST_F(CVTest, MultiThreadFeatureExtractTest) {
  if (!enable_test_) {
    return;
  }

  cv::Mat descriptors1, descriptors2;
  extractor_->ExtractFeatures(img1_, &keypoints1_, &descriptors1);
  extractor_->ExtractFeatures(img2_, &keypoints2_, &descriptors2);

  // Matching descriptors using BFMatcher
  // cv::BFMatcher matcher(cv::NORM_HAMMING, true);
  // matcher.match(descriptors1, descriptors2, matches_);

  cv::BFMatcher matcher(cv::NORM_HAMMING, true);
  matcher.match(descriptors1, descriptors2, matches_);

  std::vector<cv::DMatch> good_matches;
  // Apply GMS algorithm
  std::vector<bool> inliers;
  cv::xfeatures2d::matchGMS(img1_.size(), img2_.size(), keypoints1_, keypoints2_, matches_,
                            good_matches, true, true);

  // Draw matches using OpenCV
  cv::Mat img_matches = DrawMatches(img1_, keypoints1_, img2_, keypoints2_, good_matches);

  // Create OpenGL window in single line
  pangolin::CreateWindowAndBind("Main", 1920, 540);

  // 3D Mouse handler requires depth testing to be enabled
  glEnable(GL_DEPTH_TEST);

  pangolin::View& d_image = pangolin::Display("image")
                                .SetBounds(0.0f, 1.0f, 0.0f, 1.0f, 1920.0 / 540.0)
                                .SetLock(pangolin::LockLeft, pangolin::LockTop);

  const int width = img_matches.cols;
  const int height = img_matches.rows;

  pangolin::GlTexture imageTexture(width, height, GL_RGB, false, 0, GL_RGB, GL_UNSIGNED_BYTE);

  cv::Mat flipped_image;
  cv::flip(img_matches, flipped_image, 0);

  // Default hooks for exiting (Esc) and fullscreen (tab).
  while (!pangolin::ShouldQuit()) {
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    // Set image data and upload to GPU
    imageTexture.Upload(flipped_image.data, GL_RGB, GL_UNSIGNED_BYTE);

    // Display the images
    d_image.Activate();
    glColor3f(1.0, 1.0, 1.0);
    imageTexture.RenderToViewport();

    pangolin::FinishFrame();
  }
}

int main(int argc, char** argv) {
  google::InitGoogleLogging(argv[0]);
  FLAGS_stderrthreshold = google::INFO;
  FLAGS_colorlogtostderr = true;

  // Initialize Google Test framework
  ::testing::InitGoogleTest(&argc, argv);
  google::ParseCommandLineFlags(&argc, &argv, true);
  // Run tests
  return RUN_ALL_TESTS();
}
