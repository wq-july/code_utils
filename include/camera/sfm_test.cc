
#include <opencv2/features2d.hpp>
#include <vector>

#include "gflags/gflags.h"
#include "glog/logging.h"
#include "gtest/gtest.h"
#include "opencv2/highgui.hpp"
#include "opencv2/xfeatures2d.hpp"
#include "pangolin/display/display.h"
#include "pangolin/display/view.h"
#include "pangolin/handler/handler.h"

#define private public
#include "camera/sfm.h"

DEFINE_string(img1, "../include/camera/config/easy1.png", "第一张测试图像");
DEFINE_string(img1_depth, "../include/camera/config/1_depth.png", "第一张测试图像的深度图");
DEFINE_string(img2, "../include/camera/config/easy2.png", "第二张测试图像");
DEFINE_string(img2_depth, "../include/camera/config/2_depth.png", "第二张测试图像的深度图");
DEFINE_string(config, "../include/camera/config/config.conf", "配置文件路径");

class SFMTest : public testing::Test {
 public:
  void SetUp() override {
    img1_ = cv::imread(FLAGS_img1, cv::IMREAD_COLOR);
    img2_ = cv::imread(FLAGS_img2, cv::IMREAD_COLOR);
    if (img1_.empty() || img2_.empty()) {
      LOG(ERROR) << "Could not open or find the images!";
      enable_test_ = false;
    }
    CameraConfig::SFMConfig config;
    Utils::LoadProtoConfig(FLAGS_config, &config);
    sfm_ = std::make_shared<Camera::SFM>(config);
  }

  cv::Mat DrawMatches(const cv::Mat& img1,
                      const std::vector<cv::KeyPoint>& keypoints1,
                      const cv::Mat& img2,
                      const std::vector<cv::KeyPoint>& keypoints2,
                      const std::vector<cv::DMatch>& matches) {
    // 将两张图像水平拼接
    cv::Mat img_matches;
    cv::hconcat(img1, img2, img_matches);

    // 绘制匹配点和连线
    for (const auto& match : matches) {
      const cv::KeyPoint& kp1 = keypoints1[match.queryIdx];
      const cv::KeyPoint& kp2 = keypoints2[match.trainIdx];
      cv::Point2f pt1 = kp1.pt;
      cv::Point2f pt2 = kp2.pt + cv::Point2f(static_cast<float>(img1.cols), 0);

      // 绘制关键点的方形框和中间的圆圈
      cv::rectangle(img_matches,
                    cv::Point(pt1.x - 3, pt1.y - 3),
                    cv::Point(pt1.x + 3, pt1.y + 3),
                    cv::Scalar(255, 0, 0),
                    1,
                    cv::LINE_AA);

      cv::circle(img_matches, pt1, 2, cv::Scalar(255, 0, 0), cv::FILLED, cv::LINE_AA);

      cv::rectangle(img_matches,
                    cv::Point(pt2.x - 3, pt2.y - 3),
                    cv::Point(pt2.x + 3, pt2.y + 3),
                    cv::Scalar(255, 0, 0),
                    1,
                    cv::LINE_AA);
      cv::circle(img_matches, pt2, 2, cv::Scalar(255, 0, 0), cv::FILLED, cv::LINE_AA);

      // 绘制匹配线
      cv::line(img_matches,
               pt1,
               pt2,
               cv::Scalar(0, 255, 0),
               2,
               cv::LINE_AA);  // 绿色线条，宽度为2，抗锯齿
    }

    return img_matches;
  }

  void PangolinShow(const cv::Mat& img_matches) {
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

  void RevertPoints(std::vector<cv::KeyPoint>* const keypoints) {
    CHECK_NOTNULL(keypoints);
    for (auto& keypoint : *keypoints) {
      keypoint.pt.x *= 2;
      keypoint.pt.y *= 2;
    }
  }

  void KeypointsToPoints2f(const std::vector<cv::KeyPoint>& keypoints,
                           std::vector<cv::Point2f>* const points) {
    points->clear();
    points->reserve(keypoints.size());
    for (const auto& kp : keypoints) {
      points->emplace_back(kp.pt);
    }
  }

  void Points2fToKeypoints(const std::vector<cv::Point2f>& points2f,
                           std::vector<cv::KeyPoint>* const keypoints) {
    keypoints->clear();
    keypoints->reserve(points2f.size());
    for (const auto& kp : points2f) {
      cv::KeyPoint point;
      point.pt = kp;
      keypoints->emplace_back(point);
    }
  }

  void StatusToDMatches(const std::vector<uchar>& status, std::vector<cv::DMatch>* const matches) {
    matches->clear();
    for (size_t i = 0; i < status.size(); ++i) {
      if (status[i] == 1) {
        matches->emplace_back(cv::DMatch(i, i, 0));  // 这里的距离设为0，因为我们只是生成匹配对
      }
    }
  }

  cv::Mat img1_;
  cv::Mat img2_;
  std::shared_ptr<Camera::SFM> sfm_ = nullptr;
  bool enable_test_ = true;
};

TEST_F(SFMTest, P2PTest) {
  if (!enable_test_) {
    return;
  }

  std::vector<cv::KeyPoint> keypoinyts1, keypoinyts2;
  cv::Mat descriptors1, descriptors2;
  std::vector<cv::DMatch> matches;

  sfm_->feature_manager_->ExtractFeatures(img1_, &keypoinyts1, &descriptors1);
  sfm_->feature_manager_->ExtractFeatures(img2_, &keypoinyts2, &descriptors2);

  cv::BFMatcher matcher(cv::NORM_HAMMING, true);
  matcher.match(descriptors1, descriptors2, matches);

  std::vector<cv::DMatch> good_matches;
  // Apply GMS algorithm
  cv::xfeatures2d::matchGMS(
      img1_.size(), img2_.size(), keypoinyts1, keypoinyts2, matches, good_matches, true, true);
  // Draw matches using OpenCV
  cv::Mat img_matches = DrawMatches(img1_, keypoinyts1, img2_, keypoinyts2, good_matches);

  // SuperPoint特征点提取和SuperGlue匹配
  // std::vector<cv::KeyPoint> keypoinyts1, keypoinyts2;
  // cv::Mat descriptors1, descriptors2;
  // std::vector<cv::DMatch> matches;
  // sfm_->feature_manager_->ExtractFeatures(img1_, &keypoinyts1, &descriptors1);
  // sfm_->feature_manager_->ExtractFeatures(img2_, &keypoinyts2, &descriptors2);

  // sfm_->feature_manager_->Match(descriptors1, descriptors2, matches);
  // // RevertPoints(&keypoinyts1);
  // // RevertPoints(&keypoinyts2);

  // std::vector<cv::DMatch> good_matches;
  // // Apply GMS algorithm
  // cv::xfeatures2d::matchGMS(
  //     img1_.size(), img2_.size(), keypoinyts1, keypoinyts2, matches, good_matches, true, true);
  // cv::Mat img_matches = DrawMatches(img1_, keypoinyts1, img2_, keypoinyts2, good_matches);

  // cv::imshow("match", img_matches);
  // cv::waitKey(0);

  // PangolinShow(img_matches);

  Eigen::Matrix3d fundamental_matrix = Eigen::Matrix3d::Identity();
  std::vector<bool> inliers;
  if (sfm_->FindFundamentalMatrix(
          keypoinyts1, keypoinyts2, matches, &fundamental_matrix, &inliers)) {
    std::cout << "基础矩阵是：\n" << fundamental_matrix << "\n";
  } else {
    std::cout << "基础矩阵计算失败！ \n";
  }
  std::vector<Eigen::Vector3d> p3d;
  Eigen::Matrix3d R = Eigen::Matrix3d::Identity();
  Eigen::Vector3d t = Eigen::Vector3d::Zero();
  sfm_->ReconstructFromFmat(
      fundamental_matrix, keypoinyts1, keypoinyts2, matches, 50, 1.0, &inliers, &p3d, &R, &t);
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
