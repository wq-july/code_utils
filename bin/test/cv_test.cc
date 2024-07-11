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
#include "camera/feature_manager.h"

DEFINE_string(img1, "../bin/data/images/tum/image0.png", "第一张测试图像");
DEFINE_string(img2, "../bin/data/images/tum/image1.png", "第二张测试图像");
DEFINE_string(config, "../bin/test/conf/feature_config.conf", "配置文件路径");
DEFINE_string(features_0, "../bin/test/conf/features_0.txt", "图像0的提取点");
DEFINE_string(features_1, "../bin/test/conf/features_1.txt", "图像1的提取点");
DEFINE_string(log_path, "../log/cv_log.txt", "日志文件路径");
class CVTest : public testing::Test {
 public:
  void SetUp() override {
    img1_ = cv::imread(FLAGS_img1, cv::IMREAD_COLOR);
    img2_ = cv::imread(FLAGS_img2, cv::IMREAD_COLOR);
    if (img1_.empty() || img2_.empty()) {
      LOG(ERROR) << "Could not open or find the images!";
      enable_test_ = false;
    }
    CameraConfig::FeatureConfig config;
    Utils::LoadProtoConfig(FLAGS_config, &config);
    feature_manager_ = std::make_shared<Camera::FeatureManger>(config);
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

      // 绘制关键点
      cv::circle(img_matches, pt1, 4, cv::Scalar(255, 0, 0), 1, cv::LINE_AA);  // 红色圆圈，抗锯齿
      cv::circle(img_matches, pt2, 4, cv::Scalar(255, 0, 0), 1, cv::LINE_AA);  // 红色圆圈，抗锯齿

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

  void SaveMatrixToFile(const Eigen::Matrix<double, 259, Eigen::Dynamic>& matrix,
                        const std::string& filename) {
    std::ofstream file(filename);

    if (file.is_open()) {
      file << matrix.rows() << " " << matrix.cols() << "\n";
      file << matrix << "\n";
      file.close();
    } else {
      std::cerr << "Unable to open file for writing: " << filename << std::endl;
    }
  }

  Eigen::Matrix<double, 259, Eigen::Dynamic> LoadMatrixFromFile(const std::string& filename) {
    std::ifstream file(filename);
    int rows, cols;
    if (file.is_open()) {
      file >> rows >> cols;
      Eigen::Matrix<double, 259, Eigen::Dynamic> matrix(rows, cols);
      for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
          file >> matrix(i, j);
        }
      }
      file.close();
      return matrix;
    } else {
      std::cerr << "Unable to open file for reading: " << filename << std::endl;
      return Eigen::Matrix<double, 259, Eigen::Dynamic>();  // 返回一个空矩阵
    }
  }

  void RevertPoints(std::vector<cv::KeyPoint>* const keypoints) {
    CHECK_NOTNULL(keypoints);
    for (auto& keypoint : *keypoints) {
      keypoint.pt.x *= 2;
      keypoint.pt.y *= 2;
    }
  }

  cv::Mat img1_;
  cv::Mat img2_;
  std::shared_ptr<Camera::FeatureManger> feature_manager_ = nullptr;
  std::vector<cv::KeyPoint> keypoints1_;
  std::vector<cv::KeyPoint> keypoints2_;
  std::vector<cv::DMatch> matches_;
  bool enable_test_ = true;
};

TEST_F(CVTest, LegancyFeatureExtractAndMatchTest) {
  if (true) {
    return;
  }
  cv::Mat descriptors1, descriptors2;
  feature_manager_->ExtractFeatures(img1_, &keypoints1_, &descriptors1);
  feature_manager_->ExtractFeatures(img2_, &keypoints2_, &descriptors2);

  cv::BFMatcher matcher(cv::NORM_HAMMING, true);
  matcher.match(descriptors1, descriptors2, matches_);

  std::vector<cv::DMatch> good_matches;
  // Apply GMS algorithm
  std::vector<bool> inliers;
  cv::xfeatures2d::matchGMS(
      img1_.size(), img2_.size(), keypoints1_, keypoints2_, matches_, good_matches, true, true);

  // Draw matches using OpenCV
  cv::Mat img_matches = DrawMatches(img1_, keypoints1_, img2_, keypoints2_, good_matches);

  PangolinShow(img_matches);
}

TEST_F(CVTest, SuperPointAndGlueTest) {
  if (!enable_test_) {
    return;
  }

  /*
    SuperPoint特征点提取和SuperGlue匹配
  */
  cv::Mat descriptors1, descriptors2;
  feature_manager_->ExtractFeatures(img1_, &keypoints1_, &descriptors1);
  feature_manager_->ExtractFeatures(img2_, &keypoints2_, &descriptors2);

  feature_manager_->Match(descriptors1, descriptors2, matches_);

  RevertPoints(&keypoints1_);
  RevertPoints(&keypoints2_);

  LOG(INFO) << "matches size is " << matches_.size();

  // std::vector<cv::DMatch> good_matches;
  // // Apply GMS algorithm
  // std::vector<bool> inliers;
  // cv::xfeatures2d::matchGMS(
  //     img1_.size(), img2_.size(), keypoints1_, keypoints2_, matches_, good_matches, true, true);
  // // Draw matches using OpenCV
  // cv::Mat img_matches = DrawMatches(img1_, keypoints1_, img2_, keypoints2_, good_matches);
  // LOG(INFO) << "good matches size is " << good_matches.size();

  cv::Mat img_matches = DrawMatches(img1_, keypoints1_, img2_, keypoints2_, matches_);

  PangolinShow(img_matches);
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
