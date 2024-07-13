#pragma once
#include <opencv2/features2d.hpp>
#include <opencv2/opencv.hpp>

#include "Eigen/Core"
#include "glog/logging.h"

#include "../protos/pb/camera.pb.h"
#include "tensorRT/generic.h"
#include "tensorRT/logger.h"
#include "tensorRT/logging.h"
#include "util/time.h"

namespace Camera {
class SuperPoint : public TensorRT::GenericInference, public cv::Feature2D {
 public:
  SuperPoint(const CameraConfig::SuperPoint& config);

  CV_WRAP static cv::Ptr<cv::Feature2D> create(const CameraConfig::SuperPoint& config);

  // TODO 进一步适配
  // void detect(cv::InputArray image,
  //             CV_OUT std::vector<cv::KeyPoint>& keypoints,
  //             cv::InputArray mask = cv::noArray()) override;

  void detectAndCompute(cv::InputArray image,
                        cv::InputArray mask,
                        CV_OUT std::vector<cv::KeyPoint>& keypoints,
                        cv::OutputArray descriptors,
                        bool useProvidedKeypoints = false) override;
  // TODO 进一步适配
  // void compute(cv::InputArray image,
  //              CV_OUT CV_IN_OUT std::vector<cv::KeyPoint>& keypoints,
  //              cv::OutputArray descriptors) override;

 private:
  void SetIBuilderConfigProfile(nvinfer1::IBuilderConfig* const config,
                                nvinfer1::IOptimizationProfile* const profile) override;
  bool VerifyEngine(const nvinfer1::INetworkDefinition* network) override;
  void SetContext(nvinfer1::IExecutionContext* const context) override;
  bool ProcessInput(const TensorRT::BufferManager& buffers) override;
  bool ProcessOutput(const TensorRT::BufferManager& buffers) override;

 private:
  void FindHighScoreIndex(std::vector<float>& scores,
                          std::vector<std::vector<int>>& keypoints,
                          int h,
                          int w,
                          double threshold);

  void RemoveBorders(std::vector<std::vector<int>>& keypoints,
                     std::vector<float>& scores,
                     int border,
                     int height,
                     int width);

  std::vector<int> SortIndexes(std::vector<float>& data);

  void TopKKeypoints(std::vector<std::vector<int>>& keypoints, std::vector<float>& scores, int k);

  void NormalizeKeypoints(const std::vector<std::vector<int>>& keypoints,
                          std::vector<std::vector<double>>& keypoints_norm,
                          int h,
                          int w,
                          int s);

  int Clip(int val, int max);

  void GridSample(const float* input,
                  std::vector<std::vector<double>>& grid,
                  std::vector<std::vector<double>>& output,
                  int dim,
                  int h,
                  int w);

  template <typename Iter_T>
  double VectorNormalize(Iter_T first, Iter_T last) {
    return sqrt(inner_product(first, last, first, 0.0));
  }

  void NormalizeDescriptors(std::vector<std::vector<double>>& dest_descriptors);

  void SampleDescriptors(std::vector<std::vector<int>>& keypoints,
                         float* descriptors,
                         std::vector<std::vector<double>>& dest_descriptors,
                         int dim,
                         int h,
                         int w,
                         int s = 8);

 private:
  nvinfer1::Dims input_dims_{};
  nvinfer1::Dims semi_dims_{};
  nvinfer1::Dims desc_dims_{};
  Utils::Timer timer_;
  CameraConfig::SuperPoint config_;
  float* output_score_ = nullptr;
  float* output_desc_ = nullptr;
  cv::Mat input_img_;
};
}  // namespace Camera
