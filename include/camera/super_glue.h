#pragma once
#pragma once
#include <opencv2/features2d.hpp>
#include <opencv2/opencv.hpp>

#include "Eigen/Core"
#include "glog/logging.h"

#include "../protos/pb/camera.pb.h"
#include "tensorRT/generic.h"
#include "util/time.h"

namespace Camera {
class SuperGlue : public TensorRT::GenericInference, public cv::DescriptorMatcher {
 public:
  SuperGlue(const CameraConfig::SuperGlue& config);
  virtual ~SuperGlue() = default;

  CV_WRAP static cv::Ptr<cv::DescriptorMatcher> create(const CameraConfig::SuperGlue& config) {
    return cv::makePtr<SuperGlue>(config);
  }

  void SuperGlueMatch(Eigen::Matrix<double, 259, Eigen::Dynamic> queryDescriptors,
                      Eigen::Matrix<double, 259, Eigen::Dynamic> trainDescriptors,
                      CV_OUT std::vector<cv::DMatch>& matches,
                      cv::InputArray mask = cv::noArray());

  bool isMaskSupported() const override {
    return true;
  }

  cv::Ptr<cv::DescriptorMatcher> clone(bool emptyTrainData = false) const override {
    return cv::makePtr<SuperGlue>(*this);
  }

  void knnMatchImpl(cv::InputArray queryDescriptors,
                    std::vector<std::vector<cv::DMatch>>& matches,
                    int k,
                    cv::InputArrayOfArrays masks = cv::noArray(),
                    bool compactResult = false) override {}

  void radiusMatchImpl(cv::InputArray queryDescriptors,
                       std::vector<std::vector<cv::DMatch>>& matches,
                       float maxDistance,
                       cv::InputArrayOfArrays masks = cv::noArray(),
                       bool compactResult = false) override {}

 private:
  void SetIBuilderConfigProfile(nvinfer1::IBuilderConfig* const config,
                                nvinfer1::IOptimizationProfile* const profile) override;
  bool VerifyEngine(const nvinfer1::INetworkDefinition* network) override;

  bool ProcessInput(const TensorRT::BufferManager& buffers) override;
  bool ProcessOutput(const TensorRT::BufferManager& buffers) override;
  void SetContext(nvinfer1::IExecutionContext* const context) override;

 private:
  void Decode(float* scores,
              int h,
              int w,
              std::vector<int>& indices0,
              std::vector<int>& indices1,
              std::vector<double>& mscores0,
              std::vector<double>& mscores1);
  void MaxMatrix(const float* data, int* indices, float* values, int h, int w, int dim);
  void EqualGather(const int* indices0, const int* indices1, int* mutual, int size);
  void WhereExp(const int* flag_data, float* data, std::vector<double>& mscores0, int size);
  void WhereGather(const int* flag_data,
                   int* indices,
                   std::vector<double>& mscores0,
                   std::vector<double>& mscores1,
                   int size);
  void AndThreshold(const int* mutual0,
                    int* valid0,
                    const std::vector<double>& mscores0,
                    double threhold);
  void AndGather(const int* mutual1, const int* valid0, const int* indices1, int* valid1, int size);
  void WhereNegativeOne(const int* flag_data, const int* data, int size, std::vector<int>& indices);
  void LogOptimalTransport(
      float* scores, float* Z, int m, int n, float alpha = 2.3457, int iters = 100);
  void LogSinkhornIterations(
      float* couplings, float* Z, int m, int n, float* log_mu, float* log_nu, int iters);

  Eigen::Matrix<double, 259, Eigen::Dynamic> NormalizeKeypoints(
      const Eigen::Matrix<double, 259, Eigen::Dynamic>& features, int width, int height);

 private:
  CameraConfig::SuperGlue config_;

  // nvinfer1::Dims keypoints_0_dims_{};
  // nvinfer1::Dims scores_0_dims_{};
  // nvinfer1::Dims descriptors_0_dims_{};
  // nvinfer1::Dims keypoints_1_dims_{};
  // nvinfer1::Dims scores_1_dims_{};
  // nvinfer1::Dims descriptors_1_dims_{};
  nvinfer1::Dims output_scores_dims_{};

  std::vector<int> indices0_;
  std::vector<int> indices1_;
  std::vector<double> scores0_;
  std::vector<double> scores1_;

  Eigen::Matrix<double, 259, Eigen::Dynamic> query_des_;
  Eigen::Matrix<double, 259, Eigen::Dynamic> train_des_;
};

}  // namespace Camera
