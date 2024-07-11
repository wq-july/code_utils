#include "camera/super_point.h"

#include "opencv2/core/eigen.hpp"
namespace Camera {

cv::Ptr<cv::Feature2D> SuperPoint::create(const CameraConfig::SuperPoint& config) {
  return cv::makePtr<SuperPoint>(config);
}

SuperPoint::SuperPoint(const CameraConfig::SuperPoint& config)
    : GenericInference(config.tensor_config()), config_(config) {
  TensorRT::SetReportableSeverity(TensorRT::Logger::Severity::kINTERNAL_ERROR);
  // 创建推理模型
  Build();
}

void SuperPoint::SetIBuilderConfigProfile(nvinfer1::IBuilderConfig* const config,
                                          nvinfer1::IOptimizationProfile* const profile) {
  CHECK(config != nullptr) << "config is nullptr!";
  CHECK(profile != nullptr) << "profile is nullptr!";
  // SuperPoint输入只有一张图像，tensor数量只有一个
  // 设置最小尺寸
  profile->setDimensions(config_.tensor_config().input_tensor_names()[0].c_str(),
                         nvinfer1::OptProfileSelector::kMIN,
                         nvinfer1::Dims4(1, 1, 100, 100));
  // 设置最优尺寸
  profile->setDimensions(config_.tensor_config().input_tensor_names()[0].c_str(),
                         nvinfer1::OptProfileSelector::kOPT,
                         nvinfer1::Dims4(1, 1, 500, 500));
  // 设置最大尺寸
  profile->setDimensions(config_.tensor_config().input_tensor_names()[0].c_str(),
                         nvinfer1::OptProfileSelector::kMAX,
                         nvinfer1::Dims4(1, 1, 1500, 1500));
  config->setMemoryPoolLimit(nvinfer1::MemoryPoolType::kWORKSPACE, 1 << 30);
  // config->setAvgTimingIterations(1);
  config->addOptimizationProfile(profile);
}

bool SuperPoint::VerifyEngine(const nvinfer1::INetworkDefinition* network) {
  CHECK(engine_->getNbIOTensors() == config_.tensor_config().input_tensor_names().size() +
                                         config_.tensor_config().output_tensor_names().size())
      << "Error tensors nums !";

  for (int i = 0; i < config_.tensor_config().input_tensor_names().size(); ++i) {
    CHECK(std::string(engine_->getIOTensorName(i)) ==
          config_.tensor_config().input_tensor_names()[i])
        << "Incorrect tensor name" << engine_->getIOTensorName(i) << " vs "
        << config_.tensor_config().input_tensor_names()[i];
  }

  CHECK(network->getNbInputs() == 1);
  input_dims_ = network->getInput(0)->getDimensions();
  CHECK(input_dims_.nbDims == 4);
  CHECK(network->getNbOutputs() == 2);
  semi_dims_ = network->getOutput(0)->getDimensions();
  CHECK(semi_dims_.nbDims == 3);
  desc_dims_ = network->getOutput(1)->getDimensions();
  CHECK(desc_dims_.nbDims == 4);

  return true;
}

void SuperPoint::SetContext(nvinfer1::IExecutionContext* const context) {
  // 一个输入，两个输出
  CHECK(engine_->getNbIOTensors() == 3);
  context->setInputShape(config_.tensor_config().input_tensor_names()[0].c_str(),
                         nvinfer1::Dims4(1, 1, config_.image_height(), config_.image_width()));
}
bool SuperPoint::ProcessInput(const TensorRT::BufferManager& buffers) {
  input_dims_.d[2] = input_img_.rows;
  input_dims_.d[3] = input_img_.cols;
  semi_dims_.d[1] = input_img_.rows;
  semi_dims_.d[2] = input_img_.cols;
  desc_dims_.d[1] = 256;
  desc_dims_.d[2] = input_img_.rows / 8;
  desc_dims_.d[3] = input_img_.cols / 8;
  auto* host_data_buffer =
      static_cast<float*>(buffers.GetHostBuffer(config_.tensor_config().input_tensor_names()[0]));
  for (int row = 0; row < input_img_.rows; ++row) {
    for (int col = 0; col < input_img_.cols; ++col) {
      host_data_buffer[row * input_img_.cols + col] =
          float(input_img_.at<unsigned char>(row, col)) / 255.0;
    }
  }
  return true;
}

bool SuperPoint::ProcessOutput(const TensorRT::BufferManager& buffers) {
  output_score_ =
      static_cast<float*>(buffers.GetHostBuffer(config_.tensor_config().output_tensor_names()[0]));
  output_desc_ =
      static_cast<float*>(buffers.GetHostBuffer(config_.tensor_config().output_tensor_names()[1]));
  if (!output_score_ || !output_desc_) {
    LOG(ERROR) << "Null output !";
    return false;
  }
  return true;
}

void SuperPoint::FindHighScoreIndex(std::vector<float>& scores,
                                    std::vector<std::vector<int>>& keypoints,
                                    int h,
                                    int w,
                                    double threshold) {
  std::vector<float> new_scores;
  for (int i = 0; i < static_cast<int>(scores.size()); ++i) {
    if (scores[i] > threshold) {
      std::vector<int> location = {int(i / w), i % w};
      keypoints.emplace_back(location);
      new_scores.emplace_back(scores[i]);
    }
  }
  scores.swap(new_scores);
}

void SuperPoint::RemoveBorders(std::vector<std::vector<int>>& keypoints,
                               std::vector<float>& scores,
                               int border,
                               int height,
                               int width) {
  std::vector<std::vector<int>> keypoints_selected;
  std::vector<float> scores_selected;
  for (int i = 0; i < static_cast<int>(keypoints.size()); ++i) {
    bool flag_h = (keypoints[i][0] >= border) && (keypoints[i][0] < (height - border));
    bool flag_w = (keypoints[i][1] >= border) && (keypoints[i][1] < (width - border));
    if (flag_h && flag_w) {
      keypoints_selected.emplace_back(std::vector<int>{keypoints[i][1], keypoints[i][0]});
      scores_selected.emplace_back(scores[i]);
    }
  }
  keypoints.swap(keypoints_selected);
  scores.swap(scores_selected);
}

std::vector<int> SuperPoint::SortIndexes(std::vector<float>& data) {
  std::vector<int> indexes(data.size());
  iota(indexes.begin(), indexes.end(), 0);
  sort(indexes.begin(), indexes.end(), [&data](int i1, int i2) {
    return data[i1] > data[i2];
  });
  return indexes;
}

void SuperPoint::TopKKeypoints(std::vector<std::vector<int>>& keypoints,
                               std::vector<float>& scores,
                               int k) {
  if (k < static_cast<int>(keypoints.size()) && k != -1) {
    std::vector<std::vector<int>> keypoints_top_k;
    std::vector<float> scores_top_k;
    std::vector<int> indexes = SortIndexes(scores);
    for (int i = 0; i < k; ++i) {
      keypoints_top_k.emplace_back(keypoints[indexes[i]]);
      scores_top_k.emplace_back(scores[indexes[i]]);
    }
    keypoints.swap(keypoints_top_k);
    scores.swap(scores_top_k);
  }
}

void SuperPoint::NormalizeKeypoints(const std::vector<std::vector<int>>& keypoints,
                                    std::vector<std::vector<double>>& keypoints_norm,
                                    int h,
                                    int w,
                                    int s) {
  for (auto& keypoint : keypoints) {
    std::vector<double> kp = {keypoint[0] - s / 2 + 0.5, keypoint[1] - s / 2 + 0.5};
    kp[0] = kp[0] / (w * s - s / 2 - 0.5);
    kp[1] = kp[1] / (h * s - s / 2 - 0.5);
    kp[0] = kp[0] * 2 - 1;
    kp[1] = kp[1] * 2 - 1;
    keypoints_norm.emplace_back(kp);
  }
}

int SuperPoint::Clip(int val, int max) {
  if (val < 0) return 0;
  return std::min(val, max - 1);
}

void SuperPoint::GridSample(const float* input,
                            std::vector<std::vector<double>>& grid,
                            std::vector<std::vector<double>>& output,
                            int dim,
                            int h,
                            int w) {
  // descriptors 1, 256, image_height/8, image_width/8
  // keypoints 1, 1, number, 2
  // out 1, 256, 1, number
  for (auto& g : grid) {
    double ix = ((g[0] + 1) / 2) * (w - 1);
    double iy = ((g[1] + 1) / 2) * (h - 1);

    int ix_nw = Clip(std::floor(ix), w);
    int iy_nw = Clip(std::floor(iy), h);

    int ix_ne = Clip(ix_nw + 1, w);
    int iy_ne = Clip(iy_nw, h);

    int ix_sw = Clip(ix_nw, w);
    int iy_sw = Clip(iy_nw + 1, h);

    int ix_se = Clip(ix_nw + 1, w);
    int iy_se = Clip(iy_nw + 1, h);

    double nw = (ix_se - ix) * (iy_se - iy);
    double ne = (ix - ix_sw) * (iy_sw - iy);
    double sw = (ix_ne - ix) * (iy - iy_ne);
    double se = (ix - ix_nw) * (iy - iy_nw);

    std::vector<double> descriptor;
    for (int i = 0; i < dim; ++i) {
      // 256x60x106 dhw
      // x * height * depth + y * depth + z
      double nw_val = input[i * h * w + iy_nw * w + ix_nw];
      double ne_val = input[i * h * w + iy_ne * w + ix_ne];
      double sw_val = input[i * h * w + iy_sw * w + ix_sw];
      double se_val = input[i * h * w + iy_se * w + ix_se];
      descriptor.emplace_back(nw_val * nw + ne_val * ne + sw_val * sw + se_val * se);
    }
    output.emplace_back(descriptor);
  }
}

void SuperPoint::NormalizeDescriptors(std::vector<std::vector<double>>& dest_descriptors) {
  for (auto& descriptor : dest_descriptors) {
    double norm_inv = 1.0 / VectorNormalize(descriptor.begin(), descriptor.end());
    std::transform(descriptor.begin(),
                   descriptor.end(),
                   descriptor.begin(),
                   std::bind1st(std::multiplies<double>(), norm_inv));
  }
}

void SuperPoint::SampleDescriptors(std::vector<std::vector<int>>& keypoints,
                                   float* descriptors,
                                   std::vector<std::vector<double>>& dest_descriptors,
                                   int dim,
                                   int h,
                                   int w,
                                   int s) {
  std::vector<std::vector<double>> keypoints_norm;
  NormalizeKeypoints(keypoints, keypoints_norm, h, w, s);
  GridSample(descriptors, keypoints_norm, dest_descriptors, dim, h, w);
  NormalizeDescriptors(dest_descriptors);
}

void SuperPoint::detectAndCompute(cv::InputArray img,
                                  cv::InputArray mask,
                                  CV_OUT std::vector<cv::KeyPoint>& keypoints,
                                  cv::OutputArray descriptors,
                                  bool useProvidedKeypoints) {
  input_img_ = img.getMat();
  if (input_img_.rows != config_.image_height() || input_img_.cols != config_.image_width()) {
    cv::resize(input_img_, input_img_, cv::Size(config_.image_width(), config_.image_height()));
    LOG(WARNING) << "Input image must have same size with network, resize img as ["
                 << config_.image_width() << " x " << config_.image_height() << " ]";
  }

  Infer();
  std::vector<std::vector<int>> keypoints_index;
  std::vector<std::vector<double>> features;
  int semi_feature_map_h = semi_dims_.d[1];
  int semi_feature_map_w = semi_dims_.d[2];
  std::vector<float> scores_vec(output_score_,
                                output_score_ + semi_feature_map_h * semi_feature_map_w);
  FindHighScoreIndex(scores_vec,
                     keypoints_index,
                     semi_feature_map_h,
                     semi_feature_map_w,
                     config_.keypoint_threshold());

  RemoveBorders(keypoints_index,
                scores_vec,
                config_.remove_borders(),
                semi_feature_map_h,
                semi_feature_map_w);
  TopKKeypoints(keypoints_index, scores_vec, config_.max_keypoints());

  Eigen::Matrix<double, 259, Eigen::Dynamic> descriptors_eigen;

  descriptors_eigen.resize(259, scores_vec.size());

  int desc_feature_dim = desc_dims_.d[1];
  int desc_feature_map_h = desc_dims_.d[2];
  int desc_feature_map_w = desc_dims_.d[3];

  SampleDescriptors(keypoints_index,
                    output_desc_,
                    features,
                    desc_feature_dim,
                    desc_feature_map_h,
                    desc_feature_map_w);

  for (int i = 0; i < static_cast<int>(scores_vec.size()); i++) {
    descriptors_eigen(0, i) = scores_vec[i];
  }

  for (int i = 1; i < 3; ++i) {
    for (int j = 0; j < static_cast<int>(keypoints_index.size()); ++j) {
      descriptors_eigen(i, j) = keypoints_index[j][i - 1];
    }
  }
  for (int m = 3; m < 259; ++m) {
    for (int n = 0; n < static_cast<int>(features.size()); ++n) {
      descriptors_eigen(m, n) = features[n][m - 3];
    }
  }
  keypoints.clear();
  for (int i = 0; i < descriptors_eigen.cols(); ++i) {
    double score = descriptors_eigen(0, i);
    double x = descriptors_eigen(1, i);
    double y = descriptors_eigen(2, i);
    keypoints.emplace_back(x, y, 8, -1, score);
  }

  // 创建一个 cv::Mat 对象，将 Eigen 矩阵的数据指针传递给它
  // cv::Mat mat(259, descriptors_eigen.cols(), CV_64F,
  // const_cast<double*>(descriptors_eigen.data())); mat.copyTo(descriptors);

  cv::eigen2cv(descriptors_eigen, descriptors);

  output_desc_ = nullptr;
  output_score_ = nullptr;
}

}  // namespace Camera