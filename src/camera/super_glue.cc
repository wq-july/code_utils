#include "camera/super_glue.h"

#include <fstream>
namespace Camera {

SuperGlue::SuperGlue(const CameraConfig::SuperGlue& config)
    : GenericInference(config.tensor_config()), config_(config) {
  TensorRT::SetReportableSeverity(TensorRT::Logger::Severity::kINTERNAL_ERROR);
  // 创建推理模型
  Build();
}

void SuperGlue::SetIBuilderConfigProfile(nvinfer1::IBuilderConfig* const config,
                                         nvinfer1::IOptimizationProfile* const profile) {
  CHECK(config != nullptr) << "config is nullptr!";
  CHECK(profile != nullptr) << "profile is nullptr!";

  profile->setDimensions(config_.tensor_config().input_tensor_names()[0].c_str(),
                         nvinfer1::OptProfileSelector::kMIN,
                         nvinfer1::Dims3(1, 256, 1));
  profile->setDimensions(config_.tensor_config().input_tensor_names()[0].c_str(),
                         nvinfer1::OptProfileSelector::kOPT,
                         nvinfer1::Dims3(1, 256, 512));
  profile->setDimensions(config_.tensor_config().input_tensor_names()[0].c_str(),
                         nvinfer1::OptProfileSelector::kMAX,
                         nvinfer1::Dims3(1, 256, 1024));

  profile->setDimensions(config_.tensor_config().input_tensor_names()[1].c_str(),
                         nvinfer1::OptProfileSelector::kMIN,
                         nvinfer1::Dims3(1, 256, 1));
  profile->setDimensions(config_.tensor_config().input_tensor_names()[1].c_str(),
                         nvinfer1::OptProfileSelector::kOPT,
                         nvinfer1::Dims3(1, 256, 512));
  profile->setDimensions(config_.tensor_config().input_tensor_names()[1].c_str(),
                         nvinfer1::OptProfileSelector::kMAX,
                         nvinfer1::Dims3(1, 256, 1024));

  profile->setDimensions(config_.tensor_config().input_tensor_names()[2].c_str(),
                         nvinfer1::OptProfileSelector::kMIN,
                         nvinfer1::Dims3(1, 1, 2));
  profile->setDimensions(config_.tensor_config().input_tensor_names()[2].c_str(),
                         nvinfer1::OptProfileSelector::kOPT,
                         nvinfer1::Dims3(1, 512, 2));
  profile->setDimensions(config_.tensor_config().input_tensor_names()[2].c_str(),
                         nvinfer1::OptProfileSelector::kMAX,
                         nvinfer1::Dims3(1, 1024, 2));

  profile->setDimensions(config_.tensor_config().input_tensor_names()[3].c_str(),
                         nvinfer1::OptProfileSelector::kMIN,
                         nvinfer1::Dims3(1, 1, 2));
  profile->setDimensions(config_.tensor_config().input_tensor_names()[3].c_str(),
                         nvinfer1::OptProfileSelector::kOPT,
                         nvinfer1::Dims3(1, 512, 2));
  profile->setDimensions(config_.tensor_config().input_tensor_names()[3].c_str(),
                         nvinfer1::OptProfileSelector::kMAX,
                         nvinfer1::Dims3(1, 1024, 2));

  profile->setDimensions(config_.tensor_config().input_tensor_names()[4].c_str(),
                         nvinfer1::OptProfileSelector::kMIN,
                         nvinfer1::Dims2(1, 1));
  profile->setDimensions(config_.tensor_config().input_tensor_names()[4].c_str(),
                         nvinfer1::OptProfileSelector::kOPT,
                         nvinfer1::Dims2(1, 512));
  profile->setDimensions(config_.tensor_config().input_tensor_names()[4].c_str(),
                         nvinfer1::OptProfileSelector::kMAX,
                         nvinfer1::Dims2(1, 1024));

  profile->setDimensions(config_.tensor_config().input_tensor_names()[5].c_str(),
                         nvinfer1::OptProfileSelector::kMIN,
                         nvinfer1::Dims2(1, 1));
  profile->setDimensions(config_.tensor_config().input_tensor_names()[5].c_str(),
                         nvinfer1::OptProfileSelector::kOPT,
                         nvinfer1::Dims2(1, 512));
  profile->setDimensions(config_.tensor_config().input_tensor_names()[5].c_str(),
                         nvinfer1::OptProfileSelector::kMAX,
                         nvinfer1::Dims2(1, 1024));

  // config->setMemoryPoolLimit(nvinfer1::MemoryPoolType::kWORKSPACE, 1 << 30);
  // config->setAvgTimingIterations(1);
  config->addOptimizationProfile(profile);
}

bool SuperGlue::VerifyEngine(const nvinfer1::INetworkDefinition* network) {
  CHECK(engine_->getNbIOTensors() == config_.tensor_config().input_tensor_names().size() +
                                         config_.tensor_config().output_tensor_names().size())
      << "Error tensors nums !";

  for (int i = 0; i < config_.tensor_config().input_tensor_names().size(); ++i) {
    CHECK(std::string(engine_->getIOTensorName(i)) ==
          config_.tensor_config().input_tensor_names()[i])
        << "Incorrect tensor name: " << engine_->getIOTensorName(i) << " vs "
        << config_.tensor_config().input_tensor_names()[i];
  }

  CHECK(network->getNbInputs() == 6) << "network->getNbInputs() = " << network->getNbInputs();

  auto keypoints_0_dims_ = network->getInput(2)->getDimensions();
  auto scores_0_dims_ = network->getInput(4)->getDimensions();
  auto descriptors_0_dims_ = network->getInput(0)->getDimensions();
  auto keypoints_1_dims_ = network->getInput(3)->getDimensions();
  auto scores_1_dims_ = network->getInput(5)->getDimensions();
  auto descriptors_1_dims_ = network->getInput(1)->getDimensions();
  CHECK(keypoints_0_dims_.d[1] == -1);
  CHECK(scores_0_dims_.d[1] == -1);
  CHECK(descriptors_0_dims_.d[2] == -1);
  CHECK(keypoints_1_dims_.d[1] == -1);
  CHECK(scores_1_dims_.d[1] == -1);
  CHECK(descriptors_1_dims_.d[2] == -1);

  return true;
}

void SuperGlue::SetContext(nvinfer1::IExecutionContext* const context) {
  CHECK_NOTNULL(context);
  // 6输入，1输出
  CHECK(engine_->getNbIOTensors() == 7);

  context->setInputShape(config_.tensor_config().input_tensor_names()[2].c_str(),
                         nvinfer1::Dims3(1, query_des_.cols(), 2));
  context->setInputShape(config_.tensor_config().input_tensor_names()[4].c_str(),
                         nvinfer1::Dims2(1, query_des_.cols()));
  context->setInputShape(config_.tensor_config().input_tensor_names()[0].c_str(),
                         nvinfer1::Dims3(1, 256, query_des_.cols()));
  context->setInputShape(config_.tensor_config().input_tensor_names()[3].c_str(),
                         nvinfer1::Dims3(1, train_des_.cols(), 2));
  context->setInputShape(config_.tensor_config().input_tensor_names()[5].c_str(),
                         nvinfer1::Dims2(1, train_des_.cols()));
  context->setInputShape(config_.tensor_config().input_tensor_names()[1].c_str(),
                         nvinfer1::Dims3(1, 256, train_des_.cols()));
  output_scores_dims_ =
      context->getTensorShape(config_.tensor_config().output_tensor_names()[0].c_str());
}

bool SuperGlue::ProcessInput(const TensorRT::BufferManager& buffers) {
  // 这是将CPU的内存数据拿出来进行处理
  auto* keypoints_0_buffer =
      static_cast<float*>(buffers.GetHostBuffer(config_.tensor_config().input_tensor_names()[2]));
  auto* scores_0_buffer =
      static_cast<float*>(buffers.GetHostBuffer(config_.tensor_config().input_tensor_names()[4]));
  auto* descriptors_0_buffer =
      static_cast<float*>(buffers.GetHostBuffer(config_.tensor_config().input_tensor_names()[0]));
  auto* keypoints_1_buffer =
      static_cast<float*>(buffers.GetHostBuffer(config_.tensor_config().input_tensor_names()[3]));
  auto* scores_1_buffer =
      static_cast<float*>(buffers.GetHostBuffer(config_.tensor_config().input_tensor_names()[5]));
  auto* descriptors_1_buffer =
      static_cast<float*>(buffers.GetHostBuffer(config_.tensor_config().input_tensor_names()[1]));

  for (int rows0 = 0; rows0 < 1; ++rows0) {
    for (int cols0 = 0; cols0 < query_des_.cols(); ++cols0) {
      scores_0_buffer[rows0 * query_des_.cols() + cols0] = query_des_(rows0, cols0);
    }
  }

  for (int colk0 = 0; colk0 < query_des_.cols(); ++colk0) {
    for (int rowk0 = 1; rowk0 < 3; ++rowk0) {
      keypoints_0_buffer[colk0 * 2 + (rowk0 - 1)] = query_des_(rowk0, colk0);
    }
  }

  for (int rowd0 = 3; rowd0 < query_des_.rows(); ++rowd0) {
    for (int cold0 = 0; cold0 < query_des_.cols(); ++cold0) {
      descriptors_0_buffer[(rowd0 - 3) * query_des_.cols() + cold0] = query_des_(rowd0, cold0);
    }
  }

  for (int rows1 = 0; rows1 < 1; ++rows1) {
    for (int cols1 = 0; cols1 < train_des_.cols(); ++cols1) {
      scores_1_buffer[rows1 * train_des_.cols() + cols1] = train_des_(rows1, cols1);
    }
  }

  for (int colk1 = 0; colk1 < train_des_.cols(); ++colk1) {
    for (int rowk1 = 1; rowk1 < 3; ++rowk1) {
      keypoints_1_buffer[colk1 * 2 + (rowk1 - 1)] = train_des_(rowk1, colk1);
    }
  }

  for (int rowd1 = 3; rowd1 < train_des_.rows(); ++rowd1) {
    for (int cold1 = 0; cold1 < train_des_.cols(); ++cold1) {
      descriptors_1_buffer[(rowd1 - 3) * train_des_.cols() + cold1] = train_des_(rowd1, cold1);
    }
  }


  return true;
}

bool SuperGlue::ProcessOutput(const TensorRT::BufferManager& buffers) {

  indices0_.clear();
  indices1_.clear();
  scores0_.clear();
  scores1_.clear();
  auto* output_score =
      static_cast<float*>(buffers.GetHostBuffer(config_.tensor_config().output_tensor_names()[0]));
  int scores_map_h = output_scores_dims_.d[1];
  int scores_map_w = output_scores_dims_.d[2];
  // auto* scores = new float[(scores_map_h + 1) * (scores_map_w + 1)];
  // log_optimal_transport(output_score, scores, scores_map_h, scores_map_w);
  // scores_map_h = scores_map_h + 1;
  // scores_map_w = scores_map_w + 1;
  Decode(output_score, scores_map_h, scores_map_w, indices0_, indices1_, scores0_, scores1_);
  return true;
}

void SuperGlue::Decode(float* scores,
                       int h,
                       int w,
                       std::vector<int>& indices0,
                       std::vector<int>& indices1,
                       std::vector<double>& mscores0,
                       std::vector<double>& mscores1) {
  auto* max_indices0 = new int[h - 1];
  auto* max_indices1 = new int[w - 1];
  auto* max_values0 = new float[h - 1];
  auto* max_values1 = new float[w - 1];
  MaxMatrix(scores, max_indices0, max_values0, h, w, 2);
  MaxMatrix(scores, max_indices1, max_values1, h, w, 1);
  auto* mutual0 = new int[h - 1];
  auto* mutual1 = new int[w - 1];
  EqualGather(max_indices1, max_indices0, mutual0, h - 1);
  EqualGather(max_indices0, max_indices1, mutual1, w - 1);
  WhereExp(mutual0, max_values0, mscores0, h - 1);
  WhereGather(mutual1, max_indices1, mscores0, mscores1, w - 1);
  auto* valid0 = new int[h - 1];
  auto* valid1 = new int[w - 1];
  AndThreshold(mutual0, valid0, mscores0, 0.2);
  AndGather(mutual1, valid0, max_indices1, valid1, w - 1);
  WhereNegativeOne(valid0, max_indices0, h - 1, indices0);
  WhereNegativeOne(valid1, max_indices1, w - 1, indices1);
  delete[] max_indices0;
  delete[] max_indices1;
  delete[] max_values0;
  delete[] max_values1;
  delete[] mutual0;
  delete[] mutual1;
  delete[] valid0;
  delete[] valid1;
}

void SuperGlue::MaxMatrix(const float* data, int* indices, float* values, int h, int w, int dim) {
  if (dim == 2) {
    for (int i = 0; i < h - 1; ++i) {
      float max_value = -FLT_MAX;
      int max_indices = 0;
      for (int j = 0; j < w - 1; ++j) {
        if (max_value < data[i * w + j]) {
          max_value = data[i * w + j];
          max_indices = j;
        }
      }
      values[i] = max_value;
      indices[i] = max_indices;
    }
  } else if (dim == 1) {
    for (int i = 0; i < w - 1; ++i) {
      float max_value = -FLT_MAX;
      int max_indices = 0;
      for (int j = 0; j < h - 1; ++j) {
        if (max_value < data[j * w + i]) {
          max_value = data[j * w + i];
          max_indices = j;
        }
      }
      values[i] = max_value;
      indices[i] = max_indices;
    }
  }
}

void SuperGlue::EqualGather(const int* indices0, const int* indices1, int* mutual, int size) {
  for (int i = 0; i < size; ++i) {
    if (indices0[indices1[i]] == i) {
      mutual[i] = 1;
    } else {
      mutual[i] = 0;
    }
  }
}

void SuperGlue::WhereExp(const int* flag_data,
                         float* data,
                         std::vector<double>& mscores0,
                         int size) {
  for (int i = 0; i < size; ++i) {
    if (flag_data[i] == 1) {
      mscores0.push_back(std::exp(data[i]));
    } else {
      mscores0.push_back(0);
    }
  }
}

void SuperGlue::WhereGather(const int* flag_data,
                            int* indices,
                            std::vector<double>& mscores0,
                            std::vector<double>& mscores1,
                            int size) {
  for (int i = 0; i < size; ++i) {
    if (flag_data[i] == 1) {
      mscores1.push_back(mscores0[indices[i]]);
    } else {
      mscores1.push_back(0);
    }
  }
}

void SuperGlue::AndThreshold(const int* mutual0,
                             int* valid0,
                             const std::vector<double>& mscores0,
                             double threhold) {
  for (uint32_t i = 0; i < mscores0.size(); ++i) {
    if (mutual0[i] == 1 && mscores0[i] > threhold) {
      valid0[i] = 1;
    } else {
      valid0[i] = 0;
    }
  }
}

void SuperGlue::AndGather(
    const int* mutual1, const int* valid0, const int* indices1, int* valid1, int size) {
  for (int i = 0; i < size; ++i) {
    if (mutual1[i] == 1 && valid0[indices1[i]] == 1) {
      valid1[i] = 1;
    } else {
      valid1[i] = 0;
    }
  }
}

void SuperGlue::WhereNegativeOne(const int* flag_data,
                                 const int* data,
                                 int size,
                                 std::vector<int>& indices) {
  for (int i = 0; i < size; ++i) {
    if (flag_data[i] == 1) {
      indices.push_back(data[i]);
    } else {
      indices.push_back(-1);
    }
  }
}

void SuperGlue::LogSinkhornIterations(
    float* couplings, float* Z, int m, int n, float* log_mu, float* log_nu, int iters) {
  auto* u = new float[m]();
  auto* v = new float[n]();
  for (int k = 0; k < iters; ++k) {
    for (int ki = 0; ki < m; ++ki) {
      float nu_expsum = 0.0;
      for (int kn = 0; kn < n; ++kn) {
        nu_expsum += std::exp(couplings[ki * n + kn] + v[kn]);
      }
      u[ki] = log_mu[ki] - std::log(nu_expsum);
    }
    for (int kj = 0; kj < n; ++kj) {
      float nu_expsum = 0.0;
      for (int km = 0; km < m; ++km) {
        nu_expsum += std::exp(couplings[km * n + kj] + u[km]);
      }
      v[kj] = log_nu[kj] - std::log(nu_expsum);
    }
  }

  for (int i = 0; i < m; ++i) {
    for (int j = 0; j < n; ++j) {
      Z[i * n + j] = couplings[i * n + j] + u[i] + v[j];
    }
  }
  delete[] u;
  delete[] v;
}

void SuperGlue::LogOptimalTransport(float* scores, float* Z, int m, int n, float alpha, int iters) {
  auto* couplings = new float[(m + 1) * (n + 1)];
  for (int i = 0; i < m + 1; ++i) {
    for (int j = 0; j < n + 1; ++j) {
      if (i == m || j == n) {
        couplings[i * (n + 1) + j] = alpha;
      } else {
        couplings[i * (n + 1) + j] = scores[i * n + j];
      }
    }
  }

  float norm = -std::log(m + n);

  auto* log_mu = new float[m + 1];
  auto* log_nu = new float[n + 1];
  for (int ii = 0; ii < m; ++ii) {
    log_mu[ii] = norm;
  }
  log_mu[m] = std::log(n) + norm;

  for (int jj = 0; jj < n; ++jj) {
    log_nu[jj] = norm;
  }
  log_nu[n] = std::log(m) + norm;

  LogSinkhornIterations(couplings, Z, m + 1, n + 1, log_mu, log_nu, iters);
  for (int ii = 0; ii < m + 1; ++ii) {
    for (int jj = 0; jj < n + 1; ++jj) {
      Z[ii * (n + 1) + jj] = Z[ii * (n + 1) + jj] - norm;
    }
  }
  delete[] couplings;
  delete[] log_mu;
  delete[] log_nu;
}

Eigen::Matrix<double, 259, Eigen::Dynamic> SuperGlue::NormalizeKeypoints(
    const Eigen::Matrix<double, 259, Eigen::Dynamic>& features, int width, int height) {
  Eigen::Matrix<double, 259, Eigen::Dynamic> norm_features;
  norm_features.resize(259, features.cols());
  norm_features = features;
  for (int col = 0; col < features.cols(); ++col) {
    norm_features(1, col) = (features(1, col) - width / 2) / (std::max(width, height) * 0.7);
    norm_features(2, col) = (features(2, col) - height / 2) / (std::max(width, height) * 0.7);
  }
  return norm_features;
}

void SuperGlue::SuperGlueMatch(Eigen::Matrix<double, 259, Eigen::Dynamic> queryDescriptors,
                               Eigen::Matrix<double, 259, Eigen::Dynamic> trainDescriptors,
                               CV_OUT std::vector<cv::DMatch>& matches,
                               cv::InputArray mask) {
  matches.clear();
  query_des_ = NormalizeKeypoints(queryDescriptors, config_.image_width(), config_.image_height());
  train_des_ = NormalizeKeypoints(trainDescriptors, config_.image_width(), config_.image_height());

  Infer();

  int num_match = 0;
  std::vector<cv::Point2f> points0, points1;
  std::vector<int> point_indexes;
  for (int i = 0; i < static_cast<int>(indices0_.size()); i++) {
    if (indices0_.at(i) < static_cast<int>(indices1_.size()) && indices0_.at(i) >= 0 &&
        indices1_.at(indices0_.at(i)) == i) {
      double d = 1.0 - (scores0_[i] + scores1_[indices0_[i]]) / 2.0;
      matches.emplace_back(i, indices0_[i], d);
      points0.emplace_back(query_des_(1, i), query_des_(2, i));
      points1.emplace_back(train_des_(1, indices0_.at(i)), train_des_(2, indices0_.at(i)));
      num_match++;
    }
  }
}

}  // namespace Camera