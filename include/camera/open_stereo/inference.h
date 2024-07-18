#ifndef INFERENCE_H
#define INFERENCE_H

#include <NvInfer.h>
#include <NvOnnxParser.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>

#include <iostream>
#include <opencv2/opencv.hpp>
#include <string>
#include <unordered_map>
#include <vector>

#include "camera/open_stereo/utils.h"

// Logger for TensorRT info/warning/errors
class Logger : public nvinfer1::ILogger {
 public:
  void log(nvinfer1::ILogger::Severity severity, const char* msg) noexcept override {
    if (severity <= nvinfer1::ILogger::Severity::kVERBOSE) {
      std::cout << msg << std::endl;
    }
  }
};

class InferenceEngine {
 public:
  InferenceEngine(const std::string& engine_file);
  ~InferenceEngine();

  std::unordered_map<std::string, cv::Mat> run(
      const std::unordered_map<std::string, cv::Mat>& sample);

 private:
  nvinfer1::IRuntime* runtime_;
  nvinfer1::ICudaEngine* engine_;
  nvinfer1::IExecutionContext* context_;

  // void* buffers_[2];
  std::vector<void*> buffers_;
  cudaStream_t stream_;
  std::vector<char> engine_data_;

  void loadEngine(const std::string& engine_file);
  void showEngineInfo();
  void allocateBuffers();
  void preprocess(const std::unordered_map<std::string, cv::Mat>& sample);
  std::unordered_map<std::string, cv::Mat> postprocess();

  static Logger gLogger;
};

#endif  // INFERENCE_H
