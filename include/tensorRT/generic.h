#pragma once

// Original code from https://github.com/enazoe/yolo-tensorrt
#include <cuda_runtime.h>
#include <unistd.h>

#include <chrono>
#include <cstdlib>
#include <ctime>
#include <experimental/filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "NvInfer.h"
#include "NvInferPlugin.h"
#include "NvOnnxConfig.h"
#include "NvOnnxParser.h"

#include "glog/logging.h"
#include "opencv2/opencv.hpp"

#include "../protos/pb/tensorRT.pb.h"
#include "tensorRT/buffers.h"
#include "tensorRT/logger.h"

namespace TensorRT {

class GenericInference {
 public:
  GenericInference(const TensorRTConfig::Config& config);
  bool Build();
  bool Infer();
  bool ConstructNetwork(TensorRTUniquePtr<nvinfer1::IBuilder>& builder,
                        TensorRTUniquePtr<nvinfer1::INetworkDefinition>& network,
                        TensorRTUniquePtr<nvinfer1::IBuilderConfig>& config,
                        TensorRTUniquePtr<nvonnxparser::IParser>& parser) const;
  virtual ~GenericInference() = default;
  virtual void SetIBuilderConfigProfile(nvinfer1::IBuilderConfig* const config,
                                        nvinfer1::IOptimizationProfile* const profile);
  virtual bool VerifyEngine(const nvinfer1::INetworkDefinition* network);
  virtual bool ProcessInput(const TensorRT::BufferManager& buffers);
  virtual bool ProcessOutput(const TensorRT::BufferManager& buffers);
  virtual void SetContext(nvinfer1::IExecutionContext* const context);

 private:
  bool LoadTRTEngine(const std::string& engin_path);
  bool LoadTRTONNXEngine(const std::string& onnx_path);
  bool SaveEngine();

 protected:
  TensorRTConfig::Config config_;
  std::shared_ptr<nvinfer1::ICudaEngine> engine_ = nullptr;
};

}  // namespace TensorRT