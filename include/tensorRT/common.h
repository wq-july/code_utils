#pragma once

#include <cuda_runtime_api.h>

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstring>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <map>
#include <memory>
#include <new>
#include <numeric>
#include <ratio>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "NvInferSafeRuntime.h"

#include "glog/logging.h"

namespace TensorRT {

#undef TCHECK
#define TCHECK(status)                                   \
  do {                                                   \
    auto ret = (status);                                 \
    if (ret != 0) {                                      \
      std::cerr << "Cuda failure: " << ret << std::endl; \
      exit(EXIT_FAILURE);                                \
    }                                                    \
  } while (0)

struct InferDeleter {
  template <typename T>
  void operator()(T* obj) const {
    delete obj;
  }
};

template <typename T>
using TensorRTUniquePtr = std::unique_ptr<T, TensorRT::InferDeleter>;

static auto StreamDeleter = [](cudaStream_t* pStream) {
  if (pStream) {
    static_cast<void>(cudaStreamDestroy(*pStream));
    delete pStream;
  }
};

inline std::unique_ptr<cudaStream_t, decltype(StreamDeleter)> MakeCudaStream() {
  std::unique_ptr<cudaStream_t, decltype(StreamDeleter)> p_stream(new cudaStream_t, StreamDeleter);
  if (cudaStreamCreateWithFlags(p_stream.get(), cudaStreamNonBlocking) != cudaSuccess) {
    p_stream.reset(nullptr);
  }
  return p_stream;
}

template <typename A, typename B>
inline A DivUp(A x, B n) {
  return (x + n - 1) / n;
}

inline int64_t Volume(nvinfer1::Dims const& d) {
  return std::accumulate(d.d, d.d + d.nbDims, int64_t{1}, std::multiplies<int64_t>{});
}

inline uint32_t GetElementSize(nvinfer1::DataType t) noexcept {
  switch (t) {
    case nvinfer1::DataType::kINT64:
      return 8;
    case nvinfer1::DataType::kINT32:
    case nvinfer1::DataType::kFLOAT:
      return 4;
    case nvinfer1::DataType::kBF16:
    case nvinfer1::DataType::kHALF:
      return 2;
    case nvinfer1::DataType::kBOOL:
    case nvinfer1::DataType::kUINT8:
    case nvinfer1::DataType::kINT8:
    case nvinfer1::DataType::kFP8:
      return 1;
    case nvinfer1::DataType::kINT4:
      LOG(FATAL) << "Element size is not implemented for sub-byte data-types";
  }
  return 0;
}

inline void EnableDLA(nvinfer1::IBuilder* builder,
                      nvinfer1::IBuilderConfig* config,
                      int useDLACore,
                      bool allowGPUFallback = true) {
  if (useDLACore >= 0) {
    if (builder->getNbDLACores() == 0) {
      std::cerr << "Trying to use DLA core " << useDLACore
                << " on a platform that doesn't have any DLA cores" << std::endl;
      assert("Error: use DLA core on a platfrom that doesn't have any DLA cores" && false);
    }
    if (allowGPUFallback) {
      config->setFlag(nvinfer1::BuilderFlag::kGPU_FALLBACK);
    }
    if (!config->getFlag(nvinfer1::BuilderFlag::kINT8)) {
      // User has not requested INT8 Mode.
      // By default run in FP16 mode. FP32 mode is not permitted.
      config->setFlag(nvinfer1::BuilderFlag::kFP16);
    }
    config->setDefaultDeviceType(nvinfer1::DeviceType::kDLA);
    config->setDLACore(useDLACore);
  }
}

// Ensures that every tensor used by a network has a dynamic range set.
//
// All tensors in a network must have a dynamic range specified if a calibrator is not used.
// This function is just a utility to globally fill in missing scales and zero-points for the entire
// network.
//
// If a tensor does not have a dyanamic range set, it is assigned inRange or outRange as follows:
//
// * If the tensor is the input to a layer or output of a pooling node, its dynamic range is derived
// from inRange.
// * Otherwise its dynamic range is derived from outRange.
//
// The default parameter values are intended to demonstrate, for final layers in the network,
// cases where dynamic ranges are asymmetric.
//
// The default parameter values choosen arbitrarily. Range values should be choosen such that
// we avoid underflow or overflow. Also range value should be non zero to avoid uniform zero scale
// tensor.
inline void SetAllDynamicRanges(nvinfer1::INetworkDefinition* network,
                                float inRange = 2.0F,
                                float outRange = 4.0F) {
  // Ensure that all layer inputs have a scale.
  for (int i = 0; i < network->getNbLayers(); i++) {
    auto layer = network->getLayer(i);
    for (int j = 0; j < layer->getNbInputs(); j++) {
      nvinfer1::ITensor* input{layer->getInput(j)};
      // Optional inputs are nullptr here and are from RNN layers.
      if (input != nullptr && !input->dynamicRangeIsSet()) {
        CHECK(input->setDynamicRange(-inRange, inRange));
      }
    }
  }

  // Ensure that all layer outputs have a scale.
  // Tensors that are also inputs to layers are ingored here
  // since the previous loop nest assigned scales to them.
  for (int i = 0; i < network->getNbLayers(); i++) {
    auto layer = network->getLayer(i);
    for (int j = 0; j < layer->getNbOutputs(); j++) {
      nvinfer1::ITensor* output{layer->getOutput(j)};
      // Optional outputs are nullptr here and are from RNN layers.
      if (output != nullptr && !output->dynamicRangeIsSet()) {
        // Pooling must have the same input and output scales.
        if (layer->getType() == nvinfer1::LayerType::kPOOLING) {
          CHECK(output->setDynamicRange(-inRange, inRange));
        } else {
          CHECK(output->setDynamicRange(-outRange, outRange));
        }
      }
    }
  }
}

}  // namespace TensorRT
