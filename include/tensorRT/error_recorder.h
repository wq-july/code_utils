#pragma once
#include <atomic>
#include <cstdint>
#include <exception>
#include <mutex>
#include <vector>

#include "NvInferRuntime.h"

#include "tensorRT/logger.h"

using nvinfer1::ErrorCode;
using nvinfer1::IErrorRecorder;

//!
//! A simple implementation of the IErrorRecorder interface for
//! use by samples. This interface also can be used as a reference
//! implementation.
//! The TensorRT Error recorder is based on a vector that pairs the error
//! code and the error string into a single element. It also uses
//! standard mutex's and atomics in order to make sure that the code
//! works in a multi-threaded environment.
//!
class ErrorRecorder : public IErrorRecorder {
  using errorPair = std::pair<ErrorCode, std::string>;
  using errorStack = std::vector<errorPair>;

 public:
  ErrorRecorder() = default;

  ~ErrorRecorder() noexcept override {}
  int32_t getNbErrors() const noexcept final {
    return mErrorStack.size();
  }
  ErrorCode getErrorCode(int32_t errorIdx) const noexcept final {
    return invalidIndexCheck(errorIdx) ? ErrorCode::kINVALID_ARGUMENT : (*this)[errorIdx].first;
  };
  IErrorRecorder::ErrorDesc getErrorDesc(int32_t errorIdx) const noexcept final {
    return invalidIndexCheck(errorIdx) ? "errorIdx out of range."
                                       : (*this)[errorIdx].second.c_str();
  }
  // This class can never overflow since we have dynamic resize via std::vector usage.
  bool hasOverflowed() const noexcept final {
    return false;
  }

  // Empty the errorStack.
  void clear() noexcept final {
    try {
      // grab a lock so that there is no addition while clearing.
      std::lock_guard<std::mutex> guard(mStackLock);
      mErrorStack.clear();
    } catch (const std::exception& e) {
      TensorRT::gLogFatal << "Internal Error: " << e.what() << std::endl;
    }
  };

  //! Simple helper function that
  bool empty() const noexcept {
    return mErrorStack.empty();
  }

  bool reportError(ErrorCode val, IErrorRecorder::ErrorDesc desc) noexcept final {
    try {
      std::lock_guard<std::mutex> guard(mStackLock);
      TensorRT::gLogError << "Error[" << static_cast<int32_t>(val) << "]: " << desc << std::endl;
      mErrorStack.push_back(errorPair(val, desc));
    } catch (const std::exception& e) {
      TensorRT::gLogFatal << "Internal Error: " << e.what() << std::endl;
    }
    // All errors are considered fatal.
    return true;
  }

  // Atomically increment or decrement the ref counter.
  IErrorRecorder::RefCount incRefCount() noexcept final {
    return ++mRefCount;
  }
  IErrorRecorder::RefCount decRefCount() noexcept final {
    return --mRefCount;
  }

 private:
  // Simple helper functions.
  const errorPair& operator[](size_t index) const noexcept {
    return mErrorStack[index];
  }

  bool invalidIndexCheck(int32_t index) const noexcept {
    // By converting signed to unsigned, we only need a single check since
    // negative numbers turn into large positive greater than the size.
    size_t sIndex = index;
    return sIndex >= mErrorStack.size();
  }
  // Mutex to hold when locking mErrorStack.
  std::mutex mStackLock;

  // Reference count of the class. Destruction of the class when mRefCount
  // is not zero causes undefined behavior.
  std::atomic<int32_t> mRefCount{0};

  // The error stack that holds the errors recorded by TensorRT.
  errorStack mErrorStack;
};  // class ErrorRecorder
