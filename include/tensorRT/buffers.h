#pragma once
#include <cuda_runtime_api.h>

#include <cassert>
#include <iostream>
#include <iterator>
#include <memory>
#include <new>
#include <numeric>
#include <string>
#include <unordered_map>
#include <vector>

#include "NvInfer.h"

#include "tensorRT/common.h"

namespace TensorRT {

// The GenericBuffer class is a templated class for buffers.
//
// This templated RAII (Resource Acquisition Is Initialization) class handles the
// allocation, deallocation, querying of buffers on both the device and the host.
// It can handle data of arbitrary types because it stores byte buffers.
// The template parameters AllocFunc and FreeFunc are used for the
// allocation and deallocation of the buffer.
// AllocFunc must be a functor that takes in (void** ptr, size_t size)
// and returns bool. ptr is a pointer to where the allocated buffer address should be
// stored. size is the amount of memory in bytes to allocate. The boolean indicates
// whether or not the memory allocation was successful. FreeFunc must be a functor that
// takes in (void* ptr) and returns void. ptr is the allocated buffer address. It must
// work with nullptr input.
template <typename AllocFunc, typename FreeFunc>
class GenericBuffer {
 public:
  // Construct an empty buffer.
  GenericBuffer(nvinfer1::DataType type = nvinfer1::DataType::kFLOAT)
      : mSize(0), mCapacity(0), mType(type), mBuffer(nullptr) {}

  // Construct a buffer with the specified allocation size in bytes.
  GenericBuffer(size_t size, nvinfer1::DataType type) : mSize(size), mCapacity(size), mType(type) {
    if (!allocFn(&mBuffer, this->nbBytes())) {
      throw std::bad_alloc();
    }
  }

  GenericBuffer(GenericBuffer&& buf)
      : mSize(buf.mSize), mCapacity(buf.mCapacity), mType(buf.mType), mBuffer(buf.mBuffer) {
    buf.mSize = 0;
    buf.mCapacity = 0;
    buf.mType = nvinfer1::DataType::kFLOAT;
    buf.mBuffer = nullptr;
  }

  ~GenericBuffer() {
    freeFn(mBuffer);
  }

  GenericBuffer& operator=(GenericBuffer&& buf) {
    if (this != &buf) {
      freeFn(mBuffer);
      mSize = buf.mSize;
      mCapacity = buf.mCapacity;
      mType = buf.mType;
      mBuffer = buf.mBuffer;
      // Reset buf.
      buf.mSize = 0;
      buf.mCapacity = 0;
      buf.mBuffer = nullptr;
    }
    return *this;
  }

  // Returns pointer to underlying array.
  void* data() {
    return mBuffer;
  }

  // Returns pointer to underlying array.
  const void* data() const {
    return mBuffer;
  }

  // Returns the size (in number of elements) of the buffer.
  size_t size() const {
    return mSize;
  }

  // Returns the size (in bytes) of the buffer.
  size_t nbBytes() const {
    return this->size() * GetElementSize(mType);
  }

  // Resizes the buffer. This is a no-op if the new size is smaller than or equal to the
  // current capacity.
  void resize(size_t newSize) {
    mSize = newSize;
    if (mCapacity < newSize) {
      freeFn(mBuffer);
      if (!allocFn(&mBuffer, this->nbBytes())) {
        throw std::bad_alloc{};
      }
      mCapacity = newSize;
    }
  }

  // Overload of resize that accepts Dims
  void resize(const nvinfer1::Dims& dims) {
    return this->resize(Volume(dims));
  }

 private:
  size_t mSize{0}, mCapacity{0};
  nvinfer1::DataType mType;
  void* mBuffer;
  AllocFunc allocFn;
  FreeFunc freeFn;
};

class DeviceAllocator {
 public:
  bool operator()(void** ptr, size_t size) const {
    cudaError_t err = cudaMalloc(ptr, size);
    if (err != cudaSuccess) {
      std::cerr << "CUDA malloc failed: " << cudaGetErrorString(err) << std::endl;
      return false;
    }
    return true;
  }
};

class DeviceFree {
 public:
  void operator()(void* ptr) const {
    cudaFree(ptr);
  }
};

class HostAllocator {
 public:
  bool operator()(void** ptr, size_t size) const {
    *ptr = malloc(size);
    return *ptr != nullptr;
  }
};

class HostFree {
 public:
  void operator()(void* ptr) const {
    free(ptr);
  }
};

using DeviceBuffer = GenericBuffer<DeviceAllocator, DeviceFree>;
using HostBuffer = GenericBuffer<HostAllocator, HostFree>;

// The ManagedBuffer class groups together a pair of corresponding device and host buffers.
class ManagedBuffer {
 public:
  DeviceBuffer device_buffer_;
  HostBuffer host_buffer_;
};

// \brief  The BufferManager class handles host and device buffer allocation and deallocation.
//
// \details This RAII class handles host and device buffer allocation and deallocation,
//          memcpy between host and device buffers to aid with inference,
//          and debugging dumps to validate inference. The BufferManager class is meant to be
//          used to simplify buffer management and any interactions between buffers and the engine.
class BufferManager {
 public:
  static const size_t kINVALID_SIZE_VALUE = ~size_t(0);

  // \brief Create a BufferManager for handling buffer interactions with engine, when the I/O
  // tensor volumes are provided
  BufferManager(std::shared_ptr<nvinfer1::ICudaEngine> engine,
                std::vector<int64_t> const& volumes,
                int32_t batchSize = 0)
      : engine_(engine), batch_size_(batchSize) {
    // Create host and device buffers
    for (int32_t i = 0; i < engine_->getNbIOTensors(); i++) {
      auto const name = engine->getIOTensorName(i);
      tensor_names_[name] = i;

      nvinfer1::DataType type = engine_->getTensorDataType(name);

      std::unique_ptr<ManagedBuffer> manBuf{new ManagedBuffer()};
      manBuf->device_buffer_ = DeviceBuffer(volumes[i], type);
      manBuf->host_buffer_ = HostBuffer(volumes[i], type);
      void* device_buffer_ = manBuf->device_buffer_.data();
      device_bindings_.emplace_back(device_buffer_);
      managed_buffers_.emplace_back(std::move(manBuf));
    }
  }

  // Create a BufferManager for handling buffer interactions with engine.
  BufferManager(std::shared_ptr<nvinfer1::ICudaEngine> engine,
                int32_t const batchSize = 0,
                nvinfer1::IExecutionContext const* context = nullptr)
      : engine_(engine), batch_size_(batchSize) {
    // Create host and device buffers
    for (int32_t i = 0, e = engine_->getNbIOTensors(); i < e; i++) {
      auto const name = engine->getIOTensorName(i);
      tensor_names_[name] = i;

      auto dims = context ? context->getTensorShape(name) : engine_->getTensorShape(name);
      size_t vol = context || !batch_size_ ? 1 : static_cast<size_t>(batch_size_);
      nvinfer1::DataType type = engine_->getTensorDataType(name);
      int32_t vecDim = engine_->getTensorVectorizedDim(name);
      if (-1 != vecDim)  // i.e., 0 != lgScalarsPerVector
      {
        int32_t scalarsPerVec = engine_->getTensorComponentsPerElement(name);
        dims.d[vecDim] = TensorRT::DivUp(dims.d[vecDim], scalarsPerVec);
        vol *= scalarsPerVec;
      }
      vol *= TensorRT::Volume(dims);
      std::unique_ptr<ManagedBuffer> manBuf{new ManagedBuffer()};
      manBuf->device_buffer_ = DeviceBuffer(vol, type);
      manBuf->host_buffer_ = HostBuffer(vol, type);
      void* device_buffer_ = manBuf->device_buffer_.data();
      device_bindings_.emplace_back(device_buffer_);
      managed_buffers_.emplace_back(std::move(manBuf));
    }
  }

  // Returns a vector of device buffers that you can use directly as
  // bindings for the execute and enqueue methods of IExecutionContext.
  std::vector<void*>& GetDeviceBindings() {
    return device_bindings_;
  }

  // Returns a vector of device buffers.
  std::vector<void*> const& GetDeviceBindings() const {
    return device_bindings_;
  }

  // Returns the device buffer corresponding to tensor_name.
  // Returns nullptr if no such tensor can be found.
  void* GetDeviceBuffer(std::string const& tensor_name) const {
    return GetBuffer(false, tensor_name);
  }

  // Returns the host buffer corresponding to tensor_name.
  // Returns nullptr if no such tensor can be found.
  void* GetHostBuffer(std::string const& tensor_name) const {
    return GetBuffer(true, tensor_name);
  }

  // Returns the size of the host and device buffers that correspond to tensor_name.
  // Returns kINVALID_SIZE_VALUE if no such tensor can be found.
  size_t Size(std::string const& tensor_name) const {
    auto record = tensor_names_.find(tensor_name);
    if (record == tensor_names_.end()) return kINVALID_SIZE_VALUE;
    return managed_buffers_[record->second]->host_buffer_.nbBytes();
  }

  //  Templated print function that dumps buffers of arbitrary type to std::ostream.
  //  rowCount parameter controls how many elements are on each line.
  //  A rowCount of 1 means that there is only 1 element on each line.
  template <typename T>
  void Print(std::ostream& os, void* buf, size_t bufSize, size_t rowCount) {
    assert(rowCount != 0);
    assert(bufSize % sizeof(T) == 0);
    T* typedBuf = static_cast<T*>(buf);
    size_t numItems = bufSize / sizeof(T);
    for (int32_t i = 0; i < static_cast<int>(numItems); i++) {
      // Handle rowCount == 1 case
      if (rowCount == 1 && i != static_cast<int>(numItems) - 1)
        os << typedBuf[i] << std::endl;
      else if (rowCount == 1)
        os << typedBuf[i];
      // Handle rowCount > 1 case
      else if (i % rowCount == 0)
        os << typedBuf[i];
      else if (i % rowCount == rowCount - 1)
        os << " " << typedBuf[i] << std::endl;
      else
        os << " " << typedBuf[i];
    }
  }

  // Copy the contents of input host buffers to input device buffers synchronously.
  void CopyInputToDevice() {
    MemcpyBuffers(true, false, false);
  }

  // Copy the contents of output device buffers to output host buffers synchronously.
  void CopyOutputToHost() {
    MemcpyBuffers(false, true, false);
  }

  // Copy the contents of input host buffers to input device buffers asynchronously.
  void CopyInputToDeviceAsync(cudaStream_t const& stream = 0) {
    MemcpyBuffers(true, false, true, stream);
  }

  // Copy the contents of output device buffers to output host buffers asynchronously.
  void CopyOutputToHostAsync(cudaStream_t const& stream = 0) {
    MemcpyBuffers(false, true, true, stream);
  }

  ~BufferManager() = default;

 private:
  void* GetBuffer(bool const isHost, std::string const& tensor_name) const {
    auto record = tensor_names_.find(tensor_name);
    if (record == tensor_names_.end()) return nullptr;
    return (isHost ? managed_buffers_[record->second]->host_buffer_.data()
                   : managed_buffers_[record->second]->device_buffer_.data());
  }

  bool TenosrIsInput(const std::string& tensor_name) const {
    return engine_->getTensorIOMode(tensor_name.c_str()) == nvinfer1::TensorIOMode::kINPUT;
  }

  void MemcpyBuffers(bool const copy_input,
                     bool const device2host,
                     bool const async,
                     cudaStream_t const& stream = 0) {
    for (auto const& n : tensor_names_) {
      void* dstPtr = device2host ? managed_buffers_[n.second]->host_buffer_.data()
                                 : managed_buffers_[n.second]->device_buffer_.data();
      void const* srcPtr = device2host ? managed_buffers_[n.second]->device_buffer_.data()
                                       : managed_buffers_[n.second]->host_buffer_.data();
      size_t const byteSize = managed_buffers_[n.second]->host_buffer_.nbBytes();
      const cudaMemcpyKind memcpyType =
          device2host ? cudaMemcpyDeviceToHost : cudaMemcpyHostToDevice;
      if ((copy_input && TenosrIsInput(n.first)) || (!copy_input && !TenosrIsInput(n.first))) {
        if (async)
          TCHECK(cudaMemcpyAsync(dstPtr, srcPtr, byteSize, memcpyType, stream));
        else
          TCHECK(cudaMemcpy(dstPtr, srcPtr, byteSize, memcpyType));
      }
    }
  }

  // The pointer to the engine
  std::shared_ptr<nvinfer1::ICudaEngine> engine_;
  // The batch size for legacy networks, 0 otherwise.
  int batch_size_;
  // The vector of pointers to managed buffers
  std::vector<std::unique_ptr<ManagedBuffer>> managed_buffers_;
  // The vector of device buffers needed for engine execution
  std::vector<void*> device_bindings_;
  // The map of tensor name and index pairs
  std::unordered_map<std::string, int32_t> tensor_names_;
};

}  // namespace TensorRT
