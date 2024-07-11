#pragma once

#include <NvInfer.h>
#include <NvOnnxParser.h>

#include <Eigen/Core>
#include <memory>
#include <opencv2/opencv.hpp>
#include <string>

#include "../thirdparty/tensorrtbuffer/include/buffers.h"
#include "util/config.h"

using tensorrt_common::TensorRTUniquePtr;

namespace Camera {
class SuperPoint {
 public:
  explicit SuperPoint(Utils::SuperPointConfig super_point_config);

  bool build();

  bool infer(const cv::Mat& image, Eigen::Matrix<double, 259, Eigen::Dynamic>& features);

  void visualization(const std::string& image_name, const cv::Mat& image);

  void save_engine();

  bool deserialize_engine();

 private:
  Utils::SuperPointConfig super_point_config_;
  nvinfer1::Dims input_dims_{};
  nvinfer1::Dims semi_dims_{};
  nvinfer1::Dims desc_dims_{};
  std::shared_ptr<nvinfer1::ICudaEngine> engine_;
  std::shared_ptr<nvinfer1::IExecutionContext> context_;
  std::vector<std::vector<int>> keypoints_;
  std::vector<std::vector<double>> descriptors_;

  bool construct_network(TensorRTUniquePtr<nvinfer1::IBuilder>& builder,
                         TensorRTUniquePtr<nvinfer1::INetworkDefinition>& network,
                         TensorRTUniquePtr<nvinfer1::IBuilderConfig>& config,
                         TensorRTUniquePtr<nvonnxparser::IParser>& parser) const;

  bool process_input(const tensorrt_buffer::BufferManager& buffers, const cv::Mat& image);

  bool process_output(const tensorrt_buffer::BufferManager& buffers,
                      Eigen::Matrix<double, 259, Eigen::Dynamic>& features);

  void remove_borders(std::vector<std::vector<int>>& keypoints,
                      std::vector<float>& scores,
                      int border,
                      int height,
                      int width);

  std::vector<size_t> sort_indexes(std::vector<float>& data);

  void top_k_keypoints(std::vector<std::vector<int>>& keypoints, std::vector<float>& scores, int k);

  void find_high_score_index(std::vector<float>& scores,
                             std::vector<std::vector<int>>& keypoints,
                             int h,
                             int w,
                             double threshold);

  void sample_descriptors(std::vector<std::vector<int>>& keypoints,
                          float* descriptors,
                          std::vector<std::vector<double>>& dest_descriptors,
                          int dim,
                          int h,
                          int w,
                          int s = 8);
};

typedef std::shared_ptr<SuperPoint> SuperPointPtr;

}  // namespace Camera

{
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
}