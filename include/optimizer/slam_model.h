#pragma once

#include "optimizer/slam_data_struct.h"
#include <map>
#include <vector>
#include <fstream>
#include <string>
#include <Eigen/StdVector>
#include <glog/logging.h>
#include <ceres/ceres.h>

namespace Optimizer {

using namespace SlamDataStruct;

enum class modetype {
  Slam2dExample,
  Slam3dExample,
};

template <typename PoseType, typename ConstraintType>
class SlamModel {
  using PoseTypeVec = std::map<int32_t,
                               PoseType,
                               std::less<int32_t>,
                               Eigen::aligned_allocator<std::pair<const int32_t, PoseType>>>;
  using ConstraintsVec = std::vector<ConstraintType, Eigen::aligned_allocator<ConstraintType>>;

 public:
  SlamModel(const std::string& g2o_file_path,
            const std::string& original_pose,
            const std::string& optimized_pose,
            modetype type,
            int32_t max_num_iterations);

  ~SlamModel() = default;

  void Process();

 private:
  bool OutputPoses(const std::string& filename, const PoseTypeVec& poses);
  void BuildOptimizationProblem(const ConstraintsVec& constraints, PoseTypeVec* poses, ceres::Problem* problem);
  bool SolveOptimizationProblem(ceres::Problem* problem);

  const std::string g2o_file_path_;
  const std::string original_pose_;
  const std::string optimized_pose_;
  const modetype type_;
  const int32_t max_num_iterations_;

  PoseTypeVec poses_;
  ConstraintsVec constraints_;
};

}  // namespace Optimizer

#include "optimizer/tpp/slam_model.tpp"
