#pragma once

#include "optimizer/slam_data_struct.hpp"

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
  SlamModel(const std::string g2o_file_path,
            const std::string original_pose,
            const std::string optimized_pose,
            const modetype type,
            const int32_t max_num_iterations)
      : g2o_file_path_(g2o_file_path),
        original_pose_(original_pose),
        optimized_pose_(optimized_pose),
        type_(type),
        max_num_iterations_(max_num_iterations) {
  }

  ~SlamModel() = default;

  void Process() {
    CHECK(Utils::ReadG2oFile(g2o_file_path_, &poses_, &constraints_))
        << "Error reading the file: " << g2o_file_path_;
    LOG(INFO) << "Number of poses: " << poses_.size() << '\n';
    LOG(INFO) << "Number of constraints: " << constraints_.size() << '\n';
    CHECK(OutputPoses(original_pose_, poses_)) << "Error outputting to " << original_pose_;
    ceres::Problem problem;
    BuildOptimizationProblem(constraints_, &poses_, &problem);
    CHECK(SolveOptimizationProblem(&problem)) << "The solve was not successful, exiting.";
    CHECK(OutputPoses(optimized_pose_, poses_)) << "Error outputting to " << optimized_pose_;
  }

 private:
  // Output the poses to the file with format: ID x y yaw_radians.
  // Output the poses to the file with format: id x y z q_x q_y q_z q_w.
  bool OutputPoses(const std::string& filename, const PoseTypeVec& poses) {
    std::fstream outfile;
    outfile.open(filename.c_str(), std::istream::out);
    if (!outfile) {
      std::cerr << "Error opening the file: " << filename << '\n';
      return false;
    }
    for (const auto& pair : poses) {
      if constexpr (std::is_same_v<PoseType, Optimizer::SlamDataStruct::Pose2d>) {
        outfile << pair.first << " " << pair.second.x << " " << pair.second.y << ' '
                << pair.second.yaw_radians << '\n';
      } else if constexpr (std::is_same_v<PoseType, Optimizer::SlamDataStruct::Pose3d>) {
        outfile << pair.first << " " << pair.second.p.transpose() << " " << pair.second.q.x() << " "
                << pair.second.q.y() << " " << pair.second.q.z() << " " << pair.second.q.w()
                << '\n';
      } else {
        // Handle other PoseType cases
        LOG(FATAL) << "Unhandled PoseType!";
      }
    }
    return true;
  }

  // !默认情况下的通用版本
  // template <typename PoseType>
  // bool OutputPoses(const std::string& filename, const PoseTypeVec& poses) {
  //   !通用的处理逻辑，如果 PoseType 没有特化，会调用这个版本
  //   LOG(FATAL) << "Unhandled PoseType!";
  //   return false;
  // }

  // ! Pose2d 特化版本
  // template <>
  // bool OutputPoses<Optimizer::SlamDataStruct::Pose2d>(const std::string& filename,
  //                                                     const PoseTypeVec& poses) {
  //   std::fstream outfile;
  //   outfile.open(filename.c_str(), std::istream::out);
  //   if (!outfile) {
  //     std::cerr << "Error opening the file: " << filename << '\n';
  //     return false;
  //   }
  //   for (const auto& pair : poses) {
  //     outfile << pair.first << " " << pair.second.x << " " << pair.second.y << ' '
  //             << pair.second.yaw_radians << '\n';
  //   }
  //   return true;
  // }

  // !Pose3d 特化版本
  // template <>
  // bool OutputPoses<Optimizer::SlamDataStruct::Pose3d>(const std::string& filename,
  //                                                     const PoseTypeVec& poses) {
  //   std::fstream outfile;
  //   outfile.open(filename.c_str(), std::istream::out);
  //   if (!outfile) {
  //     std::cerr << "Error opening the file: " << filename << '\n';
  //     return false;
  //   }
  //   for (const auto& pair : poses) {
  //     outfile << pair.first << " " << pair.second.p.transpose() << " " << pair.second.q.x() << "
  //     "
  //             << pair.second.q.y() << " " << pair.second.q.z() << " " << pair.second.q.w() <<
  //             '\n';
  //   }
  //   return true;
  // }

  // Constructs the nonlinear least squares optimization problem from the pose
  // graph constraints.
  void BuildOptimizationProblem(const ConstraintsVec& constraints,
                                PoseTypeVec* poses,
                                ceres::Problem* problem) {
    CHECK(poses != nullptr);
    CHECK(problem != nullptr);
    if (constraints.empty()) {
      LOG(INFO) << "No constraints, no problem to optimize.";
      return;
    }

    ceres::LossFunction* loss_function = nullptr;
    ceres::Manifold* state_manifold = nullptr;

    if (type_ == modetype::Slam2dExample) {
      state_manifold = AngleManifold::Create();
    } else if (type_ == modetype::Slam3dExample) {
      state_manifold = new ceres::EigenQuaternionManifold;
    }

    for (const auto& constraint : constraints) {
      auto pose_begin_iter = poses->find(constraint.id_begin);
      CHECK(pose_begin_iter != poses->end())
          << "Pose with ID: " << constraint.id_begin << " not found.";
      auto pose_end_iter = poses->find(constraint.id_end);
      CHECK(pose_end_iter != poses->end())
          << "Pose with ID: " << constraint.id_end << " not found.";

      //   sqrt_information = constraint.information.llt().matrixL();
      // Ceres will take ownership of the pointer.
      ceres::CostFunction* cost_function = nullptr;
      if constexpr (std::is_same_v<PoseType, Optimizer::SlamDataStruct::Pose2d>) {
        cost_function = PoseGraph2dErrorTerm::Create(constraint.x,
                                                     constraint.y,
                                                     constraint.yaw_radians,
                                                     constraint.information.llt().matrixL());
        problem->AddResidualBlock(cost_function,
                                  loss_function,
                                  &pose_begin_iter->second.x,
                                  &pose_begin_iter->second.y,
                                  &pose_begin_iter->second.yaw_radians,
                                  &pose_end_iter->second.x,
                                  &pose_end_iter->second.y,
                                  &pose_end_iter->second.yaw_radians);
        problem->SetManifold(&pose_begin_iter->second.yaw_radians, state_manifold);
        problem->SetManifold(&pose_end_iter->second.yaw_radians, state_manifold);

      } else if constexpr (std::is_same_v<PoseType, Optimizer::SlamDataStruct::Pose3d>) {
        cost_function =
            PoseGraph3dErrorTerm::Create(constraint.t_be, constraint.information.llt().matrixL());
        problem->AddResidualBlock(cost_function,
                                  loss_function,
                                  pose_begin_iter->second.p.data(),
                                  pose_begin_iter->second.q.coeffs().data(),
                                  pose_end_iter->second.p.data(),
                                  pose_end_iter->second.q.coeffs().data());

        problem->SetManifold(pose_begin_iter->second.q.coeffs().data(), state_manifold);
        problem->SetManifold(pose_end_iter->second.q.coeffs().data(), state_manifold);
      }
    }
  }

  // Returns true if the solve was successful.
  bool SolveOptimizationProblem(ceres::Problem* problem) {
    CHECK(problem != nullptr);
    ceres::Solver::Options options;
    options.max_num_iterations = max_num_iterations_;
    options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
    ceres::Solver::Summary summary;
    ceres::Solve(options, problem, &summary);
    LOG(INFO) << summary.FullReport();
    return summary.IsSolutionUsable();
  }

 private:
  const std::string g2o_file_path_;
  const std::string original_pose_;
  const std::string optimized_pose_;
  const modetype type_;
  const int32_t max_num_iterations_;

  PoseTypeVec poses_;
  ConstraintsVec constraints_;

  PoseType* PoseType_;
};

}  // namespace Optimizer