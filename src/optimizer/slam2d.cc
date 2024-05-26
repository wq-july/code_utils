#include "optimizer/slam2d.h"

namespace Optimizer {
namespace SLAM2D {

Slam2d::Slam2d(const std::string g2o_file_path,
               const std::string original_pose,
               const std::string optimized_pose) {
  CHECK(Utils::ReadG2oFile(g2o_file_path, &poses_, &constraints_))
      << "Error reading the file: " << g2o_file_path;
  LOG(INFO) << "Number of poses: " << poses_.size() << '\n';
  LOG(INFO) << "Number of constraints: " << constraints_.size() << '\n';
  CHECK(OutputPoses(original_pose, poses_)) << "Error outputting to " << original_pose;
  ceres::Problem problem;
  BuildOptimizationProblem(constraints_, &poses_, &problem);
  CHECK(SolveOptimizationProblem(&problem)) << "The solve was not successful, exiting.";
  CHECK(OutputPoses(optimized_pose, poses_)) << "Error outputting to " << optimized_pose;
}

Slam2d::~Slam2d() {
}

// Output the poses to the file with format: ID x y yaw_radians.
bool Slam2d::OutputPoses(const std::string& filename, const std::map<int32_t, Pose2d>& poses) {
  std::fstream outfile;
  outfile.open(filename.c_str(), std::istream::out);
  if (!outfile) {
    std::cerr << "Error opening the file: " << filename << '\n';
    return false;
  }
  for (const auto& pair : poses) {
    outfile << pair.first << " " << pair.second.x << " " << pair.second.y << ' '
            << pair.second.yaw_radians << '\n';
  }
  return true;
}

// Constructs the nonlinear least squares optimization problem from the pose
// graph constraints.
void Slam2d::BuildOptimizationProblem(const std::vector<Constraint2d>& constraints,
                                      std::map<int32_t, Pose2d>* poses,
                                      ceres::Problem* problem) {
  CHECK(poses != nullptr);
  CHECK(problem != nullptr);
  if (constraints.empty()) {
    LOG(INFO) << "No constraints, no problem to optimize.";
    return;
  }

  ceres::LossFunction* loss_function = nullptr;
  ceres::Manifold* angle_manifold = AngleManifold::Create();

  for (const auto& constraint : constraints) {
    auto pose_begin_iter = poses->find(constraint.id_begin);
    CHECK(pose_begin_iter != poses->end())
        << "Pose with ID: " << constraint.id_begin << " not found.";
    auto pose_end_iter = poses->find(constraint.id_end);
    CHECK(pose_end_iter != poses->end()) << "Pose with ID: " << constraint.id_end << " not found.";

    const Eigen::Matrix3d sqrt_information = constraint.information.llt().matrixL();
    // Ceres will take ownership of the pointer.
    ceres::CostFunction* cost_function = PoseGraph2dErrorTerm::Create(
        constraint.x, constraint.y, constraint.yaw_radians, sqrt_information);
    problem->AddResidualBlock(cost_function,
                              loss_function,
                              &pose_begin_iter->second.x,
                              &pose_begin_iter->second.y,
                              &pose_begin_iter->second.yaw_radians,
                              &pose_end_iter->second.x,
                              &pose_end_iter->second.y,
                              &pose_end_iter->second.yaw_radians);

    problem->SetManifold(&pose_begin_iter->second.yaw_radians, angle_manifold);
    problem->SetManifold(&pose_end_iter->second.yaw_radians, angle_manifold);
  }

  // The pose graph optimization problem has three DOFs that are not fully
  // constrained. This is typically referred to as gauge freedom. You can apply
  // a rigid body transformation to all the nodes and the optimization problem
  // will still have the exact same cost. The Levenberg-Marquardt algorithm has
  // internal damping which mitigate this issue, but it is better to properly
  // constrain the gauge freedom. This can be done by setting one of the poses
  // as constant so the optimizer cannot change it.
  auto pose_start_iter = poses->begin();
  CHECK(pose_start_iter != poses->end()) << "There are no poses.";
  problem->SetParameterBlockConstant(&pose_start_iter->second.x);
  problem->SetParameterBlockConstant(&pose_start_iter->second.y);
  problem->SetParameterBlockConstant(&pose_start_iter->second.yaw_radians);
}

// Returns true if the solve was successful.
bool Slam2d::SolveOptimizationProblem(ceres::Problem* problem) {
  CHECK(problem != nullptr);

  ceres::Solver::Options options;
  options.max_num_iterations = 100;
  options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;

  ceres::Solver::Summary summary;
  ceres::Solve(options, problem, &summary);

  std::cout << summary.FullReport() << '\n';

  return summary.IsSolutionUsable();
}

}  // namespace SLAM2D

}  // namespace Optimizer
