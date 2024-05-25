#pragma once
#include "ceres/ceres.h"
#include "optimizer/optimizer.h"

namespace Optimizer {
namespace SLAM2D {

// The state for each vertex in the pose graph.
struct Pose2d {
  double x;
  double y;
  double yaw_radians;

  // The name of the data type in the g2o file format.
  static std::string name() {
    return "VERTEX_SE2";
  }
};

inline std::istream& operator>>(std::istream& input, Pose2d& pose) {
  input >> pose.x >> pose.y >> pose.yaw_radians;
  // Normalize the angle between -pi to pi.
  pose.yaw_radians = Utils::Math::NormalizeAngle(pose.yaw_radians);
  return input;
}

// The constraint between two vertices in the pose graph. The constraint is the
// transformation from vertex id_begin to vertex id_end.
struct Constraint2d {
  int32_t id_begin;
  int32_t id_end;

  double x;
  double y;
  double yaw_radians;

  // The inverse of the covariance matrix for the measurement. The order of the
  // entries are x, y, and yaw.
  Eigen::Matrix3d information;

  // The name of the data type in the g2o file format.
  static std::string name() {
    return "EDGE_SE2";
  }
};

inline std::istream& operator>>(std::istream& input, Constraint2d& constraint) {
  input >> constraint.id_begin >> constraint.id_end >> constraint.x >> constraint.y >>
      constraint.yaw_radians >> constraint.information(0, 0) >> constraint.information(0, 1) >>
      constraint.information(0, 2) >> constraint.information(1, 1) >>
      constraint.information(1, 2) >> constraint.information(2, 2);

  // Set the lower triangular part of the information matrix.
  constraint.information(1, 0) = constraint.information(0, 1);
  constraint.information(2, 0) = constraint.information(0, 2);
  constraint.information(2, 1) = constraint.information(1, 2);

  // Normalize the angle between -pi to pi.
  constraint.yaw_radians = Utils::Math::NormalizeAngle(constraint.yaw_radians);
  return input;
}

// Defines a manifold for updating the angle to be constrained in [-pi to pi).
struct AngleManifold {
 public:
  template <typename T>
  bool Plus(const T* x_radians, const T* delta_radians, T* x_plus_delta_radians) const {
    *x_plus_delta_radians = Utils::Math::NormalizeAngle(*x_radians + *delta_radians);
    return true;
  }

  template <typename T>
  bool Minus(const T* y_radians, const T* x_radians, T* y_minus_x_radians) const {
    *y_minus_x_radians =
        Utils::Math::NormalizeAngle(*y_radians) - Utils::Math::NormalizeAngle(*x_radians);

    return true;
  }

  static ceres::Manifold* Create() {
    return new ceres::AutoDiffManifold<AngleManifold, 1, 1>;
  }
};

// 误差模型，这个和之前曲线拟合部分结构上是完全对的上的
// Computes the error term for two poses that have a relative pose measurement
// between them. Let the hat variables be the measurement.
//
// residual =  information^{1/2} * [  r_a^T * (p_b - p_a) - \hat{p_ab}   ]
//                                 [ Normalize(yaw_b - yaw_a - \hat{yaw_ab}) ]
//
// where r_a is the rotation matrix that rotates a vector represented in frame A
// into the global frame, and Normalize(*) ensures the angles are in the range
// [-pi, pi).
class PoseGraph2dErrorTerm {
 public:
  PoseGraph2dErrorTerm(double x_ab,
                       double y_ab,
                       double yaw_ab_radians,
                       const Eigen::Matrix3d& sqrt_information)
      : p_ab_(x_ab, y_ab), yaw_ab_radians_(yaw_ab_radians), sqrt_information_(sqrt_information) {
  }

  template <typename T>
  bool operator()(const T* const x_a,
                  const T* const y_a,
                  const T* const yaw_a,
                  const T* const x_b,
                  const T* const y_b,
                  const T* const yaw_b,
                  T* residuals_ptr) const {
    const Eigen::Matrix<T, 2, 1> p_a(*x_a, *y_a);
    const Eigen::Matrix<T, 2, 1> p_b(*x_b, *y_b);

    Eigen::Map<Eigen::Matrix<T, 3, 1>> residuals_map(residuals_ptr);

    residuals_map.template head<2>() =
        Utils::RotationMatrix2D(*yaw_a).transpose() * (p_b - p_a) - p_ab_.cast<T>();
    residuals_map(2) =
        Utils::Math::NormalizeAngle((*yaw_b - *yaw_a) - static_cast<T>(yaw_ab_radians_));

    // Scale the residuals by the square root information matrix to account for
    // the measurement uncertainty.
    residuals_map = sqrt_information_.template cast<T>() * residuals_map;

    return true;
  }

  static ceres::CostFunction* Create(double x_ab,
                                     double y_ab,
                                     double yaw_ab_radians,
                                     const Eigen::Matrix3d& sqrt_information) {
    return new ceres::AutoDiffCostFunction<PoseGraph2dErrorTerm, 3, 1, 1, 1, 1, 1, 1>(
        x_ab, y_ab, yaw_ab_radians, sqrt_information);
  }

  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

 private:
  // The position of B relative to A in the A frame.
  const Eigen::Vector2d p_ab_;
  // The orientation of frame B relative to frame A.
  const double yaw_ab_radians_;
  // The inverse square root of the measurement covariance matrix.
  const Eigen::Matrix3d sqrt_information_;
};

class Slam2d {
 public:
  Slam2d(const std::string g2o_file_path,
         const std::string original_pose,
         const std::string optimized_pose);

  ~Slam2d();

  // Output the poses to the file with format: ID x y yaw_radians.
  bool OutputPoses(const std::string& filename, const std::map<int32_t, Pose2d>& poses);

  // Constructs the nonlinear least squares optimization problem from the pose
  // graph constraints.
  void BuildOptimizationProblem(const std::vector<Constraint2d>& constraints,
                                std::map<int32_t, Pose2d>* poses,
                                ceres::Problem* problem);

  // Returns true if the solve was successful.
  bool SolveOptimizationProblem(ceres::Problem* problem);

 private:
  std::map<int32_t, Pose2d> poses_;
  std::vector<Constraint2d> constraints_;
};

}  // namespace SLAM2D

}  // namespace Optimizer
