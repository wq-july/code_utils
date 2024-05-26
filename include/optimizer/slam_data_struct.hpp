#pragma once
#include "ceres/autodiff_cost_function.h"
#include "ceres/ceres.h"
#include "optimizer/optimizer.h"

namespace Optimizer {
namespace SlamDataStruct {

// ================================ 2D Slam Data Struct ================================
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

// ================================ 3D Slam Data Struct ================================

struct Pose3d {
  Eigen::Vector3d p;
  Eigen::Quaterniond q;

  // The name of the data type in the g2o file format.
  static std::string name() {
    return "VERTEX_SE3:QUAT";
  }

  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};

inline std::istream& operator>>(std::istream& input, Pose3d& pose) {
  input >> pose.p.x() >> pose.p.y() >> pose.p.z() >> pose.q.x() >> pose.q.y() >> pose.q.z() >>
      pose.q.w();
  // Normalize the quaternion to account for precision loss due to
  // serialization.
  pose.q.normalize();
  return input;
}


// The constraint between two vertices in the pose graph. The constraint is the
// transformation from vertex id_begin to vertex id_end.
struct Constraint3d {
  int32_t id_begin;
  int32_t id_end;

  // The transformation that represents the pose of the end frame E w.r.t. the
  // begin frame B. In other words, it transforms a vector in the E frame to
  // the B frame.
  Pose3d t_be;

  // The inverse of the covariance matrix for the measurement. The order of the
  // entries are x, y, z, delta orientation.
  Eigen::Matrix<double, 6, 6> information;

  // The name of the data type in the g2o file format.
  static std::string name() {
    return "EDGE_SE3:QUAT";
  }

  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};

inline std::istream& operator>>(std::istream& input, Constraint3d& constraint) {
  Pose3d& t_be = constraint.t_be;
  input >> constraint.id_begin >> constraint.id_end >> t_be;

  for (int32_t i = 0; i < 6 && input.good(); ++i) {
    for (int32_t j = i; j < 6 && input.good(); ++j) {
      input >> constraint.information(i, j);
      if (i != j) {
        constraint.information(j, i) = constraint.information(i, j);
      }
    }
  }
  return input;
}

// ================================ Slam Error Models ================================

// -------------------------------- 2D Slam --------------------------------
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

// -------------------------------- 3D Slam --------------------------------
class PoseGraph3dErrorTerm {
 public:
  PoseGraph3dErrorTerm(Pose3d t_ab_measured, Eigen::Matrix<double, 6, 6> sqrt_information)
      : t_ab_measured_(std::move(t_ab_measured)), sqrt_information_(std::move(sqrt_information)) {
  }

  template <typename T>
  bool operator()(const T* const p_a_ptr,
                  const T* const q_a_ptr,
                  const T* const p_b_ptr,
                  const T* const q_b_ptr,
                  T* residuals_ptr) const {
    Eigen::Map<const Eigen::Matrix<T, 3, 1>> p_a(p_a_ptr);
    Eigen::Map<const Eigen::Quaternion<T>> q_a(q_a_ptr);

    Eigen::Map<const Eigen::Matrix<T, 3, 1>> p_b(p_b_ptr);
    Eigen::Map<const Eigen::Quaternion<T>> q_b(q_b_ptr);

    // Compute the relative transformation between the two frames.
    Eigen::Quaternion<T> q_a_inverse = q_a.conjugate();
    Eigen::Quaternion<T> q_ab_estimated = q_a_inverse * q_b;

    // Represent the displacement between the two frames in the A frame.
    Eigen::Matrix<T, 3, 1> p_ab_estimated = q_a_inverse * (p_b - p_a);

    // Compute the error between the two orientation estimates.
    Eigen::Quaternion<T> delta_q = t_ab_measured_.q.template cast<T>() * q_ab_estimated.conjugate();

    // Compute the residuals.
    // [ position         ]   [ delta_p          ]
    // [ orientation (3x1)] = [ 2 * delta_q(0:2) ]
    Eigen::Map<Eigen::Matrix<T, 6, 1>> residuals(residuals_ptr);
    residuals.template block<3, 1>(0, 0) = p_ab_estimated - t_ab_measured_.p.template cast<T>();
    residuals.template block<3, 1>(3, 0) = T(2.0) * delta_q.vec();

    // Scale the residuals by the measurement uncertainty.
    residuals.applyOnTheLeft(sqrt_information_.template cast<T>());

    return true;
  }

  static ceres::CostFunction* Create(const Pose3d& t_ab_measured,
                                     const Eigen::Matrix<double, 6, 6>& sqrt_information) {
    return new ceres::AutoDiffCostFunction<PoseGraph3dErrorTerm, 6, 3, 4, 3, 4>(t_ab_measured,
                                                                                sqrt_information);
  }

  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

 private:
  // The measurement for the position of B relative to A in the A frame.
  const Pose3d t_ab_measured_;
  // The square root of the measurement information matrix.
  const Eigen::Matrix<double, 6, 6> sqrt_information_;
};

}  // namespace SlamDataStruct

}  // namespace Optimizer
