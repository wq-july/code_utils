#pragma once

#include <string>

#include "Eigen/Dense"

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
  static std::string name();
};

std::istream& operator>>(std::istream& input, Pose2d& pose);

// The constraint between two vertices in the pose graph. The constraint is the
// transformation from vertex id_begin to vertex id_end.
struct Constraint2d {
  int32_t id_begin;
  int32_t id_end;

  double x;
  double y;
  double yaw_radians;

  Eigen::Matrix3d information;

  static std::string name();
};

std::istream& operator>>(std::istream& input, Constraint2d& constraint);

// Defines a manifold for updating the angle to be constrained in [-pi to pi).
struct AngleManifold {
 public:
  template <typename T>
  bool Plus(const T* x_radians, const T* delta_radians, T* x_plus_delta_radians) const;

  template <typename T>
  bool Minus(const T* y_radians, const T* x_radians, T* y_minus_x_radians) const;

  static ceres::Manifold* Create();
};

// ================================ 3D Slam Data Struct ================================

struct Pose3d {
  Eigen::Vector3d p;
  Eigen::Quaterniond q;

  static std::string name();

  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};

std::istream& operator>>(std::istream& input, Pose3d& pose);

struct Constraint3d {
  int32_t id_begin;
  int32_t id_end;

  Pose3d t_be;

  Eigen::Matrix<double, 6, 6> information;

  static std::string name();

  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};

std::istream& operator>>(std::istream& input, Constraint3d& constraint);

// ================================ Slam Error Models ================================

// -------------------------------- 2D Slam --------------------------------
class PoseGraph2dErrorTerm {
 public:
  PoseGraph2dErrorTerm(double x_ab,
                       double y_ab,
                       double yaw_ab_radians,
                       const Eigen::Matrix3d& sqrt_information);

  template <typename T>
  bool operator()(const T* const x_a,
                  const T* const y_a,
                  const T* const yaw_a,
                  const T* const x_b,
                  const T* const y_b,
                  const T* const yaw_b,
                  T* residuals_ptr) const;

  static ceres::CostFunction* Create(double x_ab,
                                     double y_ab,
                                     double yaw_ab_radians,
                                     const Eigen::Matrix3d& sqrt_information);

  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

 private:
  const Eigen::Vector2d p_ab_;
  const double yaw_ab_radians_;
  const Eigen::Matrix3d sqrt_information_;
};

// -------------------------------- 3D Slam --------------------------------
class PoseGraph3dErrorTerm {
 public:
  PoseGraph3dErrorTerm(Pose3d t_ab_measured, Eigen::Matrix<double, 6, 6> sqrt_information);

  template <typename T>
  bool operator()(const T* const p_a_ptr,
                  const T* const q_a_ptr,
                  const T* const p_b_ptr,
                  const T* const q_b_ptr,
                  T* residuals_ptr) const;

  static ceres::CostFunction* Create(const Pose3d& t_ab_measured,
                                     const Eigen::Matrix<double, 6, 6>& sqrt_information);

  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

 private:
  const Pose3d t_ab_measured_;
  const Eigen::Matrix<double, 6, 6> sqrt_information_;
};

}  // namespace SlamDataStruct

}  // namespace Optimizer
