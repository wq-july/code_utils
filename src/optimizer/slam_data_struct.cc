#include "optimizer/slam_data_struct.h"

namespace Optimizer {
namespace SlamDataStruct {

// ================================ 2D Slam Data Struct ================================

std::string Pose2d::name() {
  return "VERTEX_SE2";
}

std::istream& operator>>(std::istream& input, Pose2d& pose) {
  input >> pose.x >> pose.y >> pose.yaw_radians;
  pose.yaw_radians = Utils::Math::NormalizeAngle(pose.yaw_radians);
  return input;
}

std::string Constraint2d::name() {
  return "EDGE_SE2";
}

std::istream& operator>>(std::istream& input, Constraint2d& constraint) {
  input >> constraint.id_begin >> constraint.id_end >> constraint.x >> constraint.y >>
      constraint.yaw_radians >> constraint.information(0, 0) >> constraint.information(0, 1) >>
      constraint.information(0, 2) >> constraint.information(1, 1) >>
      constraint.information(1, 2) >> constraint.information(2, 2);

  constraint.information(1, 0) = constraint.information(0, 1);
  constraint.information(2, 0) = constraint.information(0, 2);
  constraint.information(2, 1) = constraint.information(1, 2);

  constraint.yaw_radians = Utils::Math::NormalizeAngle(constraint.yaw_radians);
  return input;
}

template <typename T>
bool AngleManifold::Plus(const T* x_radians,
                         const T* delta_radians,
                         T* x_plus_delta_radians) const {
  *x_plus_delta_radians = Utils::Math::NormalizeAngle(*x_radians + *delta_radians);
  return true;
}

template <typename T>
bool AngleManifold::Minus(const T* y_radians, const T* x_radians, T* y_minus_x_radians) const {
  *y_minus_x_radians =
      Utils::Math::NormalizeAngle(*y_radians) - Utils::Math::NormalizeAngle(*x_radians);

  return true;
}

ceres::Manifold* AngleManifold::Create() {
  return new ceres::AutoDiffManifold<AngleManifold, 1, 1>;
}

// ================================ 3D Slam Data Struct ================================

std::string Pose3d::name() {
  return "VERTEX_SE3:QUAT";
}

std::istream& operator>>(std::istream& input, Pose3d& pose) {
  input >> pose.p.x() >> pose.p.y() >> pose.p.z() >> pose.q.x() >> pose.q.y() >> pose.q.z() >>
      pose.q.w();
  pose.q.normalize();
  return input;
}

std::string Constraint3d::name() {
  return "EDGE_SE3:QUAT";
}

std::istream& operator>>(std::istream& input, Constraint3d& constraint) {
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
PoseGraph2dErrorTerm::PoseGraph2dErrorTerm(double x_ab,
                                           double y_ab,
                                           double yaw_ab_radians,
                                           const Eigen::Matrix3d& sqrt_information)
    : p_ab_(x_ab, y_ab), yaw_ab_radians_(yaw_ab_radians), sqrt_information_(sqrt_information) {
}

template <typename T>
bool PoseGraph2dErrorTerm::operator()(const T* const x_a,
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

  residuals_map = sqrt_information_.template cast<T>() * residuals_map;

  return true;
}

ceres::CostFunction* PoseGraph2dErrorTerm::Create(double x_ab,
                                                  double y_ab,
                                                  double yaw_ab_radians,
                                                  const Eigen::Matrix3d& sqrt_information) {
  return new ceres::AutoDiffCostFunction<PoseGraph2dErrorTerm, 3, 1, 1, 1, 1, 1, 1>(
      x_ab, y_ab, yaw_ab_radians, sqrt_information);
}

// -------------------------------- 3D Slam --------------------------------
PoseGraph3dErrorTerm::PoseGraph3dErrorTerm(Pose3d t_ab_measured,
                                           Eigen::Matrix<double, 6, 6> sqrt_information)
    : t_ab_measured_(std::move(t_ab_measured)), sqrt_information_(std::move(sqrt_information)) {
}

template <typename T>
bool PoseGraph3dErrorTerm::operator()(const T* const p_a_ptr,
                                      const T* const q_a_ptr,
                                      const T* const p_b_ptr,
                                      const T* const q_b_ptr,
                                      T* residuals_ptr) const {
  Eigen::Map<const Eigen::Matrix<T, 3, 1>> p_a(p_a_ptr);
  Eigen::Map<const Eigen::Quaternion<T>> q_a(q_a_ptr);

  Eigen::Map<const Eigen::Matrix<T, 3, 1>> p_b(p_b_ptr);
  Eigen::Map<const Eigen::Quaternion<T>> q_b(q_b_ptr);

  Eigen::Quaternion<T> q_a_inverse = q_a.conjugate();
  Eigen::Quaternion<T> q_ab_estimated = q_a_inverse * q_b;

  Eigen::Matrix<T, 3, 1> p_ab_estimated = q_a_inverse * (p_b - p_a);

  Eigen::Quaternion<T> delta_q = t_ab_measured_.q.template cast<T>() * q_ab_estimated.conjugate();

  Eigen::Map<Eigen::Matrix<T, 6, 1>> residuals(residuals_ptr);
  residuals.template block<3, 1>(0, 0) = p_ab_estimated - t_ab_measured_.p.template cast<T>();
  residuals.template block<3, 1>(3, 0) = T(2.0) * delta_q.vec();

  residuals.applyOnTheLeft(sqrt_information_.template cast<T>());

  return true;
}

ceres::CostFunction* PoseGraph3dErrorTerm::Create(
    const Pose3d& t_ab_measured, const Eigen::Matrix<double, 6, 6>& sqrt_information) {
  return new ceres::AutoDiffCostFunction<PoseGraph3dErrorTerm, 6, 3, 4, 3, 4>(t_ab_measured,
                                                                              sqrt_information);
}

}  // namespace SlamDataStruct
}  // namespace Optimizer
