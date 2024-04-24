#pragma once

#include "Eigen/Core"
#include "Eigen/Geometry"
#include "Eigen/StdVector"

namespace Utils {

namespace {
constexpr double EPS = 1.0e-10;
}  // namespace

class SO3 {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

 public:
  SO3();

  SO3(const SO3 &other);

  explicit SO3(const Eigen::Matrix3d &R);

  explicit SO3(const Eigen::Quaterniond &unit_quaternion);

  SO3(const double euler_x, const double euler_y, const double euler_z);

  SO3 Inverse() const;

  Eigen::Matrix3d GetMatrix() const;

  Eigen::Matrix3d Adj() const;

  Eigen::Matrix3d Generator(int32_t i);

  Eigen::Vector3d Log() const;

  void SetQuaternion(const Eigen::Quaterniond &quaternion);

  const Eigen::Quaterniond &unit_quaternion() const { return unit_quaternion_; }

 public:
  static Eigen::Matrix3d JacobianRight(const Eigen::Vector3d &w);

  static Eigen::Matrix3d JacobianRightInverse(const Eigen::Vector3d &w);

  // Jl, left jacobian of SO3, Jl(x) = Jr(-x)
  static Eigen::Matrix3d JacobianLeft(const Eigen::Vector3d &w);

  static Eigen::Matrix3d JacobianLeftInverse(const Eigen::Vector3d &w);

  static SO3 Exp(const Eigen::Vector3d &omega);

  static SO3 ExpAndTheta(const Eigen::Vector3d &omega, double *theta);

  static Eigen::Vector3d Log(const SO3 &so3);

  static Eigen::Vector3d LogAndTheta(const SO3 &so3, double *theta);

  static Eigen::Matrix3d Hat(const Eigen::Vector3d &omega);  // 向量=>反对称矩阵

  static Eigen::Vector3d Vee(const Eigen::Matrix3d &Omega);  // 反对称矩阵=>向量

  static Eigen::Vector3d LieBracket(const Eigen::Vector3d &omega1,
                                    const Eigen::Vector3d &omega2);

  static Eigen::Matrix3d DLieBrackeTabByDa(const Eigen::Vector3d &b);

 public:
  void operator=(const SO3 &so3);

  SO3 operator*(const SO3 &so3) const;

  void operator*=(const SO3 &so3);

  Eigen::Vector3d operator*(const Eigen::Vector3d &xyz) const;

 public:
  static const int32_t DOF = 3;

 protected:
  Eigen::Quaterniond unit_quaternion_;
};

inline std::ostream &operator<<(std::ostream &out_str, const SO3 &so3) {
  out_str << so3.Log().transpose();
  return out_str;
}

}  // namespace Utils