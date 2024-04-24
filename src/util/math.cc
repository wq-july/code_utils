#include "util/math.h"

namespace Utils {

Eigen::Matrix3d SO3::JacobianRight(const Eigen::Vector3d &w) {
  Eigen::Matrix3d Jr = Eigen::Matrix3d::Identity();
  double theta = w.norm();  // 旋转角
  if (theta < 0.00001)      // 旋转角很小
  {
    return Jr;  // 返回单位阵
  } else {
    Eigen::Vector3d k = w.normalized();  // 旋转向量的单位向量.
    Eigen::Matrix3d K = SO3::Hat(k);     // 反对称

    Jr = Eigen::Matrix3d::Identity() - (1 - cos(theta)) / theta * K +
         (1 - sin(theta) / theta) * K * K;
  }

  return Jr;
}

Eigen::Matrix3d SO3::JacobianRightInverse(const Eigen::Vector3d &w) {
  Eigen::Matrix3d Jrinv = Eigen::Matrix3d::Identity();
  double theta = w.norm();
  if (theta < 0.00001) {
    return Jrinv;
  }

  else {
    Eigen::Vector3d k = w.normalized();
    Eigen::Matrix3d K = SO3::Hat(k);
    Jrinv = Eigen::Matrix3d::Identity() + 0.5 * SO3::Hat(w) +
            (1.0 - (1.0 + cos(theta)) * theta / (2.0 * sin(theta))) * K * K;
  }

  return Jrinv;
}

Eigen::Matrix3d SO3::JacobianLeft(const Eigen::Vector3d &w) {
  return JacobianRight(-w);
}

Eigen::Matrix3d SO3::JacobianLeftInverse(const Eigen::Vector3d &w) {
  return JacobianRightInverse(-w);
}

SO3::SO3() { unit_quaternion_.setIdentity(); }

SO3::SO3(const SO3 &other) : unit_quaternion_(other.unit_quaternion_) {
  unit_quaternion_.normalize();
}

SO3::SO3(const Eigen::Matrix3d &R) : unit_quaternion_(R) {
  unit_quaternion_.normalize();
}

SO3::SO3(const Eigen::Quaterniond &quat) : unit_quaternion_(quat) {
  assert(unit_quaternion_.squaredNorm() > EPS);
  unit_quaternion_.normalize();
}

// 绕三轴的旋转
SO3::SO3(double rot_x, double rot_y, double rot_z) {
  unit_quaternion_ = (SO3::Exp(Eigen::Vector3d(rot_x, 0.f, 0.f)) *
                      SO3::Exp(Eigen::Vector3d(0.f, rot_y, 0.f)) *
                      SO3::Exp(Eigen::Vector3d(0.f, 0.f, rot_z)))
                         .unit_quaternion_;
}

// 符号重载
void SO3::operator=(const SO3 &other) {
  this->unit_quaternion_ = other.unit_quaternion_;
}

SO3 SO3::operator*(const SO3 &other) const {
  SO3 result(*this);
  result.unit_quaternion_ *= other.unit_quaternion_;
  result.unit_quaternion_.normalize();
  return result;
}

void SO3::operator*=(const SO3 &other) {
  unit_quaternion_ *= other.unit_quaternion_;
  unit_quaternion_.normalize();
}

// 用于和向量相乘
Eigen::Vector3d SO3::operator*(const Eigen::Vector3d &xyz) const {
  return unit_quaternion_._transformVector(xyz);
}

// SO3的逆
SO3 SO3::Inverse() const {
  return SO3(unit_quaternion_.conjugate());  // 返回共轭四元数
}

// 转换为旋转矩阵
Eigen::Matrix3d SO3::GetMatrix() const {
  return unit_quaternion_.toRotationMatrix();
}

// 返回旋转矩阵的伴随矩阵
Eigen::Matrix3d SO3::Adj() const { return GetMatrix(); }

Eigen::Matrix3d SO3::Generator(int32_t i) {
  // 断言函数,参数条件为假则终止
  assert(i >= 0 && i < 3);
  Eigen::Vector3d e;
  e.setZero();
  e[i] = 1.f;
  return Hat(e);
}

// SO3的映射
Eigen::Vector3d SO3::Log() const { return SO3::Log(*this); }

Eigen::Vector3d SO3::Log(const SO3 &other) {
  double theta;
  return LogAndTheta(other, &theta);
}

// 映射为旋转向量和旋转角
Eigen::Vector3d SO3::LogAndTheta(const SO3 &other, double *theta) {
  double n = other.unit_quaternion_.vec().norm();  // 四元数虚部的模长
  double w = other.unit_quaternion_.w();           // 四元数实部
  double squared_w = w * w;

  double two_atan_nbyw_by_n;
  // 变换原理
  // C. Hertzberg et al.:
  // "Integrating Generic Sensor Fusion Algorithms with Sound State
  // Representation through Encapsulation of Manifolds"
  // Information Fusion, 2011
  if (n < EPS) {
    // 四元数归一化后,n=1,w=1.
    assert(std::fabs(w) > EPS);

    two_atan_nbyw_by_n = 2. / w - 2. * (n * n) / (w * squared_w);
  } else {
    if (std::fabs(w) < EPS) {
      if (w > 0) {
        two_atan_nbyw_by_n = M_PI / n;
      } else {
        two_atan_nbyw_by_n = -M_PI / n;
      }
    }
    two_atan_nbyw_by_n = 2 * atan(n / w) / n;
  }

  *theta = two_atan_nbyw_by_n * n;
  return two_atan_nbyw_by_n * other.unit_quaternion_.vec();
}

SO3 SO3::Exp(const Eigen::Vector3d &omega) {
  double theta;
  return ExpAndTheta(omega, &theta);
}

// 旋转向量转换成四元数
SO3 SO3::ExpAndTheta(const Eigen::Vector3d &omega, double *theta) {
  *theta = omega.norm();
  double half_theta = 0.5 * (*theta);

  double imag_factor;                    // 虚部系数
  double real_factor = cos(half_theta);  //
  if ((*theta) < EPS) {
    double theta_sq = (*theta) * (*theta);
    double theta_po4 = theta_sq * theta_sq;
    imag_factor = 0.5 - 0.0208333 * theta_sq + 0.000260417 * theta_po4;
  } else {
    double sin_half_theta = sin(half_theta);
    imag_factor = sin_half_theta / (*theta);
  }

  return SO3(Eigen::Quaterniond(real_factor, imag_factor * omega.x(),
                                imag_factor * omega.y(),
                                imag_factor * omega.z()));
}

Eigen::Matrix3d SO3::Hat(const Eigen::Vector3d &v) {
  Eigen::Matrix3d Omega;
  Omega << 0, -v(2), v(1), v(2), 0, -v(0), -v(1), v(0), 0;
  return Omega;
}

// 反对称矩阵到向量
Eigen::Vector3d SO3::Vee(const Eigen::Matrix3d &Omega) {
  assert(fabs(Omega(2, 1) + Omega(1, 2)) < EPS);
  assert(fabs(Omega(0, 2) + Omega(2, 0)) < EPS);
  assert(fabs(Omega(1, 0) + Omega(0, 1)) < EPS);
  return Eigen::Vector3d(Omega(2, 1), Omega(0, 2), Omega(1, 0));
}

// 李括号操作，向量叉乘
Eigen::Vector3d SO3::LieBracket(const Eigen::Vector3d &omega1,
                                const Eigen::Vector3d &omega2) {
  return omega1.cross(omega2);
}

// 是把一个角速度转化为角速度矩阵的函数
Eigen::Matrix3d SO3::DLieBrackeTabByDa(const Eigen::Vector3d &b) {
  return -Hat(b);
}

void SO3::SetQuaternion(const Eigen::Quaterniond &quaternion) {
  assert(quaternion.norm() != 0);
  unit_quaternion_ = quaternion;
  unit_quaternion_.normalize();
}

}  // namespace Utils
