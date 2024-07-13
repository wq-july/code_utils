#pragma once

#include "Eigen/Dense"

namespace Common {

// 简单的 3D 点结构体，基于 Eigen::Vector3d
struct PointXYZ {
  double x = 0.0;
  double y = 0.0;
  double z = 0.0;

  // 默认构造函数
  inline PointXYZ() {}

  // 构造函数，接受 x、y、z 坐标
  inline PointXYZ(double _x, double _y, double _z) : x(_x), y(_y), z(_z) {}

  // 构造函数，接受 Eigen::Vector3d 对象
  inline PointXYZ(const Eigen::Vector3d& vec) : x(vec(0)), y(vec(1)), z(vec(2)) {}

  // 复制构造函数
  inline PointXYZ(const PointXYZ& other) : x(other.x), y(other.y), z(other.z) {}

  // 将点转为 Eigen::Vector3d
  inline Eigen::Vector3d ToVector3d() const {
    return Eigen::Vector3d(x, y, z);
  }

  // 加法运算符重载
  inline PointXYZ operator+(const PointXYZ& other) const {
    return PointXYZ(x + other.x, y + other.y, z + other.z);
  }

  // 减法运算符重载
  inline PointXYZ operator-(const PointXYZ& other) const {
    return PointXYZ(x - other.x, y - other.y, z - other.z);
  }

  // 除法运算符重载
  inline PointXYZ operator/(double scalar) const {
    return PointXYZ(x / scalar, y / scalar, z / scalar);
  }

  // 乘法运算符重载
  inline PointXYZ operator*(double scalar) const {
    return PointXYZ(x * scalar, y * scalar, z * scalar);
  }

  // 加法赋值运算符重载
  inline PointXYZ& operator+=(const PointXYZ& other) {
    x += other.x;
    y += other.y;
    z += other.z;
    return *this;
  }

  // 减法赋值运算符重载
  inline PointXYZ& operator-=(const PointXYZ& other) {
    x -= other.x;
    y -= other.y;
    z -= other.z;
    return *this;
  }

  // 赋值运算符重载
  inline PointXYZ& operator=(const PointXYZ& other) {
    if (this != &other) {
      x = other.x;
      y = other.y;
      z = other.z;
    }
    return *this;
  }
};

// 带强度信息的 3D 点结构体，继承自 PointXYZ，如果需要用到强度信息的算法的时候在进一步的扩展
struct PointXYZI : public PointXYZ {
  float intensity = 0.f;  // 强度信息

  inline PointXYZI() {}

  inline PointXYZI(double _x, double _y, double _z, float _intensity = 0.f)
      : PointXYZ(_x, _y, _z), intensity(_intensity) {}

  inline PointXYZI(const Eigen::Vector3d& vec, float _intensity = 0.f)
      : PointXYZ(vec), intensity(_intensity) {}

  inline PointXYZI(const PointXYZ& point, float _intensity = 0.f)
      : PointXYZ(point), intensity(_intensity) {}

  // 复制构造函数
  inline PointXYZI(const PointXYZI& other) : PointXYZ(other), intensity(other.intensity) {}

  // 赋值运算符重载
  inline PointXYZI& operator=(const PointXYZI& other) {
    if (this != &other) {
      static_cast<PointXYZ&>(*this) = other;
      intensity = other.intensity;
    }
    return *this;
  }

  // 除法运算符重载
  inline PointXYZI operator/(double scalar) const {
    return PointXYZI(static_cast<PointXYZ>(*this) / scalar, intensity);
  }

  // 乘法运算符重载
  inline PointXYZI operator*(double scalar) const {
    return PointXYZI(static_cast<PointXYZ>(*this) * scalar, intensity);
  }
};

}  // namespace Common