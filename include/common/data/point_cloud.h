#pragma once

#include <vector>

#include "Eigen/Core"

#include "common/data/point.h"

namespace Common {
namespace Data {

class PointCloud {
 public:
  // 插入点
  void emplace_back(float x, float y, float z) {
    points_.emplace_back(PointXYZ{x, y, z});
  }

  // 返回点的数量
  int32_t size() const {
    return points_.size();
  }

  // 检查是否为空
  bool empty() const {
    return points_.empty();
  }

  // 预留空间
  void reserve(int32_t n) {
    points_.reserve(n);
  }

  // 清空点云
  void clear() {
    points_.clear();
  }

  // 通过索引获取点
  PointXYZ& at(int32_t index) {
    if (index >= points_.size()) {
      throw std::out_of_range("Index out of range");
    }
    return points_[index];
  }

  const PointXYZ& at(int32_t index) const {
    if (index >= points_.size()) {
      throw std::out_of_range("Index out of range");
    }
    return points_[index];
  }

  // 返回所有点
  const std::vector<PointXYZ>& points() const {
    return points_;
  }

 private:
  std::vector<PointXYZ> points_;
};

// TODO 进一步提供均值和方差的函数以及成员变量

}  // namespace Data

}  // namespace Common
