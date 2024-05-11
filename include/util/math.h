#pragma once

#include <math.h>

#include <cassert>
#include <cstdint>
#include <numeric>

#include "Eigen/Core"
#include "Eigen/Dense"

namespace Utils {

constexpr double kGravity = -9.81;             // 重力
constexpr double kDegreeToRad = M_PI / 180.0;  // 角度转为弧度
constexpr double kRadToDegree = 180.0 / M_PI;  // 弧度转角度
// 非法定义
constexpr uint32_t kINVALID = std::numeric_limits<uint32_t>::max();

class Math {
 public:
  template <typename ContainerType, typename Getter>
  static void ComputeMeanAndCovDiag(const ContainerType& data, Eigen::Vector3d* const mean,
                                   Eigen::Vector3d* const cov_diag, Getter&& getter) {
    uint32_t length = data.size();
    assert(length > 1);

    // clang-format off
    // 实际上这个就是提供容器开始和结束位置，然后提供加法运算规则，最后输出结果类型就可以
    *mean = std::accumulate(data.begin(), data.end(), Eigen::Vector3d::Zero().eval(),
        [&getter](const Eigen::Vector3d& sum, const auto& data) {
          return sum + getter(data);
        }) / length;

    *cov_diag = std::accumulate(data.begin(), data.end(), Eigen::Vector3d::Zero().eval(),
        [&mean, &getter](const Eigen::Vector3d& sum, const auto &data) {
          return sum + (getter(data) - *mean).cwiseAbs2().eval();
        }) / (length - 1);
    // clang-format on
  }
};

}  // namespace Utils