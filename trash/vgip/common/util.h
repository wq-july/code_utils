/****************************************************************************
 *
 * Copyright (c) 2021 ZelosTech.com, Inc. All Rights Reserved
 *
 ***************************************************************************/
/**
 * @file util.h
 **/

#pragma once

#include <cstdint>
#include "Eigen/Core"

namespace zelos {
namespace zoe {
namespace localization {

// 会频繁的调用这个函数，设置成内联函数
/// @brief Fast floor (https://stackoverflow.com/questions/824118/why-is-floor-so-slow).
/// @param pt  Double vector
/// @return    Floored int vector
inline Eigen::Array3i FastFloor(const Eigen::Array3d& pt) {
  const Eigen::Array3i ncoord = pt.cast<int>();
  return ncoord - (pt < ncoord.cast<double>()).cast<int>();
};

/**
 * @brief Spatial hashing function.
 *        Teschner et al., "Optimized Spatial Hashing for Collision Detection of Deformable Objects", VMV2003.
 */
struct XORVector3iHash {
public:
  int64_t operator()(const Eigen::Vector3i& x) const {
    const int64_t p1 = 73856093;
    const int64_t p2 = 19349669;  // 19349663 was not a prime number
    const int64_t p3 = 83492791;
    return static_cast<int64_t>((x[0] * p1) ^ (x[1] * p2) ^ (x[2] * p3));
  }

  static int64_t Hash(const Eigen::Vector3i& x) { return XORVector3iHash()(x); }
  static bool Equal(const Eigen::Vector3i& x1, const Eigen::Vector3i& x2) { return x1 == x2; }
};


enum struct NearbyType {
    CENTER,  // 只考虑中心
    // for 3D
    NEARBY6,  // 上下左右前后
};

}  // namespace localization
}  // namespace zoe
}  // namespace zelos

/* vim: set expandtab ts=2 sw=2 sts=2 tw=100: */
