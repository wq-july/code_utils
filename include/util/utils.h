#pragma once

#include <cstdint>
#include "Eigen/Core"

namespace Utils {

inline Eigen::Array3i FastFloor(const Eigen::Array3d& pt) {
  const Eigen::Array3i ncoord = pt.cast<int>();
  return ncoord - (pt < ncoord.cast<double>()).cast<int>();
};

// vg-icp
struct VgIcpHash {
public:
  inline int64_t operator()(const Eigen::Vector3i& x) const {
    const int64_t p1 = 73856093;
    const int64_t p2 = 19349669;  // 19349663 was not a prime number
    const int64_t p3 = 83492791;
    return static_cast<int64_t>((x[0] * p1) ^ (x[1] * p2) ^ (x[2] * p3));
  }

  static int64_t Hash(const Eigen::Vector3i& x) { return VgIcpHash()(x); }
  static bool Equal(const Eigen::Vector3i& x1, const Eigen::Vector3i& x2) { return x1 == x2; }
};

// zelos
struct ZelosHash {
  inline int64_t operator()(const Eigen::Vector3i& key) const {
    return ((key.x() * 73856093) ^ (key.y() * 471943) ^ (key.z() * 83492791)) & ((1 << 20) - 1);
  }
};

// kiss-icp
struct KissIcpHash {
  inline  int64_t operator()(const Eigen::Vector3i &voxel) const {
        const uint32_t *vec = reinterpret_cast<const uint32_t *>(voxel.data());
        return ((1 << 20) - 1) & (vec[0] * 73856093 ^ vec[1] * 19349669 ^ vec[2] * 83492791);
    }
};

}