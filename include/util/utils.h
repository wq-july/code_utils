#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <execution>
#include <filesystem>
#include <fstream>
#include <functional>
#include <map>
#include <numeric>
#include <random>
#include <stdexcept>
#include <vector>

#include "ceres/ceres.h"
#include "Eigen/Core"
#include "pcl/point_cloud.h"
#include "pcl/point_types.h"

#include "glog/logging.h"

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
  static int64_t Hash(const Eigen::Vector3i& x) {
    return VgIcpHash()(x);
  }
  static bool Equal(const Eigen::Vector3i& x1, const Eigen::Vector3i& x2) {
    return x1 == x2;
  }
};

// zelos
struct ZelosHash {
  inline int64_t operator()(const Eigen::Vector3i& key) const {
    return ((key.x() * 73856093) ^ (key.y() * 471943) ^ (key.z() * 83492791)) & ((1 << 20) - 1);
  }
};

// kiss-icp
struct KissIcpHash {
  // kiss-icp
  inline int64_t operator()(const Eigen::Vector3i& voxel) const {
    const uint32_t* vec = reinterpret_cast<const uint32_t*>(voxel.data());
    return ((1 << 20) - 1) & (vec[0] * 73856093 ^ vec[1] * 19349669 ^ vec[2] * 83492791);
  }
};

inline double ComputeDistance(const Eigen::Vector3d& p1, const Eigen::Vector3d& p2) {
  return (p1 - p2).norm();
}

// TODO，后期将对这个函数封装到一个类中用作pcl的接口, 此外可以指定并行线程的数量，以及使用omp等效率
inline std::vector<Eigen::Vector3d> PclToEigen3d(
    const pcl::PointCloud<pcl::PointXYZ>::Ptr& input_cloud) {
  if (!input_cloud || input_cloud->empty()) {
    throw std::invalid_argument("Input cloud is empty!");
  }
  std::vector<Eigen::Vector3d> out_points(input_cloud->points.size());  // 预分配内存
  std::transform(std::execution::par,
                 input_cloud->points.begin(),
                 input_cloud->points.end(),
                 out_points.begin(),
                 [](const pcl::PointXYZ& point) -> Eigen::Vector3d {
                   return point.getArray3fMap().cast<double>();
                 });

  return out_points;
}

void GenerateRandomCoefficientsAndData(
    const std::function<double(const Eigen::VectorXd&, const double)>& func,
    const int32_t param_count,
    const int32_t data_size,
    const std::pair<double, double>& param_range,
    const std::pair<double, double>& noise_range,
    const std::string log_path,
    Eigen::VectorXd* const parameters,
    std::vector<std::pair<double, double>>* const data);

// 将yaw旋转角变换成2D矩阵
template <typename T>
inline Eigen::Matrix<T, 2, 2> RotationMatrix2D(T yaw_radians) {
  const T cos_yaw = ceres::cos(yaw_radians);
  const T sin_yaw = ceres::sin(yaw_radians);

  Eigen::Matrix<T, 2, 2> rotation;
  rotation << cos_yaw, -sin_yaw, sin_yaw, cos_yaw;
  return rotation;
}

// Reads a single pose from the input and inserts it into the map. Returns false
// if there is a duplicate entry.
template <typename Pose, typename Allocator>
bool ReadVertex(std::ifstream* infile, std::map<int, Pose, std::less<int>, Allocator>* poses) {
  int id;
  Pose pose;
  *infile >> id >> pose;

  // Ensure we don't have duplicate poses.
  if (poses->find(id) != poses->end()) {
    LOG(ERROR) << "Duplicate vertex with ID: " << id;
    return false;
  }
  (*poses)[id] = pose;

  return true;
}

// Reads the constraints between two vertices in the pose graph
template <typename Constraint, typename Allocator>
void ReadConstraint(std::ifstream* infile, std::vector<Constraint, Allocator>* constraints) {
  Constraint constraint;
  *infile >> constraint;

  constraints->push_back(constraint);
}

// Reads a file in the g2o filename format that describes a pose graph
// problem. The g2o format consists of two entries, vertices and constraints.
//
// In 2D, a vertex is defined as follows:
//
// VERTEX_SE2 ID x_meters y_meters yaw_radians
//
// A constraint is defined as follows:
//
// EDGE_SE2 ID_A ID_B A_x_B A_y_B A_yaw_B I_11 I_12 I_13 I_22 I_23 I_33
//
// where I_ij is the (i, j)-th entry of the information matrix for the
// measurement.
//
//
// In 3D, a vertex is defined as follows:
//
// VERTEX_SE3:QUAT ID x y z q_x q_y q_z q_w
//
// where the quaternion is in Hamilton form.
// A constraint is defined as follows:
//
// EDGE_SE3:QUAT ID_a ID_b x_ab y_ab z_ab q_x_ab q_y_ab q_z_ab q_w_ab I_11 I_12 I_13 ... I_16 I_22 I_23 ... I_26 ... I_66 // NOLINT
//
// where I_ij is the (i, j)-th entry of the information matrix for the
// measurement. Only the upper-triangular part is stored. The measurement order
// is the delta position followed by the delta orientation.
template <typename Pose, typename Constraint, typename MapAllocator, typename VectorAllocator>
bool ReadG2oFile(const std::string& filename,
                 std::map<int, Pose, std::less<int>, MapAllocator>* poses,
                 std::vector<Constraint, VectorAllocator>* constraints) {
  CHECK(poses != nullptr);
  CHECK(constraints != nullptr);

  poses->clear();
  constraints->clear();

  std::ifstream infile(filename.c_str());
  if (!infile) {
    return false;
  }

  std::string data_type;
  while (infile.good()) {
    // Read whether the type is a node or a constraint.
    infile >> data_type;
    if (data_type == Pose::name()) {
      if (!ReadVertex(&infile, poses)) {
        return false;
      }
    } else if (data_type == Constraint::name()) {
      ReadConstraint(&infile, constraints);
    } else {
      LOG(ERROR) << "Unknown data type: " << data_type;
      return false;
    }

    // Clear any trailing whitespace from the line.
    infile >> std::ws;
  }

  return true;
}


}  // namespace Utils