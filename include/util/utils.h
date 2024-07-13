#pragma once

#include <dirent.h>
#include <sys/stat.h>

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

#include "Eigen/Core"
#include "Eigen/Dense"
#include "ceres/ceres.h"
#include "glog/logging.h"
#include "google/protobuf/text_format.h"
#include "opencv2/core.hpp"
#include "pcl/point_cloud.h"
#include "pcl/point_types.h"

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

inline double ComputeDistance(const cv::Point2f& pt1, const cv::Point2f& pt2) {
  double dx = pt1.x - pt2.x;
  double dy = pt1.y - pt2.y;
  return sqrt(dx * dx + dy * dy);
}

// From vins-fusion
inline bool IsInBorder(const cv::Mat& img, const cv::Point2f& pt) {
  const int BORDER_SIZE = 1;
  int img_x = cvRound(pt.x);
  int img_y = cvRound(pt.y);
  return BORDER_SIZE <= img_x && img_x < img.cols - BORDER_SIZE && BORDER_SIZE <= img_y &&
         img_y < img.rows - BORDER_SIZE;
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
inline bool ReadVertex(std::ifstream* infile,
                       std::map<int, Pose, std::less<int>, Allocator>* poses) {
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
inline void ReadConstraint(std::ifstream* infile, std::vector<Constraint, Allocator>* constraints) {
  Constraint constraint;
  *infile >> constraint;

  constraints->push_back(constraint);
}

template <typename Pose, typename Constraint, typename MapAllocator, typename VectorAllocator>
bool ReadG2oFile(const std::string& filename,
                 std::map<int32_t, Pose, std::less<int32_t>, MapAllocator>* poses,
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

// TODO，后期将对这个函数封装到一个类中用作pcl的接口, 此外可以指定并行线程的数量，以及使用omp等效率
template <typename PointType>
inline std::vector<Eigen::Vector3d> PclToVec3d(
    const typename pcl::PointCloud<PointType>::Ptr& pcl_cloud_ptr) {
  if (!pcl_cloud_ptr || pcl_cloud_ptr->empty()) {
    throw std::invalid_argument("Input cloud is empty!");
  }
  std::vector<Eigen::Vector3d> eigen_points(pcl_cloud_ptr->points.size());  // 预分配内存
  std::transform(std::execution::par,
                 pcl_cloud_ptr->points.begin(),
                 pcl_cloud_ptr->points.end(),
                 eigen_points.begin(),
                 [](const PointType& point) -> Eigen::Vector3d {
                   return point.getArray3fMap().template cast<double>();
                 });

  return eigen_points;
}

inline Eigen::Vector3d Matrix3dToEuler(const Eigen::Matrix3d& rotation) {
  return rotation.eulerAngles(0, 1, 2);
}

// static bool GetFileNames(const std::string& path, std::vector<std::string>& filenames) {
//   DIR* pDir;
//   struct dirent* ptr;
//   if (!(pDir = opendir(path.c_str()))) {
//     std::cerr << "Current folder doesn't exist!" << std::endl;
//     return false;
//   }
//   while ((ptr = readdir(pDir)) != nullptr) {
//     if (strcmp(ptr->d_name, ".") != 0 && strcmp(ptr->d_name, "..") != 0) {
//       filenames.push_back(path + "/" + ptr->d_name);
//     }
//   }
//   closedir(pDir);
//   std::sort(filenames.begin(), filenames.end());
//   return true;
// }

// static bool FileExists(const std::string& file) {
//   struct stat file_status {};
//   if (stat(file.c_str(), &file_status) == 0 && (file_status.st_mode & S_IFREG)) {
//     return true;
//   }
//   return false;
// }

// static void ConcatenateFolderAndFileNameBase(const std::string& folder,
//                                              const std::string& file_name,
//                                              std::string* path) {
//   *path = folder;
//   if (path->back() != '/') {
//     *path += '/';
//   }
//   *path = *path + file_name;
// }

// static std::string ConcatenateFolderAndFileName(const std::string& folder,
//                                                 const std::string& file_name) {
//   std::string path;
//   ConcatenateFolderAndFileNameBase(folder, file_name, &path);
//   return path;
// }

// 模板函数，用于加载配置文件
template <typename T>
bool LoadProtoConfig(const std::string& file_path, T* const config) {
  // 读取配置文件内容
  std::ifstream configFile(file_path);
  if (!configFile.is_open()) {
    std::cerr << "Failed to open config file: " << file_path << std::endl;
    return false;
  }
  std::string config_str((std::istreambuf_iterator<char>(configFile)),
                         std::istreambuf_iterator<char>());
  // 使用TextFormat解析配置文件内容
  if (!google::protobuf::TextFormat::ParseFromString(config_str, config)) {
    std::cerr << "Failed to parse config file: " << file_path << std::endl;
    return false;
  }
  return true;
}

}  // namespace Utils