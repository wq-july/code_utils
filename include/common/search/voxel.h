#pragma once

#include <vector>

#include "Eigen/Core"
#include "Eigen/Geometry"

namespace Common {

// 每个最小voxel单元，其中存储着点
struct Voxel {
  Voxel(const Eigen::Vector3d& point, const uint32_t max_num) : max_nums_(max_num) {
    AddPoint(point);
  };
  ~Voxel() = default;

  void AddPoint(const Eigen::Vector3d& point);

  // buffer of points with a max limit of n_points
  std::vector<Eigen::Vector3d> points_;
  uint32_t max_nums_;
};

struct GaussianVoxel : public Voxel {
 public:
  GaussianVoxel() = default;
  ~GaussianVoxel() = default;
  // inline void AddPoint(const Eigen::Vector3d& point);

  void AddPoint(const Eigen::Vector3d& points, const Eigen::Matrix3d& cov,
                const Eigen::Isometry3d& transformation);

  void Finalize();
  Eigen::Vector3d GetMean() const;
  Eigen::Matrix3d GetCov() const;
  uint32_t GetPointsNum() const;
  uint32_t GetIndex() const;

 public:
  uint32_t num_points_ = 0;
  std::vector<Eigen::Vector3d> points_;

 private:
  bool finalized_ = false;
  uint32_t index_ = 0u;
  Eigen::Vector3d mean_ = Eigen::Vector3d::Zero();
  Eigen::Matrix3d cov_ = Eigen::Matrix3d::Zero();
};

}  // namespace Common
