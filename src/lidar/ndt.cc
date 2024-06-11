#include "lidar/ndt.h"

using namespace Common;
namespace Lidar {

NDT::NDT(const double voxel_size,
         const int32_t max_iters,
         const double break_dx,
         const int32_t min_effect_points,
         const bool use_downsample,
         const double outlier_th,
         const int32_t min_effective_points)
    : voxel_size_(voxel_size),
      max_iters_(max_iters),
      break_dx_(break_dx),
      min_effect_points_(min_effect_points),
      use_downsample_(use_downsample),
      outlier_th_(outlier_th),
      min_effective_points_(min_effective_points) {
  voxel_map_ = std::make_shared<Common::VoxelMap>(voxel_size, 100, 100);
  filter_ = std::make_shared<Lidar::PointCloudFilter>(0.3);
}

// 核心问题，通过最近邻方法找到对应关系，然后构建最小二乘问题，反复交替迭代
bool NDT::Align(const Eigen::Isometry3d& pred_pose,
                const PointCloudPtr& source_cloud,
                const PointCloudPtr& target_cloud,
                Eigen::Isometry3d* const final_pose) {
  CHECK_NOTNULL(source_cloud);
  CHECK_NOTNULL(target_cloud);
  CHECK_NOTNULL(final_pose);
  if (use_downsample_) {
    PointCloudPtr filtered_source_cloud(new PointCloud);
    filter_->Filter(source_cloud, filtered_source_cloud.get());
    SetSourceCloud(filtered_source_cloud);
  } else {
    SetSourceCloud(source_cloud);
  }
  SetTargetCloud(target_cloud);

  timer_.StartTimer("Build VoxelMap");
  voxel_map_->AddPoints(*target_cloud);
  timer_.StopTimer();
  timer_.PrintElapsedTime();
  // 删除点数不够的体素
  voxel_map_->RemoveFewerPointsVoxel();

  pose_ = Sophus::SE3d(pred_pose.rotation(), pred_pose.translation());

  timer_.StartTimer("Align");
  bool res = NDTAlign();
  timer_.StopTimer();
  timer_.PrintElapsedTime();

  if (res) {
    final_pose->translation() = pose_.translation();
    final_pose->linear() = pose_.so3().matrix();
  }
  return res;
}

bool NDT::SetSourceCloud(const PointCloudPtr& source) {
  if (!source || source->empty()) {
    LOG(FATAL) << "Set Target Point Cloud Failed, Beacuse of NULL !";
    return false;
  }
  source_cloud_ = std::make_shared<PointCloud>(source);
  // 计算点云中心的函数之后需要调整到工具库中，不要冗余在data中，不合理
  source_center_ = source_cloud_->ComputeCentroid();
  return true;
}

bool NDT::SetTargetCloud(const PointCloudPtr& target) {
  if (!target || target->empty()) {
    LOG(FATAL) << "Set Target Point Cloud Failed, Beacuse of NULL !";
    return false;
  }
  target_cloud_ = std::make_shared<PointCloud>(target);
  // 计算点云中心的函数之后需要调整到工具库中，不要冗余在data中，不合理
  target_center_ = target_cloud_->ComputeCentroid();
  return true;
}

bool NDT::NDTAlign() {
  CHECK_NOTNULL(source_cloud_);
  CHECK_NOTNULL(target_cloud_);
  LOG(INFO) << "Processing Multi-threaded NDT";
  // 对点的索引，预先生成
  std::vector<int32_t> index(source_cloud_->size());
  for (uint32_t i = 0; i < index.size(); ++i) {
    index.at(i) = i;
  }

  int32_t nearby_size = 7;
  int32_t effective_nums = 0;
  for (int32_t iter = 0; iter < max_iters_; ++iter) {
    // 并发，统计有效点，统计各个块的雅可比矩阵和残差，最终都是要加在一起的
    // 用空间换时间，不能在for_each中进行迭代，锁很耗时
    // 使用临近的前后左右上下和中心7个体素

    int32_t total_size = index.size() * nearby_size;
    std::vector<bool> effect_points(total_size, false);

    std::vector<Eigen::Matrix<double, 3, 6>> jacobians(total_size);
    std::vector<Eigen::Matrix3d> inv_covs_(total_size);
    std::vector<Eigen::Vector3d> errs(total_size);
    std::vector<Eigen::Matrix3d> infos(total_size);
    // 这里使用高斯牛顿，也可以使用其他优化方法，这里应该可以定义成自己写的优化类
    // 这里的操作是针对每个点的
    std::for_each(std::execution::par_unseq, index.begin(), index.end(), [&](const int32_t idx) {
      // 1. 首先根据上一步得到的位姿变换点，将点变换到对应位置
      auto pt = source_cloud_->at(idx);
      Eigen::Vector3d transformed_pt = pose_ * pt;
      std::vector<NDTVoxel> nearby_voxels;
      voxel_map_->GetNeighborVoxels(transformed_pt, &nearby_voxels);
      for (uint32_t i = 0; i < nearby_voxels.size(); ++i) {
        int real_idx = idx * nearby_size + i;
        Eigen::Vector3d err = transformed_pt - nearby_voxels.at(i).mean_;
        // check chi2 th
        double res = err.transpose() * nearby_voxels.at(i).inv_cov_ * err;
        if (std::isnan(res) || res > outlier_th_) {
          continue;
        }
        // build residual
        Eigen::Matrix<double, 3, 6> jac;
        jac.block<3, 3>(0, 0) = -pose_.so3().matrix() * Sophus::SO3d::hat(pt);
        jac.block<3, 3>(0, 3) = Eigen::Matrix3d::Identity();
        jacobians.at(real_idx) = jac;
        errs.at(real_idx) = err;
        infos.at(real_idx) = nearby_voxels.at(i).inv_cov_;
        effect_points.at(real_idx) = true;
      }
    });

    Eigen::Matrix<double, 6, 6> sum_hessian = Eigen::Matrix<double, 6, 6>::Zero();
    Eigen::Matrix<double, 6, 1> sum_b = Eigen::Matrix<double, 6, 1>::Zero();
    effective_nums = 0;
    double sum_errs = 0.0;
    // 5. 整理之前并发得到的结果，需要叠加和计算
    for (uint32_t i = 0; i < effect_points.size(); ++i) {
      if (!effect_points.at(i)) {
        continue;
      }
      ++effective_nums;
      sum_hessian += jacobians.at(i).transpose() * jacobians.at(i);
      sum_b -= jacobians.at(i).transpose() * errs.at(i);
      sum_errs += errs.at(i).dot(errs.at(i));
    }

    int32_t effect_nums = effective_nums / nearby_size;
    if (effect_nums < min_effective_points_) {
      LOG(FATAL) << "effective num too small: " << effect_nums;
      return false;
    }

    // 6. 计算求解
    Eigen::Matrix<double, 6, 1> dx =
        Optimizer::NonlinearOptimizer::Solver(sum_hessian, sum_b, Optimizer::SolverType::INVERSE);
    pose_.so3() = pose_.so3() * Sophus::SO3d::exp(dx.head<3>());
    pose_.translation() += dx.tail<3>();

    if (dx.norm() < break_dx_) {
      LOG(INFO) << "converged, dx = " << dx.transpose();
      break;
    }
  }
  LOG(INFO) << "Effect Points Nums is " << effective_nums / nearby_size;
  return true;
}

}  // namespace Lidar