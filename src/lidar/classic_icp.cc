#include "lidar/classic_icp.h"

#include "pcl/registration/icp.h"
#include "pcl/registration/ndt.h"

#include "optimizer/optimizer.h"
#include "util/math.h"
namespace {
constexpr double Lambda = 1.0e-6;
constexpr double BreakError = 1.0e-4;
constexpr int32_t MaxIters = 100;
}  // namespace

namespace Lidar {

ClassicICP::ClassicICP(const double voxel_size,
                       const double max_map_distance,
                       const int32_t max_num_per_voxel,
                       const AlignMethod align_method,
                       const SearchMethod search_method,
                       const int32_t max_iters,
                       const double break_dx,
                       const int32_t min_effect_points,
                       const bool use_downsample)
    : align_method_(align_method),
      search_method_(search_method),
      max_iters_(max_iters),
      break_dx_(break_dx),
      min_effect_points_(min_effect_points),
      use_downsample_(use_downsample) {
  kdtree_ = std::make_shared<Common::KdTree>();
  voxel_map_ = std::make_shared<Common::VoxelMap>(voxel_size, max_map_distance, max_num_per_voxel);
  filter_ = std::make_shared<Lidar::PointCloudFilter>(0.3);
}

bool ClassicICP::Align(const Eigen::Isometry3d& pred_pose,
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
  switch (search_method_) {
    case SearchMethod::KDTREE:
      timer_.StartTimer("Build KdTree");
      kdtree_->BuildTree(target_cloud);
      timer_.StopTimer();
      timer_.PrintElapsedTime();
      break;

    case SearchMethod::VOXEL_MAP:
      timer_.StartTimer("Build VoxelMap");
      voxel_map_->AddPoints(*target_cloud);
      timer_.StopTimer();
      timer_.PrintElapsedTime();
      break;

    default:
      timer_.StartTimer("Build VoxelMap");
      voxel_map_->AddPoints(*target_cloud);
      timer_.StopTimer();
      timer_.PrintElapsedTime();
      break;
  }

  pose_ = Sophus::SE3d(pred_pose.rotation(), pred_pose.translation());
  bool res = false;
  timer_.StartTimer("Align");
  switch (align_method_) {
    case AlignMethod::PCL_ICP:
      res = PclICP();
      break;
    case AlignMethod::SVD_ICP:
      res = SvdIcp();
      break;
    case AlignMethod::POINT_TO_POINT_ICP:
      res = PointToPointICP();
      break;
    case AlignMethod::POINT_TO_LINE_ICP:
      res = PointToLineICP();
      break;
    case AlignMethod::POINT_TO_PLANE_ICP:
      res = PointToPlaneICP();
      break;
    case AlignMethod::GENERAL_ICP:
      res = GeneralICP();
      break;
    default:
      res = GeneralICP();
      break;
  }
  timer_.StopTimer();
  timer_.PrintElapsedTime();

  if (res) {
    final_pose->translation() = pose_.translation();
    final_pose->linear() = pose_.so3().matrix();
  }
  return res;
}

bool ClassicICP::SetSourceCloud(const PointCloudPtr& source) {
  if (!source || source->empty()) {
    LOG(FATAL) << "Set Target Point Cloud Failed, Beacuse of NULL !";
    return false;
  }
  source_cloud_ = std::make_shared<PointCloud>(source);
  source_center_ = source_cloud_->ComputeCentroid();
  LOG(INFO) << "Source Cloud Center Is " << source_center_.transpose();
  return true;
}

bool ClassicICP::SetTargetCloud(const PointCloudPtr& target) {
  if (!target || target->empty()) {
    LOG(FATAL) << "Set Target Point Cloud Failed, Beacuse of NULL !";
    return false;
  }
  target_cloud_ = std::make_shared<PointCloud>(target);
  target_center_ = target_cloud_->ComputeCentroid();
  LOG(INFO) << "Target Cloud Center Is " << target_center_.transpose();
  return true;
}

bool ClassicICP::SvdIcp() {
  // 使用SVD方法进行匹配，但是这个感觉意义不是很大，可以先使用聚类的方法，即将粗略对应关系找好，然后在进行匹配
  // final_pose->translation() = pred_pose.translation();
  return true;
}

bool ClassicICP::PointToLineICP() {
  CHECK_NOTNULL(source_cloud_);
  CHECK_NOTNULL(target_cloud_);
  LOG(INFO) << "Processing Multi-threaded Point to Line ICP";
  // 对点的索引，预先生成
  std::vector<int32_t> index(source_cloud_->size());
  for (uint32_t i = 0; i < index.size(); ++i) {
    index.at(i) = i;
  }
  int32_t effective_nums = 0;
  for (int32_t iter = 0; iter < max_iters_; ++iter) {
    // 并发，统计有效点，统计各个块的雅可比矩阵和残差，最终都是要加在一起的
    // 用空间换时间，不能在for_each中进行迭代，锁很耗时
    std::vector<bool> effect_points(index.size(), false);
    std::vector<Eigen::Matrix<double, 3, 6>> jacobians(index.size());
    std::vector<Eigen::Vector3d> errs(index.size());
    // 这里使用高斯牛顿，也可以使用其他优化方法，这里应该可以定义成自己写的优化类
    // 这里的操作是针对每个点的
    std::for_each(std::execution::par_unseq, index.begin(), index.end(), [&](const int32_t idx) {
      // 1. 首先根据上一步得到的位姿变换点，将点变换到对应位置
      auto pt = source_cloud_->at(idx);
      Eigen::Vector3d transformed_pt = pose_ * pt;
      // 2. 使用最近邻方法搜素最近点
      bool searched = false;
      std::vector<Eigen::Vector3d> searched_pts;
      std::vector<std::pair<uint32_t, double>> kdtree_res;
      std::vector<std::pair<Eigen::Vector3d, double>> voxelmap_res;
      switch (search_method_) {
        case SearchMethod::KDTREE:
          kdtree_->GetClosestPoint(transformed_pt, &kdtree_res, 5);
          if (kdtree_res.size() > 3) {
            for (uint32_t i = 0u; i < kdtree_res.size(); ++i) {
              searched_pts.emplace_back(target_cloud_->at(kdtree_res.at(i).first));
            }
            searched = true;
          }
          break;
        case SearchMethod::VOXEL_MAP:
          voxel_map_->GetClosestNeighbor(transformed_pt, &voxelmap_res, 5);
          if (voxelmap_res.size() > 3) {
            for (uint32_t i = 0u; i < voxelmap_res.size(); ++i) {
              searched_pts.emplace_back(voxelmap_res.at(i).first);
            }
            searched = true;
          }
          break;
      }

      Eigen::Vector3d err = Eigen::Vector3d::Zero();

      Eigen::Vector3d coeffs = Eigen::Vector3d::Zero();
      Eigen::Vector3d dis = Eigen::Vector3d::Zero();
      if (searched && Utils::Math::Line3DFit(searched_pts, &coeffs, &dis)) {
        err = Sophus::SO3d::hat(dis) * (transformed_pt - coeffs);
        if (err.norm() > max_point_line_distance_) {
          return;
        }
      } else {
        // 平面拟合失败！
        return;
      }
      effect_points.at(idx) = true;
      // 4. 计算雅可比矩阵和残差
      // 观测量维度 x 状态量维度, 因为点到面的距离是一维标量
      Eigen::Matrix<double, 3, 6> jac = Eigen::Matrix<double, 3, 6>::Zero();
      jac.block<3, 3>(0, 0) =
          -Sophus::SO3d::hat(dis) * pose_.so3().matrix() * Sophus::SO3d::hat(pt);
      jac.block<3, 3>(0, 3) = Sophus::SO3d::hat(dis);
      // 将结果都保存起来，用空间换时间，否则可能需要加上线程锁，这样实际上是很耗时的
      jacobians.at(idx) = jac;
      errs.at(idx) = err;
    });

    Eigen::Matrix<double, 6, 6> sum_hessian = Eigen::Matrix<double, 6, 6>::Zero();
    Eigen::Matrix<double, 6, 1> sum_b = Eigen::Matrix<double, 6, 1>::Zero();
    effective_nums = 0;
    double sum_errs = 0.0;
    // 5. 整理之前并发得到的结果，需要叠加和计算
    for (uint32_t i = 0; i < index.size(); ++i) {
      if (!effect_points.at(i)) {
        continue;
      }
      ++effective_nums;
      sum_hessian += jacobians.at(i).transpose() * jacobians.at(i);
      sum_b -= jacobians.at(i).transpose() * errs.at(i);
      sum_errs += errs.at(i).dot(errs.at(i));
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
  LOG(INFO) << "Effect Points Nums is " << effective_nums;
  return true;
}

bool ClassicICP::PointToPlaneICP() {
  CHECK_NOTNULL(source_cloud_);
  CHECK_NOTNULL(target_cloud_);
  LOG(INFO) << "Processing Multi-threaded Point to Plane ICP";
  // 对点的索引，预先生成
  std::vector<int32_t> index(source_cloud_->size());
  for (uint32_t i = 0; i < index.size(); ++i) {
    index.at(i) = i;
  }

  int32_t effective_nums = 0;
  for (int32_t iter = 0; iter < max_iters_; ++iter) {
    // 并发，统计有效点，统计各个块的雅可比矩阵和残差，最终都是要加在一起的
    // 用空间换时间，不能在for_each中进行迭代，锁很耗时
    std::vector<bool> effect_points(index.size(), false);
    std::vector<Eigen::Matrix<double, 1, 6>> jacobians(index.size());
    std::vector<double> errs(index.size());
    // 这里使用高斯牛顿，也可以使用其他优化方法，这里应该可以定义成自己写的优化类
    // 这里的操作是针对每个点的
    std::for_each(std::execution::par_unseq, index.begin(), index.end(), [&](const int32_t idx) {
      // 1. 首先根据上一步得到的位姿变换点，将点变换到对应位置
      auto pt = source_cloud_->at(idx);
      Eigen::Vector3d transformed_pt = pose_ * pt;
      // 2. 使用最近邻方法搜素最近点
      bool searched = false;
      std::vector<Eigen::Vector3d> searched_pts;
      std::vector<std::pair<uint32_t, double>> kdtree_res;
      std::vector<std::pair<Eigen::Vector3d, double>> voxelmap_res;
      switch (search_method_) {
        case SearchMethod::KDTREE:
          // 注意这里需要搜索5个点，这样才能拟合平面
          kdtree_->GetClosestPoint(transformed_pt, &kdtree_res, 5);
          // 以为只有超过三个点才能拟合一个平面
          if (kdtree_res.size() > 3) {
            for (uint32_t i = 0u; i < kdtree_res.size(); ++i) {
              searched_pts.emplace_back(target_cloud_->at(kdtree_res.at(i).first));
            }
            searched = true;
          }
          break;
        case SearchMethod::VOXEL_MAP:
          // 注意这里需要搜索5个点，这样才能拟合平面
          voxel_map_->GetClosestNeighbor(transformed_pt, &voxelmap_res, 5);
          if (voxelmap_res.size() > 3) {
            for (uint32_t i = 0u; i < voxelmap_res.size(); ++i) {
              searched_pts.emplace_back(voxelmap_res.at(i).first);
            }
            searched = true;
          }
          break;
      }

      double dis = 0.0;
      Eigen::Vector4d coeffs = Eigen::Vector4d::Zero();
      if (searched && Utils::Math::PlaneFit(searched_pts, &coeffs)) {
        dis = coeffs.head<3>().dot(transformed_pt) + coeffs[3];
        if (fabs(dis) > max_point_plane_distance_) {
          return;
        }
      } else {
        // 平面拟合失败！
        return;
      }
      effect_points.at(idx) = true;

      // 4. 计算雅可比矩阵和残差
      // 观测量维度 x 状态量维度, 因为点到面的距离是一维标量
      Eigen::Matrix<double, 1, 6> jac = Eigen::Matrix<double, 1, 6>::Zero();
      jac.block<1, 3>(0, 0) =
          -coeffs.head<3>().transpose() * pose_.so3().matrix() * Sophus::SO3d::hat(pt);
      jac.block<1, 3>(0, 3) = coeffs.head<3>().transpose();
      // 将结果都保存起来，用空间换时间，否则可能需要加上线程锁，这样实际上是很耗时的
      jacobians.at(idx) = jac;
      errs.at(idx) = dis;
    });

    Eigen::Matrix<double, 6, 6> sum_hessian = Eigen::Matrix<double, 6, 6>::Zero();
    Eigen::Matrix<double, 6, 1> sum_b = Eigen::Matrix<double, 6, 1>::Zero();
    effective_nums = 0;
    double sum_errs = 0.0;
    // 5. 整理之前并发得到的结果，需要叠加和计算
    for (uint32_t i = 0; i < index.size(); ++i) {
      if (!effect_points.at(i)) {
        continue;
      }
      ++effective_nums;
      sum_hessian += jacobians.at(i).transpose() * jacobians.at(i);
      sum_b -= jacobians.at(i).transpose() * errs.at(i);
      sum_errs += errs.at(i) * errs.at(i);
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
  LOG(INFO) << "Effect Points Nums is " << effective_nums;
  return true;
}

bool ClassicICP::GeneralICP() {
  return true;
}

bool ClassicICP::PclICP() {
  // ---------------------------------------------------
  // PCL ICP
  pcl::IterativeClosestPoint<pcl::PointXYZ, pcl::PointXYZ> icp;
  pcl::PointCloud<pcl::PointXYZ>::Ptr filtered_source_cloud(new pcl::PointCloud<pcl::PointXYZ>);
  pcl::PointCloud<pcl::PointXYZ>::Ptr filtered_target_cloud(new pcl::PointCloud<pcl::PointXYZ>);
  filter_->Filter(source_cloud_->ToPCLPointCloud(), filtered_source_cloud.get());
  filter_->Filter(target_cloud_->ToPCLPointCloud(), filtered_target_cloud.get());

  icp.setInputSource(filtered_source_cloud);
  icp.setInputTarget(filtered_target_cloud);

  icp.setInputSource(source_cloud_->ToPCLPointCloud());
  icp.setInputTarget(target_cloud_->ToPCLPointCloud());

  pcl::PointCloud<pcl::PointXYZ>::Ptr aligned(new pcl::PointCloud<pcl::PointXYZ>);
  icp.align(*aligned, pose_.matrix().cast<float>());

  if (icp.hasConverged()) {
    LOG(INFO) << "ICP has converged, score: " << icp.getFitnessScore();
  } else {
    LOG(FATAL) << "ICP did not converge.";
    return false;
  }
  return true;
}

bool ClassicICP::PointToPointICP() {
  CHECK_NOTNULL(source_cloud_);
  CHECK_NOTNULL(target_cloud_);
  LOG(INFO) << "Processing Multi-threaded Point to Point ICP";
  // 对点的索引，预先生成
  std::vector<int32_t> index(source_cloud_->size());
  for (uint32_t i = 0; i < index.size(); ++i) {
    index.at(i) = i;
  }

  int32_t effective_nums = 0;
  for (int32_t iter = 0; iter < max_iters_; ++iter) {
    // 并发，统计有效点，统计各个块的雅可比矩阵和残差，最终都是要加在一起的
    // 用空间换时间，不能在for_each中进行迭代，锁很耗时
    std::vector<bool> effect_points(index.size(), false);
    std::vector<Eigen::Matrix<double, 3, 6>> jacobians(index.size());
    std::vector<Eigen::Vector3d> errs(index.size());
    // 这里使用高斯牛顿，也可以使用其他优化方法，这里应该可以定义成自己写的优化类
    // 这里的操作是针对每个点的
    std::for_each(std::execution::par_unseq, index.begin(), index.end(), [&](const int32_t idx) {
      // 1. 首先根据上一步得到的位姿变换点，将点变换到对应位置
      auto pt = source_cloud_->at(idx);
      Eigen::Vector3d transformed_pt = pose_ * pt;
      // 2. 使用最近邻方法搜素最近点
      bool searched = false;
      Eigen::Vector3d searched_pt = Eigen::Vector3d::Zero();
      double dis_square = 0.0;
      std::vector<std::pair<uint32_t, double>> kdtree_res;
      std::vector<std::pair<Eigen::Vector3d, double>> voxelmap_res;
      switch (search_method_) {
        case SearchMethod::KDTREE:
          kdtree_->GetClosestPoint(transformed_pt, &kdtree_res);
          if (!kdtree_res.empty()) {
            searched_pt = target_cloud_->at(kdtree_res.front().first);
            dis_square = kdtree_res.front().second;
            searched = true;
          }
          break;
        case SearchMethod::VOXEL_MAP:
          voxel_map_->GetClosestNeighbor(transformed_pt, &voxelmap_res);
          if (!voxelmap_res.empty()) {
            searched_pt = voxelmap_res.front().first;
            dis_square = voxelmap_res.front().second;
            searched = true;
          }
          break;
      }

      // 3. 筛除异常数据
      if (!searched || dis_square > max_point_point_distance_) {
        // for_each中并行处理数据的时候，使用return替代for循环中continue的操作
        return;
      }
      effect_points.at(idx) = true;

      // 4. 计算雅可比矩阵和残差
      Eigen::Vector3d err = searched_pt - transformed_pt;
      // 观测量维度 x 状态量维度
      Eigen::Matrix<double, 3, 6> jac = Eigen::Matrix<double, 3, 6>::Zero();
      jac.block<3, 3>(0, 0) = pose_.so3().matrix() * Sophus::SO3d::hat(pt);
      jac.block<3, 3>(0, 3) = -Eigen::Matrix3d::Identity();

      // 将结果都保存起来，用空间换时间，否则可能需要加上线程锁，这样实际上是很耗时的
      jacobians.at(idx) = jac;
      errs.at(idx) = err;
    });

    Eigen::Matrix<double, 6, 6> sum_hessian = Eigen::Matrix<double, 6, 6>::Zero();
    Eigen::Matrix<double, 6, 1> sum_b = Eigen::Matrix<double, 6, 1>::Zero();
    effective_nums = 0;
    double sum_errs = 0.0;
    // 5. 整理之前并发得到的结果，需要叠加和计算
    for (uint32_t i = 0; i < index.size(); ++i) {
      if (!effect_points.at(i)) {
        continue;
      }
      ++effective_nums;
      sum_hessian += jacobians.at(i).transpose() * jacobians.at(i);
      sum_b -= jacobians.at(i).transpose() * errs.at(i);
      sum_errs += errs.at(i).dot(errs.at(i));
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
  LOG(INFO) << "Effect Points Nums is " << effective_nums;
  return true;
}

}  // namespace Lidar
