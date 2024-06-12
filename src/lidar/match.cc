#include "lidar/match.h"

#include "glog/logging.h"
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

Matcher::Matcher(const double voxel_size,
                 const double max_map_distance,
                 const int32_t max_num_per_voxel,
                 const AlignMethod align_method,
                 const SearchMethod search_method,
                 const int32_t max_iters,
                 const double break_dx,
                 const double outlier_th,
                 const int32_t min_effect_points,
                 const bool use_downsample)
    : align_method_(align_method),
      search_method_(search_method),
      max_iters_(max_iters),
      break_dx_(break_dx),
      min_effect_points_(min_effect_points),
      use_downsample_(use_downsample),
      outlier_th_(outlier_th) {
  kdtree_ = std::make_shared<Common::KdTree>();
  voxel_map_ = std::make_shared<Common::VoxelMap>(voxel_size, max_map_distance, max_num_per_voxel);
  filter_ = std::make_shared<Lidar::PointCloudFilter>(0.3);
}

bool Matcher::Align(const Eigen::Isometry3d& pred_pose,
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
  res = GeneralMatch(align_method_);
  timer_.StopTimer();
  timer_.PrintElapsedTime();

  if (res) {
    final_pose->translation() = pose_.translation();
    final_pose->linear() = pose_.so3().matrix();
  }
  return res;
}

bool Matcher::SetSourceCloud(const PointCloudPtr& source) {
  if (!source || source->empty()) {
    LOG(FATAL) << "Set Target Point Cloud Failed, Beacuse of NULL !";
    return false;
  }
  source_cloud_ = std::make_shared<PointCloud>(source);
  source_center_ = source_cloud_->ComputeCentroid();
  LOG(INFO) << "Source Cloud Center Is " << source_center_.transpose();
  return true;
}

bool Matcher::SetTargetCloud(const PointCloudPtr& target) {
  if (!target || target->empty()) {
    LOG(FATAL) << "Set Target Point Cloud Failed, Beacuse of NULL !";
    return false;
  }
  target_cloud_ = std::make_shared<PointCloud>(target);
  target_center_ = target_cloud_->ComputeCentroid();
  LOG(INFO) << "Target Cloud Center Is " << target_center_.transpose();
  return true;
}

bool Matcher::KnnSearch(const SearchMethod search_method,
                        const Eigen::Vector3d& pt,
                        const int32_t k_nums,
                        std::vector<std::pair<Eigen::Vector3d, double>>* const res) {
  CHECK_NOTNULL(res);
  std::vector<std::pair<uint32_t, double>> kdtree_res;
  switch (search_method_) {
    case SearchMethod::KDTREE:
      kdtree_->GetClosestPoint(pt, &kdtree_res, k_nums);
      for (uint32_t i = 0; i < kdtree_res.size(); ++i) {
        res->emplace_back(
            std::make_pair(target_cloud_->at(kdtree_res.at(i).first), kdtree_res.at(i).second));
      }
      break;
    case SearchMethod::VOXEL_MAP:
      voxel_map_->GetClosestNeighbor(pt, res, k_nums);
      break;
    default:
      voxel_map_->GetClosestNeighbor(pt, res, k_nums);
      break;
  }
  if (res->empty()) {
    return false;
  }
  return true;
}

bool Matcher::GeneralMatch(const AlignMethod match_method) {
  CHECK_NOTNULL(source_cloud_);
  CHECK_NOTNULL(target_cloud_);

  std::vector<int32_t> index(source_cloud_->size());
  for (uint32_t i = 0; i < index.size(); ++i) {
    index.at(i) = i;
  }

  int32_t effective_nums = 0;
  int32_t nearby_size = (match_method == AlignMethod::NDT) ? 7 : 1;
  for (int32_t iter = 0; iter < max_iters_; ++iter) {
    int32_t total_size = index.size() * nearby_size;
    std::vector<bool> effect_points(total_size, false);
    std::vector<Eigen::Matrix<double, 3, 6>> jacobians_3d(total_size);
    std::vector<Eigen::Matrix<double, 1, 6>> jacobians_1d(total_size);
    std::vector<Eigen::Vector3d> errs_3d(total_size);
    std::vector<double> errs_1d(total_size);
    std::vector<Eigen::Matrix3d> infos(total_size);

    std::for_each(std::execution::par_unseq, index.begin(), index.end(), [&](const int32_t idx) {
      auto pt = source_cloud_->at(idx);
      Eigen::Vector3d transformed_pt = pose_ * pt;
      std::vector<Eigen::Vector3d> searched_pts;
      std::vector<std::pair<Eigen::Vector3d, double>> search_res;
      bool searched = false;

      switch (match_method) {
        case AlignMethod::POINT_TO_POINT_ICP: {
          searched = KnnSearch(search_method_, transformed_pt, 1, &search_res);
          if (!searched) return;
          if (search_res.front().second > max_point_point_distance_) return;
          Eigen::Vector3d searched_pt = search_res.front().first;
          Eigen::Vector3d err = searched_pt - transformed_pt;
          Eigen::Matrix<double, 3, 6> jac = Eigen::Matrix<double, 3, 6>::Zero();
          jac.block<3, 3>(0, 0) = pose_.so3().matrix() * Sophus::SO3d::hat(pt);
          jac.block<3, 3>(0, 3) = -Eigen::Matrix3d::Identity();
          effect_points.at(idx) = true;
          jacobians_3d.at(idx) = jac;
          errs_3d.at(idx) = err;
          break;
        }
        case AlignMethod::POINT_TO_LINE_ICP: {
          searched = KnnSearch(search_method_, transformed_pt, 5, &search_res);
          if (!searched) return;
          for (const auto& res : search_res) {
            searched_pts.emplace_back(res.first);
          }
          Eigen::Vector3d coeffs, dis;
          if (Utils::Math::Line3DFit(searched_pts, &coeffs, &dis)) {
            Eigen::Vector3d err = Sophus::SO3d::hat(dis) * (transformed_pt - coeffs);
            if (err.norm() > max_point_line_distance_) return;
            Eigen::Matrix<double, 3, 6> jac = Eigen::Matrix<double, 3, 6>::Zero();
            jac.block<3, 3>(0, 0) =
                -Sophus::SO3d::hat(dis) * pose_.so3().matrix() * Sophus::SO3d::hat(pt);
            jac.block<3, 3>(0, 3) = Sophus::SO3d::hat(dis);
            effect_points.at(idx) = true;
            jacobians_3d.at(idx) = jac;
            errs_3d.at(idx) = err;
          }
          break;
        }
        case AlignMethod::POINT_TO_PLANE_ICP: {
          searched = KnnSearch(search_method_, transformed_pt, 5, &search_res);
          if (!searched) return;
          for (const auto& res : search_res) {
            searched_pts.emplace_back(res.first);
          }
          Eigen::Vector4d coeffs;
          if (Utils::Math::PlaneFit(searched_pts, &coeffs)) {
            double dis = coeffs.head<3>().dot(transformed_pt) + coeffs[3];
            if (fabs(dis) > max_point_plane_distance_) return;
            Eigen::Matrix<double, 1, 6> jac = Eigen::Matrix<double, 1, 6>::Zero();
            jac.block<1, 3>(0, 0) =
                -coeffs.head<3>().transpose() * pose_.so3().matrix() * Sophus::SO3d::hat(pt);
            jac.block<1, 3>(0, 3) = coeffs.head<3>().transpose();
            effect_points.at(idx) = true;
            jacobians_1d.at(idx) = jac;
            errs_1d.at(idx) = dis;
          }
          break;
        }
        case AlignMethod::NDT: {
          std::vector<Common::GaussianVoxel> nearby_voxels;
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
            jacobians_3d.at(real_idx) = jac;
            errs_3d.at(real_idx) = err;
            infos.at(real_idx) = nearby_voxels.at(i).inv_cov_;
            effect_points.at(real_idx) = true;
          }
          break;
        }
      }
    });

    Eigen::Matrix<double, 6, 6> sum_hessian = Eigen::Matrix<double, 6, 6>::Zero();
    Eigen::Matrix<double, 6, 1> sum_b = Eigen::Matrix<double, 6, 1>::Zero();
    effective_nums = 0;
    double sum_errs = 0.0;

    for (int32_t i = 0; i < total_size; ++i) {
      if (!effect_points.at(i)) continue;
      ++effective_nums;
      if (match_method == AlignMethod::POINT_TO_POINT_ICP ||
          match_method == AlignMethod::POINT_TO_LINE_ICP) {
        sum_hessian += jacobians_3d.at(i).transpose() * jacobians_3d.at(i);
        sum_b -= jacobians_3d.at(i).transpose() * errs_3d.at(i);
        sum_errs += errs_3d.at(i).dot(errs_3d.at(i));
      } else if (match_method == AlignMethod::POINT_TO_PLANE_ICP) {
        sum_hessian += jacobians_1d.at(i).transpose() * jacobians_1d.at(i);
        sum_b -= jacobians_1d.at(i).transpose() * errs_1d.at(i);
        sum_errs += errs_1d.at(i) * errs_1d.at(i);
      } else if (match_method == AlignMethod::NDT) {
        // 可能还是需要找一下bug，本该乘上的信息矩阵，结果效果并不好，不如乘上单位矩阵？
        // sum_hessian += jacobians.at(i).transpose() * infos.at(i) * jacobians.at(i);
        // sum_b -= jacobians.at(i).transpose() * infos.at(i) * errs.at(i);
        // sum_errs += errs.at(i).transpose() * infos.at(i) * errs.at(i);
        sum_hessian += jacobians_3d.at(i).transpose() * jacobians_3d.at(i);
        sum_b -= jacobians_3d.at(i).transpose() * errs_3d.at(i);
        sum_errs += errs_3d.at(i).transpose() * errs_3d.at(i);
      }
    }

    int32_t effect_nums = effective_nums / nearby_size;
    if (effect_nums < min_effect_points_) {
      LOG(FATAL) << "effective num too small: " << effect_nums;
      return false;
    }

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
