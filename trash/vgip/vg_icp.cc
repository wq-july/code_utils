/****************************************************************************
 *
 * Copyright (c) 2023 ZelosTech.com, Inc. All Rights Reserved
 *
 ***************************************************************************/
/**
 * @file vg_icp.cc
 **/

#include "localization/matching/registration/vg_icp/vg_icp.h"

#include "Eigen/Geometry"

#include "zlog/logger.h"

namespace zelos {
namespace zoe {
namespace localization {

VGICP::VGICP(const ::zelos::zoe::localization::proto::VGICPConfig& config) {
  config_ = config;
  num_threads_ = config_.num_threads();
  rotation_eps_ = config_.rotation_eps();
  translation_eps_ = config_.translation_eps();
  lambda_ = config_.lambda();
  map_ = std::make_shared<GaussianVoxelMap>(config_.vgicp_map_config());
  voxelnn_ = std::make_shared<VoxelNN>(0.5, NearbyType::NEARBY6);
}

void VGICP::Align(const Eigen::Isometry3d& guess_matrix,
    const pcl::PointCloud<pcl::PointXYZI>::Ptr& input_cloud,
    const std::shared_ptr<GaussianVoxelMap>& gaussian_map,
    Eigen::Isometry3d* const final_matrix) {
  ZCHECK_NOTNULL(input_cloud);
  ZCHECK_NOTNULL(gaussian_map);
  voxelnn_->SetPointCloud(input_cloud);
  *final_matrix = guess_matrix;

  for (int32_t i = 0; i < config_.max_iterations(); ++i) {
    Eigen::Matrix<double, 6, 6> sum_H = Eigen::Matrix<double, 6, 6>::Zero();
    Eigen::Matrix<double, 6, 1> sum_b = Eigen::Matrix<double, 6, 1>::Zero();
    
    int32_t count = 0;

    for (int32_t i = 0; i < voxelnn_->Size(); i++) {
      std::pair<Eigen::Vector3i, double> res;
      Eigen::Vector3d transed_source_pt = (*final_matrix) * voxelnn_->GetCloud()->at(i);
      if (!gaussian_map->NeaestNeighborSearch(transed_source_pt, &res)) {
        continue;
      }
      count++;

      const Eigen::Matrix3d RCR = gaussian_map->GetCov(res.first) +
          final_matrix->rotation() * voxelnn_->GetCloud()->cov(i) * final_matrix->rotation().transpose();

      // std::cout << "res.first is " << res.second << "\n";
      // std::cout << "gaussian_map->GetCov(res.first) is \n" << gaussian_map->GetCov(res.first) << "\n";

      // std::cout << "RCR is \n" << RCR << "\n";

      Eigen::Matrix4d mahalanobis = Eigen::Matrix4d::Zero();
      mahalanobis.block<3, 3>(0, 0) = RCR.inverse();

      // std::cout << "mahalanobis is \n" << mahalanobis << "\n";

      const Eigen::Vector3d tmp_residual = gaussian_map->GetMean(res.first) - transed_source_pt;

      const Eigen::Vector4d residual =
          Eigen::Vector4d(tmp_residual.x(), tmp_residual.y(), tmp_residual.z(), 0.0);

      Eigen::Matrix<double, 4, 6> J = Eigen::Matrix<double, 4, 6>::Zero();
      J.block<3, 3>(0, 0) = final_matrix->linear() * SkewSymmetricMatrix(voxelnn_->GetCloud()->at(i));
      J.block<3, 3>(0, 3) = -final_matrix->linear();

      // std::cout << "voxelnn_->GetCloud()->at(i) is " << voxelnn_->GetCloud()->at(i).transpose() << "\n";
      // std::cout << "SkewSymmetricMatrix(voxelnn_->GetCloud()->at(i)) " << SkewSymmetricMatrix(voxelnn_->GetCloud()->at(i)) << "\n";
      // std::cout << "-final_matrix->linear()  " << -final_matrix->linear() << "\n";

      // std::cout << "J is \n" << J << "\n";
      // std::cout << "J trans is " << J.transpose() << "\n";


      Eigen::Matrix<double, 6, 6> H = J.transpose() * mahalanobis * J;
      Eigen::Matrix<double, 6, 1> b = J.transpose() * mahalanobis * residual;
      // double e = 0.5 * residual.transpose() * mahalanobis * residual;

      // std::cout << "H is \n" << H << "\n";
      // std::cout << "b is " << b.transpose() << "\n";

      sum_H += H;
      sum_b += b;
    }

    std::cout << "sum_H is \n" << sum_H << "\n";
    std::cout << "sum_b is " << sum_b.transpose() << "\n";
    std::cout << "effect nums is " << count << "\n";

    // Solve linear system
    const Eigen::Matrix<double, 6, 1> delta =
        (sum_H + lambda_ * Eigen ::Matrix<double, 6, 6>::Identity()).ldlt().solve(-sum_b);
    std::cout << "delta is " << delta.transpose() << "\n";
    *final_matrix = (*final_matrix) * SE3Exp(delta);

    if (delta.template head<3>().norm() < rotation_eps_ && delta.template tail<3>().norm() < translation_eps_) {
      break;
    }
  }
}

std::shared_ptr<GaussianPointCloud> VGICP::PreProcessPclCloud(const pcl::PointCloud<pcl::PointXYZI>::Ptr& input_cloud) {
  if (!input_cloud) {
    std::cout << "nullptr !";
  }
  voxelnn_->SetPointCloud(input_cloud);
  return voxelnn_->GetCloud();
}

}  // namespace localization
}  // namespace zoe
}  // namespace zelos

/* vim: set expandtab ts=2 sw=2 sts=2 tw=100: */
