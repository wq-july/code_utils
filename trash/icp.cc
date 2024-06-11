// 单线程版本的icp，代码被弃用了！
bool ICP::PointToPointICP() {
  CHECK_NOTNULL(source_cloud_);
  CHECK_NOTNULL(target_cloud_);

  LOG(INFO) << "Processing Single Thread Point to Point ICP";
  for (int32_t iter = 0; iter < max_iters_; ++iter) {
    Eigen::Matrix<double, 6, 6> sum_hessian = Eigen::Matrix<double, 6, 6>::Zero();
    Eigen::Matrix<double, 6, 1> sum_b = Eigen::Matrix<double, 6, 1>::Zero();
    int32_t effective_nums = 0;
    double sum_errs = 0.0;

    for (uint32_t i = 0u; i < source_cloud_->size(); ++i) {
      // 1. 首先根据上一步得到的位姿变换点，将点变换到对应位置
      auto pt = source_cloud_->at(i);
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
            searched_pt = source_cloud_->at(kdtree_res.front().first);
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
      if (!searched || dis_square > max_distance_) {
        continue;
      }

      // 4. 计算雅可比矩阵和残差
      Eigen::Vector3d err = searched_pt - transformed_pt;
      // 观测量维度 x 状态量维度
      Eigen::Matrix<double, 3, 6> jac = Eigen::Matrix<double, 3, 6>::Zero();
      jac.block<3, 3>(0, 0) = pose_.so3().matrix() * Sophus::SO3d::hat(pt);
      jac.block<3, 3>(0, 3) = -Eigen::Matrix3d::Identity();
      ++effective_nums;
      sum_hessian += jac.transpose() * jac;
      sum_b -= jac.transpose() * err;
      sum_errs += err.dot(err);
    }

    if (effective_nums < min_effect_points_) {
      LOG(ERROR) << "Effective num is too small: " << effective_nums;
      return false;
    }
    // 6. 计算求解
    Eigen::Matrix<double, 6, 1> dx =
        Optimizer::NonlinearOptimizer::Solver(sum_hessian, sum_b, Optimizer::SolverType::INVERSE);
    pose_.so3() = pose_.so3() * Sophus::SO3d::exp(dx.head<3>());
    pose_.translation() += dx.tail<3>();
    if (dx.norm() < 1.0e-6) {
      LOG(INFO) << "converged, dx = " << dx.transpose();
      break;
    }
  }
  return true;
}