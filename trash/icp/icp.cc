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
      std::vector<Eigen::Vector3d> searched_pts;
      std::vector<std::pair<Eigen::Vector3d, double>> search_res;
      bool searched = KnnSearch(search_method_, transformed_pt, 5, &search_res);
      if (searched && search_res.size() > 3) {
        for (const auto& iter : search_res) {
          searched_pts.emplace_back(iter.first);
        }
      } else {
        return;
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
      std::vector<Eigen::Vector3d> searched_pts;
      std::vector<std::pair<Eigen::Vector3d, double>> search_res;
      bool searched = KnnSearch(search_method_, transformed_pt, 5, &search_res);
      if (searched && search_res.size() > 3) {
        for (const auto& iter : search_res) {
          searched_pts.emplace_back(iter.first);
        }
      } else {
        return;
      }

      // 3. 筛除异常数据
      double dis = 0.0;
      Eigen::Vector4d coeffs = Eigen::Vector4d::Zero();
      if (Utils::Math::PlaneFit(searched_pts, &coeffs)) {
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
      Eigen::Vector3d searched_pt = Eigen::Vector3d::Zero();
      std::vector<std::pair<Eigen::Vector3d, double>> search_res;
      bool searched = KnnSearch(search_method_, transformed_pt, 1, &search_res);
      // 3. 筛除异常数据
      if (searched) {
        searched_pt = search_res.front().first;
        if (search_res.front().second > max_point_point_distance_) {
          return;
        }
        effect_points.at(idx) = true;
      } else {
        // for_each中并行处理数据的时候，使用return替代for循环中continue的操作
        return;
      }
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