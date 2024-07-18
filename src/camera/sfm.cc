#include "camera/sfm.h"
namespace Camera {
#define SIGN(X) ((X) >= 0 ? (1) : (-1))

SFM::SFM(const CameraConfig::SFMConfig config) : config_(config) {
  pnp_solver_ = std::make_shared<PnpSolver>(config_.pnp_solver());
  camera_model_ = std::make_shared<CameraBase>(config_.camera_model());
  feature_manager_ = std::make_shared<FeatureManager>(config_.feature_config());
}

double SFM::FindFundamentalMatrix(const std::vector<cv::KeyPoint>& p2ds_1,
                                  const std::vector<cv::KeyPoint>& p2ds_2,
                                  const std::vector<cv::DMatch>& matches,
                                  Eigen::Matrix3d* const fundamental_mat,
                                  std::vector<bool>* const inliers) const {
  CHECK(p2ds_1.size() > 8 && p2ds_2.size() > 8 && matches.size() > 8 && !fundamental_mat)
      << "Check Input !";
  double score = 0.0;
  double sum_err = std::numeric_limits<double>::max();
  if (config_.enable_cv_p2p()) {
    // 相机内参, cv::eigen2cv转换可能存在问题，如果有问题启用
    // auto K = camera_model_->K_;
    // cv::Mat K =
    //     (cv::Mat_<double>(3, 3) << K(0, 0), 0.0, K(0, 2), 0.0, K(1, 1), K(1, 2), 0, 0, 1.0);
    cv::Mat K;
    cv::eigen2cv(camera_model_->K_, K);

    //-- 把匹配点转换为vector<Point2f>的形式
    std::vector<cv::Point2f> points1;
    std::vector<cv::Point2f> points2;

    for (uint32_t i = 0; i < matches.size(); ++i) {
      points1.emplace_back(p2ds_1[matches[i].queryIdx].pt);
      points2.emplace_back(p2ds_2[matches[i].trainIdx].pt);
    }

    //-- 计算基础矩阵
    cv::Mat fundamental_matrix;
    // 8点法
    fundamental_matrix = cv::findFundamentalMat(points1, points2, cv::FM_8POINT);
    cv::cv2eigen(fundamental_matrix, *fundamental_mat);
    score =
        CheckFundamentalMat(p2ds_1, p2ds_2, matches, *fundamental_mat, inliers, config_.sigma());
    sum_err = ProjectError(p2ds_1, p2ds_2, matches, *fundamental_mat, inliers);
  } else {
    // 使用Ransac算法计算最优结果
    // 随机生成器
    std::random_device rd;
    std::mt19937 gen(rd());
    // 初始化索引容器
    std::vector<int32_t> indices(matches.size());
    // 依次递增填充数据
    std::iota(indices.begin(), indices.end(), 0);

    std::vector<bool> inliers;
    for (int32_t iter = 0; iter < config_.ransac_iterations(); ++iter) {
      // 随机选择8个匹配点对
      std::vector<Eigen::Vector2d> points1, points2;
      for (int32_t i = 0; i < 8; ++i) {
        std::uniform_int_distribution<> dis(0, indices.size() - 1);
        int32_t idx = dis(gen);
        // 提取匹配点
        int32_t match_idx = indices[idx];
        points1.emplace_back(p2ds_1[matches[match_idx].queryIdx].pt);
        points2.emplace_back(p2ds_2[matches[match_idx].trainIdx].pt);
        // 移除已选择的索引
        indices.erase(indices.begin() + idx);
      }
      Eigen::Matrix3d Fmat = ComputeFundamentalMat21(points1, points2);
      std::vector<bool> cur_inliers;
      double cur_score =
          CheckFundamentalMat(p2ds_1, p2ds_2, matches, Fmat, &cur_inliers, config_.sigma());
      double cur_sum_err = ProjectError(p2ds_1, p2ds_2, matches, Fmat, &cur_inliers);
      if (cur_score > score && cur_sum_err < sum_err) {
        *fundamental_mat = Fmat;
        inliers = cur_inliers;
        score = cur_score;
        sum_err = cur_sum_err;
      }
    }
    LOG(INFO) << "FindFundamentalMatrix: score = " << score << ", sum_err = " << sum_err;
  }
  return score;
}

bool SFM::FindEssentialMatrix(const std::vector<cv::KeyPoint>& p2ds_1,
                              const std::vector<cv::KeyPoint>& p2ds_2,
                              const std::vector<cv::DMatch>& matches,
                              Eigen::Matrix3d* const essential_mat,
                              std::vector<bool>* const inliers) const {
  CHECK(p2ds_1.size() > 8 && p2ds_2.size() > 8 && matches.size() > 8 && !essential_mat)
      << "Check Input !";
  if (config_.enable_cv_p2p()) {
    auto K = camera_model_->K_mat_;
    //-- 计算本质矩阵
    // cv::Point2d principal_point(K(0, 2), K(1, 2));       // 相机光心
    // double focal_length = camera_model_->focal_length_;  // 相机焦距

    //-- 把匹配点转换为vector<Point2f>的形式
    std::vector<cv::Point2f> points1;
    std::vector<cv::Point2f> points2;

    for (uint32_t i = 0; i < matches.size(); ++i) {
      points1.emplace_back(p2ds_1[matches[i].queryIdx].pt);
      points2.emplace_back(p2ds_2[matches[i].trainIdx].pt);
    }

    cv::Mat E = cv::findEssentialMat(points1, points2, K, cv::RANSAC, 0.999, 1.0);
    cv::cv2eigen(E, *essential_mat);

    // 恢复姿态 (R: 旋转矩阵, t: 平移向量)
    cv::Mat R, t;
    int32_t inliers = cv::recoverPose(E, points1, points2, K, R, t);

    if (inliers < 12) {
      return false;
    }

  } else {
    /* 原理和基础矩阵计算基本一致，两者相差外参， F = K^T * E * K
     * ，代码参考FindFundamentalMatrix函数 */
    LOG(INFO) << "Please Use FindFundamentalMatrix or Enable OpenCV Functions !";
    Eigen::Matrix3d Fmat = Eigen::Matrix3d::Identity();
    // TODO, 统计一下这个得分多少合适，作为一个阈值，返回false
    double score = FindFundamentalMatrix(p2ds_1, p2ds_2, matches, &Fmat, inliers);
    *essential_mat = (camera_model_->K_).transpose() * Fmat * (camera_model_->K_);
    if (score < 1.0) {
      return false;
    }
  }
  return true;
}

double SFM::FindHomography(const std::vector<cv::KeyPoint>& p2ds_1,
                           const std::vector<cv::KeyPoint>& p2ds_2,
                           const std::vector<cv::DMatch>& matches,
                           Eigen::Matrix3d* const homography_mat,
                           std::vector<bool>* const inliers) const {
  CHECK(p2ds_1.size() > 3 && p2ds_2.size() > 3 && matches.size() > 3 && !homography_mat)
      << "Check Input !";
  double score = 0.0;
  if (config_.enable_cv_p2p()) {
    //-- 把匹配点转换为vector<Point2f>的形式
    std::vector<cv::Point2f> points1;
    std::vector<cv::Point2f> points2;
    for (uint32_t i = 0; i < matches.size(); ++i) {
      points1.emplace_back(p2ds_1[matches[i].queryIdx].pt);
      points2.emplace_back(p2ds_2[matches[i].trainIdx].pt);
    }
    //-- 计算单应矩阵
    //-- 但是本例中场景不是平面，单应矩阵意义不大
    cv::Mat homography_matrix;
    homography_matrix = cv::findHomography(points1, points2, cv::RANSAC, config_.reproj_err_th());
    cv::cv2eigen(homography_matrix, *homography_mat);
    score = CheckHomographyMat(p2ds_1, p2ds_2, matches, *homography_mat, inliers, config_.sigma());
  } else {
    // 使用Ransac算法计算最优结果
    // 随机生成器
    std::random_device rd;
    std::mt19937 gen(rd());
    // 初始化索引容器
    std::vector<int32_t> indices(matches.size());
    // 依次递增填充数据
    std::iota(indices.begin(), indices.end(), 0);

    for (int32_t iter = 0; iter < config_.ransac_iterations(); ++iter) {
      // 随机选择8个匹配点对
      std::vector<Eigen::Vector2d> points1, points2;
      for (int32_t i = 0; i < 8; ++i) {
        std::uniform_int_distribution<> dis(0, indices.size() - 1);
        int32_t idx = dis(gen);
        // 提取匹配点
        int32_t match_idx = indices[idx];
        points1.emplace_back(p2ds_1[matches[match_idx].queryIdx].pt);
        points2.emplace_back(p2ds_2[matches[match_idx].trainIdx].pt);
        // 移除已选择的索引
        indices.erase(indices.begin() + idx);
      }
      Eigen::Matrix3d Hmat = ComputeHomographyMat21(points1, points2);
      std::vector<bool> cur_inliers;
      double cur_score =
          CheckHomographyMat(p2ds_1, p2ds_2, matches, Hmat, &cur_inliers, config_.sigma());

      if (cur_score > score) {
        *homography_mat = Hmat;
        *inliers = cur_inliers;
        score = cur_score;
      }
    }
  }
  return score;
}

// 分解得到四组解，需要进一步进行判断，ORB-SLAM中是对这几组解进行三角化，以三角化点最多的那组解为正解
bool SFM::DecomposeEssentialMatrix(const Eigen::Matrix3d& essential_mat,
                                   std::vector<Sophus::SE3d>* const relative_pose) const {
  Eigen::Matrix3d R1 = Eigen::Matrix3d::Identity();
  Eigen::Matrix3d R2 = Eigen::Matrix3d::Identity();
  Eigen::Vector3d t = Eigen::Vector3d::Zero();
  DecomposeE(essential_mat, &R1, &R2, &t);
  Sophus::SE3d solution1(R1, t);
  Sophus::SE3d solution2(R1, -t);
  Sophus::SE3d solution3(R2, t);
  Sophus::SE3d solution4(R2, -t);
  relative_pose->emplace_back(solution1);
  relative_pose->emplace_back(solution2);
  relative_pose->emplace_back(solution3);
  relative_pose->emplace_back(solution4);
  return true;
}

// 分解得到多组解，需要进一步判断
bool SFM::DecomposeHomographyMatrix(
    const Eigen::Matrix3d& homography_mat,
    std::vector<std::tuple<Eigen::Matrix3d, Eigen::Vector3d, Eigen::Vector3d>>* const relative_pose)
    const {
  CHECK(!relative_pose) << "Nullptr !";
  if (config_.enable_cv_p2p()) {
    // 分解单应矩阵
    std::vector<cv::Mat> Rs, ts, normals;
    cv::Mat H;
    cv::eigen2cv(homography_mat, H);
    cv::decomposeHomographyMat(H, camera_model_->K_mat_, Rs, ts, normals);
  } else {
    switch (config_.decompose_h_method()) {
      case CameraConfig::SFMConfig::DecomposeHMethod::SFMConfig_DecomposeHMethod_ZHANG:
        DecomposeH_Zhang(homography_mat, camera_model_->K_, relative_pose);
        break;

      case CameraConfig::SFMConfig::DecomposeHMethod::SFMConfig_DecomposeHMethod_Ezio_Malis:
        DecomposeH_EM(homography_mat, camera_model_->K_, relative_pose);
        break;

      default:
        DecomposeH_Zhang(homography_mat, camera_model_->K_, relative_pose);
        break;
    }
  }
  // TODO, 简单检验一下RT，可以计算一下重投影误差，粗略筛选一下合适的解；
  return true;
}

/**
 * @brief 从特征点匹配求homography（normalized DLT）
 * |x'|     | h1 h2 h3 ||x|
 * |y'| = a | h4 h5 h6 ||y|  简写: x' = a H x, a为一个尺度因子
 * |1 |     | h7 h8 h9 ||1|
 * 使用DLT(direct linear tranform)求解该模型
 * x' = a H x
 * ---> (x') 叉乘 (H x)  = 0
 * ---> Ah = 0
 * A = | 0  0  0 -x -y -1 xy' yy' y'|  h = | h1 h2 h3 h4 h5 h6 h7 h8 h9 |
 *     |-x -y -1  0  0  0 xx' yx' x'|
 * 通过SVD求解Ah = 0，A'A最小特征值对应的特征向量即为解
 * @param  p1s 归一化后的点, in reference frame
 * @param  p2s 归一化后的点, in current frame
 * @return     单应矩阵
 * @see        Multiple View Geometry in Computer Vision - Algorithm 4.2 p109
 */
Eigen::Matrix3d SFM::ComputeHomographyMat21(const std::vector<Eigen::Vector2d>& p1s,
                                            const std::vector<Eigen::Vector2d>& p2s) const {
  // Normalize coordinates
  std::vector<Eigen::Vector2d> pts1, pts2;
  Eigen::Matrix3d T1, T2;
  Normalize(p1s, &pts1, &T1);
  Normalize(p2s, &pts2, &T2);
  Eigen::Matrix3d T2t = T2.transpose();

  const int32_t N = p1s.size();
  Eigen::MatrixXd A(2 * N, 9);

  for (int32_t i = 0; i < N; i++) {
    const double u1 = pts1[i].x();
    const double v1 = pts1[i].y();
    const double u2 = pts2[i].x();
    const double v2 = pts2[i].y();

    A(2 * i, 0) = 0.0;
    A(2 * i, 1) = 0.0;
    A(2 * i, 2) = 0.0;
    A(2 * i, 3) = -u1;
    A(2 * i, 4) = -v1;
    A(2 * i, 5) = -1;
    A(2 * i, 6) = v2 * u1;
    A(2 * i, 7) = v2 * v1;
    A(2 * i, 8) = v2;

    A(2 * i + 1, 0) = u1;
    A(2 * i + 1, 1) = v1;
    A(2 * i + 1, 2) = 1;
    A(2 * i + 1, 3) = 0.0;
    A(2 * i + 1, 4) = 0.0;
    A(2 * i + 1, 5) = 0.0;
    A(2 * i + 1, 6) = -u2 * u1;
    A(2 * i + 1, 7) = -u2 * v1;
    A(2 * i + 1, 8) = -u2;
  }

  Eigen::JacobiSVD<Eigen::MatrixXd> svd(A, Eigen::ComputeFullV);

  Eigen::Matrix<double, 3, 3, Eigen::RowMajor> H(svd.matrixV().col(8).data());

  return T2t * H * T1;
}

/**
 * @brief 从特征点匹配求fundamental matrix（normalized 8点法）
 * x'Fx = 0 整理可得：Af = 0
 * A = | x'x x'y x' y'x y'y y' x y 1 |, f = | f1 f2 f3 f4 f5 f6 f7 f8 f9 |
 * 通过SVD求解Af = 0，A'A最小特征值对应的特征向量即为解
 * @param  p1s 归一化后的点, in reference frame
 * @param  p2s 归一化后的点, in current frame
 * @return     基础矩阵
 * @see        Multiple View Geometry in Computer Vision - Algorithm 11.1 p282 (中文版 p191)
 */
Eigen::Matrix3d SFM::ComputeFundamentalMat21(const std::vector<Eigen::Vector2d>& p1s,
                                             const std::vector<Eigen::Vector2d>& p2s) const {
  // Normalize coordinates
  std::vector<Eigen::Vector2d> pts1, pts2;
  Eigen::Matrix3d T1, T2;
  Normalize(p1s, &pts1, &T1);
  Normalize(p2s, &pts2, &T2);
  Eigen::Matrix3d T2t = T2.transpose();

  const int32_t N = p1s.size();
  Eigen::MatrixXd A(N, 9);
  for (int32_t i = 0; i < N; i++) {
    const double u1 = pts1[i].x();
    const double v1 = pts1[i].y();
    const double u2 = pts2[i].x();
    const double v2 = pts2[i].y();

    A(i, 0) = u2 * u1;
    A(i, 1) = u2 * v1;
    A(i, 2) = u2;
    A(i, 3) = v2 * u1;
    A(i, 4) = v2 * v1;
    A(i, 5) = v2;
    A(i, 6) = u1;
    A(i, 7) = v1;
    A(i, 8) = 1;
  }

  Eigen::JacobiSVD<Eigen::MatrixXd> svd(A, Eigen::ComputeFullU | Eigen::ComputeFullV);
  Eigen::Matrix<double, 3, 3, Eigen::RowMajor> Fpre(svd.matrixV().col(8).data());

  Eigen::JacobiSVD<Eigen::Matrix3d> svd2(Fpre, Eigen::ComputeFullU | Eigen::ComputeFullV);
  Eigen::Vector3d w = svd2.singularValues();
  // 这里注意计算完要强制让第三个奇异值为0
  w(2) = 0;

  auto fundamental_mat =
      T2t * svd2.matrixU() * Eigen::DiagonalMatrix<double, 3>(w) * svd2.matrixV().transpose() * T1;
  return fundamental_mat;
}

/**
 * @brief 像素坐标标准化，计算点集的横纵均值，与均值偏差的均值。最后返回的是变化矩阵T
 * 直接乘以像素坐标的齐次向量即可获得去中心去均值后的特征点坐标
 * @param oringinal_pts 特征点
 * @param norm_pts 去中心去均值后的特征点坐标
 * @param T  变化矩阵
 */
void SFM::Normalize(const std::vector<Eigen::Vector2d>& oringinal_pts,
                    std::vector<Eigen::Vector2d>* const norm_pts,
                    Eigen::Matrix3d* const T) const {
  double mean_x = 0;
  double mean_y = 0;
  const int32_t N = oringinal_pts.size();
  norm_pts->clear();
  norm_pts->resize(N);

  for (int32_t i = 0; i < N; i++) {
    mean_x += oringinal_pts[i].x();
    mean_y += oringinal_pts[i].y();
  }

  // 1. 求均值
  mean_x = mean_x / N;
  mean_y = mean_y / N;

  double mean_dev_x = 0;
  double mean_dev_y = 0;

  for (int32_t i = 0; i < N; i++) {
    (*norm_pts)[i].x() = oringinal_pts[i].x() - mean_x;
    (*norm_pts)[i].y() = oringinal_pts[i].y() - mean_y;

    mean_dev_x += fabs((*norm_pts)[i].x());
    mean_dev_y += fabs((*norm_pts)[i].y());
  }

  // 2. 确定新原点后计算与新原点的距离均值
  mean_dev_x = mean_dev_x / N;
  mean_dev_y = mean_dev_y / N;

  // 3. 去均值化
  double sx = 1.0 / mean_dev_x;
  double sy = 1.0 / mean_dev_y;

  for (int32_t i = 0; i < N; i++) {
    (*norm_pts)[i].x() = (*norm_pts)[i].x() * sx;
    (*norm_pts)[i].y() = (*norm_pts)[i].y() * sy;
  }

  // 4. 计算变化矩阵
  T->setZero();
  (*T)(0, 0) = sx;
  (*T)(1, 1) = sy;
  (*T)(0, 2) = -mean_x * sx;
  (*T)(1, 2) = -mean_y * sy;
  (*T)(2, 2) = 1.0;
}

/**
 * @brief 检查结果
 * @param F21 顾名思义
 * @param inliers 匹配是否合法，大小为matches
 * @param sigma 默认为1
 */
double SFM::CheckFundamentalMat(const std::vector<cv::KeyPoint>& pts1,
                                const std::vector<cv::KeyPoint>& pts2,
                                const std::vector<cv::DMatch>& matches,
                                const Eigen::Matrix3d& F21,
                                std::vector<bool>* const inliers,
                                const double sigma = 1.0) const {
  const int32_t N = matches.size();

  const double f11 = F21(0, 0);
  const double f12 = F21(0, 1);
  const double f13 = F21(0, 2);
  const double f21 = F21(1, 0);
  const double f22 = F21(1, 1);
  const double f23 = F21(1, 2);
  const double f31 = F21(2, 0);
  const double f32 = F21(2, 1);
  const double f33 = F21(2, 2);

  inliers->resize(N);

  double score = 0;

  // 基于卡方检验计算出的阈值 自由度为1的卡方分布，显著性水平为0.05，对应的临界阈值
  const double th = 3.841;
  // 基于卡方检验计算出的阈值 自由度为2的卡方分布，显著性水平为0.05，对应的临界阈值
  const double th_score = 5.991;

  const double inv_sigma_square = 1.0 / (sigma * sigma);

  for (int32_t i = 0; i < N; i++) {
    double score1 = 0.0;
    double score2 = 0.0;

    const auto& kp1 = pts1[matches[i].queryIdx];
    const auto& kp2 = pts1[matches[i].trainIdx];

    const double u1 = kp1.pt.x;
    const double v1 = kp1.pt.y;
    const double u2 = kp2.pt.x;
    const double v2 = kp2.pt.y;

    // Reprojection error in second image
    // l2=F21x1=(a2,b2,c2)
    // 计算 img1 上的点在 img2 上投影得到的极线 l2 = F21 * p1 = (a2,b2,c2)
    const double a2 = f11 * u1 + f12 * v1 + f13;
    const double b2 = f21 * u1 + f22 * v1 + f23;
    const double c2 = f31 * u1 + f32 * v1 + f33;

    // 计算误差 e = (a * p2.x + b * p2.y + c) /  sqrt(a * a + b * b)
    const double num2 = a2 * u2 + b2 * v2 + c2;

    const double squareDist1 = num2 * num2 / (a2 * a2 + b2 * b2);

    const double chi_square1 = squareDist1 * inv_sigma_square;

    // 自由度为1是因为这里的计算是点到线的距离，判定分数自由度为2的原因可能是为了与H矩阵持平
    if (chi_square1 > th) {
      inliers->at(i) = false;
      continue;
    } else {
      score1 = th_score - chi_square1;
    }

    // Reprojection error in second image
    // l1 =x2tF21=(a1,b1,c1)
    // 与上面相同只不过反过来了
    const double a1 = f11 * u2 + f21 * v2 + f31;
    const double b1 = f12 * u2 + f22 * v2 + f32;
    const double c1 = f13 * u2 + f23 * v2 + f33;

    const double num1 = a1 * u1 + b1 * v1 + c1;

    const double square_dist2 = num1 * num1 / (a1 * a1 + b1 * b1);

    const double chi_square2 = square_dist2 * inv_sigma_square;

    if (chi_square2 > th) {
      inliers->at(i) = false;
      continue;
    } else {
      score2 = th_score - chi_square2;
    }

    inliers->at(i) = true;
    score += (score1 + score2);
  }

  return score;
}

/**
 * @brief 检查结果
 * @param H21 顾名思义
 * @param H12 顾名思义
 * @param inliers 匹配是否合法，大小为matches
 * @param sigma 默认为1
 */
double SFM::CheckHomographyMat(const std::vector<cv::KeyPoint>& pts1,
                               const std::vector<cv::KeyPoint>& pts2,
                               const std::vector<cv::DMatch>& matches,
                               const Eigen::Matrix3d& H21,
                               std::vector<bool>* const inliers,
                               double sigma = 1.0) const {
  const int32_t N = matches.size();

  const double h11 = H21(0, 0);
  const double h12 = H21(0, 1);
  const double h13 = H21(0, 2);
  const double h21 = H21(1, 0);
  const double h22 = H21(1, 1);
  const double h23 = H21(1, 2);
  const double h31 = H21(2, 0);
  const double h32 = H21(2, 1);
  const double h33 = H21(2, 2);

  auto H12 = H21.inverse();
  const double h11inv = H12(0, 0);
  const double h12inv = H12(0, 1);
  const double h13inv = H12(0, 2);
  const double h21inv = H12(1, 0);
  const double h22inv = H12(1, 1);
  const double h23inv = H12(1, 2);
  const double h31inv = H12(2, 0);
  const double h32inv = H12(2, 1);
  const double h33inv = H12(2, 2);

  inliers->resize(N);

  double score = 0;
  // 基于卡方检验计算出的阈值 自由度为2的卡方分布，显著性水平为0.05，对应的临界阈值
  const double th = 5.991;

  const double invSigmaSquare = 1.0 / (sigma * sigma);

  for (int32_t i = 0; i < N; i++) {
    bool bIn = true;

    const auto& kp1 = pts1[matches[i].queryIdx];
    const auto& kp2 = pts2[matches[i].trainIdx];

    const double u1 = kp1.pt.x;
    const double v1 = kp1.pt.y;
    const double u2 = kp2.pt.x;
    const double v2 = kp2.pt.y;

    // Reprojection error in first image
    // x2in1 = H12*x2

    // 计算投影误差，2投1
    // 1投2这么做，计算累计的卡方检验分数，分数越高证明内点与误差越优，这么做为了平衡误差与内点个数，不是说内点个数越高越好，也不是说误差越小越好
    const double w2in1inv = 1.0 / (h31inv * u2 + h32inv * v2 + h33inv);
    const double u2in1 = (h11inv * u2 + h12inv * v2 + h13inv) * w2in1inv;
    const double v2in1 = (h21inv * u2 + h22inv * v2 + h23inv) * w2in1inv;

    const double squareDist1 = (u1 - u2in1) * (u1 - u2in1) + (v1 - v2in1) * (v1 - v2in1);

    const double chiSquare1 = squareDist1 * invSigmaSquare;

    if (chiSquare1 > th)
      bIn = false;
    else
      score += th - chiSquare1;

    // Reprojection error in second image
    // x1in2 = H21*x1

    const double w1in2inv = 1.0 / (h31 * u1 + h32 * v1 + h33);
    const double u1in2 = (h11 * u1 + h12 * v1 + h13) * w1in2inv;
    const double v1in2 = (h21 * u1 + h22 * v1 + h23) * w1in2inv;

    const double squareDist2 = (u2 - u1in2) * (u2 - u1in2) + (v2 - v1in2) * (v2 - v1in2);

    const double chiSquare2 = squareDist2 * invSigmaSquare;

    if (chiSquare2 > th)
      bIn = false;
    else
      score += th - chiSquare2;

    if (bIn)
      inliers->at(i) = true;
    else
      inliers->at(i) = false;
  }

  return score;
}

double SFM::ProjectError(const std::vector<cv::KeyPoint>& pts1,
                         const std::vector<cv::KeyPoint>& pts2,
                         const std::vector<cv::DMatch>& matches,
                         const Eigen::Matrix3d& M21,
                         std::vector<bool>* const inliers) const {
  CHECK(!pts1.empty() && pts2.empty() && !inliers) << "Check Input !";
  const int32_t N = matches.size();
  bool traversal = false;
  if (inliers->size() != matches.size()) {
    inliers->clear();
    inliers->resize(false, N);
    traversal = true;
  }
  Eigen::Vector2d sum_err = Eigen::Vector2d::Zero();
  for (int32_t i = 0; i < N; ++i) {
    if (!traversal && !inliers->at(i)) {
      continue;
    }
    auto err =
        (M21 * Eigen::Vector3d(
                   pts1.at(matches[i].queryIdx).pt.x, pts1.at(matches[i].queryIdx).pt.y, 1.0) -
         Eigen::Vector3d(pts2.at(matches.at(i).trainIdx).pt.x,
                         pts2.at(matches.at(i).trainIdx).pt.y))
            .cwiseAbs();
    if (err.norm() < config_.reproj_err_th()) {
      inliers->at(i) = true;
      sum_err += err;
    }
  }
  return sum_err.norm();
}

bool SFM::Triangulate(const Eigen::Vector3d& x_c1,
                      const Eigen::Vector3d& x_c2,
                      const Eigen::Matrix<double, 3, 4>& Tc1w,
                      const Eigen::Matrix<double, 3, 4>& Tc2w,
                      Eigen::Vector3d* const x3D) {
  Eigen::Matrix4d A;
  // x = a*P*X， 左右两面乘Pc的反对称矩阵 a*[x]^ * P *X = 0
  // 构成了A矩阵，中间涉及一个尺度a，因为都是归一化平面，但右面是0所以直接可以约掉不影响最后的尺度
  //  0 -1 v    P(0)     -P.row(1) + v*P.row(2)
  //  1 0 -u *  P(1)  =   P.row(0) - u*P.row(2)
  // -v u  0    P(2)    u*P.row(1) - v*P.row(0)
  // 发现上述矩阵线性相关，所以取前两维，两个点构成了4行的矩阵，就是如下的操作，求出的是4维的结果[X,Y,Z,A]，所以需要除以最后一维使之为1，就成了[X,Y,Z,1]这种齐次形式
  A.block<1, 4>(0, 0) = x_c1(0) * Tc1w.block<1, 4>(2, 0) - Tc1w.block<1, 4>(0, 0);
  A.block<1, 4>(1, 0) = x_c1(1) * Tc1w.block<1, 4>(2, 0) - Tc1w.block<1, 4>(1, 0);
  A.block<1, 4>(2, 0) = x_c2(0) * Tc2w.block<1, 4>(2, 0) - Tc2w.block<1, 4>(0, 0);
  A.block<1, 4>(3, 0) = x_c2(1) * Tc2w.block<1, 4>(2, 0) - Tc2w.block<1, 4>(1, 0);

  // 解方程 AX=0
  Eigen::JacobiSVD<Eigen::Matrix4d> svd(A, Eigen::ComputeFullV);

  Eigen::Vector4d x3Dh = svd.matrixV().col(3);

  if (std::fabs(x3Dh(3)) < 1e-6) {
    LOG(ERROR) << "Triangulate failed !";
    return false;
  }

  // Euclidean coordinates
  (*x3D) = x3Dh.head(3) / x3Dh(3);

  return true;
}

void SFM::DecomposeE(const Eigen::Matrix3d& E,
                     Eigen::Matrix3d* const R1,
                     Eigen::Matrix3d* const R2,
                     Eigen::Vector3d* const t) const {
  Eigen::JacobiSVD<Eigen::Matrix3d> svd(E, Eigen::ComputeFullU | Eigen::ComputeFullV);

  // 对 t 有归一化，但是这个地方并没有决定单目整个SLAM过程的尺度
  // 因为CreateInitialMapMonocular函数对3D点深度会缩放，然后反过来对 t 有改变
  Eigen::Matrix3d U = svd.matrixU();
  Eigen::Matrix3d Vt = svd.matrixV().transpose();

  *t = U.col(2);
  *t = *t / t->norm();

  Eigen::Matrix3d W;
  W.setZero();
  W(0, 1) = -1;
  W(1, 0) = 1;
  W(2, 2) = 1;

  *R1 = U * W * Vt;
  // 旋转矩阵有行列式为1的约束
  if (R1->determinant() < 0) {
    *R1 = -*R1;
  }
  *R2 = U * W.transpose() * Vt;
  if (R2->determinant() < 0) {
    *R2 = -*R2;
  }
}

double SFM::OppositeOfMinor(const Eigen::Matrix3d& M, const int32_t row, const int32_t col) const {
  int32_t x1 = col == 0 ? 1 : 0;
  int32_t x2 = col == 2 ? 1 : 2;
  int32_t y1 = row == 0 ? 1 : 0;
  int32_t y2 = row == 2 ? 1 : 2;

  return (M(y1, x2) * M(y2, x1) - M(y1, x1) * M(y2, x2));
}

void SFM::DecomposeH_EM(
    const Eigen::Matrix3d& H,
    const Eigen::Matrix3d& K,
    std::vector<std::tuple<Eigen::Matrix3d, Eigen::Vector3d, Eigen::Vector3d>>* const solutions)
    const {
  // normalize homography matrix with intrinsic camera matrix
  auto Hnorm = K.inverse() * H * K;
  // remove scale of the normalized homography
  Eigen::JacobiSVD<Eigen::Matrix3d> svd(Hnorm, Eigen::ComputeThinU | Eigen::ComputeThinV);
  auto w = svd.singularValues();
  // 归一化_Hnorm
  Hnorm *= (1.0 / w(1));

  const double epsilon = 1.0e-3;

  Eigen::Matrix3d S;

  // S = H'H - I
  S = Hnorm.transpose() * Hnorm;
  S(0, 0) -= 1.0;
  S(1, 1) -= 1.0;
  S(2, 2) -= 1.0;

  // check if H is rotation matrix
  if (S.cwiseAbs().rowwise().sum().maxCoeff() < epsilon) {
    solutions->emplace_back(
        std::make_tuple(Hnorm, Eigen::Vector3d(0, 0, 0), Eigen::Vector3d(0, 0, 0)));
    return;
  }

  Eigen::Vector3d npa, npb;

  double M00 = OppositeOfMinor(S, 0, 0);
  double M11 = OppositeOfMinor(S, 1, 1);
  double M22 = OppositeOfMinor(S, 2, 2);

  double rtM00 = std::sqrt(M00);
  double rtM11 = std::sqrt(M11);
  double rtM22 = std::sqrt(M22);

  double M01 = OppositeOfMinor(S, 0, 1);
  double M12 = OppositeOfMinor(S, 1, 2);
  double M02 = OppositeOfMinor(S, 0, 2);

  int32_t e12 = SIGN(M12);
  int32_t e02 = SIGN(M02);
  int32_t e01 = SIGN(M01);

  double nS00 = std::abs(S(0, 0));
  double nS11 = std::abs(S(1, 1));
  double nS22 = std::abs(S(2, 2));

  // find max( |Sii| ), i=0, 1, 2
  int32_t indx = 0;
  if (nS00 < nS11) {
    indx = 1;
    if (nS11 < nS22) indx = 2;
  } else {
    if (nS00 < nS22) indx = 2;
  }

  switch (indx) {
    case 0:
      npa[0] = S(0, 0), npb[0] = S(0, 0);
      npa[1] = S(0, 1) + rtM22, npb[1] = S(0, 1) - rtM22;
      npa[2] = S(0, 2) + e12 * rtM11, npb[2] = S(0, 2) - e12 * rtM11;
      break;
    case 1:
      npa[0] = S(0, 1) + rtM22, npb[0] = S(0, 1) - rtM22;
      npa[1] = S(1, 1), npb[1] = S(1, 1);
      npa[2] = S(1, 2) - e02 * rtM00, npb[2] = S(1, 2) + e02 * rtM00;
      break;
    case 2:
      npa[0] = S(0, 2) + e01 * rtM11, npb[0] = S(0, 2) - e01 * rtM11;
      npa[1] = S(1, 2) + rtM00, npb[1] = S(1, 2) - rtM00;
      npa[2] = S(2, 2), npb[2] = S(2, 2);
      break;
    default:
      break;
  }

  double traceS = S(0, 0) + S(1, 1) + S(2, 2);
  double v = 2.0 * sqrtf(1 + traceS - M00 - M11 - M22);

  double ESii = SIGN(S(indx, indx));
  double r_2 = 2.0 + traceS + v;
  double nt_2 = 2.0 + traceS - v;

  double r = sqrt(r_2);
  double n_t = sqrt(nt_2);

  Eigen::Vector3d na = npa / npa.norm();
  Eigen::Vector3d nb = npb / npb.norm();

  double half_nt = 0.5 * n_t;
  double esii_t_r = ESii * r;

  Eigen::Vector3d ta_star = half_nt * (esii_t_r * nb - n_t * na);
  Eigen::Vector3d tb_star = half_nt * (esii_t_r * na - n_t * nb);

  // computes R = H( I - (2/v)*te_star*ne_t )
  // Ra, ta, na
  Eigen::Matrix3d Ra = Hnorm * (Eigen::Matrix3d::Identity() - (2.0 / v) * ta_star * na.transpose());
  Eigen::Vector3d ta = Ra * ta_star;

  // Rb, tb, nb
  Eigen::Matrix3d Rb = Hnorm * (Eigen::Matrix3d::Identity() - (2.0 / v) * tb_star * nb.transpose());
  Eigen::Vector3d tb = Rb * tb_star;

  solutions->emplace_back(std::make_tuple(Ra, ta, na));
  solutions->emplace_back(std::make_tuple(Ra, -ta, -na));
  solutions->emplace_back(std::make_tuple(Rb, tb, nb));
  solutions->emplace_back(std::make_tuple(Rb, -tb, -nb));
}

/*
R = H * inv(I + tstar*transpose(n) );
t = R * tstar;
*/
bool SFM::SolveMotion(
    const Eigen::Matrix3d& Hnorm,
    const Eigen::Vector3d& tstar,
    const Eigen::Vector3d& n,
    std::tuple<Eigen::Matrix3d, Eigen::Vector3d, Eigen::Vector3d>* const solution) const {
  Eigen::Matrix3d temp = tstar * n.transpose();
  temp(0, 0) += 1.0;
  temp(1, 1) += 1.0;
  temp(2, 2) += 1.0;
  Eigen::Matrix3d R = Hnorm * temp.inverse();
  Eigen::Vector3d t = R * tstar;
  *solution = std::make_tuple(R, t, n);
  return n.transpose() * R.transpose() * t > -1.0;
}

void SFM::DecomposeH_Zhang(
    const Eigen::Matrix3d& H,
    const Eigen::Matrix3d& K,
    std::vector<std::tuple<Eigen::Matrix3d, Eigen::Vector3d, Eigen::Vector3d>>* const solutions)
    const {
  // normalize homography matrix with intrinsic camera matrix
  auto Hnorm = K.inverse() * H * K;

  // 计算SVD
  Eigen::JacobiSVD<Eigen::Matrix3d> svd(Hnorm, Eigen::ComputeFullU | Eigen::ComputeFullV);
  auto W = svd.singularValues();
  auto Vt = svd.matrixV().transpose();  // Eigen的Vt是转置后的V

  // 获取奇异值lambda1和lambda3
  double lambda1 = W(0);
  double lambda3 = W(2);

  double lambda1m3 = (lambda1 - lambda3);
  double lambda1m3_2 = lambda1m3 * lambda1m3;
  double lambda1t3 = lambda1 * lambda3;

  double t1 = 1.0 / (2.0 * lambda1t3);
  double t2 = std::sqrt(1.0 + 4.0 * lambda1t3 / lambda1m3_2);
  double t12 = t1 * t2;

  double e1 = -t1 + t12;  // t1*(-1.0f + t2 );
  double e3 = -t1 - t12;  // t1*(-1.0f - t2);

  double e1_2 = e1 * e1;
  double e3_2 = e3 * e3;

  double nv1p = std::sqrt(e1_2 * lambda1m3_2 + 2 * e1 * (lambda1t3 - 1) + 1.0);
  double nv3p = std::sqrt(e3_2 * lambda1m3_2 + 2 * e3 * (lambda1t3 - 1) + 1.0);

  // 定义v1p和v3p数组
  Eigen::Vector3d v1p, v3p;

  // 计算v1p
  v1p[0] = Vt(0, 0) * nv1p;
  v1p[1] = Vt(0, 1) * nv1p;
  v1p[2] = Vt(0, 2) * nv1p;

  // 计算v3p
  v3p[0] = Vt(2, 0) * nv3p;
  v3p[1] = Vt(2, 1) * nv3p;
  v3p[2] = Vt(2, 2) * nv3p;

  /*The eight solutions are
   (A): tstar = +- (v1p - v3p)/(e1 -e3), n = +- (e1*v3p - e3*v1p)/(e1-e3)
   (B): tstar = +- (v1p + v3p)/(e1 -e3), n = +- (e1*v3p + e3*v1p)/(e1-e3)
       */
  double v1pmv3p[3], v1ppv3p[3];
  double e1v3me3v1[3], e1v3pe3v1[3];
  double inv_e1me3 = 1.0 / (e1 - e3);

  for (int32_t kk = 0; kk < 3; ++kk) {
    v1pmv3p[kk] = v1p[kk] - v3p[kk];
    v1ppv3p[kk] = v1p[kk] + v3p[kk];
  }
  for (int32_t kk = 0; kk < 3; ++kk) {
    double e1v3 = e1 * v3p[kk];
    double e3v1 = e3 * v1p[kk];
    e1v3me3v1[kk] = e1v3 - e3v1;
    e1v3pe3v1[kk] = e1v3 + e3v1;
  }

  Eigen::Vector3d tstar_p, tstar_n;
  Eigen::Vector3d n_p, n_n;
  std::tuple<Eigen::Matrix3d, Eigen::Vector3d, Eigen::Vector3d> solution;

  //=================================================================================
  /// Solution group A
  for (int32_t kk = 0; kk < 3; ++kk) {
    tstar_p[kk] = v1pmv3p[kk] * inv_e1me3;
    tstar_n[kk] = -tstar_p[kk];

    n_p[kk] = e1v3me3v1[kk] * inv_e1me3;
    n_n[kk] = -n_p[kk];
  }

  //(A) Four different combinations for solution A
  // (i)  (+, +)
  if (SolveMotion(Hnorm, tstar_p, n_p, &solution)) solutions->emplace_back(solution);

  // (ii)  (+, -)
  if (SolveMotion(Hnorm, tstar_p, n_n, &solution)) solutions->emplace_back(solution);

  // (iii)  (-, +)
  if (SolveMotion(Hnorm, tstar_n, n_p, &solution)) solutions->emplace_back(solution);

  // (iv)  (-, -)
  if (SolveMotion(Hnorm, tstar_n, n_n, &solution)) solutions->emplace_back(solution);

  //=================================================================================
  /// Solution group B
  for (int32_t kk = 0; kk < 3; ++kk) {
    tstar_p[kk] = v1ppv3p[kk] * inv_e1me3;
    tstar_n[kk] = -tstar_p[kk];

    n_p[kk] = e1v3pe3v1[kk] * inv_e1me3;
    n_n[kk] = -n_p[kk];
  }

  //(B) Four different combinations for solution B
  // (i)  (+, +)
  // (i)  (+, +)
  if (SolveMotion(Hnorm, tstar_p, n_p, &solution)) solutions->emplace_back(solution);

  // (ii)  (+, -)
  if (SolveMotion(Hnorm, tstar_p, n_n, &solution)) solutions->emplace_back(solution);

  // (iii)  (-, +)
  if (SolveMotion(Hnorm, tstar_n, n_p, &solution)) solutions->emplace_back(solution);

  // (iv)  (-, -)
  if (SolveMotion(Hnorm, tstar_n, n_n, &solution)) solutions->emplace_back(solution);
  //=================================================================================
}

}  // namespace Camera