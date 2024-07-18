#include "camera/sfm.h"

#include <glog/logging.h>

#include <cstdint>
#include <future>
#include <tuple>
#include <utility>
#include <vector>

#include "sophus/se3.hpp"

#include "camera/camera_model/pinhole.h"
namespace Camera {
#define SIGN(X) ((X) >= 0 ? (1) : (-1))

SFM::SFM(const CameraConfig::SFMConfig config) : config_(config) {
  // TODO, 根据camera_model中的参数进行有选择的初始化这个指针
  camera_model_ = std::make_shared<Pinhole>(config_.camera_model().pinhole_config());
  pnp_solver_ = std::make_shared<PnpSolver>(config_.pnp_solver(), camera_model_);
  feature_manager_ = std::make_shared<FeatureManager>(config_.feature_config());
}

double SFM::FindFundamentalMatrix(const std::vector<cv::KeyPoint>& p2ds_1,
                                  const std::vector<cv::KeyPoint>& p2ds_2,
                                  const std::vector<cv::DMatch>& matches,
                                  Eigen::Matrix3d* const fundamental_matrix3d,
                                  std::vector<bool>* const inliers) const {
  CHECK(p2ds_1.size() > 8) << "Check p2ds_1 !";
  CHECK(p2ds_2.size() > 8) << "Check p2ds_2 !";
  CHECK(matches.size() > 8) << "Check matches !";
  CHECK(fundamental_matrix3d != nullptr) << "Check fundamental_matrix3d !";

  double score = 0.0;
  double sum_err = std::numeric_limits<double>::max();
  if (config_.enable_cv_p2p()) {
    // 相机内参, cv::eigen2cv转换可能存在问题，如果有问题启用
    // auto K = camera_model_->K_;
    // cv::Mat K =
    //     (cv::Mat_<double>(3, 3) << K(0, 0), 0.0, K(0, 2), 0.0, K(1, 1), K(1, 2), 0, 0, 1.0);
    cv::Mat K = camera_model_->K_mat_;

    //-- 把匹配点转换为vector<Point2f>的形式
    std::vector<cv::Point2f> points1;
    std::vector<cv::Point2f> points2;

    for (uint32_t i = 0; i < matches.size(); ++i) {
      points1.emplace_back(p2ds_1[matches[i].queryIdx].pt);
      points2.emplace_back(p2ds_2[matches[i].trainIdx].pt);
    }

    //-- 计算基础矩阵
    cv::Mat fundamental_mat;
    // 8点法
    fundamental_mat = cv::findFundamentalMat(points1, points2, cv::FM_8POINT);
    cv::cv2eigen(fundamental_mat, *fundamental_matrix3d);
    score = CheckFundamentalMat(
        p2ds_1, p2ds_2, matches, *fundamental_matrix3d, inliers, config_.sigma());
    sum_err = ProjectErrorFmat(p2ds_1, p2ds_2, matches, *fundamental_matrix3d, inliers);
  } else {
    // 使用Ransac算法计算最优结果
    // 随机生成器
    std::random_device rd;
    std::mt19937 gen(rd());
    // 初始化索引容器
    std::vector<int32_t> indices(matches.size());
    // 依次递增填充数据
    std::iota(indices.begin(), indices.end(), 0);

    // double max_var1 = 0.0, max_var2 = 0.0;

    // 注意这里也使用了随机选择8个点的方法，但是不像ORB-SLAM3中那样，使用过的点直接就删除，我觉得这样不是很合理
    // 此外，这里我觉得应该对选择的8个点进行方差的统计，我认为方差大的可能更好
    std::vector<bool> inliers;
    for (int32_t iter = 0; iter < config_.ransac_iterations(); ++iter) {
      // 用于存放结果的vector
      std::vector<int32_t> sampled_vector;
      // 使用std::sample进行随机选择
      std::sample(indices.begin(), indices.end(), std::back_inserter(sampled_vector), 8, gen);
      // 随机选择8个匹配点对
      std::vector<Eigen::Vector2d> points1, points2;
      for (int32_t i = 0; i < 8; ++i) {
        // 提取匹配点
        int32_t match_idx = sampled_vector[i];
        auto pt1 = p2ds_1[matches[match_idx].queryIdx].pt;
        auto pt2 = p2ds_2[matches[match_idx].trainIdx].pt;
        points1.emplace_back(Eigen::Vector2d(pt1.x, pt1.y));
        points2.emplace_back(Eigen::Vector2d(pt2.x, pt2.y));
      }

      // double var1 = Utils::Math::CalculateVariance(points1);
      // double var2 = Utils::Math::CalculateVariance(points2);
      // // 统计一下点的方差和计算基础矩阵的效果的关系，我觉得方差大一些，相对来说矩阵应该更优
      // if (var1 > max_var1) {
      //   max_var1 = var1;
      // }
      // if (var2 > max_var2) {
      //   max_var2 = var2;
      // }

      Eigen::Matrix3d Fmat = ComputeFundamentalMat21(points1, points2);
      // std::cout << "ORB-SLAM3计算基础矩阵是 \n" << Fmat << "\n";

      std::vector<bool> cur_inliers;
      double cur_score =
          CheckFundamentalMat(p2ds_1, p2ds_2, matches, Fmat, &cur_inliers, config_.sigma());
      double cur_sum_err = ProjectErrorFmat(p2ds_1, p2ds_2, matches, Fmat, &cur_inliers);

      if (cur_score > score && cur_sum_err < sum_err) {
        *fundamental_matrix3d = Fmat;
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
  CHECK(p2ds_1.size() > 8) << "Check p2ds_1 !";
  CHECK(p2ds_2.size() > 8) << "Check p2ds_2 !";
  CHECK(matches.size() > 8) << "Check matches !";
  CHECK(essential_mat != nullptr) << "Check essential_mat !";

  if (config_.enable_cv_p2p()) {
    auto K = camera_model_->K_mat_;
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
    LOG(INFO) << "Please Use FindFundamentalMatrix or Enable OpenCV Functions ! Default: "
                 "FindFundamentalMatrix";
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
        auto pt1 = p2ds_1[matches[match_idx].queryIdx].pt;
        auto pt2 = p2ds_2[matches[match_idx].trainIdx].pt;
        points1.emplace_back(Eigen::Vector2d(pt1.x, pt1.y));
        points2.emplace_back(Eigen::Vector2d(pt2.x, pt2.y));
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

      case CameraConfig::SFMConfig::DecomposeHMethod::SFMConfig_DecomposeHMethod_ORBSLAM3:
        DecomposeH_ORBSLAM3(homography_mat, camera_model_->K_, relative_pose);
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

  auto fundamental_matrix3d =
      T2t * svd2.matrixU() * Eigen::DiagonalMatrix<double, 3>(w) * svd2.matrixV().transpose() * T1;
  return fundamental_matrix3d;
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
                                const double sigma) const {
  // 说明：在已值n维观测数据误差服从N(0，sigma）的高斯分布时
  // 其误差加权最小二乘结果为  sum_error = SUM(e(i)^T * Q^(-1) * e(i))
  // 其中：e(i) = [e_x,e_y,...]^T, Q维观测数据协方差矩阵，即sigma * sigma组成的协方差矩阵
  // 误差加权最小二次结果越小，说明观测数据精度越高
  // 那么，score = SUM((th - e(i)^T * Q^(-1) * e(i)))的分数就越高
  // 算法目标：检查基础矩阵
  // 检查方式：利用对极几何原理 p2^T * F * p1 = 0
  // 假设：三维空间中的点 P 在 img1 和 img2 两图像上的投影分别为 p1 和 p2（两个为同名点）
  //   则：p2 一定存在于极线 l2 上，即 p2*l2 = 0. 而l2 = F*p1 = (a, b, c)^T
  //      所以，这里的误差项 e 为 p2 到 极线 l2 的距离，如果在直线上，则 e = 0
  //      根据点到直线的距离公式：d = (ax + by + c) / sqrt(a * a + b * b)
  //      所以，e =  (a * p2.x + b * p2.y + c) /  sqrt(a * a + b * b)

  // 算法流程
  // input: 基础矩阵 F 左右视图匹配点集 mkps1
  //    do:
  //        for p1(i), p2(i) in mvKeys:
  //           l2 = F * p1(i)
  //           l1 = p2(i) * F
  //           error_i1 = dist_point_to_line(x2,l2)
  //           error_i2 = dist_point_to_line(x1,l1)
  //
  //           w1 = 1 / sigma / sigma
  //           w2 = 1 / sigma / sigma
  //
  //           if error1 < th
  //              score +=   thScore - error_i1 * w1
  //           if error2 < th
  //              score +=   thScore - error_i2 * w2
  //
  //           if error_1i > th or error_2i > th
  //              p1(i), p2(i) are inner points
  //              inliers(i) = true
  //           else
  //              p1(i), p2(i) are outliers
  //              inliers(i) = false
  //           end
  //        end
  //   output: score, inliers

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

  inliers->resize(N, false);

  double score = 0;

  // 基于卡方检验计算出的阈值 自由度为1的卡方分布，显著性水平为0.05，对应的临界阈值
  const double th = 3.841;
  // 基于卡方检验计算出的阈值 自由度为2的卡方分布，显著性水平为0.05，对应的临界阈值
  const double th_score = 5.991;

  const double inv_sigma_square = 1.0 / (sigma * sigma);
  int32_t count = 0;
  std::cout << "匹配点对数量为：" << N << "\n";
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
      continue;
    } else {
      score2 = th_score - chi_square2;
    }

    inliers->at(i) = true;
    count++;
    score += (score1 + score2);
  }
  std::cout << "内点数量为：" << count << "\n";
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
                               double sigma) const {
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

// 代码符合预期
double SFM::ProjectErrorFmat(const std::vector<cv::KeyPoint>& pts1,
                             const std::vector<cv::KeyPoint>& pts2,
                             const std::vector<cv::DMatch>& matches,
                             const Eigen::Matrix3d& M21,
                             std::vector<bool>* const inliers) const {
  CHECK(!pts1.empty()) << "Check pts1 !";
  CHECK(!pts2.empty()) << "Check pts2 !";
  CHECK(inliers != nullptr) << "Check inliers !";
  const int32_t N = matches.size();
  bool traversal = false;
  if (inliers->size() != matches.size()) {
    inliers->clear();
    inliers->resize(false, N);
    traversal = true;
  }
  double sum_err = 0.0;
  for (int32_t i = 0; i < N; ++i) {
    if (!traversal && !inliers->at(i)) {
      continue;
    }
    double err =
        Eigen::Vector3d(
            pts2.at(matches.at(i).trainIdx).pt.x, pts2.at(matches.at(i).trainIdx).pt.y, 1.0)
            .transpose() *
        M21 *
        Eigen::Vector3d(pts1.at(matches[i].queryIdx).pt.x, pts1.at(matches[i].queryIdx).pt.y, 1.0);
    if (err < config_.reproj_err_th()) {
      inliers->at(i) = true;
      sum_err += std::fabs(err);
    }
  }
  return sum_err;
}

bool SFM::Triangulate(const Eigen::Vector3d& x_c1,
                      const Eigen::Vector3d& x_c2,
                      const Eigen::Matrix<double, 3, 4>& Tc1w,
                      const Eigen::Matrix<double, 3, 4>& Tc2w,
                      Eigen::Vector3d* const x3D) const {
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
  Eigen::Matrix3d Hnorm = K.inverse() * H * K;
  // remove scale of the normalized homography
  Eigen::JacobiSVD<Eigen::Matrix3d> svd(Hnorm, Eigen::ComputeThinU | Eigen::ComputeThinV);
  Eigen::Vector3d w = svd.singularValues();
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
  CHECK_NOTNULL(solutions);
  solutions->reserve(8);
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

void SFM::DecomposeH_ORBSLAM3(
    const Eigen::Matrix3d& H,
    const Eigen::Matrix3d& K,
    std::vector<std::tuple<Eigen::Matrix3d, Eigen::Vector3d, Eigen::Vector3d>>* const solutions)
    const {
  // 目的 ：通过单应矩阵H恢复两帧图像之间的旋转矩阵R和平移向量T
  // 参考 ：Motion and structure from motion in a piecewise plannar environment.
  //        International Journal of Pattern Recognition and Artificial Intelligence, 1988
  // https://www.researchgate.net/publication/243764888_Motion_and_Structure_from_Motion_in_a_Piecewise_Planar_Environment

  // 流程:
  //      1. 根据H矩阵的奇异值d'= d2 或者 d' = -d2 分别计算 H 矩阵分解的 8 组解
  //        1.1 讨论 d' > 0 时的 4 组解
  //        1.2 讨论 d' < 0 时的 4 组解
  //      2. 对 8 组解进行验证，并选择产生相机前方最多3D点的解为最优解

  // We recover 8 motion hypotheses using the method of Faugeras et al.
  // Motion and structure from motion in a piecewise planar environment.
  // International Journal of Pattern Recognition and Artificial Intelligence, 1988

  // 参考SLAM十四讲第二版p170-p171
  // H = K * (R - t * n / d) * K_inv
  // 其中: K表示内参数矩阵
  //       K_inv 表示内参数矩阵的逆
  //       R 和 t 表示旋转和平移向量
  //       n 表示平面法向量
  // 令 H = K * A * K_inv
  // 则 A = k_inv * H * k

  Eigen::Matrix3d invK = K.inverse();
  Eigen::Matrix3d A = invK * H * K;

  // 对矩阵A进行SVD分解
  // A 等待被进行奇异值分解的矩阵
  // w 奇异值矩阵
  // U 奇异值分解左矩阵
  // Vt 奇异值分解右矩阵，注意函数返回的是转置
  // cv::SVD::FULL_UV 全部分解
  // A = U * w * Vt
  // cv::Mat U, w, Vt, V;
  // cv::SVD::compute(A, w, U, Vt, cv::SVD::FULL_UV);

  // 计算SVD
  Eigen::JacobiSVD<Eigen::Matrix3d> svd(A, Eigen::ComputeFullU | Eigen::ComputeFullV);
  auto W = svd.singularValues();
  auto U = svd.matrixU();
  auto Vt = svd.matrixV().transpose();  // Eigen的Vt是转置后的V
  // Eigen的V矩阵与OpenCV的Vt不同，直接使用svd.matrixV()即可
  auto V = svd.matrixV();

  // 根据文献eq(8)，计算关联变量
  // 计算变量s = det(U) * det(V)
  // 因为det(V)==det(Vt), 所以 s = det(U) * det(Vt)

  double s = U.determinant() * V.determinant();

  // 取得矩阵的各个奇异值
  double d1 = W(0);
  double d2 = W(1);
  double d3 = W(2);

  // SVD分解正常情况下特征值di应该是正的，且满足d1>=d2>=d3
  if (d1 / d2 < 1.00001 || d2 / d3 < 1.00001) {
    LOG(ERROR) << "Singular Values Error !";
    return;
  }

  // 在ORBSLAM中没有对奇异值 d1 d2 d3按照论文中描述的关系进行分类讨论, 而是直接进行了计算
  // 定义8中情况下的旋转矩阵、平移向量和空间向量
  solutions->reserve(8);

  // Step 1.1 讨论 d' > 0 时的 4 组解
  // 根据论文eq.(12)有
  // x1 = e1 * sqrt((d1 * d1 - d2 * d2) / (d1 * d1 - d3 * d3))
  // x2 = 0
  // x3 = e3 * sqrt((d2 * d2 - d2 * d2) / (d1 * d1 - d3 * d3))
  // 令 aux1 = sqrt((d1*d1-d2*d2)/(d1*d1-d3*d3))
  //    aux3 = sqrt((d2*d2-d3*d3)/(d1*d1-d3*d3))
  // 则
  // x1 = e1 * aux1
  // x3 = e3 * aux2

  // 因为 e1,e2,e3 = 1 or -1
  // 所以有x1和x3有四种组合
  // x1 =  {aux1,aux1,-aux1,-aux1}
  // x3 =  {aux3,-aux3,aux3,-aux3}

  double aux1 = sqrt((d1 * d1 - d2 * d2) / (d1 * d1 - d3 * d3));
  double aux3 = sqrt((d2 * d2 - d3 * d3) / (d1 * d1 - d3 * d3));
  double x1[] = {aux1, aux1, -aux1, -aux1};
  double x3[] = {aux3, -aux3, aux3, -aux3};

  // 根据论文eq.(13)有
  // sin(theta) = e1 * e3 * sqrt(( d1 * d1 - d2 * d2) * (d2 * d2 - d3 * d3)) /(d1 + d3)/d2
  // cos(theta) = (d2* d2 + d1 * d3) / (d1 + d3) / d2
  // 令  aux_stheta = sqrt((d1*d1-d2*d2)*(d2*d2-d3*d3))/((d1+d3)*d2)
  // 则  sin(theta) = e1 * e3 * aux_stheta
  //     cos(theta) = (d2*d2+d1*d3)/((d1+d3)*d2)
  // 因为 e1 e2 e3 = 1 or -1
  // 所以 sin(theta) = {aux_stheta, -aux_stheta, -aux_stheta, aux_stheta}
  double aux_stheta = sqrt((d1 * d1 - d2 * d2) * (d2 * d2 - d3 * d3)) / ((d1 + d3) * d2);

  double ctheta = (d2 * d2 + d1 * d3) / ((d1 + d3) * d2);
  double stheta[] = {aux_stheta, -aux_stheta, -aux_stheta, aux_stheta};

  // 计算旋转矩阵 R'
  // 根据不同的e1 e3组合所得出来的四种R t的解
  //      | ctheta      0   -aux_stheta|       | aux1|
  // Rp = |    0        1       0      |  tp = |  0  |
  //      | aux_stheta  0    ctheta    |       |-aux3|

  //      | ctheta      0    aux_stheta|       | aux1|
  // Rp = |    0        1       0      |  tp = |  0  |
  //      |-aux_stheta  0    ctheta    |       | aux3|

  //      | ctheta      0    aux_stheta|       |-aux1|
  // Rp = |    0        1       0      |  tp = |  0  |
  //      |-aux_stheta  0    ctheta    |       |-aux3|

  //      | ctheta      0   -aux_stheta|       |-aux1|
  // Rp = |    0        1       0      |  tp = |  0  |
  //      | aux_stheta  0    ctheta    |       | aux3|
  // 开始遍历这四种情况中的每一种
  for (int i = 0; i < 4; i++) {
    // 生成Rp，就是eq.(8) 的 R'
    Eigen::Matrix3d Rp = Eigen::Matrix3d::Identity();
    Rp(0, 0) = ctheta;
    Rp(0, 2) = -stheta[i];
    Rp(2, 0) = stheta[i];
    Rp(2, 2) = ctheta;

    // eq.(8) 计算R
    Eigen::Matrix3d R = s * U * Rp * Vt;

    // eq. (14) 生成tp
    Eigen::Vector3d tp = Eigen::Vector3d::Identity();
    tp(0) = x1[i];
    tp(1) = 0;
    tp(2) = -x3[i];
    tp *= d1 - d3;

    // 这里虽然对t有归一化，并没有决定单目整个SLAM过程的尺度
    // 因为CreateInitialMapMonocular函数对3D点深度会缩放，然后反过来对 t 有改变
    // eq.(8)恢复原始的t
    Eigen::Vector3d t = U * tp;
    t /= t.norm();

    // 构造法向量np
    Eigen::Vector3d np = Eigen::Vector3d::Identity();
    np(0) = x1[i];
    np(1) = 0;
    np(2) = x3[i];

    // eq.(8) 恢复原始的法向量
    Eigen::Vector3d n = V * np;
    // 保持平面法向量向上
    if (n(2) < 0) {
      n = -n;
    }
    solutions->emplace_back(std::make_tuple(R, t, n));
  }

  // Step 1.2 讨论 d' < 0 时的 4 组解
  double aux_sphi = sqrt((d1 * d1 - d2 * d2) * (d2 * d2 - d3 * d3)) / ((d1 - d3) * d2);
  // cos_theta项
  double cphi = (d1 * d3 - d2 * d2) / ((d1 - d3) * d2);
  // 考虑到e1,e2的取值，这里的sin_theta有两种可能的解
  double sphi[] = {aux_sphi, -aux_sphi, -aux_sphi, aux_sphi};

  // 对于每种由e1 e3取值的组合而形成的四种解的情况
  for (int i = 0; i < 4; i++) {
    // 计算旋转矩阵 R'
    Eigen::Matrix3d Rp = Eigen::Matrix3d::Identity();

    Rp(0, 0) = cphi;
    Rp(0, 2) = sphi[i];
    Rp(1, 1) = -1;
    Rp(2, 0) = sphi[i];
    Rp(2, 2) = -cphi;

    // 恢复出原来的R
    Eigen::Matrix3d R = s * U * Rp * Vt;

    // 构造tp
    Eigen::Vector3d tp = Eigen::Vector3d::Identity();
    tp(0) = x1[i];
    tp(1) = 0;
    tp(2) = x3[i];
    tp *= d1 + d3;

    // 恢复出原来的t
    Eigen::Vector3d t = U * tp;
    // 归一化之后加入到vector中,要提供给上面的平移矩阵都是要进行过归一化的
    t /= t.norm();

    // 构造法向量np
    Eigen::Vector3d np = Eigen::Vector3d::Identity();
    np(0) = x1[i];
    np(1) = 0;
    np(2) = x3[i];

    // 恢复出原来的法向量
    Eigen::Vector3d n = V * np;
    // 保证法向量指向上方
    if (n(2) < 0) {
      n = -n;
    }
    // 添加到vector中
    solutions->emplace_back(std::make_tuple(R, t, n));
  }
}

bool SFM::ReconstructFromFmat(const Eigen::Matrix3d& F21,
                              const std::vector<cv::KeyPoint>& kp1,
                              const std::vector<cv::KeyPoint>& kp2,
                              const std::vector<cv::DMatch>& matches,
                              const int32_t min_triangulated_pts,
                              const double min_parallax,
                              std::vector<bool>* const inliers,
                              std::vector<Eigen::Vector3d>* const p3d,
                              Eigen::Matrix3d* const R21,
                              Eigen::Vector3d* const t21) {
  // 统计了合法的匹配，后面用于对比重建出的点数
  int32_t N = 0;
  for (int32_t i = 0, iend = inliers->size(); i < iend; i++) {
    if ((*inliers)[i]) {
      N++;
    }
  }
  std::cout << "内点数量为： " << N << "\n";

  auto K = camera_model_->K_;
  Eigen::Matrix3d E21 = K.transpose() * F21 * K;

  // Step 3 从本质矩阵求解两个R解和两个t解，共四组解
  // 不过由于两个t解互为相反数，因此这里先只获取一个
  // 虽然这个函数对t有归一化，但并没有决定单目整个SLAM过程的尺度.
  // 因为 CreateInitialMapMonocular 函数对3D点深度会缩放，然后反过来对 t 有改变.
  // 注意下文中的符号“'”表示矩阵的转置
  //                          |0 -1  0|
  // E = U Sigma V'   let W = |1  0  0|
  //                          |0  0  1|
  // 得到4个解 E = [R|t]
  // R1 = UWV' R2 = UW'V' t1 = U3 t2 = -U3
  std::vector<Sophus::SE3d> solutions;
  DecomposeEssentialMatrix(E21, &solutions);

  // 启用多线程来同步进行检验R和t矩阵
  // 原理：若某一组合使恢复得到的3D点位于相机正前方的数量最多，那么该组合就是最佳组合
  // 实现：根据计算的解组合成为四种情况, 多线程同步调用CheckRT()
  // 进行检查,得到可以进行三角化测量的点的数目
  // 定义四组解分别在对同一匹配点集进行三角化测量之后的特征点空间坐标
  std::vector<std::future<std::pair<int32_t, int32_t>>> futures;
  std::vector<std::vector<bool>> inliers_list(solutions.size(), *inliers);
  std::vector<std::vector<Eigen::Vector3d>> p3ts_list(solutions.size());
  std::vector<double> parallaxes(solutions.size());
  double th2 = config_.reproj_err_th();
  for (int32_t i = 0, iend = solutions.size(); i < iend; ++i) {
    futures.push_back(std::async(std::launch::async,
                                 [this,
                                  &solutions,
                                  &kp1,
                                  &kp2,
                                  &matches,
                                  &K,
                                  th2,
                                  &inliers_list,
                                  &p3ts_list,
                                  &parallaxes,
                                  i]() {
                                   int32_t good_pts = this->CheckRT(solutions[i].so3().matrix(),
                                                                    solutions[i].translation(),
                                                                    kp1,
                                                                    kp2,
                                                                    matches,
                                                                    K,
                                                                    th2,
                                                                    &inliers_list[i],
                                                                    &p3ts_list[i],
                                                                    &parallaxes[i]);
                                   return std::make_pair(good_pts, i);
                                 }));
  }
  std::vector<std::pair<int32_t, int32_t>> results;
  for (auto& future : futures) {
    results.push_back(future.get());
  }

  // Step 4.2 选取最大可三角化测量的点的数目
  auto max_good_pair = *std::max_element(
      results.begin(),
      results.end(),
      [](const std::pair<int32_t, int32_t>& a, const std::pair<int32_t, int32_t>& b) {
        return a.first < b.first;
      });

  int32_t max_good_pt = max_good_pair.first;
  // 统计四组解中重建的有效3D点个数 > 0.7 * max_good_pt 的解的数目
  // 如果有多个解同时满足该条件，认为结果太接近，nsimilar++，nsimilar>1就认为有问题了，后面返回false
  int32_t nsimilar = 0;
  for (const auto& res : results) {
    if (res.first > 0.7 * max_good_pt) {
      nsimilar++;
    }
  }

  // Step 4.3 确定最小的可以三角化的点数
  // 在0.9倍的内点数 和 指定值minTriangulated =50 中取最大的，也就是说至少50个
  int migood_pt = std::max(static_cast<int>(0.9 * N), min_triangulated_pts);

  // Step 4.4 四个结果中如果没有明显的最优结果，或者没有足够数量的三角化点，则返回失败
  // 条件1: 如果四组解能够重建的最多3D点个数小于所要求的最少3D点个数（mMigood_pt），失败
  // 条件2: 如果存在两组及以上的解能三角化出 > 0.7*max_good_pt的点，说明没有明显最优结果，失败
  if (max_good_pt < migood_pt || nsimilar > 1) {
    return false;
  }

  //  Step 4.5 选择最佳解记录结果
  // 条件1: 有效重建最多的3D点，即max_good_pt == good_ptx，也即是位于相机前方的3D点个数最多
  // 条件2: 三角化视差角 parallax 必须大于最小视差角 minParallax，角度越大3D点越稳定
  int32_t best_index = max_good_pair.second;
  if (parallaxes.at(best_index) > min_triangulated_pts) {
    // 存储3D坐标
    *p3d = p3ts_list.at(best_index);

    // 获取特征点向量的三角化测量标记
    *inliers = inliers_list.at(best_index);

    // 存储相机姿态
    *R21 = solutions.at(best_index).so3().matrix();
    *t21 = solutions.at(best_index).translation();

    // 结束
    return true;
  }

  // 如果有最优解但是不满足对应的parallax>minParallax，那么返回false表示求解失败
  return false;
}

bool SFM::ReconstructH(const Eigen::Matrix3d& H21,
                       const std::vector<cv::KeyPoint>& kp1,
                       const std::vector<cv::KeyPoint>& kp2,
                       const std::vector<cv::DMatch>& matches,
                       const int32_t min_triangulated_pts,
                       const double min_parallax,
                       std::vector<bool>* const inliers,
                       std::vector<Eigen::Vector3d>* const p3d,
                       Eigen::Matrix3d* const R21,
                       Eigen::Vector3d* const t21,
                       Eigen::Vector3d* const n21) const {
  // 统计匹配的特征点对中属于内点(Inlier)或有效点个数
  int N = 0;
  for (int32_t i = 0, iend = inliers->size(); i < iend; i++) {
    if ((*inliers)[i]) {
      N++;
    }
  }

  // 分解单应矩阵得到几组解
  std::vector<std::tuple<Eigen::Matrix3d, Eigen::Vector3d, Eigen::Vector3d>> solutions;
  DecomposeHomographyMatrix(H21, &solutions);

  // Instead of applying the visibility constraints proposed in the Faugeras' paper (which could
  // fail for points seen with low parallax) We reconstruct all hypotheses and check in terms of
  // triangulated points and parallax
  // 多线程同步进行检验
  std::vector<std::future<std::pair<int32_t, int32_t>>> futures;
  std::vector<std::vector<bool>> inliers_list(solutions.size(), *inliers);
  std::vector<std::vector<Eigen::Vector3d>> p3ts_list(solutions.size());
  std::vector<double> parallaxes(solutions.size());
  double th2 = config_.reproj_err_th();
  auto K = camera_model_->K_;
  for (int32_t i = 0, iend = solutions.size(); i < iend; ++i) {
    futures.push_back(std::async(std::launch::async,
                                 [this,
                                  &solutions,
                                  &kp1,
                                  &kp2,
                                  &matches,
                                  &K,
                                  th2,
                                  &inliers_list,
                                  &p3ts_list,
                                  &parallaxes,
                                  i]() {
                                   int32_t good_pts = this->CheckRT(std::get<0>(solutions[i]),
                                                                    std::get<1>(solutions[i]),
                                                                    kp1,
                                                                    kp2,
                                                                    matches,
                                                                    K,
                                                                    th2,
                                                                    &inliers_list[i],
                                                                    &p3ts_list[i],
                                                                    &parallaxes[i]);
                                   return std::make_pair(good_pts, i);
                                 }));
  }
  std::vector<std::pair<int32_t, int32_t>> results;
  for (auto& future : futures) {
    results.push_back(future.get());
  }

  auto best_good = std::make_pair(-1, -1);
  auto second_best_good = std::make_pair(-1, -1);
  for (const auto& result : results) {
    if (result.first > best_good.first) {
      second_best_good = best_good;
      best_good = result;
    } else if (result.first > second_best_good.first) {
      second_best_good = result;
    }
  }
  int32_t best_index = best_good.second;

  // Step 3 选择最优解。要满足下面的四个条件
  // 1. good点数最优解明显大于次优解，这里取0.75经验值
  // 2. 视角差大于规定的阈值
  // 3. good点数要大于规定的最小的被三角化的点数量
  // 4. good数要足够多，达到总数的90%以上
  if (second_best_good.first < 0.75 * best_good.first &&
      parallaxes.at(best_index) >= min_parallax && best_good.first > min_triangulated_pts &&
      best_good.first > 0.9 * N) {
    // 存储3D坐标
    *p3d = p3ts_list.at(best_index);

    // 获取特征点向量的三角化测量标记
    *inliers = inliers_list.at(best_index);

    // 存储相机姿态
    *R21 = std::get<0>(solutions.at(best_index));
    *t21 = std::get<1>(solutions.at(best_index));
    *n21 = std::get<2>(solutions.at(best_index));

    // 返回真，找到了最好的解
    return true;
  }

  return false;
}

/**
 * @brief 进行cheirality check，从而进一步找出F分解后最合适的解
 * @param R 旋转
 * @param t 平移
 * @param kps1 特征点
 * @param kps2 特征点
 * @param matches 匹配关系
 * @param inliers 匹配关系是否有效
 * @param K 内参
 * @param p3ts 三维点
 * @param th2 误差半径
 * @param parallax
 */
int32_t SFM::CheckRT(const Eigen::Matrix3d& R,
                     const Eigen::Vector3d& t,
                     const std::vector<cv::KeyPoint>& kps1,
                     const std::vector<cv::KeyPoint>& kps2,
                     const std::vector<cv::DMatch>& matches,
                     const Eigen::Matrix3d& K,
                     const double th2,
                     std::vector<bool>* const inliers,
                     std::vector<Eigen::Vector3d>* const p3ts,
                     double* const parallax) const {
  // 首先拿到相机内参
  const double fx = K(0, 0);
  const double fy = K(1, 1);
  const double cx = K(0, 2);
  const double cy = K(1, 2);

  // 三角化成功的3d点按照inliers来进行处理，这样比较优雅
  p3ts->resize(inliers->size());

  // 视差角，视差大一些相对来说更加鲁棒精确一些
  std::vector<double> cos_parallax_v;
  cos_parallax_v.reserve(inliers->size());

  // Camera 1 Projection Matrix K[I|0]
  // 步骤1：得到一个相机的投影矩阵
  // 以第一个相机的光心作为世界坐标系
  Eigen::Matrix<double, 3, 4> P1;
  P1.setZero();
  P1.block<3, 3>(0, 0) = K;

  Eigen::Vector3d O1;
  O1.setZero();

  // Camera 2 Projection Matrix K[R|t]
  // 步骤2：得到第二个相机的投影矩阵
  Eigen::Matrix<double, 3, 4> P2;
  P2.block<3, 3>(0, 0) = R;
  P2.block<3, 1>(0, 3) = t;
  P2 = K * P2;

  // 第二个相机的光心在世界坐标系下的坐标
  Eigen::Vector3d O2 = -R.transpose() * t;

  int32_t good_pt = 0;
  for (int32_t i = 0, iend = inliers->size(); i < iend; i++) {
    if (!(*inliers)[i]) {
      continue;
    }
    // kp1和kp2是匹配特征点
    const auto& kp1 = kps1[matches[i].queryIdx];
    const auto& kp2 = kps2[matches[i].trainIdx];
    Eigen::Vector3d p3ds;
    Eigen::Vector3d x_p1(kp1.pt.x, kp1.pt.y, 1);
    Eigen::Vector3d x_p2(kp2.pt.x, kp2.pt.y, 1);
    // 步骤3：利用三角法恢复三维点p3ds
    Triangulate(x_p1, x_p2, P1, P2, &p3ds);
    // 检验三角化的点结果
    if (!std::isfinite(p3ds(0)) || !std::isfinite(p3ds(1)) || !std::isfinite(p3ds(2))) {
      // 自始至终都使用inliers来跟踪标记好的点对
      inliers->at(i) = false;
      continue;
    }

    // Check parallax
    // 步骤4：计算视差角余弦值
    Eigen::Vector3d normal1 = p3ds - O1;
    double dist1 = normal1.norm();

    Eigen::Vector3d normal2 = p3ds - O2;
    double dist2 = normal2.norm();

    double cos_parallax = normal1.dot(normal2) / (dist1 * dist2);

    // 步骤5：判断3D点是否在两个摄像头前方
    // Check depth in front of first camera (only if enough parallax, as "infinite" points can
    // easily go to negative depth) 步骤5.1：3D点深度为负，在第一个摄像头后方，淘汰
    if (p3ds(2) <= 0 && cos_parallax < 0.99998) {
      inliers->at(i) = false;
      continue;
    }

    // Check depth in front of second camera (only if enough parallax, as "infinite" points can
    // easily go to negative depth) 步骤5.2：3D点深度为负，在第二个摄像头后方，淘汰
    Eigen::Vector3d p3dC2 = R * p3ds + t;

    if (p3dC2(2) <= 0 && cos_parallax < 0.99998) {
      inliers->at(i) = false;
      continue;
    };

    // 步骤6：计算重投影误差
    // Check reprojection error in first image
    // 计算3D点在第一个图像上的投影误差
    double im1x, im1y;
    double invZ1 = 1.0 / p3ds(2);
    im1x = fx * p3ds(0) * invZ1 + cx;
    im1y = fy * p3ds(1) * invZ1 + cy;

    double squareError1 =
        (im1x - kp1.pt.x) * (im1x - kp1.pt.x) + (im1y - kp1.pt.y) * (im1y - kp1.pt.y);

    // 步骤6.1：重投影误差太大，跳过淘汰
    // 一般视差角比较小时重投影误差比较大
    if (squareError1 > th2) {
      inliers->at(i) = false;
      continue;
    }

    // Check reprojection error in second image
    // 计算3D点在第二个图像上的投影误差
    double im2x, im2y;
    double invZ2 = 1.0 / p3dC2(2);
    im2x = fx * p3dC2(0) * invZ2 + cx;
    im2y = fy * p3dC2(1) * invZ2 + cy;

    double squareError2 =
        (im2x - kp2.pt.x) * (im2x - kp2.pt.x) + (im2y - kp2.pt.y) * (im2y - kp2.pt.y);

    // 步骤6.1：重投影误差太大，跳过淘汰
    // 一般视差角比较小时重投影误差比较大
    if (squareError2 > th2) {
      inliers->at(i) = false;
      continue;
    }
    // 步骤7：统计经过检验的3D点个数，记录3D点视差角
    cos_parallax_v.emplace_back(cos_parallax);
    p3ts->at(i) = Eigen::Vector3d(p3ds(0), p3ds(1), p3ds(2));
    good_pt++;
  }

  // 7 得到3D点中较小的视差角，并且转换成为角度制表示
  if (good_pt > 0) {
    // 从小到大排序，注意cos_parallax_v值越大，视差越小
    std::sort(cos_parallax_v.begin(), cos_parallax_v.end());

    // !排序后并没有取最小的视差角，而是取一个较小的视差角
    // 作者的做法：如果经过检验过后的有效3D点小于50个，那么就取最后那个最小的视差角(cos值最大)
    // 如果大于50个，就取排名第50个的较小的视差角即可，为了避免3D点太多时出现太小的视差角
    size_t idx = std::min(50, int(cos_parallax_v.size() - 1));
    // 将这个选中的角弧度制转换为角度制
    *parallax = std::acos(cos_parallax_v[idx]) * 180 / CV_PI;
  } else {
    *parallax = 0.0;
  }

  return good_pt;
}

}  // namespace Camera
