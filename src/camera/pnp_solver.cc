#include "camera/pnp_solver.h"

namespace Camera {

PnpSolver::PnpSolver(const CameraConfig::PnpSolverConfig& config,
                     const std::shared_ptr<Camera::CameraBase>& cam_model_ptr)
    : camera_model_ptr_(cam_model_ptr) {}

bool PnpSolver::Solve(const std::vector<Eigen::Vector3d>& p3ds,
                      const std::vector<Eigen::Vector2d>& pt2s,
                      Sophus::SE3d* const relative_pose) {
  timer_.StartTimer("PNP Solver");
  bool res = false;
  switch (config_.pnp_solve_method()) {
    case CameraConfig::PnpSolverConfig::PnpSolveMethod::PnpSolverConfig_PnpSolveMethod_DLT: {
      res = DLTSolve(p3ds, pt2s, relative_pose);
      break;
    }
    case CameraConfig::PnpSolverConfig::PnpSolveMethod::PnpSolverConfig_PnpSolveMethod_EPNP: {
      res = P3PSolve(p3ds, pt2s, relative_pose);
      break;
    }

    case CameraConfig::PnpSolverConfig::PnpSolveMethod::PnpSolverConfig_PnpSolveMethod_MLPNP: {
      res = EPNPSolve(p3ds, pt2s, relative_pose);
      break;
    }

    case CameraConfig::PnpSolverConfig::PnpSolveMethod::
        PnpSolverConfig_PnpSolveMethod_OpenCV_SOLVEPNP_ITERATIVE: {
      res = MLPNPSolve(p3ds, pt2s, relative_pose);
      break;
    }

    case CameraConfig::PnpSolverConfig::PnpSolveMethod::
        PnpSolverConfig_PnpSolveMethod_OpenCV_SOLVEPNP_EPNP: {
      res = CvIterative(p3ds, pt2s, relative_pose);
      break;
    }

    case CameraConfig::PnpSolverConfig::PnpSolveMethod::
        PnpSolverConfig_PnpSolveMethod_OpenCV_SOLVEPNP_P3P: {
      res = CvEPNP(p3ds, pt2s, relative_pose);
      break;
    }

    case CameraConfig::PnpSolverConfig::PnpSolveMethod::
        PnpSolverConfig_PnpSolveMethod_OpenCV_SOLVEPNP_DLS: {
      res = CvP3P(p3ds, pt2s, relative_pose);
      break;
    }

    case CameraConfig::PnpSolverConfig::PnpSolveMethod::
        PnpSolverConfig_PnpSolveMethod_OpenCV_SOLVEPNP_UPNP: {
      res = CvDLS(p3ds, pt2s, relative_pose);
      break;
    }

    case CameraConfig::PnpSolverConfig::PnpSolveMethod::
        PnpSolverConfig_PnpSolveMethod_OpenCV_SOLVEPNP_AP3P: {
      res = CvUPNP(p3ds, pt2s, relative_pose);
      break;
    }

    case CameraConfig::PnpSolverConfig::PnpSolveMethod::
        PnpSolverConfig_PnpSolveMethod_OpenCV_SOLVEPNP_IPPE: {
      res = CvAP3P(p3ds, pt2s, relative_pose);
      break;
    }

    case CameraConfig::PnpSolverConfig::PnpSolveMethod::
        PnpSolverConfig_PnpSolveMethod_OpenCV_SOLVEPNP_IPPE_SQUARE: {
      res = CvIPPE(p3ds, pt2s, relative_pose);
      break;
    }

    case CameraConfig::PnpSolverConfig::PnpSolveMethod::
        PnpSolverConfig_PnpSolveMethod_OpenCV_SOLVEPNP_SQPNP: {
      res = CvSquare(p3ds, pt2s, relative_pose);
      break;
    }

    default: {
      res = CvSQPNP(p3ds, pt2s, relative_pose);
      break;
    }
  }
  timer_.StopTimer();
  timer_.PrintElapsedTime();
  return res;
}

bool PnpSolver::DLTSolve(const std::vector<Eigen::Vector3d>& p3ds,
                         const std::vector<Eigen::Vector2d>& pt2s,
                         Sophus::SE3d* const relative_pose) const {
  // 检查输入是否为空，以及相对位姿指针是否为空
  CHECK(!p3ds.empty() && !pt2s.empty() && relative_pose != nullptr) << "Empty input";
  CHECK(p3ds.size() == pt2s.size() && p3ds.size() >= 6) << "Input size error !";

  // 获取相机参数矩阵K
  const Eigen::Matrix3d K = camera_model_ptr_->K_;
  const double fx = K(0, 0);
  const double fy = K(1, 1);
  const double cx = K(0, 2);
  const double cy = K(1, 2);

  const int32_t n = p3ds.size();

  // 步骤1：构造矩阵A，大小为2n x 12
  Eigen::MatrixXd A(2 * n, 12);
  for (int32_t i = 0; i < n; ++i) {
    const Eigen::Vector3d& pt3d = p3ds[i];
    const Eigen::Vector2d& pt2d = pt2s[i];

    const double& x = pt3d[0];
    const double& y = pt3d[1];
    const double& z = pt3d[2];
    const double& u = pt2d[0];
    const double& v = pt2d[1];

    A(2 * i, 0) = x * fx;
    A(2 * i, 1) = y * fx;
    A(2 * i, 2) = z * fx;
    A(2 * i, 3) = fx;
    A(2 * i, 4) = 0.0;
    A(2 * i, 5) = 0.0;
    A(2 * i, 6) = 0.0;
    A(2 * i, 7) = 0.0;
    A(2 * i, 8) = x * cx - u * x;
    A(2 * i, 9) = y * cx - u * y;
    A(2 * i, 10) = z * cx - u * z;
    A(2 * i, 11) = cx - u;

    A(2 * i + 1, 0) = 0.0;
    A(2 * i + 1, 1) = 0.0;
    A(2 * i + 1, 2) = 0.0;
    A(2 * i + 1, 3) = 0.0;
    A(2 * i + 1, 4) = x * fy;
    A(2 * i + 1, 5) = y * fy;
    A(2 * i + 1, 6) = z * fy;
    A(2 * i + 1, 7) = fy;
    A(2 * i + 1, 8) = x * cy - v * x;
    A(2 * i + 1, 9) = y * cy - v * y;
    A(2 * i + 1, 10) = z * cy - v * z;
    A(2 * i + 1, 11) = cy - v;
  }

  // 步骤2：通过SVD求解Ax = 0
  Eigen::JacobiSVD<Eigen::MatrixXd> svd_A(A, Eigen::ComputeThinV);
  const Eigen::MatrixXd& V_A = svd_A.matrixV();
  const Eigen::VectorXd& Sigma_A = svd_A.singularValues();

  // 检查SVD是否成功
  if (Sigma_A.minCoeff() < 1e-10) {
    LOG(ERROR) << "SVD solve failed !";
    return false;
  }

  // 获取a1-a12
  Eigen::VectorXd a = V_A.col(11);
  Eigen::Matrix3d R_bar;
  R_bar << a(0), a(1), a(2), a(4), a(5), a(6), a(8), a(9), a(10);

  // 步骤3：重构旋转矩阵R和比例因子beta
  Eigen::JacobiSVD<Eigen::Matrix3d> svd_R(R_bar, Eigen::ComputeFullU | Eigen::ComputeFullV);
  Eigen::Matrix3d U_R = svd_R.matrixU();
  Eigen::Matrix3d V_R = svd_R.matrixV();
  Eigen::Vector3d V_Sigma = svd_R.singularValues();

  // 检查R的SVD分解是否成功
  if (V_Sigma.minCoeff() < 1e-10) {
    LOG(ERROR) << "SVD solve failed !";
    return false;
  }

  Eigen::Matrix3d rotation = U_R * V_R.transpose();
  double beta = 1.0 / V_Sigma.mean();

  // 步骤4：计算平移向量t
  Eigen::Vector3d t_bar(a(3), a(7), a(11));
  Eigen::Vector3d translation = beta * t_bar;

  // 检查正负
  int32_t num_positive = 0;
  int32_t num_negative = 0;
  for (const auto& pt3d : p3ds) {
    double lambda = beta * (pt3d[0] * a(8) + pt3d[1] * a(9) + pt3d[2] * a(10) + a(11));
    if (lambda >= 0) {
      ++num_positive;
    } else {
      ++num_negative;
    }
  }

  if (num_positive < num_negative) {
    rotation = -rotation;
    translation = -translation;
  }

  relative_pose->so3() = Sophus::SO3d(rotation);
  relative_pose->translation() = translation;

  return true;
}

bool PnpSolver::P3PSolve(const std::vector<Eigen::Vector3d>& p3ds,
                         const std::vector<Eigen::Vector2d>& pt2s,
                         Sophus::SE3d* const relative_pose) const {
  return true;
}

bool PnpSolver::MLPNPSolve(const std::vector<Eigen::Vector3d>& p3ds,
                           const std::vector<Eigen::Vector2d>& pt2s,
                           Sophus::SE3d* const relative_pose) const {
  return true;
}

// =====================================================================================
// EPNP原理参考：https://zhuanlan.zhihu.com/p/59070440
bool PnpSolver::EPNPSolve(const std::vector<Eigen::Vector3d>& p3ds,
                          const std::vector<Eigen::Vector2d>& pt2s,
                          Sophus::SE3d* const relative_pose) const {
  // 检查输入是否为空，以及相对位姿指针是否为空
  CHECK(!p3ds.empty() && !pt2s.empty() && relative_pose != nullptr) << "Empty input";
  CHECK(p3ds.size() == pt2s.size() && p3ds.size() >= 4) << "Input size error !";
  const Eigen::Matrix3d K = camera_model_ptr_->K_;

  // 选择世界坐标系下的控制点
  std::vector<Eigen::Vector3d> world_control_p3ds;
  SelectControlPoints(p3ds, &world_control_p3ds);

  // 计算齐次重心坐标
  std::vector<Eigen::Vector4d> hb_coordinates;
  ComputeHBCoordinates(p3ds, world_control_p3ds, &hb_coordinates);

  // 计算相机坐标系下的控制点
  // 构造 Mx = 0
  Eigen::MatrixXd M;
  ConstructM(K, hb_coordinates, pt2s, &M);
  Eigen::Matrix<double, 12, 4> eigen_vectors;
  GetFourEigenVectors(M, &eigen_vectors);

  // 构造 L * \beta = \rho
  Eigen::Matrix<double, 6, 10> L;
  ComputeL(eigen_vectors, &L);

  Eigen::Matrix<double, 6, 1> rho;
  ComputeRho(world_control_p3ds, &rho);

  Eigen::Matrix3d R;
  Eigen::Vector3d t;
  double min_error = std::numeric_limits<double>::max();

  // 处理不同的N值情况
  for (int N : {2, 3, 4}) {
    Eigen::Vector4d betas;
    Eigen::Matrix3d tmp_R;
    Eigen::Vector3d tmp_t;

    SolveAndOptimizeBeta(N, eigen_vectors, L, rho, &betas);
    ComputeRtFromBetas(eigen_vectors, hb_coordinates, betas, p3ds, &tmp_R, &tmp_t);

    double error = ReprojectionError(K, p3ds, pt2s, tmp_R, tmp_t);
    if (error < min_error) {
      min_error = error;
      R = tmp_R;
      t = tmp_t;
    }
  }

  relative_pose->so3() = Sophus::SO3d(R);
  relative_pose->translation() = t;

  return true;
}

void PnpSolver::SelectControlPoints(const std::vector<Eigen::Vector3d>& p3ds,
                                    std::vector<Eigen::Vector3d>* const control_p3ds) const {
  const int32_t n = p3ds.size();
  control_p3ds->reserve(4);

  // select the center points
  Eigen::Vector3d cw1(0.0, 0.0, 0.0);
  for (const auto& pt : p3ds) {
    cw1 += pt;
  }
  cw1 /= static_cast<double>(n);

  // PCA
  Eigen::MatrixXd A;
  A.resize(n, 3);
  for (int32_t i = 0; i < n; i++) {
    A.block(i, 0, 1, 3) = (p3ds.at(i) - cw1).transpose();
  }
  Eigen::Matrix3d ATA = A.transpose() * A;

  Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> es(ATA);
  Eigen::Vector3d D = es.eigenvalues();
  Eigen::MatrixXd V = es.eigenvectors();

  Eigen::Vector3d cw2 = cw1 + sqrt(D(0) / n) * V.block(0, 0, 3, 1);
  Eigen::Vector3d cw3 = cw1 + sqrt(D(1) / n) * V.block(0, 1, 3, 1);
  Eigen::Vector3d cw4 = cw1 + sqrt(D(2) / n) * V.block(0, 2, 3, 1);

  control_p3ds->emplace_back(cw1);
  control_p3ds->emplace_back(cw2);
  control_p3ds->emplace_back(cw3);
  control_p3ds->emplace_back(cw4);
}

void PnpSolver::ComputeHBCoordinates(const std::vector<Eigen::Vector3d>& pts3d,
                                     const std::vector<Eigen::Vector3d>& control_points,
                                     std::vector<Eigen::Vector4d>* const hb_coordinates) const {
  const int32_t n = pts3d.size();
  hb_coordinates->clear();
  hb_coordinates->reserve(n);

  // construct C
  Eigen::Matrix4d C;
  for (int i = 0; i < 4; i++) {
    C.block(0, i, 3, 1) = control_points.at(i);
  }
  C.block(3, 0, 1, 4) = Eigen::Vector4d(1.0, 1.0, 1.0, 1.0);

  Eigen::Matrix4d C_inv = C.inverse();

  // compute \alpha_ij for all points
  for (int i = 0; i < n; i++) {
    Eigen::Vector4d ptw(0.0, 0.0, 0.0, 1.0);
    ptw.block(0, 0, 3, 1) = pts3d.at(i);
    hb_coordinates->emplace_back(C_inv * ptw);
  }
}

void PnpSolver::ConstructM(const Eigen::Matrix3d& K,
                           const std::vector<Eigen::Vector4d>& hb_coordinates,
                           const std::vector<Eigen::Vector2d>& pts2d,
                           Eigen::MatrixXd* const M) const {
  // get camera intrinsics
  const double fx = K(0, 0);
  const double fy = K(1, 1);
  const double cx = K(0, 2);
  const double cy = K(1, 2);

  // init M
  const int32_t n = pts2d.size();
  M->resize(2 * n, 12);

  // Fill M
  for (int32_t i = 0; i < n; i++) {
    // get alphas
    const Eigen::Vector4d& alphas = hb_coordinates.at(i);

    const double& alpha_i1 = alphas(0);
    const double& alpha_i2 = alphas(1);
    const double& alpha_i3 = alphas(2);
    const double& alpha_i4 = alphas(3);

    // get uv
    const double& u = pts2d.at(i)(0);
    const double& v = pts2d.at(i)(1);

    // idx
    const int id0 = 2 * i;
    const int id1 = id0 + 1;

    // the first line
    (*M)(id0, 0) = alpha_i1 * fx;
    (*M)(id0, 1) = 0.0;
    (*M)(id0, 2) = alpha_i1 * (cx - u);

    (*M)(id0, 3) = alpha_i2 * fx;
    (*M)(id0, 4) = 0.0;
    (*M)(id0, 5) = alpha_i2 * (cx - u);

    (*M)(id0, 6) = alpha_i3 * fx;
    (*M)(id0, 7) = 0.0;
    (*M)(id0, 8) = alpha_i3 * (cx - u);

    (*M)(id0, 9) = alpha_i4 * fx;
    (*M)(id0, 10) = 0.0;
    (*M)(id0, 11) = alpha_i4 * (cx - u);

    // for the second line
    (*M)(id1, 0) = 0.0;
    (*M)(id1, 1) = alpha_i1 * fy;
    (*M)(id1, 2) = alpha_i1 * (cy - v);

    (*M)(id1, 3) = 0.0;
    (*M)(id1, 4) = alpha_i2 * fy;
    (*M)(id1, 5) = alpha_i2 * (cy - v);

    (*M)(id1, 6) = 0.0;
    (*M)(id1, 7) = alpha_i3 * fy;
    (*M)(id1, 8) = alpha_i3 * (cy - v);

    (*M)(id1, 9) = 0.0;
    (*M)(id1, 10) = alpha_i4 * fy;
    (*M)(id1, 11) = alpha_i4 * (cy - v);
  }
}

void PnpSolver::GetFourEigenVectors(const Eigen::MatrixXd& M,
                                    Eigen::Matrix<double, 12, 4>* const eigen_vectors) const {
  Eigen::Matrix<double, 12, 12> MTM = M.transpose() * M;
  Eigen::SelfAdjointEigenSolver<Eigen::Matrix<double, 12, 12>> es(MTM);
  Eigen::MatrixXd e_vectors = es.eigenvectors();
  eigen_vectors->block(0, 0, 12, 4) = e_vectors.block(0, 0, 12, 4);
}

void PnpSolver::ComputeL(const Eigen::Matrix<double, 12, 4>& eigen_vectors,
                         Eigen::Matrix<double, 6, 10>* const L) const {
  //[B11 B12 B22 B13 B23 B33 B14 B24 B34 B44]
  const int idx0[6] = {0, 0, 0, 1, 1, 2};
  const int idx1[6] = {1, 2, 3, 2, 3, 3};

  for (int i = 0; i < 6; i++) {
    const int idi = idx0[i] * 3;
    const int idj = idx1[i] * 3;

    // the first control point.
    const Eigen::Vector3d v1i = eigen_vectors.block(idi, 0, 3, 1);
    const Eigen::Vector3d v2i = eigen_vectors.block(idi, 1, 3, 1);
    const Eigen::Vector3d v3i = eigen_vectors.block(idi, 2, 3, 1);
    const Eigen::Vector3d v4i = eigen_vectors.block(idi, 3, 3, 1);

    // the second control point
    const Eigen::Vector3d v1j = eigen_vectors.block(idj, 0, 3, 1);
    const Eigen::Vector3d v2j = eigen_vectors.block(idj, 1, 3, 1);
    const Eigen::Vector3d v3j = eigen_vectors.block(idj, 2, 3, 1);
    const Eigen::Vector3d v4j = eigen_vectors.block(idj, 3, 3, 1);

    Eigen::Vector3d S1 = v1i - v1j;
    Eigen::Vector3d S2 = v2i - v2j;
    Eigen::Vector3d S3 = v3i - v3j;
    Eigen::Vector3d S4 = v4i - v4j;

    Eigen::Matrix<double, 1, 3> S1_T = S1.transpose();
    Eigen::Matrix<double, 1, 3> S2_T = S2.transpose();
    Eigen::Matrix<double, 1, 3> S3_T = S3.transpose();
    Eigen::Matrix<double, 1, 3> S4_T = S4.transpose();

    //[B11 B12 B22 B13 B23 B33 B14 B24 B34 B44]
    (*L)(i, 0) = S1_T * S1;
    (*L)(i, 1) = 2 * S1_T * S2;
    (*L)(i, 2) = S2_T * S2;
    (*L)(i, 3) = 2 * S1_T * S3;
    (*L)(i, 4) = 2 * S2_T * S3;
    (*L)(i, 5) = S3_T * S3;
    (*L)(i, 6) = 2 * S1_T * S4;
    (*L)(i, 7) = 2 * S2_T * S4;
    (*L)(i, 8) = 2 * S3_T * S4;
    (*L)(i, 9) = S4_T * S4;
  }
}

void PnpSolver::ComputeRho(const std::vector<Eigen::Vector3d>& control_points,
                           Eigen::Matrix<double, 6, 1>* const rho) const {
  const int idx0[6] = {0, 0, 0, 1, 1, 2};
  const int idx1[6] = {1, 2, 3, 2, 3, 3};
  for (int i = 0; i < 6; i++) {
    Eigen::Vector3d v01 = control_points.at(idx0[i]) - control_points.at(idx1[i]);
    (*rho)(i, 0) = (v01.transpose() * v01);
  }
}

void PnpSolver::SolveBetaN2(const Eigen::Matrix<double, 12, 4>& eigen_vectors,
                            const Eigen::Matrix<double, 6, 10>& L,
                            const Eigen::Matrix<double, 6, 1>& rho,
                            Eigen::Vector4d* const betas) const {
  //                  [B11 B12 B22 B13 B23 B33 B14 B24 B34 B44]
  // betas_approx_2 = [B11 B12 B22                            ]

  const Eigen::Matrix<double, 6, 3>& L_approx = L.block(0, 0, 6, 3);
  Eigen::Vector3d b3 = L_approx.fullPivHouseholderQr().solve(rho);

  if (b3[0] < 0) {
    (*betas)(0) = sqrt(-b3[0]);
    (*betas)(1) = (b3[2] < 0) ? sqrt(-b3[2]) : 0.0;
  } else {
    (*betas)(0) = sqrt(b3[0]);
    (*betas)(1) = (b3[2] > 0) ? sqrt(b3[2]) : 0.0;
  }

  if (b3[1] < 0) {
    (*betas)(0) = -(*betas)(0);
  }

  (*betas)(2) = 0.0;
  (*betas)(3) = 0.0;

  // Check betas.
  std::vector<Eigen::Vector3d> camera_control_points;
  ComputeCameraControlPoints(eigen_vectors, *betas, &camera_control_points);
  if (IsGoodBetas(camera_control_points) == false) {
    *betas = -*betas;
  }
}

void PnpSolver::SolveBetaN3(const Eigen::Matrix<double, 12, 4>& eigen_vectors,
                            const Eigen::Matrix<double, 6, 10>& L,
                            const Eigen::Matrix<double, 6, 1>& rho,
                            Eigen::Vector4d* const betas) const {
  //                  [B11 B12 B22 B13 B23 B33 B14 B24 B34 B44]
  // betas_approx_3 = [B11 B12 B22 B13 B23                    ]

  const Eigen::Matrix<double, 6, 5>& L_approx = L.block(0, 0, 6, 5);
  Eigen::Matrix<double, 5, 1> b5 = L_approx.fullPivHouseholderQr().solve(rho);

  if (b5[0] < 0) {
    (*betas)(0) = sqrt(-b5[0]);
    (*betas)(1) = (b5[2] < 0) ? sqrt(-b5[2]) : 0.0;
  } else {
    (*betas)(0) = sqrt(b5[0]);
    (*betas)(1) = (b5[2] > 0) ? sqrt(b5[2]) : 0.0;
  }
  if (b5[1] < 0) {
    (*betas)(0) = -(*betas)(0);
  }
  (*betas)(2) = b5[3] / (*betas)(0);
  (*betas)(3) = 0.0;

  // Check betas.
  std::vector<Eigen::Vector3d> camera_control_points;
  ComputeCameraControlPoints(eigen_vectors, *betas, &camera_control_points);
  if (IsGoodBetas(camera_control_points) == false) {
    *betas = -*betas;
  }
}

void PnpSolver::SolveBetaN4(const Eigen::Matrix<double, 12, 4>& eigen_vectors,
                            const Eigen::Matrix<double, int(6), int(10)>& L,
                            const Eigen::Matrix<double, int(6), int(1)>& rho,
                            Eigen::Vector4d* const betas) const {
  // betas10        = [B11 B12 B22 B13 B23 B33 B14 B24 B34 B44]
  // betas_approx_1 = [B11 B12     B13         B14]

  Eigen::Matrix<double, 6, 4> L_approx;
  L_approx.block(0, 0, 6, 2) = L.block(0, 0, 6, 2);
  L_approx.block(0, 2, 6, 1) = L.block(0, 3, 6, 1);
  L_approx.block(0, 3, 6, 1) = L.block(0, 6, 6, 1);

  Eigen::Vector4d b4 = L_approx.fullPivHouseholderQr().solve(rho);

  if (b4[0] < 0) {
    (*betas)(0) = sqrt(-b4[0]);
    (*betas)(1) = -b4[1] / (*betas)(0);
    (*betas)(2) = -b4[2] / (*betas)(0);
    (*betas)(3) = -b4[3] / (*betas)(0);
  } else {
    (*betas)(0) = sqrt(b4[0]);
    (*betas)(1) = b4[1] / (*betas)(0);
    (*betas)(2) = b4[2] / (*betas)(0);
    (*betas)(3) = b4[3] / (*betas)(0);
  }

  // Check betas.
  std::vector<Eigen::Vector3d> camera_control_points;
  ComputeCameraControlPoints(eigen_vectors, *betas, &camera_control_points);
  if (IsGoodBetas(camera_control_points) == false) {
    *betas = -*betas;
  }
}

void PnpSolver::OptimizeBeta(const Eigen::Matrix<double, int(6), int(10)>& L,
                             const Eigen::Matrix<double, int(6), int(1)>& rho,
                             Eigen::Vector4d* const betas) const {
  const int iter_num = 5;

  for (int nit = 0; nit < iter_num; nit++) {
    // construct J
    Eigen::Matrix<double, 6, 4> J;
    for (int i = 0; i < 6; i++) {
      // [B11 B12 B22 B13 B23 B33 B14 B24 B34 B44]
      J(i, 0) = 2 * (*betas)(0) * L(i, 0) + (*betas)(1) * L(i, 1) + (*betas)(2) * L(i, 3) +
                (*betas)(3) * L(i, 6);
      J(i, 1) = (*betas)(0) * L(i, 1) + 2 * (*betas)(1) * L(i, 2) + (*betas)(2) * L(i, 3) +
                (*betas)(3) * L(i, 7);
      J(i, 2) = (*betas)(0) * L(i, 3) + (*betas)(1) * L(i, 4) + 2 * (*betas)(2) * L(i, 5) +
                (*betas)(3) * L(i, 8);
      J(i, 3) = (*betas)(0) * L(i, 6) + (*betas)(1) * L(i, 7) + (*betas)(2) * L(i, 8) +
                2 * (*betas)(3) * L(i, 9);
    }

    Eigen::Matrix<double, 4, 6> J_T = J.transpose();
    Eigen::Matrix<double, 4, 4> H = J_T * J;

    // Compute residual
    // [B11 B12 B22 B13 B23 B33 B14 B24 B34 B44]
    // [B00 B01 B11 B02 B12 B22 B03 B13 B23 B33]
    Eigen::Matrix<double, 10, 1> bs;
    bs << (*betas)(0) * (*betas)(0), (*betas)(0) * (*betas)(1), (*betas)(1) * (*betas)(1),
        (*betas)(0) * (*betas)(2), (*betas)(1) * (*betas)(2), (*betas)(2) * (*betas)(2),
        (*betas)(0) * (*betas)(3), (*betas)(1) * (*betas)(3), (*betas)(2) * (*betas)(3),
        (*betas)(3) * (*betas)(3);
    Eigen::Matrix<double, 6, 1> residual = L * bs - rho;

    // std::cout << "Error " << residual.transpose() * residual << "\n";

    // Solve J^T * J \delta_beta = -J^T * residual;
    Eigen::Matrix<double, 4, 1> delta_betas = H.fullPivHouseholderQr().solve(-J_T * residual);

    // update betas;
    *betas += delta_betas;
  }  // iter n times.
}  //

void PnpSolver::ComputeCameraControlPoints(
    const Eigen::Matrix<double, int(12), int(4)>& eigen_vectors,
    const Eigen::Vector4d& betas,
    std::vector<Eigen::Vector3d>* const camera_control_points) const {
  camera_control_points->clear();
  camera_control_points->reserve(4);

  Eigen::Matrix<double, 12, 1> vec =
      betas(0) * eigen_vectors.block(0, 0, 12, 1) + betas(1) * eigen_vectors.block(0, 1, 12, 1) +
      betas(2) * eigen_vectors.block(0, 2, 12, 1) + betas(3) * eigen_vectors.block(0, 3, 12, 1);

  for (int i = 0; i < 4; i++) {
    camera_control_points->emplace_back(vec.block(i * 3, 0, 3, 1));
  }
}

bool PnpSolver::IsGoodBetas(const std::vector<Eigen::Vector3d>& camera_control_points) const {
  int num_positive = 0;
  int num_negative = 0;

  for (int i = 0; i < 4; i++) {
    if (camera_control_points.at(i)[2] > 0) {
      num_positive++;
    } else {
      num_negative++;
    }
  }

  if (num_negative >= num_positive) {
    return false;
  }

  return true;
}

void PnpSolver::RebuiltPts3dCamera(const std::vector<Eigen::Vector3d>& camera_control_points,
                                   const std::vector<Eigen::Vector4d>& hb_coordinates,
                                   std::vector<Eigen::Vector3d>* const pts3d_camera) const {
  const int32_t n = hb_coordinates.size();
  pts3d_camera->clear();
  pts3d_camera->reserve(n);

  for (int i = 0; i < n; i++) {
    Eigen::Vector4d alphas = hb_coordinates.at(i);

    Eigen::Vector3d ptc =
        camera_control_points[0] * alphas[0] + camera_control_points[1] * alphas[1] +
        camera_control_points[2] * alphas[2] + camera_control_points[3] * alphas[3];
    pts3d_camera->emplace_back(ptc);
  }
}

void PnpSolver::ComputeRt(const std::vector<Eigen::Vector3d>& pts3d_camera,
                          const std::vector<Eigen::Vector3d>& pts3d_world,
                          Eigen::Matrix3d* const R,
                          Eigen::Vector3d* const t) const {
  const int32_t n = pts3d_camera.size();
  // step 1. compute center points
  Eigen::Vector3d pcc(0.0, 0.0, 0.0);
  Eigen::Vector3d pcw(0.0, 0.0, 0.0);

  for (int i = 0; i < n; i++) {
    pcc += pts3d_camera.at(i);
    pcw += pts3d_world.at(i);
  }

  pcc /= (double)n;
  pcw /= (double)n;

  // step 2. remove centers.
  Eigen::MatrixXd Pc, Pw;
  Pc.resize(n, 3);
  Pw.resize(n, 3);

  for (int i = 0; i < n; i++) {
    Pc.block(i, 0, 1, 3) = (pts3d_camera.at(i) - pcc).transpose();

    Pw.block(i, 0, 1, 3) = (pts3d_world.at(i) - pcw).transpose();
  }

  // step 3. compute R.
  Eigen::Matrix3d W = Pc.transpose() * Pw;

  Eigen::JacobiSVD<Eigen::Matrix3d> svd(W, Eigen::ComputeFullU | Eigen::ComputeFullV);
  Eigen::Matrix3d U = svd.matrixU();
  Eigen::Matrix3d V = svd.matrixV();

  *R = U * V.transpose();

  if (R->determinant() < 0) {
    R->block(2, 0, 1, 3) = -R->block(2, 0, 1, 3);
  }

  // step 3. compute t
  *t = pcc - *R * pcw;
}

double PnpSolver::ReprojectionError(const Eigen::Matrix3d& K,
                                    const std::vector<Eigen::Vector3d>& pts3d_world,
                                    const std::vector<Eigen::Vector2d>& pts2d,
                                    const Eigen::Matrix3d& R,
                                    const Eigen::Vector3d& t) const {
  const int32_t n = pts3d_world.size();
  double sum_err2 = 0.0;
  for (size_t i = 0; i < pts3d_world.size(); i++) {
    const Eigen::Vector3d& ptw = pts3d_world.at(i);
    Eigen::Vector3d lamda_uv = K * (R * ptw + t);
    Eigen::Vector2d uv = lamda_uv.block(0, 0, 2, 1) / lamda_uv(2);

    Eigen::Vector2d e_uv = pts2d.at(i) - uv;
    sum_err2 += e_uv.transpose() * e_uv;
  }

  return sqrt(sum_err2 / (double)n);
}

void PnpSolver::SolveAndOptimizeBeta(int N,
                                     const Eigen::Matrix<double, 12, 4>& eigen_vectors,
                                     const Eigen::Matrix<double, 6, 10>& L,
                                     const Eigen::Matrix<double, 6, 1>& rho,
                                     Eigen::Vector4d* betas) const {
  switch (N) {
    case 2:
      SolveBetaN2(eigen_vectors, L, rho, betas);
      break;
    case 3:
      SolveBetaN3(eigen_vectors, L, rho, betas);
      break;
    case 4:
      SolveBetaN4(eigen_vectors, L, rho, betas);
      break;
    default:
      LOG(ERROR) << "Unsupported N value: " << N;
      return;
  }
  OptimizeBeta(L, rho, betas);
}

void PnpSolver::ComputeRtFromBetas(const Eigen::Matrix<double, 12, 4>& eigen_vectors,
                                   const std::vector<Eigen::Vector4d>& hb_coordinates,
                                   const Eigen::Vector4d& betas,
                                   const std::vector<Eigen::Vector3d>& p3ds,
                                   Eigen::Matrix3d* const R,
                                   Eigen::Vector3d* const t) const {
  std::vector<Eigen::Vector3d> camera_control_p3ds;
  ComputeCameraControlPoints(eigen_vectors, betas, &camera_control_p3ds);
  std::vector<Eigen::Vector3d> p3ds_camera;
  RebuiltPts3dCamera(camera_control_p3ds, hb_coordinates, &p3ds_camera);
  ComputeRt(p3ds_camera, p3ds, R, t);
}

// 以下是调用OpenCV库实现PNP求解
bool PnpSolver::CvIterative(const std::vector<Eigen::Vector3d>& p3ds,
                            const std::vector<Eigen::Vector2d>& pt2s,
                            Sophus::SE3d* const relative_pose) const {
  return true;
}

bool PnpSolver::CvEPNP(const std::vector<Eigen::Vector3d>& p3ds,
                       const std::vector<Eigen::Vector2d>& pt2s,
                       Sophus::SE3d* const relative_pose) const {
  return true;
}

bool PnpSolver::CvP3P(const std::vector<Eigen::Vector3d>& p3ds,
                      const std::vector<Eigen::Vector2d>& pt2s,
                      Sophus::SE3d* const relative_pose) const {
  return true;
}

bool PnpSolver::CvDLS(const std::vector<Eigen::Vector3d>& p3ds,
                      const std::vector<Eigen::Vector2d>& pt2s,
                      Sophus::SE3d* const relative_pose) const {
  return true;
}

bool PnpSolver::CvUPNP(const std::vector<Eigen::Vector3d>& p3ds,
                       const std::vector<Eigen::Vector2d>& pt2s,
                       Sophus::SE3d* const relative_pose) const {
  return true;
}

bool PnpSolver::CvAP3P(const std::vector<Eigen::Vector3d>& p3ds,
                       const std::vector<Eigen::Vector2d>& pt2s,
                       Sophus::SE3d* const relative_pose) const {
  return true;
}

bool PnpSolver::CvIPPE(const std::vector<Eigen::Vector3d>& p3ds,
                       const std::vector<Eigen::Vector2d>& pt2s,
                       Sophus::SE3d* const relative_pose) const {
  return true;
}

bool PnpSolver::CvSquare(const std::vector<Eigen::Vector3d>& p3ds,
                         const std::vector<Eigen::Vector2d>& pt2s,
                         Sophus::SE3d* const relative_pose) const {
  return true;
}

bool PnpSolver::CvSQPNP(const std::vector<Eigen::Vector3d>& p3ds,
                        const std::vector<Eigen::Vector2d>& pt2s,
                        Sophus::SE3d* const relative_pose) const {
  return true;
}

}  // namespace Camera