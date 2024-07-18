#pragma once
#include <vector>

#include "sophus/se3.hpp"

#include "../protos/pb/camera.pb.h"
#include "camera/camera_model/camera_model.h"
#include "util/time.h"

namespace Camera {
// 这个类只用来求解2D-3D问题
class PnpSolver {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
 public:
  PnpSolver(const CameraConfig::PnpSolverConfig& config,
            const std::shared_ptr<Camera::CameraBase>& cam_model_ptr);

  ~PnpSolver() = default;

  bool Solve(const std::vector<Eigen::Vector3d>& p3ds,
             const std::vector<Eigen::Vector2d>& pt2s,
             Sophus::SE3d* const relative_pose);

 private:
  bool DLTSolve(const std::vector<Eigen::Vector3d>& p3ds,
                const std::vector<Eigen::Vector2d>& pt2s,
                Sophus::SE3d* const relative_pose) const;

  bool BAGaussNewtonSolve(const std::vector<Eigen::Vector3d>& p3ds,
                          const std::vector<Eigen::Vector2d>& pt2s,
                          Sophus::SE3d* const relative_pose) const;

  bool EPNPSolve(const std::vector<Eigen::Vector3d>& p3ds,
                 const std::vector<Eigen::Vector2d>& pt2s,
                 Sophus::SE3d* const relative_pose) const;

  bool MLPNPSolve(const std::vector<Eigen::Vector3d>& p3ds,
                  const std::vector<Eigen::Vector2d>& pt2s,
                  Sophus::SE3d* const relative_pose) const;

  // EPNP相关的工具函数，原理参考：https://zhuanlan.zhihu.com/p/59070440
 private:
  void SelectControlPoints(const std::vector<Eigen::Vector3d>& p3ds,
                           std::vector<Eigen::Vector3d>* const control_p3ds) const;

  void ComputeHBCoordinates(const std::vector<Eigen::Vector3d>& pts3d,
                            const std::vector<Eigen::Vector3d>& control_points,
                            std::vector<Eigen::Vector4d>* const hb_coordinates) const;

  void ConstructM(const Eigen::Matrix3d& K,
                  const std::vector<Eigen::Vector4d>& hb_coordinates,
                  const std::vector<Eigen::Vector2d>& pts2d,
                  Eigen::MatrixXd* const M) const;

  void GetFourEigenVectors(const Eigen::MatrixXd& M,
                           Eigen::Matrix<double, 12, 4>* const eigen_vectors) const;

  void ComputeL(const Eigen::Matrix<double, 12, 4>& eigen_vectors,
                Eigen::Matrix<double, 6, 10>* const L) const;

  void ComputeRho(const std::vector<Eigen::Vector3d>& control_points,
                  Eigen::Matrix<double, 6, 1>* const rho) const;

  void SolveBetaN2(const Eigen::Matrix<double, 12, 4>& eigen_vectors,
                   const Eigen::Matrix<double, 6, 10>& L,
                   const Eigen::Matrix<double, 6, 1>& rho,
                   Eigen::Vector4d* const betas) const;

  void SolveBetaN3(const Eigen::Matrix<double, 12, 4>& eigen_vectors,
                   const Eigen::Matrix<double, 6, 10>& L,
                   const Eigen::Matrix<double, 6, 1>& rho,
                   Eigen::Vector4d* const betas) const;

  void SolveBetaN4(const Eigen::Matrix<double, 12, 4>& eigen_vectors,
                   const Eigen::Matrix<double, int(6), int(10)>& L,
                   const Eigen::Matrix<double, int(6), int(1)>& rho,
                   Eigen::Vector4d* const betas) const;

  void OptimizeBeta(const Eigen::Matrix<double, int(6), int(10)>& L,
                    const Eigen::Matrix<double, int(6), int(1)>& rho,
                    Eigen::Vector4d* const betas) const;

  void ComputeCameraControlPoints(const Eigen::Matrix<double, int(12), int(4)>& eigen_vectors,
                                  const Eigen::Vector4d& betas,
                                  std::vector<Eigen::Vector3d>* const camera_control_points) const;

  bool IsGoodBetas(const std::vector<Eigen::Vector3d>& camera_control_points) const;

  void RebuiltPts3dCamera(const std::vector<Eigen::Vector3d>& camera_control_points,
                          const std::vector<Eigen::Vector4d>& hb_coordinates,
                          std::vector<Eigen::Vector3d>* const pts3d_camera) const;

  void ComputeRt(const std::vector<Eigen::Vector3d>& pts3d_camera,
                 const std::vector<Eigen::Vector3d>& pts3d_world,
                 Eigen::Matrix3d* const R,
                 Eigen::Vector3d* const t) const;

  double ReprojectionError(const Eigen::Matrix3d& K,
                           const std::vector<Eigen::Vector3d>& pts3d_world,
                           const std::vector<Eigen::Vector2d>& pts2d,
                           const Eigen::Matrix3d& R,
                           const Eigen::Vector3d& t) const;

  void SolveAndOptimizeBeta(int N,
                            const Eigen::Matrix<double, 12, 4>& eigen_vectors,
                            const Eigen::Matrix<double, 6, 10>& L,
                            const Eigen::Matrix<double, 6, 1>& rho,
                            Eigen::Vector4d* betas) const;

  void ComputeRtFromBetas(const Eigen::Matrix<double, 12, 4>& eigen_vectors,
                          const std::vector<Eigen::Vector4d>& hb_coordinates,
                          const Eigen::Vector4d& betas,
                          const std::vector<Eigen::Vector3d>& p3ds,
                          Eigen::Matrix3d* const R,
                          Eigen::Vector3d* const t) const;

  // 以下是调用OpenCV库实现PNP求解
 private:
  bool CvIterative(const std::vector<Eigen::Vector3d>& p3ds,
                   const std::vector<Eigen::Vector2d>& pt2s,
                   Sophus::SE3d* const relative_pose) const;

  bool CvEPNP(const std::vector<Eigen::Vector3d>& p3ds,
              const std::vector<Eigen::Vector2d>& pt2s,
              Sophus::SE3d* const relative_pose) const;

  bool CvP3P(const std::vector<Eigen::Vector3d>& p3ds,
             const std::vector<Eigen::Vector2d>& pt2s,
             Sophus::SE3d* const relative_pose) const;

  bool CvDLS(const std::vector<Eigen::Vector3d>& p3ds,
             const std::vector<Eigen::Vector2d>& pt2s,
             Sophus::SE3d* const relative_pose) const;

  bool CvUPNP(const std::vector<Eigen::Vector3d>& p3ds,
              const std::vector<Eigen::Vector2d>& pt2s,
              Sophus::SE3d* const relative_pose) const;

  bool CvAP3P(const std::vector<Eigen::Vector3d>& p3ds,
              const std::vector<Eigen::Vector2d>& pt2s,
              Sophus::SE3d* const relative_pose) const;

  bool CvIPPE(const std::vector<Eigen::Vector3d>& p3ds,
              const std::vector<Eigen::Vector2d>& pt2s,
              Sophus::SE3d* const relative_pose) const;

  bool CvSquare(const std::vector<Eigen::Vector3d>& p3ds,
                const std::vector<Eigen::Vector2d>& pt2s,
                Sophus::SE3d* const relative_pose) const;

  bool CvSQPNP(const std::vector<Eigen::Vector3d>& p3ds,
               const std::vector<Eigen::Vector2d>& pt2s,
               Sophus::SE3d* const relative_pose) const;

 private:
  CameraConfig::PnpSolverConfig config_;
  Utils::Timer timer_;
  std::shared_ptr<Camera::CameraBase> camera_model_ptr_ = nullptr;
};

}  // namespace Camera