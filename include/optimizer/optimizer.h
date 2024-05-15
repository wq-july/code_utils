#pragma once

#include "Eigen/Dense"

namespace Optimizer {

enum class OptimizeMethod {
  GAUSS_NEWTON,
  LEVENBERG_MARQUARDT,
  DOGLEG,
};

class NonlinearOptimizer {
 public:
  NonlinearOptimizer(const OptimizeMethod& method, const uint32_t max_iters,
                     const double brake_threshold) {}
  ~NonlinearOptimizer() = default;

  void SetParameter(const Eigen::VectorXd& parameter) { parameter_ = parameter; }

  void SetJacobian(const Eigen::MatrixXd& jacobian) { jacobian_ = jacobian; }

  void SetResidual(const Eigen::VectorXd& residuals) { residuals_ = residuals; }

  void GaussNewton() {}

  void LevenbergMarquardt();

  void DogLeg();

  void Optimize() {
    {
      // 检查需要参数是不是已经设置好，否则弹出一些日志或者错误信息
    }

    for (uint32_t i = 0; i < max_iters_; ++i) {
      // Eigen::VectorXd delta_x = Eigen::VectorXd::Zero(parameter_dims_);

      switch (method_) {
        case OptimizeMethod::GAUSS_NEWTON:
          GaussNewton();
          break;
        case OptimizeMethod::LEVENBERG_MARQUARDT:
          LevenbergMarquardt();
          break;
        case OptimizeMethod::DOGLEG:
          DogLeg();
          break;
        default:
          GaussNewton();
          break;
      }

      // if (delta_x.norm() < brake_threshold_) {
      //   break;
      // }
    }
  }

 private:
  uint32_t parameter_dims_ = 0;
  uint32_t residuals_dims_ = 0;

 private:
  // 待优化变量，状态量
  Eigen::VectorXd parameter_;
  // 残差量
  Eigen::VectorXd residuals_;
  // 雅可比矩阵块
  Eigen::MatrixXd jacobian_;

 private:
  OptimizeMethod method_ = OptimizeMethod::GAUSS_NEWTON;
  uint32_t max_iters_ = 3;
  double brake_threshold_ = 0.001;  // 截止的阈值
  double lambda_ = 1.0;             // LM方法的阻尼参数
};

}  // namespace Optimizer
