#include "optimizer/curve_fitter.h"

namespace Optimizer {

CurveFittingOptimizer::CurveFittingOptimizer(const uint32_t max_iters, const double brake_threshold)
    : NonlinearOptimizer(max_iters, brake_threshold) {
  initialized_ = true;
  parameter_dims_ = 3;
  residuals_dims_ = 1000;
  x_data_.resize(residuals_dims_);
  y_data_.resize(residuals_dims_);
  x_ = Eigen::VectorXd::Zero(parameter_dims_);
  jac_.resize(residuals_dims_, parameter_dims_);
  sum_err_ = Eigen::VectorXd::Zero(1);
  last_sum_err_ = Eigen::VectorXd::Zero(1);
}

// 重写代价函数
Eigen::VectorXd CurveFittingOptimizer::CostFunction(const Eigen::VectorXd& x) {
  // 这里假设我们要拟合的曲线函数是一个二次多项式 y = p(0) + p(1) * x + p(2) * x^2;
  Eigen::VectorXd residuals(residuals_dims_);
  for (uint32_t i = 0; i < residuals_dims_; ++i) {
    residuals(i) = curve_function_(x, x_data_(i)) - y_data_(i);  // 残差 = 预测值 - 实际值
  }
  return residuals;
}

// 重写雅可比矩阵计算方法
Eigen::MatrixXd CurveFittingOptimizer::ComputeJacobian(const Eigen::VectorXd& x) {
  // 雅可比矩阵的维度是 (residuals_dims_ parameter_dims_)
  Eigen::MatrixXd jac(residuals_dims_, parameter_dims_);
  for (uint32_t i = 0; i < residuals_dims_; ++i) {
    for (uint32_t j = 0; j < parameter_dims_; ++j) {
      // 计算每个残差对于参数的偏导数
      jac(i, j) = std::pow(x_data_(i), j) * curve_function_(x, x_data_(i));  // 对 x(j) 的偏导数
    }
  }
  return jac;
}

// 设置拟合所需的数据
void CurveFittingOptimizer::SetData(const std::vector<std::pair<double, double>>& data) {
  for (uint32_t i = 0; i < residuals_dims_; ++i) {
    x_data_(i) = data.at(i).first;
    y_data_(i) = data.at(i).second;
  }
}

void CurveFittingOptimizer::SetCurveFunction(
    const std::function<double(const Eigen::VectorXd&, const double)>& curve_function) {
  curve_function_ = curve_function;
}

void CurveFittingOptimizer::SetRealParams(const Eigen::VectorXd& params) { real_paras_ = params; }

// bool CurveFittingOptimizer::IsConverge(const Eigen::VectorXd& dx) {
//   bool res = (sum_err_.norm() > last_sum_err_.norm()) && (dx.norm() < 1e-6);
//   last_sum_err_ = sum_err_;
//   return res;
// }

}  // namespace Optimizer
