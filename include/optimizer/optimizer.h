#pragma once

#include <algorithm>

#include "Eigen/Core"
#include "Eigen/Dense"

#include "glog/logging.h"
#include "util/math.h"
#include "util/time.h"
#include "util/utils.h"

namespace Optimizer {

enum class OptimizeMethod {
  GAUSS_NEWTON,
  LEVENBERG_MARQUARDT,
  DOGLEG,
};

enum class SolverType { QR, SVD, INVERSE };

class NonlinearOptimizer {
 public:
  NonlinearOptimizer(const int32_t max_iters, const double brake_threshold);

  ~NonlinearOptimizer();

  // 通常是一个核函数
  virtual Eigen::MatrixXd LossFunction(const Eigen::VectorXd& err);

  virtual Eigen::VectorXd CostFunction(const Eigen::VectorXd& x);
  virtual Eigen::MatrixXd ComputeJacobian(const Eigen::VectorXd& x);

  virtual Eigen::VectorXd Update(const Eigen::VectorXd& dx);

  // 这两个函数应该可以搞成通用可选的方法来判断；
  virtual bool IsConverge(const Eigen::VectorXd& dx);

  virtual bool ResultCheckout(const Eigen::VectorXd& dx);

  Eigen::VectorXd GaussNewton();

  Eigen::VectorXd LevenbergMarquardt();

  Eigen::VectorXd DogLeg();

  bool Optimize(OptimizeMethod method = OptimizeMethod::GAUSS_NEWTON);

  Eigen::VectorXd GetX() const {
    return x_;
  }

  Eigen::MatrixXd GetJacobian() const {
    return jac_;
  }

  Eigen::VectorXd Solver(const Eigen::MatrixXd& H, const Eigen::VectorXd& b, SolverType solverType);

 public:
  bool initialized_ = false;

  // 待优化状态量的维度
  uint32_t parameter_dims_ = 0;
  // 残差量的维度，也就是观测量的维度，等式方程维度
  uint32_t residuals_dims_ = 0;

  // 最大迭代次数
  int32_t max_iters_ = 0;

  // 迭代截止阈值，迭代截止有多种截止标准，到时可以具体展开
  double break_threshold_ = 1.0e-6;
  bool first_iter_ = true;

  // LM方法的阻尼参数
  double lambda_ = 1.0;

  // 计算初始的λ值，初值比较好，t取小；初值未知，t取大，取H对角线最大元素作为初值
  // 1e-6  1e-3  或者 1.0
  double init_tao_ = 1.0;

  // 用来调整阻尼系数的参数
  double meu_ = 2.0;

 public:
  // 待优化变量，状态量
  Eigen::VectorXd x_;

  // 雅可比矩阵块
  Eigen::MatrixXd jac_;

  Eigen::VectorXd sum_err_;
  Eigen::VectorXd last_sum_err_;

 private:
  SolverType solve_type_ = SolverType::INVERSE;

 private:
  Utils::Timer timer_;
};

}  // namespace Optimizer
