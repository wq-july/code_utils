#include "optimizer/optimizer.h"

namespace Optimizer {

NonlinearOptimizer::NonlinearOptimizer(const int32_t max_iters, const double breake_threshold)
    : max_iters_(max_iters), break_threshold_(breake_threshold) {
  initialized_ = true;
  // TODO，初始化的部分参数改用配置实现
  // TODO，截止条件判断和结果分析用提供默认方法进行选择就好，不需要自定义；
}

NonlinearOptimizer::~NonlinearOptimizer() {}

Eigen::VectorXd NonlinearOptimizer::GaussNewton() {

  Eigen::MatrixXd H = Eigen::MatrixXd::Zero(parameter_dims_, parameter_dims_);
  Eigen::VectorXd b = Eigen::VectorXd::Zero(parameter_dims_);

  // 1. 根据初始状态量，计算残差量，这个残差量用于估计迭代方向δx;
  // f(x) -> err = f(x) - observes
  Eigen::VectorXd err = CostFunction(x_);

  // 2. 计算雅可比矩阵，得到等式 J * J^T = -J^T * err， 其中， err -> f(x)
  jac_ = ComputeJacobian(x_);

  H = jac_.transpose() * jac_;
  b = -jac_.transpose() * err;

  sum_err_ = LossFunction(err);

  return Solver(H, b, solve_type_);
}

Eigen::VectorXd NonlinearOptimizer::LevenbergMarquardt() {

  Eigen::MatrixXd H = Eigen::MatrixXd::Zero(parameter_dims_, parameter_dims_);
  Eigen::VectorXd b = Eigen::VectorXd::Zero(parameter_dims_);

  // 1. 根据初始状态量，计算残差量，这个残差量用于估计迭代方向δx;
  // f(x) -> err = f(x) - observes
  Eigen::VectorXd err = CostFunction(x_);

  // 2. 计算雅可比矩阵，得到等式 J * J^T = -J^T * err， 其中， err -> f(x)
  jac_ = ComputeJacobian(x_);

  H = jac_.transpose() * jac_;
  b = -jac_.transpose() * err;

  sum_err_ = err.transpose() * err;

  // 计算初始的λ值，初值比较好，t取小；初值未知，t取大，取H对角线最大元素作为初值
  if (first_iter_) {
    first_iter_ = false;
    Eigen::VectorXd diagonal = H.diagonal();
    double max_diagonal = diagonal.maxCoeff();
    lambda_ = init_tao_ * max_diagonal;
  }

  // 计算dx_
  Eigen::VectorXd dx = Solver(
      H + lambda_ * Eigen::MatrixXd::Identity(parameter_dims_, parameter_dims_), b, solve_type_);

  // 计算ρ(增益率), 也就是一阶泰勒展开相似度， 实际下降量 / 近似模型下降量
  // 对于最小二乘问题 F(x) = || f(x + △x) ||^2 ≈ || f(x) + J(x)△x ||^2，
  // 其中，L(x) = || f(x) + J(x)△x ||^2 = f^T * f + △x^T * J(x) * f(x) + △x^T * J(x)^T * J(x) * △x;

  // 2. ρ = F(x+△x) - F(x) / L(△x) - L(0)  -> 个人更倾向于这种
  // 分子表示实际下降量，而分母则是近似模型的下降量

  // 注意：ρ ≈ 1.0，近似可靠；ρ >> 1.0，实际模型下降更大，可以放大增量范围；ρ
  // << 1.0，近似模型下降大，需要缩小增量可信度范围；衡量这个近似模型的精确性；

  // 计算F(x+△x) - F(x)
  Eigen::MatrixXd actual_reduction =
      LossFunction(CostFunction(Update(dx))) - LossFunction(CostFunction(x_));
  Eigen::MatrixXd approx_reduction =
      dx.transpose() * jac_.transpose() * err + dx.transpose() * jac_.transpose() * jac_ * dx;
  double rho = actual_reduction.norm() / approx_reduction.norm();
  LOG(INFO) << "ρ is " << rho;

  // 通过计算得到的ρ来验证之前选用的λ是否合适，然后调整λ
  // 1. Marquardt (1963)
  // if (rho < 0.25) {
  //   lambda_ *= 2.0;
  // } else if (rho > 0.75) {
  //   lambda_ *= static_cast<double>(1.0 / 3.0);
  // }

  // 2. Nielsen (1999)
  if (rho > 0.0) {
    lambda_ *= std::max(static_cast<double>(1.0 / 3.0), 1.0 - std::pow((2.0 * rho - 1.0), 3));
    meu_ = 2.0;
  } else {
    lambda_ *= meu_;
    meu_ *= 2.0;
  }

  return dx;
}

Eigen::VectorXd NonlinearOptimizer::DogLeg() {

  Eigen::MatrixXd H = Eigen::MatrixXd::Zero(parameter_dims_, parameter_dims_);
  Eigen::VectorXd b = Eigen::VectorXd::Zero(parameter_dims_);

  // 1. 根据初始状态量，计算残差量，这个残差量用于估计迭代方向δx;
  // f(x) -> err = f(x) - observes
  Eigen::VectorXd err = CostFunction(x_);

  // 2. 计算雅可比矩阵，得到等式 J * J^T = -J^T * err， 其中， err -> f(x)
  jac_ = ComputeJacobian(x_);

  H = jac_.transpose() * jac_;
  b = -jac_.transpose() * err;

  sum_err_ = err.transpose() * err;

  // 计算初始的λ值，初值比较好，t取小；初值未知，t取大，取H对角线最大元素作为初值
  if (first_iter_) {
    first_iter_ = false;
    Eigen::VectorXd diagonal = H.diagonal();
    double max_diagonal = diagonal.maxCoeff();
    lambda_ = init_tao_ * max_diagonal;
  }

  // 计算dx_
  Eigen::VectorXd dx = Solver(
      H + lambda_ * Eigen::MatrixXd::Identity(parameter_dims_, parameter_dims_), b, solve_type_);

  // 计算ρ(增益率), 也就是一阶泰勒展开相似度， 实际下降量 / 近似模型下降量
  // 对于最小二乘问题 F(x) = || f(x + △x) ||^2 ≈ || f(x) + J(x)△x ||^2，
  // 其中，L(x) = || f(x) + J(x)△x ||^2 = f^T * f + △x^T * J(x) * f(x) + △x^T * J(x)^T * J(x) * △x;

  // 2. ρ = F(x+△x) - F(x) / L(△x) - L(0)  -> 个人更倾向于这种
  // 分子表示实际下降量，而分母则是近似模型的下降量

  // 注意：ρ ≈ 1.0，近似可靠；ρ >> 1.0，实际模型下降更大，可以放大增量范围；ρ
  // << 1.0，近似模型下降大，需要缩小增量可信度范围；衡量这个近似模型的精确性；

  // 计算F(x+△x) - F(x)
  Eigen::MatrixXd actual_reduction =
      LossFunction(CostFunction(Update(dx))) - LossFunction(CostFunction(x_));
  Eigen::MatrixXd approx_reduction =
      dx.transpose() * jac_.transpose() * err + dx.transpose() * jac_.transpose() * jac_ * dx;
  double rho = actual_reduction.norm() / approx_reduction.norm();
  // LOG(INFO) << "ρ is " << rho << ", λ is " << lambda_;
  std::cout << "rho is " << rho << ", lambda is " << lambda_;

  // 通过计算得到的ρ来验证之前选用的λ是否合适，然后调整λ
  // 1. Marquardt (1963)
  // if (rho < 0.25) {
  //   lambda_ *= 2.0;
  // } else if (rho > 0.75) {
  //   lambda_ *= static_cast<double>(1.0 / 3.0);
  // }

  // 2. Nielsen (1999)
  if (rho > 0.0) {
    lambda_ *= std::max(static_cast<double>(1.0 / 3.0), 1.0 - std::pow((2.0 * rho - 1.0), 3));
    meu_ = 2.0;
  } else {
    lambda_ *= meu_;
    meu_ *= 2.0;
  }

  return dx;
}

bool NonlinearOptimizer::Optimize(OptimizeMethod method) {
  // 1. 初始值 x0 -> f(x0) 得到误差量 b(x0) = y - F(x0)
  // 2. b(x0) -> GN / LM / DogLeg -> δx
  // 3. x0 + δx -> x1
  // 4. 继续第一步，一直循环，直到 δx < break_threshold
  // 因为在最优解附近，只能往最优解方向一点一点挪移

  if (!initialized_) {
    LOG(FATAL) << "Have not initialized!!";
    return false;
  }

  int32_t iter = 0u;
  while (iter < max_iters_) {
    Eigen::VectorXd dx = Eigen::VectorXd::Zero(parameter_dims_);
    switch (method) {
      case OptimizeMethod::GAUSS_NEWTON:
        dx = GaussNewton();
        break;
      case OptimizeMethod::LEVENBERG_MARQUARDT:
        dx = LevenbergMarquardt();
        break;
      case OptimizeMethod::DOGLEG:
        dx = DogLeg();
        break;
      default:
        dx = GaussNewton();
        break;
    }

    if (!ResultCheckout(dx)) {
      LOG(ERROR) << "dx is nan!!";
      ++iter;
      continue;
    }

    if (IsConverge(dx) && iter > 0) {
      LOG(INFO) << "Break!";
      break;
    }

    x_ = Update(dx);

    ++iter;
  }
  LOG(INFO) << "Iter time is " << iter << ", x_ is " << x_.transpose();
  return true;
}

Eigen::VectorXd NonlinearOptimizer::Solver(const Eigen::MatrixXd& H, const Eigen::VectorXd& b,
                                           SolverType solverType) {
  Eigen::VectorXd dx;
  switch (solverType) {
    case SolverType::QR:
      dx = H.colPivHouseholderQr().solve(b);
      break;

    case SolverType::SVD:
      dx = H.ldlt().solve(b);
      break;

    case SolverType::INVERSE:
      dx = H.inverse() * b;
      break;

    default:
      LOG(ERROR) << "Unknown solver type!";
      break;
  }
  return dx;
}

Eigen::MatrixXd NonlinearOptimizer::LossFunction(const Eigen::VectorXd& err) {
  return err.transpose() * err;
}

Eigen::VectorXd NonlinearOptimizer::CostFunction(const Eigen::VectorXd& x) {
  Eigen::VectorXd err;
  return err;
}

Eigen::MatrixXd NonlinearOptimizer::ComputeJacobian(const Eigen::VectorXd& x) {
  Eigen::MatrixXd jacobian;
  return jacobian;
}

Eigen::VectorXd NonlinearOptimizer::Update(const Eigen::VectorXd& dx) { return x_ + dx; }

// 这两个函数应该可以搞成通用可选的方法来判断；
bool NonlinearOptimizer::IsConverge(const Eigen::VectorXd& dx) {
  if (dx.norm() < 1e-6) {
    return true;
  }
  return false;
}
bool NonlinearOptimizer::ResultCheckout(const Eigen::VectorXd& dx) {
  bool normal = true;
  for (uint32_t i = 0; i < dx.size(); ++i) {
    if (std::isnan(dx(i))) {
      normal = false;
      break;
    }
  }
  return normal;
}

}  // namespace Optimizer
