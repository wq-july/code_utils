#pragma once

#include "ceres/ceres.h"
#include "optimizer/optimizer.h"
#include "util/utils.h"

namespace Optimizer {

class CurveFittingOptimizer : public NonlinearOptimizer {
 public:
  CurveFittingOptimizer(const uint32_t max_iters, const double brake_threshold);
  // 重写代价函数
  Eigen::VectorXd CostFunction(const Eigen::VectorXd& x) override;

  // 重写雅可比矩阵计算方法
  Eigen::MatrixXd ComputeJacobian(const Eigen::VectorXd& x) override;

  // bool IsConverge(const Eigen::VectorXd& dx) override;

  // 设置拟合所需的数据
  void SetData(const std::vector<std::pair<double, double>>& data);

  void SetCurveFunction(
      const std::function<double(const Eigen::VectorXd&, const double)>& curve_function);

  void SetRealParams(const Eigen::VectorXd& params);

  void GaussNewtonInstance();

 private:
  Eigen::VectorXd x_data_;      // 样本点 x 坐标
  Eigen::VectorXd y_data_;      // 样本点 y 坐标
  Eigen::VectorXd real_paras_;  // 随机生成的真值
  std::function<double(const Eigen::VectorXd&, const double)> curve_function_;
};

  // ======================================================================================
  //  ceres实现曲线拟合
  // 需要自定义一个结构体
  struct CurveFitter {
   public:
    CurveFitter(double x, double y) : x_(x), y_(y) {
    }
    template <typename T>
    bool operator()(const T* const p, T* redisual) const {
      redisual[0] = y_ - exp(p[0] + p[1] * x_ + p[2] * x_ * x_);
      return true;
    }

   private:
    const double x_;
    const double y_;
  };


}  // namespace Optimizer
