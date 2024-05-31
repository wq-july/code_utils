#include "optimizer/optimizer.h"

namespace Optimizer {

class CloudMatchOptimizer : public Optimizer::NonlinearOptimizer {
 public:
  CloudMatchOptimizer(const uint32_t max_iters, const double brake_threshold)
      : NonlinearOptimizer(max_iters, brake_threshold) {}

  // 可以考虑套一层核函数
  Eigen::MatrixXd LossFunction(const Eigen::VectorXd& err);

  // 负责计算残差，但是也负责寻找对应关系，不然也没法计算残差！
  Eigen::VectorXd CostFunction(const Eigen::VectorXd& x);
  virtual Eigen::MatrixXd ComputeJacobian(const Eigen::VectorXd& x);

  virtual Eigen::VectorXd Update(const Eigen::VectorXd& dx);

  // 这两个函数应该可以搞成通用可选的方法来判断；
  virtual bool IsConverge(const Eigen::VectorXd& dx);

  virtual bool ResultCheckout(const Eigen::VectorXd& dx);
};

}  // namespace Optimizer