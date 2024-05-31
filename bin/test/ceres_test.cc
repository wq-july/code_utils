#include "ceres/ceres.h"

#include <gflags/gflags.h>
#include <glog/logging.h>
#include <gtest/gtest.h>

#include "optimizer/curve_fitter.h"
#include "util/math.h"
#include "util/utils.h"

DEFINE_string(curve_path, "../log/ceres_curve.txt", "曲线拟合原始数据路径");

class CeresOptimizerTest : public testing::Test {
 public:
  void SetUp() override {}

  // ======================================================================================
  // 1. ceres hello world， 基于结构体，使用仿函数来实现代价函数f(x)
  // 个人猜想，都是指针传入，也就是都是数组，那么底层内部实际上状态量参数和误差量的维度都可以根据输入来自动确定了；
  struct CostFunctor {
    template <typename T>
    bool operator()(const T* const x, T* residual) const {
      residual[0] = 10.0 - x[0];
      return true;
    }
  };

  // 2. ceres hello world, 自定义解析求导方式
  // A CostFunction implementing analytically derivatives for the
  // function f(x) = 10 - x.
  class QuadraticCostFunction : public ceres::SizedCostFunction<1 /* number of residuals */,
                                                                1 /* size of first parameter */> {
   public:
    bool Evaluate(double const* const* parameters,
                  double* residuals,
                  double** jacobians) const override {
      double x = parameters[0][0];

      // f(x) = 10 - x.
      residuals[0] = 10 - x;

      // f'(x) = -1. Since there's only 1 parameter and that parameter
      // has 1 dimension, there is only 1 element to fill in the
      // jacobians.
      //
      // Since the Evaluate function can be called with the jacobians
      // pointer equal to nullptr, the Evaluate function must check to see
      // if jacobians need to be computed.
      //
      // For this simple problem it is overkill to check if jacobians[0]
      // is nullptr, but in general when writing more complex
      // CostFunctions, it is possible that Ceres may only demand the
      // derivatives w.r.t. a subset of the parameter blocks.
      if (jacobians != nullptr && jacobians[0] != nullptr) {
        jacobians[0][0] = -1;
      }

      return true;
    }
  };
  // ======================================================================================
  // Powell方程，也就是求解多维方程组，之后可以升级成矩阵求解
  struct F1 {
    template <typename T>
    bool operator()(const T* const x1, const T* const x2, T* residual) const {
      // f1 = x1 + 10 * x2
      residual[0] = x1[0] + 10.0 * x2[0];
      return true;
    }
  };

  struct F2 {
    template <typename T>
    bool operator()(const T* const x3, const T* const x4, T* residuals) const {
      // f2 = sqrt(5) (x3 - x4)
      residuals[0] = std::sqrt(5.0) * (x3[0] - x4[0]);
      return true;
    }
  };

  struct F3 {
    template <typename T>
    bool operator()(const T* const x2, const T* const x3, T* residual) const {
      // f3 = (x2 - 2 x3)^2
      residual[0] = (x2[0] - 2.0 * x3[0]) * (x2[0] - 2.0 * x3[0]);
      return true;
    }
  };

  struct F4 {
    template <typename T>
    bool operator()(const T* const x1, const T* const x4, T* residual) const {
      // f4 = sqrt(10) (x1 - x4)^2
      residual[0] = sqrt(10.0) * (x1[0] - x4[0]) * (x1[0] - x4[0]);
      return true;
    }
  };
  // ======================================================================================
  // 将上述的powell函数合并成一个结构体，实现类似矩阵的运算
  struct PowellCostFunction {
    template <typename T>
    bool operator()(const T* const x, T* residual) const {
      residual[0] = x[0] + 10.0 * x[1];
      residual[1] = std::sqrt(5.0) * (x[2] - x[3]);
      residual[2] = (x[1] - 2.0 * x[2]) * (x[1] - 2.0 * x[2]);
      residual[3] = sqrt(10.0) * (x[0] - x[3]) * (x[0] - x[3]);
      return true;
    }
  };

  bool enable_test_ = true;
};

TEST_F(CeresOptimizerTest, CeresHelloWorldTest) {
  if (!enable_test_) {
    return;
  }

  double initial_x = 5.0;
  double x = initial_x;

  // 1. 首先需要定义一个ceres问题；
  ceres::Problem problem;

  // 2. 创建代价函数指针；
  // 然后需要定义一个代价函数f(x)（也就是计算残差的函数）,在定义这个函数的时候，同时需要定义雅可比矩阵求导的函数，
  // 但是ceres内部是存在自动求导的方法的，这里是可以选的；
  // 定义的代价函数是指针，因为ceres函数处理的变量一般都是指针，在new的时候选定了求导方式，这里选择自动求导
  // 第一个size表示的是残差量的维度，第二个参数表示的是状态量的维度，这里是平面曲线，所以都是一维的

  // ceres::CostFunction* cost_function =
  //     new ceres::AutoDiffCostFunction<CostFunctor, 1, 1>(new CostFunctor);

  // 上面一种是自动求导，接下来考虑数值求导，但是一般来说，我们推荐自动微分而不是数值微分。c++模板的使用使自动微分变得高效，
  // 而数值微分代价高昂，容易出现数值错误，并导致较慢的收敛。

  // ceres::CostFunction* cost_function_numeric =
  //     new ceres::NumericDiffCostFunction<CostFunctor, ceres::CENTRAL, 1, 1>(new CostFunctor);

  ceres::CostFunction* cost_function_analys = new QuadraticCostFunction;

  // 3. 初始化problem，添加残差块,
  // 核函数，状态量参数，在优化的过程中，这个状态量会进行迭代改变
  problem.AddResidualBlock(cost_function_analys, nullptr, &x);

  // 4. 开始优化，创建优化器
  // 5. 优化选项，用于确定优化过程中的一些方法，比如矩阵分解方法，优化算法等
  ceres::Solver::Options options;
  options.linear_solver_type = ceres::DENSE_QR;
  // 用于将优化的步骤日志输出到控制台中；
  options.minimizer_progress_to_stdout = true;

  // 6. 最终的优化结果以及相关的日志
  ceres::Solver::Summary summary;
  // 7. 执行优化求解
  ceres::Solve(options, &problem, &summary);

  // LOG(INFO) << summary.BriefReport();
  LOG(INFO) << "x: " << initial_x << " -> " << x;
}

TEST_F(CeresOptimizerTest, PowellTest) {
  if (!enable_test_) {
    return;
  }

  // 设置初始参数
  double x1 = 3.0;
  double x2 = -1.0;
  double x3 = 0.0;
  double x4 = 1.0;

  double x[4] = {3.0, -1.0, 0.0, 1.0};

  // 1. 创建ceres问题
  ceres::Problem problem;
  ceres::Problem problem2;

  // 2. 创建代价函数指针；
  ceres::CostFunction* f1 = new ceres::AutoDiffCostFunction<F1, 1, 1, 1>(new F1);
  ceres::CostFunction* f2 = new ceres::AutoDiffCostFunction<F2, 1, 1, 1>(new F2);
  ceres::CostFunction* f3 = new ceres::AutoDiffCostFunction<F3, 1, 1, 1>(new F3);
  ceres::CostFunction* f4 = new ceres::AutoDiffCostFunction<F4, 1, 1, 1>(new F4);
  ceres::CostFunction* powell =
      new ceres::AutoDiffCostFunction<PowellCostFunction, 4, 4>(new PowellCostFunction);

  // 3. 将上述的残差函数添加到problem中
  problem.AddResidualBlock(f1, nullptr, &x1, &x2);
  problem.AddResidualBlock(f2, nullptr, &x3, &x4);
  problem.AddResidualBlock(f3, nullptr, &x2, &x3);
  problem.AddResidualBlock(f4, nullptr, &x1, &x4);

  problem2.AddResidualBlock(powell, nullptr, x);

  ceres::Solver::Options options;
  options.max_num_iterations = 100;
  options.linear_solver_type = ceres::DENSE_QR;
  options.minimizer_progress_to_stdout = true;

  // Run the solver!
  ceres::Solver::Summary summary;
  ceres::Solver::Summary summary2;
  ceres::Solve(options, &problem, &summary);
  ceres::Solve(options, &problem2, &summary2);

  // std::cout << summary.FullReport() << "\n";

  LOG(INFO) << "Final x1 = " << x1 << ", x2 = " << x2 << ", x3 = " << x3 << ", x4 = " << x4;
  LOG(INFO) << "Final x1 = " << x[0] << ", x2 = " << x[1] << ", x3 = " << x[2] << ", x4 = " << x[3];
}

TEST_F(CeresOptimizerTest, CurveFitterTest) {
  if (!enable_test_) {
    return;
  }

  auto curve_function = [](const Eigen::VectorXd& params, const double data_x) -> double {
    return std::exp(params(0) + params(1) * data_x + params(2) * data_x * data_x);
  };

  Eigen::VectorXd parameters = Eigen::VectorXd::Zero(3);
  std::vector<std::pair<double, double>> data;
  Utils::GenerateRandomCoefficientsAndData(curve_function,
                                           3,           // Number of parameters
                                           1000,        // Number of data points
                                           {0.0, 1.0},  // Parameter range
                                           {0.1, 0.1},  // Noise range
                                           FLAGS_curve_path,
                                           &parameters,
                                           &data);

  double x[3] = {0.0, 0.0, 0.0};
  ceres::Problem problem;
  for (uint32_t i = 0; i < data.size(); ++i) {
    ceres::CostFunction* costfunctor =
        new ceres::AutoDiffCostFunction<Optimizer::CurveFitter, 1, 3>(
            new Optimizer::CurveFitter(data[i].first, data[i].second));
    problem.AddResidualBlock(costfunctor, new ceres::CauchyLoss(0.35), x);
  }

  ceres::Solver::Options options;
  options.max_num_iterations = 20;
  options.linear_solver_type = ceres::DENSE_QR;
  options.minimizer_progress_to_stdout = true;

  ceres::Solver::Summary summary;
  ceres::Solve(options, &problem, &summary);

  // LOG(INFO) << summary.BriefReport() << "\n";
  LOG(INFO) << "Real params is " << parameters(0) << ", " << parameters(1) << ", " << parameters(2);
  LOG(INFO) << "Final x: " << x[0] << ", " << x[1] << ", " << x[2] << "\n";
}

TEST_F(CeresOptimizerTest, SLAM2DTest) {
  if (!enable_test_) {
    return;
  }
}

int main(int argc, char** argv) {
  google::InitGoogleLogging(argv[0]);
  FLAGS_stderrthreshold = google::INFO;
  FLAGS_colorlogtostderr = true;

  // Initialize Google Test framework
  ::testing::InitGoogleTest(&argc, argv);
  google::ParseCommandLineFlags(&argc, &argv, true);
  // Run tests
  return RUN_ALL_TESTS();
}
