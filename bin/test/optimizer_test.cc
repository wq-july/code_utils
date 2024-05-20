#include <cmath>
#include <filesystem>
#include <fstream>
#include <random>
#include <vector>

#include "Eigen/Core"
#include "Eigen/Dense"
#include "gflags/gflags.h"
#include "gtest/gtest.h"

#include "ceres/ceres.h"
#include "glog/logging.h"
#include "util/math.h"
#include "util/time.h"

#define private public

#include "optimizer/curve_fitter.h"
#include "optimizer/optimizer.h"

DEFINE_string(curve_path, "../log/curve.txt", "拟合曲线原始数据路径");

class OptimizerTest : public testing::Test {
 public:
  void SetUp() override {
    optimizer_ = std::make_shared<Optimizer::CurveFittingOptimizer>(15u, 0.0001);

    // Define a lambda function for the polynomial a0 + a1*x + a2*x^2
    curve_function_ = [](const Eigen::VectorXd& params, const double x) -> double {
      double result = 0.0;
      for (uint32_t i = 0; i < params.size(); ++i) {
        result += params[i] * std::pow(x, i);
      }
      return std::exp(result);
    };

    Utils::GenerateRandomCoefficientsAndData(curve_function_,
                                             3,           // Number of parameters
                                             1000,        // Number of data points
                                             {0.0, 1.0},  // Parameter range
                                             {0.1, 0.1},  // Noise range
                                             FLAGS_curve_path,
                                             &real_paras_,
                                             &data_);
    optimizer_->SetData(data_);
    optimizer_->SetCurveFunction(curve_function_);
    optimizer_->SetRealParams(real_paras_);
  }

  Utils::Timer timer_;
  std::shared_ptr<Optimizer::CurveFittingOptimizer> optimizer_ = nullptr;
  std::function<double(const Eigen::VectorXd&, const double)> curve_function_;
  std::vector<std::pair<double, double>> data_;
  Eigen::VectorXd real_paras_;  // 随机生成的真值
  bool debug_ = true;
};

TEST_F(OptimizerTest, GaussNewtonTest) {
  if (!debug_) {
    return;
  }
  optimizer_->Optimize();
  double x[3] = {0.0, 0.0, 0.0};
  ceres::Problem problem;
  for (uint32_t i = 0; i < data_.size(); ++i) {
    ceres::CostFunction* costfunctor =
        new ceres::AutoDiffCostFunction<Optimizer::CurveFitter, 1, 3>(
            new Optimizer::CurveFitter(data_[i].first, data_[i].second));
    problem.AddResidualBlock(costfunctor, nullptr, x);
  }

  ceres::Solver::Options options;
  options.max_num_iterations = 20;
  options.linear_solver_type = ceres::DENSE_QR;
  options.minimizer_progress_to_stdout = true;

  ceres::Solver::Summary summary;
  ceres::Solve(options, &problem, &summary);

  // LOG(INFO) << summary.BriefReport() << "\n";
  LOG(INFO) << "Final x: " << x[0] << ", " << x[1] << ", " << x[2] << "\n";

  for (uint32_t i = 0; i < optimizer_->parameter_dims_; ++i) {
    EXPECT_LE(std::fabs(optimizer_->GetX()(i) - optimizer_->real_paras_(i)), 0.1);
  }
}

TEST_F(OptimizerTest, LMTest) {
  if (!debug_) {
    return;
  }

  optimizer_->Optimize(Optimizer::OptimizeMethod::LEVENBERG_MARQUARDT);

  for (uint32_t i = 0; i < optimizer_->parameter_dims_; ++i) {
    EXPECT_LE(std::fabs(optimizer_->GetX()(i) - optimizer_->real_paras_(i)), 0.1);
  }
}

// TEST_F(OptimizerTest, DogLegTest) {
//   if (!debug_) {
//     return;
//   }

//   optimizer_->Optimize(Optimizer::OptimizeMethod::DOGLEG);

//   for (uint32_t i = 0; i < optimizer_->parameter_dims_; ++i) {
//     EXPECT_LE(std::fabs(optimizer_->GetX()(i) - optimizer_->real_paras_(i)), 0.1);
//   }
// }

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