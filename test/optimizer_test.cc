#include <gflags/gflags.h>
#include <glog/logging.h>
#include <gtest/gtest.h>

#include <cmath>
#include <filesystem>
#include <fstream>
#include <random>
#include <vector>

#include "Eigen/Core"
#include "Eigen/Dense"
#include "opencv2/opencv.hpp"
#include "pcl/io/pcd_io.h"
#include "util/math.h"
#include "util/time.h"

#define private public

#include "optimizer/curve_fitter.h"
#include "optimizer/optimizer.h"

DEFINE_string(curve_path, "../conf/imu_config.yaml", "imu的配置文件");

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
    std::vector<std::pair<double, double>> data;
    Utils::GenerateRandomCoefficientsAndData(curve_function_,
                                             3,           // Number of parameters
                                             1000,        // Number of data points
                                             {0.0, 1.0},  // Parameter range
                                             {0.1, 0.1},  // Noise range
                                             FLAGS_curve_path, &real_paras_, &data);
    optimizer_->SetData(data);
    optimizer_->SetCurveFunction(curve_function_);
    optimizer_->SetRealParams(real_paras_);
  }

  Utils::Timer timer_;
  std::shared_ptr<Optimizer::CurveFittingOptimizer> optimizer_ = nullptr;
  std::function<double(const Eigen::VectorXd&, const double)> curve_function_;
  Eigen::VectorXd real_paras_;  // 随机生成的真值
  bool debug_ = true;
};

TEST_F(OptimizerTest, GaussNewtonTest) {
  if (!debug_) {
    return;
  }
  optimizer_->Optimize();
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
