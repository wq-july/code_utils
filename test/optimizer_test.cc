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
#include "pcl/io/pcd_io.h"
#include "util/time.h"

#define private public

#include "optimizer/optimizer.h"

DEFINE_string(config_path, "../conf/imu_config.yaml", "imu的配置文件");
DEFINE_string(scan_pcd_path, "../test/data/lidar/scan.pcd", "scan点云路径");

class OptimizerTest : public testing::Test {
 public:
  void SetUp() override {
    scan_.reset(new pcl::PointCloud<pcl::PointXYZ>());
    pcl::io::loadPCDFile<pcl::PointXYZ>(FLAGS_scan_pcd_path, *scan_);
    optimizer_ = std::make_shared<Optimizer::NonlinearOptimizer>(
        Optimizer::OptimizeMethod::GAUSS_NEWTON, 3u, 0.01);
    GenerateRandomCoefficientsAndData(&real_parameter_, &data_with_noise_);
  }

  void GenerateRandomCoefficientsAndData(std::vector<double>* parameters,
                                         std::vector<double>* data) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> gen_coeff(-1.0, 1.0);  // 系数范围[-10, 10]
    std::uniform_real_distribution<> gen_noise(-0.1, 0.1);    // 数据的噪声范围[-0.5, 0.5]

    uint32_t size = 3;
    parameters->clear();
    parameters->resize(size);
    for (uint i = 0; i < size; ++i) {
      parameters->at(i) = gen_coeff(gen);
    }

    uint32_t data_size = 1000;
    data->clear();
    data->resize(data_size);

    for (uint32_t i = 0; i < data_size; ++i) {
      double polynomial = 0.0;
      for (uint32_t j = 0; j < size; ++j) {
        polynomial += parameters->at(j) * std::pow(static_cast<double>(i) / 1000.0, j);
      }
      data->at(i) = std::exp(polynomial) + gen_noise(gen);
    }

    std::cout <<  "f(x) = exp(";
    for (uint32_t i = 0; i < size; ++i) {
      if (i == 0) {
        std::cout << parameters->at(i);
      } else {
        std::cout << parameters->at(i) << " * x^" << i;
      }
      if (i != size - 1) {
        std::cout << " + ";
      } else {
        std::cout << ") \n";
      }
    }
  }

  pcl::PointCloud<pcl::PointXYZ>::Ptr scan_ = nullptr;
  Utils::Timer timer_;
  std::shared_ptr<Optimizer::NonlinearOptimizer> optimizer_ = nullptr;
  std::vector<double> real_parameter_;
  std::vector<double> data_with_noise_;
  bool debug_ = true;
};

TEST_F(OptimizerTest, GaussNewtonTest) {
  if (!debug_) {
    return;
  }

  // 1. 设置参数，迭代次数，截止阈值等
  uint32_t max_iters = 20;
  double break_threshold = 0.00001;

  // 1. 初始值 x0 -> f(x0) 得到误差量 b(x0) = y - F(x0)
  // 2. b(x0) -> gauss_newton -> δx
  // 3. x0 + δx -> x1
  // 4. 继续第一步，一直循环，直到 δx < break_threshold
  // 因为在最优解附近，只能往最优解方向一点一点挪移

  auto para_size = real_parameter_.size();
  auto data_size = data_with_noise_.size();

  // 估计的最优状态量
  Eigen::VectorXd x = Eigen::VectorXd::Zero(para_size);

  // m x n 残差量维数 x 状态量维数
  Eigen::MatrixXd jacobian = Eigen::MatrixXd::Zero(1, para_size);

  // 更新量，这个可以用不同的方法来更新
  Eigen::VectorXd dx = Eigen::VectorXd::Zero(para_size);

  double sum_err = 0.0;
  double last_sum_err = 0.0;

  uint32_t iters = 0;
  while (iters < max_iters) {
    // 由于存在很多次观测，因此都加到一起去；
    Eigen::MatrixXd H = Eigen::MatrixXd::Zero(para_size, para_size);
    Eigen::VectorXd b = Eigen::VectorXd::Zero(para_size);

    for (uint32_t i = 0; i < data_size; ++i) {
      // 1. 根据初始状态量，计算残差量，这个残差量用于估计迭代方向δx;
      double estimated_data = 0.0;
      double polynomial = 0.0;
      for (uint32_t j = 0; j < para_size; ++j) {
        polynomial += x(j) * std::pow(static_cast<double>(i) / 1000.0, j);
        estimated_data = std::exp(polynomial);
      }
      double err = estimated_data - data_with_noise_.at(i);

      // 2. 计算雅可比矩阵，得到等式 J * J^T = -J^T * err， 其中， err -> f(x)
      for (uint32_t j = 0; j < para_size; ++j) {
        jacobian(0, j) = estimated_data * std::pow(static_cast<double>(i) / 1000.0, j);
      }

      H += jacobian.transpose() * jacobian;
      b += -jacobian.transpose() * err;

      sum_err += err * err;
    }

    // timer_.StartTimer("QR solver!");
    // // QR分解，可加快求解速度
    // dx = H.colPivHouseholderQr().solve(b);
    // LOG(INFO) << "QR solver, dx = " << dx.transpose();
    // timer_.StartTimer();
    // timer_.PrintElapsedTime();

    // timer_.StartTimer("SVD solver!");
    // // ldlt分解，可加快求解速度
    // dx = H.ldlt().solve(b);
    // LOG(INFO) << "SVD solver, dx = " << dx.transpose();
    // timer_.StartTimer();
    // timer_.PrintElapsedTime();

    timer_.StartTimer("Just inverse!");
    // 3. 计算 δx
    dx = H.inverse() * b;
    LOG(INFO) << "Just inverse, dx = " << dx.transpose();
    timer_.StartTimer();
    timer_.PrintElapsedTime(Utils::Timer::Microseconds);

    // 6.更新状态量 x0 + δx -> x1, 注意x中有旋转的话，需要另外更新了
    x += dx;

    LOG(INFO) << "cur sum_err is " << sum_err << ", last sum_err is " << last_sum_err;

    last_sum_err = sum_err;
    ++iters;

    if (dx.norm() < break_threshold) {
      break;
    }
  }

  LOG(INFO) << "iter times is " << iters;

  for (uint32_t i = 0; i < para_size; ++i) {
    EXPECT_LE(std::fabs(x(i) - real_parameter_[i]), 0.1);
  }
}


TEST_F(OptimizerTest, LMTest) {
  if (!debug_) {
    return;
  }

  // 1. 设置参数，迭代次数，截止阈值等
  uint32_t max_iters = 20;
  double break_threshold = 0.00001;

  // 1. 初始值 x0 -> f(x0) 得到误差量 b(x0) = y - F(x0)
  // 2. b(x0) -> gauss_newton -> δx
  // 3. x0 + δx -> x1
  // 4. 继续第一步，一直循环，直到 δx < break_threshold
  // 因为在最优解附近，只能往最优解方向一点一点挪移

  auto para_size = real_parameter_.size();
  auto data_size = data_with_noise_.size();

  // 估计的最优状态量
  Eigen::VectorXd x = Eigen::VectorXd::Zero(para_size);

  // m x n 残差量维数 x 状态量维数
  Eigen::MatrixXd jacobian = Eigen::MatrixXd::Zero(1, para_size);

  // 更新量，这个可以用不同的方法来更新
  Eigen::VectorXd dx = Eigen::VectorXd::Zero(para_size);

  double sum_err = 0.0;
  double last_sum_err = 0.0;

  uint32_t iters = 0;
  while (iters < max_iters) {
    // 由于存在很多次观测，因此都加到一起去；
    Eigen::MatrixXd H = Eigen::MatrixXd::Zero(para_size, para_size);
    Eigen::VectorXd b = Eigen::VectorXd::Zero(para_size);

    for (uint32_t i = 0; i < data_size; ++i) {
      // 1. 根据初始状态量，计算残差量，这个残差量用于估计迭代方向δx;
      double estimated_data = 0.0;
      double polynomial = 0.0;
      for (uint32_t j = 0; j < para_size; ++j) {
        polynomial += x(j) * std::pow(static_cast<double>(i) / 1000.0, j);
        estimated_data = std::exp(polynomial);
      }
      double err = estimated_data - data_with_noise_.at(i);

      // 2. 计算雅可比矩阵，得到等式 J * J^T = -J^T * err， 其中， err -> f(x)
      for (uint32_t j = 0; j < para_size; ++j) {
        jacobian(0, j) = estimated_data * std::pow(static_cast<double>(i) / 1000.0, j);
      }

      H += jacobian.transpose() * jacobian;
      b += -jacobian.transpose() * err;

      sum_err += err * err;
    }

    // timer_.StartTimer("QR solver!");
    // // QR分解，可加快求解速度
    // dx = H.colPivHouseholderQr().solve(b);
    // LOG(INFO) << "QR solver, dx = " << dx.transpose();
    // timer_.StartTimer();
    // timer_.PrintElapsedTime();

    // timer_.StartTimer("SVD solver!");
    // // ldlt分解，可加快求解速度
    // dx = H.ldlt().solve(b);
    // LOG(INFO) << "SVD solver, dx = " << dx.transpose();
    // timer_.StartTimer();
    // timer_.PrintElapsedTime();

    timer_.StartTimer("Just inverse!");
    // 3. 计算 δx
    dx = H.inverse() * b;
    LOG(INFO) << "Just inverse, dx = " << dx.transpose();
    timer_.StartTimer();
    timer_.PrintElapsedTime(Utils::Timer::Microseconds);

    // 6.更新状态量 x0 + δx -> x1, 注意x中有旋转的话，需要另外更新了
    x += dx;

    LOG(INFO) << "cur sum_err is " << sum_err << ", last sum_err is " << last_sum_err;

    last_sum_err = sum_err;
    ++iters;

    if (dx.norm() < break_threshold) {
      break;
    }
  }

  LOG(INFO) << "iter times is " << iters;

  for (uint32_t i = 0; i < para_size; ++i) {
    EXPECT_LE(std::fabs(x(i) - real_parameter_[i]), 0.1);
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
