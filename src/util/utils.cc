#include "util/utils.h"

namespace Utils {

void GenerateRandomCoefficientsAndData(
    const std::function<double(const Eigen::VectorXd&, const double)>& func,
    const int32_t param_count,
    const int32_t data_size,
    const std::pair<double, double>& param_range,
    const std::pair<double, double>& noise_range,
    const std::string log_path,
    Eigen::VectorXd* const parameters,
    std::vector<std::pair<double, double>>* const data) {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<> gen_param(param_range.first, param_range.second);
  std::uniform_real_distribution<> gen_noise(noise_range.first, noise_range.second);

  // Generate random parameters
  parameters->resize(param_count);
  for (int32_t i = 0; i < param_count; ++i) {
    (*parameters)(i) = gen_param(gen);
  }

  if (std::filesystem::exists(log_path)) {
    std::filesystem::remove(log_path);
  }

  std::fstream log;
  if (!log_path.empty()) {
    log.open(log_path, std::ios::app);
  }

  // Generate data with noise
  data->clear();
  data->resize(data_size);
  for (int32_t i = 0; i < data_size; ++i) {
    double x = static_cast<double>(i) / data_size;
    double y = func(*parameters, x) + gen_noise(gen);

    data->at(i) = {x, y};
    if (!log_path.empty()) {
      log << x << " " << y << "\n";
    }
  }

  LOG(INFO) << "Generated function with parameters: ";

  for (int32_t i = 0; i < param_count; ++i) {
    LOG(INFO) << (*parameters)(i);
  }
}

}  // namespace Utils