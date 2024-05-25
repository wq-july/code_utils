#include "glog/logging.h"

int main(int argc, char** argv) {
  google::InitGoogleLogging(argv[0]);
  FLAGS_stderrthreshold = google::INFO;
  FLAGS_colorlogtostderr = true;

  // clang-format off
  LOG(INFO) << "\n"
            << "=============== \033[1;31mH\033[1;32me\033[1;33ml\033[1;34ml\033[1;35mo "
               "\033[1;36mW\033[1;33mo\033[1;31mr\033[1;32ml\033[1;34md\033[0m! ===============";
  // clang-format on

  return 0;
}