#include <iostream>

#include "sensor/imu.h"

int main(int argc, char** argv) {
  if (argc < 2) {
    std::cout << "Usage: " << argv[0] << " <imu_data_file_path>" << std::endl;
    return 1;
  }
  std::string file_path = argv[1];
  Sensor::IMU imu_processer;
  imu_processer.ReadData(file_path);
  std::cout << "imu date size is " << imu_processer.GetVecData().size() << "\n";
  return 0;
}
