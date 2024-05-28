#pragma once

#include <Eigen/Core>
#include <iostream>

#include "sophus/so3.hpp"

namespace Common {

struct Constant {
  // static error state index
  static constexpr uint32_t DIM_STATE = 27u;
  static constexpr uint32_t DIM_State = 15u;

  static constexpr uint32_t POS = 0u;
  static constexpr uint32_t VEL = 3u;
  static constexpr uint32_t ROT = 6u;
  static constexpr uint32_t BG = 9u;
  static constexpr uint32_t BA = 12u;
  static constexpr uint32_t GRA = 15u;
  static constexpr uint32_t R_L_I = 18u;
  static constexpr uint32_t T_L_I = 21u;
  static constexpr uint32_t R_B_I = 24u;
  // useless
  static constexpr uint32_t T_B_I = 27u;

  // static noise index
  static constexpr uint32_t DIM_NOISE = 12u;
  static constexpr uint32_t N_G = 0u;
  static constexpr uint32_t N_A = 3u;
  static constexpr uint32_t N_BG = 6u;
  static constexpr uint32_t N_BA = 9u;

  static constexpr double Gravity = -9.81;
};

}  // namespace Common
