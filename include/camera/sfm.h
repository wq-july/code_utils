#pragma once

#include "Eigen/Core"
#include "Eigen/Dense"
#include "glog/logging.h"
#include "opencv2/calib3d.hpp"

#include "../protos/pb/camera.pb.h"

namespace Camera {

class SFM {
 public:
  SFM(const CameraConfig::SFM config) : config_(config) {}
  ~SFM() = default;

  bool SolvePNP();

  /**
   * From ORB-SLAM3
   * @brief 三角化获得三维点
   * @param x_c1 点在关键帧1下的归一化坐标
   * @param x_c2 点在关键帧2下的归一化坐标
   * @param Tc1w 关键帧1投影矩阵  [K*R | K*t]
   * @param Tc2w 关键帧2投影矩阵  [K*R | K*t]
   * @param x3D 三维点坐标，作为结果输出
   */
  bool Triangulate(const Eigen::Vector3d& x_c1,
                   const Eigen::Vector3d& x_c2,
                   const Eigen::Matrix<double, 3, 4>& Tc1w,
                   const Eigen::Matrix<double, 3, 4>& Tc2w,
                   Eigen::Vector3d* const x3D) {
    Eigen::Matrix4d A;
    // x = a*P*X， 左右两面乘Pc的反对称矩阵 a*[x]^ * P *X = 0
    // 构成了A矩阵，中间涉及一个尺度a，因为都是归一化平面，但右面是0所以直接可以约掉不影响最后的尺度
    //  0 -1 v    P(0)     -P.row(1) + v*P.row(2)
    //  1 0 -u *  P(1)  =   P.row(0) - u*P.row(2)
    // -v u  0    P(2)    u*P.row(1) - v*P.row(0)
    // 发现上述矩阵线性相关，所以取前两维，两个点构成了4行的矩阵，就是如下的操作，求出的是4维的结果[X,Y,Z,A]，所以需要除以最后一维使之为1，就成了[X,Y,Z,1]这种齐次形式
    A.block<1, 4>(0, 0) = x_c1(0) * Tc1w.block<1, 4>(2, 0) - Tc1w.block<1, 4>(0, 0);
    A.block<1, 4>(1, 0) = x_c1(1) * Tc1w.block<1, 4>(2, 0) - Tc1w.block<1, 4>(1, 0);
    A.block<1, 4>(2, 0) = x_c2(0) * Tc2w.block<1, 4>(2, 0) - Tc2w.block<1, 4>(0, 0);
    A.block<1, 4>(3, 0) = x_c2(1) * Tc2w.block<1, 4>(2, 0) - Tc2w.block<1, 4>(1, 0);

    // 解方程 AX=0
    Eigen::JacobiSVD<Eigen::Matrix4d> svd(A, Eigen::ComputeFullV);

    Eigen::Vector4d x3Dh = svd.matrixV().col(3);

    if (std::fabs(x3Dh(3)) < 1e-6) {
      LOG(ERROR) << "Triangulate failed !";
      return false;
    }

    // Euclidean coordinates
    (*x3D) = x3Dh.head(3) / x3Dh(3);

    return true;
  }

 private:
  CameraConfig::SFM config_;
};

}  // namespace Camera