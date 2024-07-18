#pragma once

#include <vector>

#include "Eigen/Dense"
#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/core/eigen.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "sophus/se3.hpp"

#include "../protos/pb/camera.pb.h"
#include "camera/camera_model/camera_model.h"
#include "camera/feature_manager.h"
#include "camera/pnp_solver.h"

namespace Camera {

class SFM {
 public:
  SFM(const CameraConfig::SFMConfig config);
  ~SFM() = default;

  double FindFundamentalMatrix(const std::vector<cv::KeyPoint>& p2ds_1,
                               const std::vector<cv::KeyPoint>& p2ds_2,
                               const std::vector<cv::DMatch>& matches,
                               Eigen::Matrix3d* const fundamental_mat,
                               std::vector<bool>* const inliers) const;

  bool FindEssentialMatrix(const std::vector<cv::KeyPoint>& p2ds_1,
                           const std::vector<cv::KeyPoint>& p2ds_2,
                           const std::vector<cv::DMatch>& matches,
                           Eigen::Matrix3d* const essential_mat,
                           std::vector<bool>* const inliers) const;

  double FindHomography(const std::vector<cv::KeyPoint>& p2ds_1,
                        const std::vector<cv::KeyPoint>& p2ds_2,
                        const std::vector<cv::DMatch>& matches,
                        Eigen::Matrix3d* const homography_mat,
                        std::vector<bool>* const inliers) const;

  bool DecomposeEssentialMatrix(const Eigen::Matrix3d& essential_mat,
                                std::vector<Sophus::SE3d>* const relative_pose) const;

  bool DecomposeHomographyMatrix(
      const Eigen::Matrix3d& homography_mat,
      std::vector<std::tuple<Eigen::Matrix3d, Eigen::Vector3d, Eigen::Vector3d>>* const
          relative_pose) const;

  /**
   * 从基础矩阵F中求解位姿R，t及三维点
   * F分解出E，E有四组解，选择计算的有效三维点（在摄像头前方、投影误差小于阈值、视差角大于阈值）最多的作为最优的解
   */
  bool ReconstructFromFmat(const Eigen::Matrix3d& F21,
                           const std::vector<cv::KeyPoint>& kp1,
                           const std::vector<cv::KeyPoint>& kp2,
                           const std::vector<cv::DMatch>& matches,
                           const int32_t min_triangulated_pts,
                           const double min_parallax,
                           std::vector<bool>* const inliers,
                           std::vector<Eigen::Vector3d>* const p3d,
                           Eigen::Matrix3d* const R21,
                           Eigen::Vector3d* const t21);

  /**
   * @brief 用H矩阵恢复R, t和三维点
   * H矩阵分解常见有两种方法：Faugeras SVD-based decomposition 和 Zhang SVD-based decomposition
   * 代码使用了Faugeras SVD-based decomposition算法，参考文献
   * Motion and structure from motion in a piecewise planar environment. International Journal of
   * Pattern Recognition and Artificial Intelligence, 1988
   */
  bool ReconstructH(const Eigen::Matrix3d& H21,
                    const std::vector<cv::KeyPoint>& kp1,
                    const std::vector<cv::KeyPoint>& kp2,
                    const std::vector<cv::DMatch>& matches,
                    const int32_t min_triangulated_pts,
                    const double min_parallax,
                    std::vector<bool>* const inliers,
                    std::vector<Eigen::Vector3d>* const p3d,
                    Eigen::Matrix3d* const R21,
                    Eigen::Vector3d* const t21,
                    Eigen::Vector3d* const n21) const;

  /**
   * @brief 从特征点匹配求homography（normalized DLT）
   * |x'|     | h1 h2 h3 ||x|
   * |y'| = a | h4 h5 h6 ||y|  简写: x' = a H x, a为一个尺度因子
   * |1 |     | h7 h8 h9 ||1|
   * 使用DLT(direct linear tranform)求解该模型
   * x' = a H x
   * ---> (x') 叉乘 (H x)  = 0
   * ---> Ah = 0
   * A = | 0  0  0 -x -y -1 xy' yy' y'|  h = | h1 h2 h3 h4 h5 h6 h7 h8 h9 |
   *     |-x -y -1  0  0  0 xx' yx' x'|
   * 通过SVD求解Ah = 0，A'A最小特征值对应的特征向量即为解
   * @param  p1s 归一化后的点, in reference frame
   * @param  p2s 归一化后的点, in current frame
   * @return     单应矩阵
   * @see        Multiple View Geometry in Computer Vision - Algorithm 4.2 p109
   */
  Eigen::Matrix3d ComputeHomographyMat21(const std::vector<Eigen::Vector2d>& p1s,
                                         const std::vector<Eigen::Vector2d>& p2s) const;

  /**
   * @brief 从特征点匹配求fundamental matrix（normalized 8点法）
   * x'Fx = 0 整理可得：Af = 0
   * A = | x'x x'y x' y'x y'y y' x y 1 |, f = | f1 f2 f3 f4 f5 f6 f7 f8 f9 |
   * 通过SVD求解Af = 0，A'A最小特征值对应的特征向量即为解
   * @param  p1s 归一化后的点, in reference frame
   * @param  p2s 归一化后的点, in current frame
   * @return     基础矩阵
   * @see        Multiple View Geometry in Computer Vision - Algorithm 11.1 p282 (中文版 p191)
   */
  Eigen::Matrix3d ComputeFundamentalMat21(const std::vector<Eigen::Vector2d>& p1s,
                                          const std::vector<Eigen::Vector2d>& p2s) const;

  /**
   * @brief 像素坐标标准化，计算点集的横纵均值，与均值偏差的均值。最后返回的是变化矩阵T
   * 直接乘以像素坐标的齐次向量即可获得去中心去均值后的特征点坐标
   * @param oringinal_pts 特征点
   * @param norm_pts 去中心去均值后的特征点坐标
   * @param T  变化矩阵
   */
  void Normalize(const std::vector<Eigen::Vector2d>& oringinal_pts,
                 std::vector<Eigen::Vector2d>* const norm_pts,
                 Eigen::Matrix3d* const T) const;
  /**
   * @brief 检查结果
   * @param F21 顾名思义
   * @param inliers 匹配是否合法，大小为matches
   * @param sigma 默认为1
   */
  double CheckFundamentalMat(const std::vector<cv::KeyPoint>& pts1,
                             const std::vector<cv::KeyPoint>& pts2,
                             const std::vector<cv::DMatch>& matches,
                             const Eigen::Matrix3d& F21,
                             std::vector<bool>* const inliers,
                             const double sigma = 1.0) const;

  /**
   * @brief 检查结果
   * @param H21 顾名思义
   * @param H12 顾名思义
   * @param inliers 匹配是否合法，大小为matches
   * @param sigma 默认为1
   */
  double CheckHomographyMat(const std::vector<cv::KeyPoint>& pts1,
                            const std::vector<cv::KeyPoint>& pts2,
                            const std::vector<cv::DMatch>& matches,
                            const Eigen::Matrix3d& H21,
                            std::vector<bool>* const inliers,
                            double sigma = 1.0) const;

  double ProjectErrorFmat(const std::vector<cv::KeyPoint>& pts1,
                          const std::vector<cv::KeyPoint>& pts2,
                          const std::vector<cv::DMatch>& matches,
                          const Eigen::Matrix3d& M21,
                          std::vector<bool>* const inliers) const;

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
                   Eigen::Vector3d* const x3D) const;

  int32_t CheckRT(const Eigen::Matrix3d& R,
                  const Eigen::Vector3d& t,
                  const std::vector<cv::KeyPoint>& kps1,
                  const std::vector<cv::KeyPoint>& kps2,
                  const std::vector<cv::DMatch>& matches,
                  const Eigen::Matrix3d& K,
                  const double th2,
                  std::vector<bool>* const inliers,
                  std::vector<Eigen::Vector3d>* const p3ts,
                  double* const parallax) const;

  /**
* @brief 分解Essential矩阵
* 解释的比较好的博客：https://blog.csdn.net/weixin_44580210/article/details/90344511
* F矩阵通过结合内参可以得到Essential矩阵，分解E矩阵将得到4组解
* 这4组解分别为[R1,t],[R1,-t],[R2,t],[R2,-t]
* ## 反对称矩阵性质
* 多视图几何上定义：一个3×3的矩阵是本质矩阵的充要条件是它的奇异值中有两个相等而第三个是0，为什么呢？
* 首先我们知道
E=[t]×​R=SR其中S为反对称矩阵，反对称矩阵有什么性质呢？
* 结论1：如果 S 是实的反对称矩阵，那么S=UBU^T，其中 B
为形如diag(a1​Z，a2​Z...am​Z，0，0...0)的分块对角阵，其中 Z = [0, 1; -1, 0]
* 反对称矩阵的特征矢量都是纯虚数并且奇数阶的反对称矩阵必是奇异的
* 那么根据这个结论我们可以将 S 矩阵写成 S=kUZU^⊤，而 Z 为
* | 0, 1, 0 |
* |-1, 0, 0 |
* | 0, 0, 0 |
* Z = diag(1, 1, 0) * W     W 为
* | 0,-1, 0 |
* | 1, 0, 0 |
* | 0, 0, 1 |
* E=SR=Udiag(1,1,0)(WU^⊤R)  这样就证明了 E 拥有两个相等的奇异值
*
* ## 恢复相机矩阵
* 假定第一个摄像机矩阵是P=[I∣0]，为了计算第二个摄像机矩阵P′，必须把 E
矩阵分解为反对成举着和旋转矩阵的乘积 SR。
* 还是根据上面的结论1，我们在相差一个常数因子的前提下有
S=UZU^T，我们假设旋转矩阵分解为UXV^T，注意这里是假设旋转矩阵分解形式为UXV^T，并不是旋转矩阵的svd分解，
* 其中 UV都是E矩阵分解出的
* Udiag(1,1,0)V^T = E = SR = (UZU^T)(UXV^⊤) = U(ZX)V^T
* 则有 ZX = diag(1,1,0)，因此 x=W或者 X=W^T
* 结论：如果 E 的SVD分解为 Udiag(1,1,0)V^⊤，E = SR有两种分解形式，分别是： S = UZU^⊤    R =
UWVTor UW^TV^⊤

* 接着分析，又因为St=0（自己和自己叉乘肯定为0嘛）以及∥t∥=1（对两个摄像机矩阵的基线的一种常用归一化），因此 t = U(0,0,1)^T = u3​，
* 即矩阵 U
的最后一列，这样的好处是不用再去求S了，应为t的符号不确定，R矩阵有两种可能，因此其分解有如下四种情况：
* P′=[UWV^T ∣ +u3​] or [UWV^T ∣ −u3​] or [UW^TV^T ∣ +u3​] or [UW^TV^T ∣
−u3​]
* @param E  Essential Matrix
* @param R1 Rotation Matrix 1
* @param R2 Rotation Matrix 2
* @param t  Translation
* @see Multiple View Geometry in Computer Vision - Result 9.19 p259
*/
  void DecomposeE(const Eigen::Matrix3d& E,
                  Eigen::Matrix3d* const R1,
                  Eigen::Matrix3d* const R2,
                  Eigen::Vector3d* const t) const;

  // 下面的几个函数用来分解单应矩阵
  double OppositeOfMinor(const Eigen::Matrix3d& M, const int32_t row, const int32_t col) const;

  void FindRmatFromTstarN(const Eigen::Vector3d& tstar,
                          const Eigen::Vector3d& n,
                          const double v,
                          Eigen::Matrix3d* const R) const;

  // Ezio Malis, et.al. "Deeper understanding of the homography decomposition for vision-based
  // control". 2007
  void DecomposeH_EM(
      const Eigen::Matrix3d& H,
      const Eigen::Matrix3d& K,
      std::vector<std::tuple<Eigen::Matrix3d, Eigen::Vector3d, Eigen::Vector3d>>* const solutions)
      const;

  // Ezio Malis, et.al. "Deeper understanding of the homography decomposition for vision-based
  // control". 2007
  void DecomposeH_Zhang(
      const Eigen::Matrix3d& H,
      const Eigen::Matrix3d& K,
      std::vector<std::tuple<Eigen::Matrix3d, Eigen::Vector3d, Eigen::Vector3d>>* const solutions)
      const;

  void DecomposeH_ORBSLAM3(
      const Eigen::Matrix3d& H,
      const Eigen::Matrix3d& K,
      std::vector<std::tuple<Eigen::Matrix3d, Eigen::Vector3d, Eigen::Vector3d>>* const solutions)
      const;

  bool SolveMotion(
      const Eigen::Matrix3d& Hnorm,
      const Eigen::Vector3d& tstar,
      const Eigen::Vector3d& n,
      std::tuple<Eigen::Matrix3d, Eigen::Vector3d, Eigen::Vector3d>* const solution) const;

 private:
  CameraConfig::SFMConfig config_;
  std::vector<Eigen::Vector3d> p3ds_;
  std::shared_ptr<PnpSolver> pnp_solver_ = nullptr;
  std::shared_ptr<CameraBase> camera_model_ = nullptr;
  std::shared_ptr<FeatureManager> feature_manager_ = nullptr;
};

}  // namespace Camera