#pragma once

#include <cstdint>
#include <execution>
#include <memory>
#include <queue>
#include <unordered_map>
#include <vector>

#include "Eigen/Core"

#include "glog/logging.h"

// #include "pcl/point_cloud.h"
// #include "pcl/point_types.h"

#include "common/data/point_cloud.h"
#include "util/math.h"

#include "util/time.h"

namespace Common {

// kdtree 节点，广义上和二叉树结构基本一致
struct KdTreeNode {
  uint32_t id_ = -1;          // 节点的id
  uint32_t point_index_ = 0;  // 点索引
  uint32_t axis_index_ = 0;  // 分割轴的索引（这个分割轴貌似是需要根据数据的高斯分布来确定的）
  double splite_thresh_ = 0.0;   // 分割位置
  KdTreeNode* left_ = nullptr;   // 左子树
  KdTreeNode* right_ = nullptr;  // 右子树

  bool IsLeaf() const {
    return left_ == nullptr && right_ == nullptr;
  }
};

struct NodeAndDistance {
  NodeAndDistance(KdTreeNode* node, const double dis_square)
      : node_(node), distance_square_(dis_square) {}

  KdTreeNode* node_ = nullptr;
  double distance_square_ = 0.0;

  bool operator<(const NodeAndDistance& other) const {
    return distance_square_ < other.distance_square_;
  }
};

class KdTree {
 public:
  explicit KdTree() = default;
  ~KdTree() {
    Clear();
  }

  bool BuildTree(const Common::Data::PointCloudPtr& cloud);

  // 获取k最近邻
  bool GetClosestPoint(const Eigen::Vector3d& pt,
                       std::vector<uint32_t>* const closest_index,
                       const uint32_t k_nums = 5);

  // 多线程并行为点云寻找最近邻
  bool GetClosestPointMT(const Common::Data::PointCloudPtr& cloud,
                         std::vector<std::pair<uint32_t, uint32_t>>* const matches,
                         const uint32_t k_nums = 5);

  // 返回节点数量
  uint32_t Size() const {
    return size_;
  }

  // 清理数据
  void Clear();

  void Reset();

  // 打印所有节点信息
  void PrintAll();

 public:
  // 这个被用于计算最近邻的倍数
  void SetEnableANN(bool use_ann = true, float alpha = 0.1) {
    approximate_ = use_ann;
    alpha_ = alpha;
  }

 private:
  static inline double DistanceSquare(const Eigen::Vector3d& p1, const Eigen::Vector3d& p2) {
    return (p1 - p2).squaredNorm();
  }

 private:
  // 数据插入
  void Insert(const std::vector<uint32_t>& points_index, KdTreeNode* const node);

  bool FindSpliteAxisAndThresh(const std::vector<uint32_t>& point_index,
                               uint32_t* const axis,
                               double* const thresh,
                               std::vector<uint32_t>* const left,
                               std::vector<uint32_t>* const right);

  void KnnSearch(const Eigen::Vector3d& pt,
                 KdTreeNode* node,
                 std::priority_queue<NodeAndDistance>* const knn_result) const;

  void ComputeDisForLeaf(const Eigen::Vector3d& pt,
                         KdTreeNode* node,
                         std::priority_queue<NodeAndDistance>* const result) const;

  bool NeedExpand(const Eigen::Vector3d& pt,
                  const KdTreeNode* node,
                  std::priority_queue<NodeAndDistance>* const knn_res) const;

 private:
  // 根节点
  std::shared_ptr<KdTreeNode> root_ = nullptr;
  // 输入的点云，那这个点云能否进一步动态扩展呢？这样就成了一个动态的点云地图，变成ikdtree?
  // std::vector<Eigen::Vector3d> cloud_;

  Common::Data::PointCloudPtr cloud_ = nullptr;

  // 使用hash打法存储节点的索引和对应的节点指针？
  std::unordered_map<uint32_t, KdTreeNode*> nodes_;

  uint32_t k_nums_ = 5;
  // 叶子节点数量
  uint32_t size_ = 0;
  uint32_t tree_node_id_ = 0;

  // 用于近似最近邻
  bool approximate_ = true;
  double alpha_ = 0.1;
};

}  // namespace Common
