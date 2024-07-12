#include "common/kdtree.h"

#include <execution>

namespace Common {

bool KdTree::BuildTree(const Common::PointCloudPtr& cloud) {
  Utils::Timer timer;
  // 基本的安全性检测判断
  if (cloud->empty()) {
    return false;
  }

  // 清空点云，并且预分配内存和空间, for pcl cloud
  // cloud_.clear();
  // uint32_t size = cloud->size();
  // cloud_.reserve(size);
  // cloud_.resize(size);
  // for (uint32_t i = 0; i < cloud->size(); ++i) {
  //   cloud_.at(i) = cloud->points.at(i);
  // }

  uint32_t size = cloud->size();

  timer.StartTimer("PointCloud Construction");
  // 注意这里显式的调用了复制构造函数而不是移动构造函数
  cloud_ = std::make_shared<Common::PointCloud>(*cloud);
  // 注意，这样写会默认的调用移动构造函数，需要理解区别
  // cloud_ = std::make_shared<Common::PointCloud>(cloud);
  timer.StopTimer();
  timer.PrintElapsedTime();

  // 清空数据，合理
  Clear();
  Reset();
  // 这个有点奇怪，index.at(i) = i的话不是有点多余吗？
  std::vector<uint32_t> index(size);
  for (uint32_t i = 0; i < size; ++i) {
    index.at(i) = i;
  }
  // 这个合理，将所有点云的索引以根节点作为入口，开始往里面插入
  Insert(index, root_.get());
  return true;
}

// 按照p162算法步骤来
void KdTree::Insert(const std::vector<uint32_t>& points, KdTreeNode* const node) {
  // ? 成员变量noed_存储着树的所有结点，函数一进来就直接将node插入到表中
  nodes_.insert({node->id_, node});

  // 递归，考虑将子集点云points插入到node中
  // Insert(processed_clouds, next_node);

  //? 1.如果子集点云是空的，说明已经插入结束到叶子节点中了， 感觉这一步没啥必要吧？
  if (points.empty()) {
    return;
  }

  // 2.如果只有一个点云，那么说明刚好是叶子节点，因为只有叶子节点才会直接存储原始数据，并且只能存储1个（抽象）
  if (points.size() == 1) {
    // 那么此时需要将此节点标记为叶子节点
    size_++;
    node->point_index_ = points.front();
    return;
  }

  // 计算子集点云在各个轴的方差，挑选方差最大的一个轴，(x,y,z)其中一个轴，然后计算均值作为分割的阈值λ,得到左右分支
  std::vector<uint32_t> left, right;
  if (!FindSpliteAxisAndThresh(points, &node->axis_index_, &node->splite_thresh_, &left, &right)) {
    // 如果无法分割，说明到了叶子节点，
    size_++;
    node->point_index_ = points.front();
    return;
  }

  const auto creat_if_not_empty = [&node, this](const std::vector<uint32_t>& index,
                                                KdTreeNode** const new_node) {
    if (!index.empty()) {
      *new_node = new KdTreeNode;
      // 这个应该是单纯用来记录节点id名称
      (*new_node)->id_ = tree_node_id_++;
      Insert(index, *new_node);
    }
  };

  creat_if_not_empty(left, &node->left_);
  creat_if_not_empty(right, &node->right_);
}

bool KdTree::GetClosestPoint(const Eigen::Vector3d& pt,
                             std::vector<std::pair<uint32_t, double>>* const closest_index,
                             const uint32_t k_nums) {
  if (k_nums > size_) {
    LOG(ERROR) << "最近邻数量不能大于点云的数量！！";
    return false;
  }
  k_nums_ = k_nums;
  std::priority_queue<NodeAndDistance> knn_result;
  KnnSearch(pt, root_.get(), &knn_result);
  closest_index->clear();
  while (!knn_result.empty()) {
    closest_index->emplace_back(
        std::make_pair(knn_result.top().node_->point_index_, knn_result.top().distance_square_));
    knn_result.pop();
  }
  // 因为 priority_queue 是最大堆，我们需要反转结果以得到最近邻的顺序
  std::reverse(closest_index->begin(), closest_index->end());
  return true;
}

bool KdTree::GetClosestPointMT(const Common::PointCloudPtr& cloud,
                               std::vector<std::pair<uint32_t, uint32_t>>* const matches,
                               const uint32_t k_nums) {
  matches->resize(cloud->size() * k_nums);
  // 索引
  std::vector<int> index(cloud->size());
  for (uint32_t i = 0; i < cloud->size(); ++i) {
    index[i] = i;
  }

  std::for_each(std::execution::par_unseq,
                index.begin(),
                index.end(),
                [this, &cloud, &matches, &k_nums](int idx) {
                  std::vector<std::pair<uint32_t, double>> closest_idx;
                  GetClosestPoint(cloud->points()[idx], &closest_idx, k_nums);
                  for (uint32_t i = 0; i < k_nums; ++i) {
                    (*matches)[idx * k_nums + i].second = idx;
                    if (i < closest_idx.size()) {
                      (*matches)[idx * k_nums + i].first = closest_idx[i].first;
                    } else {
                      (*matches)[idx * k_nums + i].first = Utils::Math::ConstMath::kINVALID;
                    }
                  }
                });

  return true;
}

void KdTree::KnnSearch(const Eigen::Vector3d& pt,
                       KdTreeNode* node,
                       std::priority_queue<NodeAndDistance>* const knn_res) const {
  // 1. 首先检查当前节点是不是叶子节点，如果是叶子节点直接就计算距离比较就行了
  if (node->IsLeaf()) {
    ComputeDisForLeaf(pt, node, knn_res);
    return;
  }

  // 不是叶子节点，那么考虑pt落在当前node的左子树还是右子树中
  KdTreeNode* this_side = nullptr;
  KdTreeNode* that_side = nullptr;

  // 比较pt中分隔轴所在的维度和分割阈值，确定点应该落在当前节点的左子树或者是右子树中
  if (pt[node->axis_index_] < node->splite_thresh_) {
    this_side = node->left_;
    that_side = node->right_;
  } else {
    this_side = node->right_;
    that_side = node->left_;
  }

  // 确认点落在哪一侧之后，继续要进行递归，可以进一步减少不必要的计算和比较
  KnnSearch(pt, this_side, knn_res);
  // 比较一下确认是否需要对另一侧进行递归
  if (NeedExpand(pt, node, knn_res)) {
    KnnSearch(pt, that_side, knn_res);
  }
}

bool KdTree::NeedExpand(const Eigen::Vector3d& pt,
                        const KdTreeNode* node,
                        std::priority_queue<NodeAndDistance>* const knn_res) const {
  // 需要找到k个近邻
  if (knn_res->size() < k_nums_) {
    return true;
  }
  // 这个表示查询点到分割面的垂直距离，因为另外一侧不可能小于这个距离
  double dis = pt[node->axis_index_] - node->splite_thresh_;
  double dis_square = dis * dis;
  if (approximate_) {
    if (dis_square < knn_res->top().distance_square_ * alpha_) {
      return true;
    } else {
      return false;
    }
  } else {
    if (dis_square < knn_res->top().distance_square_) {
      return true;
    } else {
      return false;
    }
  }
}

void KdTree::ComputeDisForLeaf(const Eigen::Vector3d& pt,
                               KdTreeNode* node,
                               std::priority_queue<NodeAndDistance>* const result) const {
  // 叶子节点中直接存储着点在点云中的索引，不在是中间节点中没有意义的序号
  double dis_square = DistanceSquare(pt, cloud_->points().at(node->point_index_));

  if (result->size() < k_nums_) {
    // 这个牛逼了，优先级队列中可以直接插入数据类型的构造函数入参就行
    result->emplace(node, dis_square);
  } else {
    // 注意是优先级队列，插入元素后，内部会自动排序
    if (dis_square < result->top().distance_square_) {
      result->emplace(node, dis_square);
      result->pop();
    }
  }
}

// 根据点云的分布计算高斯分布的均值和方差来确定分割面和分割值
bool KdTree::FindSpliteAxisAndThresh(const std::vector<uint32_t>& point_index,
                                     uint32_t* const axis,
                                     double* const thresh,
                                     std::vector<uint32_t>* const left,
                                     std::vector<uint32_t>* const right) {
  if (point_index.empty()) {
    LOG(ERROR) << "point index is empty!!";
  }
  // 使用方差最大的那个轴作为分割轴，对应的均值作为分割阈值
  Eigen::Vector3d var = Eigen::Vector3d::Zero();
  Eigen::Vector3d mean = Eigen::Vector3d::Zero();

  Utils::Math::ComputeMeanAndVariance(point_index, cloud_->points(), &mean, &var);
  var.maxCoeff(axis);
  *thresh = mean[*axis];

  for (const auto& index : point_index) {
    if (cloud_->points()[index][*axis] < *thresh) {
      left->emplace_back(index);
    } else {
      right->emplace_back(index);
    }
  }

  // 边界情况检查：输入的points等于同一个值，上面的判定是>=号，所以都进了右侧
  // 这种情况不需要继续展开，直接将当前节点设为叶子就行
  if (point_index.size() > 1 && (left->empty() || right->empty())) {
    return false;
  }

  return true;
}

void KdTree::Reset() {
  tree_node_id_ = 0;
  root_.reset(new KdTreeNode());
  root_->id_ = tree_node_id_++;
  size_ = 0;
}

void KdTree::Clear() {
  for (const auto& np : nodes_) {
    if (np.second != root_.get()) {
      delete np.second;
    }
  }

  nodes_.clear();
  root_ = nullptr;
  size_ = 0;
  tree_node_id_ = 0;
}

void KdTree::PrintAll() {
  for (const auto& np : nodes_) {
    auto node = np.second;
    if (node->left_ == nullptr && node->right_ == nullptr) {
      LOG(INFO) << "leaf node: " << node->id_ << ", idx: " << node->point_index_;
    } else {
      LOG(INFO) << "node: " << node->id_ << ", axis: " << node->axis_index_
                << ", th: " << node->splite_thresh_;
    }
  }
}

}  // namespace Common