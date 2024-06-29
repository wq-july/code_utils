
namespace Common {

template <typename KeyType, typename ValueType>
DlistNode<KeyType, ValueType>::DlistNode() : prev_node_(nullptr), next_node_(nullptr) {}

template <typename KeyType, typename ValueType>
DlistNode<KeyType, ValueType>::DlistNode(const KeyType& key, const ValueType& value)
    : key_(key), value_(value), prev_node_(nullptr), next_node_(nullptr) {}

template <typename KeyType, typename ValueType>
DlistNode<KeyType, ValueType>::~DlistNode() = default;

template <typename KeyType, typename ValueType, typename Hash, typename KeyEqual>
LRUCache<KeyType, ValueType, Hash, KeyEqual>::LRUCache(const int32_t capacity) {
  capacity_ = capacity;
  size_ = 0;
  head_ = new DlistNode<KeyType, ValueType>();
  tail_ = new DlistNode<KeyType, ValueType>();
  head_->next_node_ = tail_;
  tail_->prev_node_ = head_;
}

template <typename KeyType, typename ValueType, typename Hash, typename KeyEqual>
LRUCache<KeyType, ValueType, Hash, KeyEqual>::~LRUCache() {
  auto node = head_;
  while (node != nullptr) {
    auto next = node->next_node_;
    delete node;
    node = next;
  }
}

template <typename KeyType, typename ValueType, typename Hash, typename KeyEqual>
std::optional<std::reference_wrapper<ValueType>>
LRUCache<KeyType, ValueType, Hash, KeyEqual>::GetData(const KeyType& key) {
  auto it = cache_.find(key);
  if (it == cache_.end()) {
    return std::nullopt;
  }
  auto node = it->second;
  Refresh(node);
  return std::ref(node->value_);
}

template <typename KeyType, typename ValueType, typename Hash, typename KeyEqual>
std::optional<ValueType> LRUCache<KeyType, ValueType, Hash, KeyEqual>::Get(const KeyType& key) {
  auto it = cache_.find(key);
  if (it == cache_.end()) {
    return std::nullopt;
  }
  auto node = it->second;
  return node->value_;
}

template <typename KeyType, typename ValueType, typename Hash, typename KeyEqual>
void LRUCache<KeyType, ValueType, Hash, KeyEqual>::Put(const KeyType& key, const ValueType& value) {
  if (cache_.find(key) == cache_.end()) {
    auto node = new DlistNode<KeyType, ValueType>(key, value);
    cache_[key] = node;
    Refresh(node);
    ++size_;
    if (size_ > capacity_) {
      auto removed_node = tail_->prev_node_;
      Remove(removed_node);
      cache_.erase(removed_node->key_);
      delete removed_node;
      --size_;
    }
  } else {
    auto node = cache_[key];
    node->value_ = value;
    Refresh(node);
  }
}

template <typename KeyType, typename ValueType, typename Hash, typename KeyEqual>
template <typename Func>
void LRUCache<KeyType, ValueType, Hash, KeyEqual>::Traverse(Func f) {
  DlistNode<KeyType, ValueType>* current = head_->next_node_;
  while (current != tail_) {
    f(current->key_, &current->value_);
    current = current->next_node_;
  }
}

template <typename KeyType, typename ValueType, typename Hash, typename KeyEqual>
void LRUCache<KeyType, ValueType, Hash, KeyEqual>::PrintCacheValues() const {
  for (auto& pair : cache_) {
    auto node = pair.second;
    std::cout << "Key: " << node->key_.transpose()
              << " - Voxel: index = " << node->value_.GetIndex() << "  num is "
              << node->value_.GetPointsNum() << std::endl;
  }
}

template <typename KeyType, typename ValueType, typename Hash, typename KeyEqual>
int32_t LRUCache<KeyType, ValueType, Hash, KeyEqual>::Size() const {
  return size_;
}

template <typename KeyType, typename ValueType, typename Hash, typename KeyEqual>
int32_t LRUCache<KeyType, ValueType, Hash, KeyEqual>::Capacity() const {
  return capacity_;
}

template <typename KeyType, typename ValueType, typename Hash, typename KeyEqual>
void LRUCache<KeyType, ValueType, Hash, KeyEqual>::Refresh(
    DlistNode<KeyType, ValueType>* const node) {
  Remove(node);
  node->prev_node_ = head_;
  node->next_node_ = head_->next_node_;
  head_->next_node_->prev_node_ = node;
  head_->next_node_ = node;
}

template <typename KeyType, typename ValueType, typename Hash, typename KeyEqual>
void LRUCache<KeyType, ValueType, Hash, KeyEqual>::Remove(
    DlistNode<KeyType, ValueType>* const node) {
  if (node->next_node_ != nullptr) {
    node->prev_node_->next_node_ = node->next_node_;
    node->next_node_->prev_node_ = node->prev_node_;
  }
}

}  // namespace Common
