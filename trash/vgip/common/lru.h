/****************************************************************************
 *
 * Copyright (c) 2021 ZelosTech.com, Inc. All Rights Reserved
 *
 ***************************************************************************/
/**
 * @file lru.h
 **/

#pragma once

#include <cstdint>
#include <optional>
#include <unordered_map>
#include <iostream>

namespace zelos {
namespace zoe {
namespace localization {

template<typename KeyType, typename ValueType>
struct DlistNode {
 public:
  DlistNode() : prev_node_(nullptr), next_node_(nullptr) {};
  DlistNode(const KeyType& key, const ValueType& value) :
      key_(key), value_(value), prev_node_(nullptr), next_node_(nullptr) {};
  ~DlistNode() = default;

  KeyType key_;
  ValueType value_;
  DlistNode<KeyType, ValueType>* prev_node_;
  DlistNode<KeyType, ValueType>* next_node_;
};

template<typename KeyType, typename ValueType, typename Hash = std::hash<KeyType>,
    typename KeyEqual = std::equal_to<KeyType>>
class LRUCache {
 public:
  LRUCache(const int32_t capacity) {
    capacity_ = capacity;
    size_ = 0;
    head_ = new DlistNode<KeyType, ValueType>();
    tail_ = new DlistNode<KeyType, ValueType>();
    head_->next_node_ = tail_;
    tail_->prev_node_ = head_;
  }

  ~LRUCache() {
    // Properly delete all nodes to prevent memory leaks
    auto node = head_;
    while (node != nullptr) {
      auto next = node->next_node_;
      delete node;
      node = next;
    }
  }

  std::optional<std::reference_wrapper<ValueType>> GetData(const KeyType& key) {
    auto it = cache_.find(key);
    if (it == cache_.end()) {
        return std::nullopt;  // 如果没有找到，返回 std::nullopt
    }
    auto node = it->second;
    Refresh(node);  // 刷新节点位置，表示最近被使用
    return std::ref(node->value_);  // 返回引用包装器，允许外部修改内部值
  }

  std::optional<ValueType> Get(const KeyType& key) {
    auto it = cache_.find(key);
    if (it == cache_.end()) {
      return std::nullopt;
    }
    auto node = it->second;
    return node->value_;
  }


  void Put(const KeyType& key, const ValueType& value) {
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

  template<typename Func>
  void Traverse(Func f) {
    DlistNode<KeyType, ValueType>* current = head_->next_node_;
    while (current != tail_) {
      f(current->key_, &current->value_);
      current = current->next_node_;
    }
  }

  void PrintCacheValues() const {
    for (auto& pair : cache_) {
      auto node = pair.second;
      std::cout << "Key: " << node->key_.transpose() << " - Voxel: index = " << node->value_.GetIndex()
                << "  num is " << node->value_.GetPointsNum() << std::endl;
    }
  }

  int32_t Size() const {
    return size_;
  }

  int32_t Capacity() const {
    return capacity_;
  }

 private:
  void Refresh(DlistNode<KeyType, ValueType>* const node) {
    Remove(node);
    node->prev_node_ = head_;
    node->next_node_ = head_->next_node_;
    head_->next_node_->prev_node_ = node;
    head_->next_node_ = node;
  }

  void Remove(DlistNode<KeyType, ValueType>* const node) {
    if (node->next_node_ != nullptr) {
      node->prev_node_->next_node_ = node->next_node_;
      node->next_node_->prev_node_ = node->prev_node_;
    }
  }

 private:
  std::unordered_map<KeyType, DlistNode<KeyType, ValueType>*, Hash, KeyEqual> cache_;
  DlistNode<KeyType, ValueType>* head_;
  DlistNode<KeyType, ValueType>* tail_;
  int32_t size_;
  int32_t capacity_;
};

}  // namespace localization
}  // namespace zoe
}  // namespace zelos

/* vim: set expandtab ts=2 sw=2 sts=2 tw=100: */
