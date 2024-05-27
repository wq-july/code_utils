#pragma once

#include <cstdint>
#include <iostream>
#include <optional>
#include <unordered_map>

namespace Common {

template <typename KeyType, typename ValueType>
struct DlistNode {
 public:
  DlistNode();
  DlistNode(const KeyType& key, const ValueType& value);
  ~DlistNode();

  KeyType key_;
  ValueType value_;
  DlistNode<KeyType, ValueType>* prev_node_;
  DlistNode<KeyType, ValueType>* next_node_;
};

template <typename KeyType, typename ValueType, typename Hash = std::hash<KeyType>,
          typename KeyEqual = std::equal_to<KeyType>>
class LRUCache {
 public:
  LRUCache(const int32_t capacity);
  ~LRUCache();

  std::optional<std::reference_wrapper<ValueType>> GetData(const KeyType& key);
  std::optional<ValueType> Get(const KeyType& key);
  void Put(const KeyType& key, const ValueType& value);

  template <typename Func>
  void Traverse(Func f);

  void PrintCacheValues() const;

  int32_t Size() const;
  int32_t Capacity() const;

 private:
  void Refresh(DlistNode<KeyType, ValueType>* const node);
  void Remove(DlistNode<KeyType, ValueType>* const node);

 private:
  std::unordered_map<KeyType, DlistNode<KeyType, ValueType>*, Hash, KeyEqual> cache_;
  DlistNode<KeyType, ValueType>* head_;
  DlistNode<KeyType, ValueType>* tail_;
  int32_t size_;
  int32_t capacity_;
};

}  // namespace Common

#include "common/lru.tpp"  // 包含模板实现文件
