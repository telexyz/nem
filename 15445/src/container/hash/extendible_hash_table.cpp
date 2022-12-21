//===----------------------------------------------------------------------===//
//
//                         BusTub
//
// extendible_hash_table.cpp
//
// Identification: src/container/hash/extendible_hash_table.cpp
//
// Copyright (c) 2022, Carnegie Mellon University Database Group
//
//===----------------------------------------------------------------------===//

#include <cassert>
#include <cstdlib>
#include <functional>
#include <list>
#include <utility>

#include "common/logger.h"
#include "container/hash/extendible_hash_table.h"
#include "storage/page/page.h"

namespace bustub {

template <typename K, typename V>
ExtendibleHashTable<K, V>::ExtendibleHashTable(size_t bucket_size)
    : global_depth_(0), bucket_size_(bucket_size), num_buckets_(1), num_inserts_(0) {
  // khởi tạo bucket đầu tiên của hashtable với local depth = global depth = 0
  dir_.push_back(std::make_shared<Bucket>(Bucket(bucket_size, 0)));
}

template <typename K, typename V>
inline auto ExtendibleHashTable<K, V>::IndexOf(const K &key) -> size_t {
  int mask = (1 << global_depth_) - 1;
  // 2^global_depth_ - 1 => lấy global_depth_ bits cuối (LSBs: least significant bits) làm mask
  return std::hash<K>()(key) & mask;
}

template <typename K, typename V>
inline auto ExtendibleHashTable<K, V>::GetGlobalDepth() const -> int {
  std::scoped_lock<std::shared_mutex> lock_shared(latch_);
  return global_depth_;
}

template <typename K, typename V>
inline auto ExtendibleHashTable<K, V>::GetLocalDepth(int dir_index) const -> int {
  std::scoped_lock<std::shared_mutex> lock_shared(latch_);
  return dir_[dir_index]->GetDepth();
}

template <typename K, typename V>
inline auto ExtendibleHashTable<K, V>::GetNumBuckets() const -> int {
  std::scoped_lock<std::shared_mutex> lock_shared(latch_);
  return num_buckets_;
}

template <typename K, typename V>
auto ExtendibleHashTable<K, V>::Find(const K &key, V &value) -> bool {
  std::scoped_lock<std::shared_mutex> lock_shared(latch_);
  auto index = IndexOf(key);
  assert(index < dir_.size());
  bool result = dir_[index]->Find(key, value);
  return result;
}

template <typename K, typename V>
auto ExtendibleHashTable<K, V>::Remove(const K &key) -> bool {
  std::scoped_lock<std::shared_mutex> lock(latch_);
  auto index = IndexOf(key);
  assert(index < dir_.size());
  bool result = dir_[index]->Remove(key);
  return result;
}

template <typename K, typename V>
void ExtendibleHashTable<K, V>::Insert(const K &key, const V &value) {
  std::scoped_lock<std::shared_mutex> lock(latch_);
  InsertInternal(key, value);
}

template <typename K, typename V>
void ExtendibleHashTable<K, V>::InsertInternal(const K &key, const V &value) {
  int index = IndexOf(key);
  auto insert_bucket = dir_[index];

  if (insert_bucket->IsFull()) {
    assert(dir_.size() == (1 << global_depth_));

    if (insert_bucket->GetDepth() == global_depth_) {
      // Redistribute directory pointers
      for (int i = 0, n = dir_.size(); i < n; i++) {
        // tham chiếu tới bucket đang có
        dir_.push_back(dir_[i]);
        assert(dir_[i + n] == dir_[i]);
      }

      global_depth_++;
      assert(dir_.size() == (1 << global_depth_));
    }

    RedistributeBucket(insert_bucket);
    InsertInternal(key, value);
  } else {
    insert_bucket->Insert(key, value);
  }
}

template <typename K, typename V>
inline void ExtendibleHashTable<K, V>::RedistributeBucket(std::shared_ptr<Bucket> bucket) {
  // Split the bucket
  bucket->IncrementDepth();
  auto new_bucket = std::make_shared<Bucket>(Bucket(bucket_size_, bucket->GetDepth()));
  num_buckets_++;

  // Bit mới được mở ra do tăng local depth
  int unlock_bit = 1 << (bucket->GetDepth() - 1);

  // Tìm dir_ index map tới new bucket
  for (int i = 0, n = dir_.size(); i < n; i++) {
    if (dir_[i] == bucket) {  // tìm thấy vị trí của bucket khi chưa split
      if ((i & unlock_bit) > 0) {
        dir_[i] = new_bucket;
      }
    }
  }

  // Redistribute
  auto before_redis_total_capacity = bucket->curr_size_ + new_bucket->curr_size_;

  size_t i = 0;
  while (i < bucket->curr_size_) {
    auto key = bucket->list_k_[i];
    int index = IndexOf(key);
    auto move_to_bucket = dir_[index];
    bool move = (move_to_bucket != bucket);
    if (move) {
      assert(move_to_bucket == new_bucket);
      move_to_bucket->Insert(key, bucket->list_v_[i]);
      bucket->RemoveIndex(i);
    } else {
      ++i;
    }
  }

  assert(before_redis_total_capacity == bucket->curr_size_ + new_bucket->curr_size_);
  assert(!bucket->IsFull() || !new_bucket->IsFull());
}

//===--------------------------------------------------------------------===//
// Bucket
//===--------------------------------------------------------------------===//
template <typename K, typename V>
ExtendibleHashTable<K, V>::Bucket::Bucket(size_t array_size, int depth) : size_(array_size), depth_(depth) {
  assert(curr_size_ == 0);
  list_k_ = new K[array_size];
  list_v_ = new V[array_size];
}

template <typename K, typename V>
auto ExtendibleHashTable<K, V>::Bucket::Find(const K &key, V &value) -> bool {
  for (size_t i = 0; i < curr_size_; i++) {
    if (list_k_[i] == key) {
      value = list_v_[i];
      return true;
    }
  }
  return false;
}

template <typename K, typename V>
auto ExtendibleHashTable<K, V>::Bucket::Remove(const K &key) -> bool {
  for (size_t i = 0; i < curr_size_; ++i) {
    if (list_k_[i] == key) {
      RemoveIndex(i);
      return true;
    }
  }
  return false;
}

template <typename K, typename V>
auto ExtendibleHashTable<K, V>::Bucket::Insert(const K &key, const V &value) -> bool {
  if (IsFull()) {  // return false if bucket is full
    return false;
  }

  for (size_t i = 0; i < curr_size_; ++i) {
    if (list_k_[i] == key) {  // if key exists,
      list_v_[i] = value;     // update value
      return true;
    }
  }

  // list_.push_back(std::pair<K, V>(key, value));
  assert(curr_size_ < size_);
  list_k_[curr_size_] = key;
  list_v_[curr_size_] = value;
  curr_size_++;

  return true;
}

template class ExtendibleHashTable<page_id_t, Page *>;
template class ExtendibleHashTable<Page *, std::list<Page *>::iterator>;
template class ExtendibleHashTable<int, int>;
// test purpose
template class ExtendibleHashTable<int, std::string>;
template class ExtendibleHashTable<int, std::list<int>::iterator>;

}  // namespace bustub
