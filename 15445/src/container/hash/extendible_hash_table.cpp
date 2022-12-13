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
auto ExtendibleHashTable<K, V>::IndexOf(const K &key) -> size_t {
  int mask = (1 << global_depth_) - 1;
  // 2^global_depth_ - 1 => lấy global_depth_ bits cuối (LSBs: least significant bits) làm mask
  return std::hash<K>()(key) & mask;
}

template <typename K, typename V>
auto ExtendibleHashTable<K, V>::GetGlobalDepth() const -> int {
  std::scoped_lock<std::mutex> lock(latch_);
  return GetGlobalDepthInternal();
}

template <typename K, typename V>
auto ExtendibleHashTable<K, V>::GetGlobalDepthInternal() const -> int {
  return global_depth_;
}

template <typename K, typename V>
auto ExtendibleHashTable<K, V>::GetLocalDepth(int dir_index) const -> int {
  std::scoped_lock<std::mutex> lock(latch_);
  return GetLocalDepthInternal(dir_index);
}

template <typename K, typename V>
auto ExtendibleHashTable<K, V>::GetLocalDepthInternal(int dir_index) const -> int {
  int dir_size = dir_.size();
  assert(dir_index < dir_size);
  return dir_[dir_index]->GetDepth();
}

template <typename K, typename V>
auto ExtendibleHashTable<K, V>::GetNumBuckets() const -> int {
  std::scoped_lock<std::mutex> lock(latch_);
  return GetNumBucketsInternal();
}

template <typename K, typename V>
auto ExtendibleHashTable<K, V>::GetNumBucketsInternal() const -> int {
  return num_buckets_;
}

template <typename K, typename V>
auto ExtendibleHashTable<K, V>::Find(const K &key, V &value) -> bool {
  auto index = IndexOf(key);
  assert(index < dir_.size());
  return dir_[index]->Find(key, value);
}

template <typename K, typename V>
auto ExtendibleHashTable<K, V>::Remove(const K &key) -> bool {
  auto index = IndexOf(key);
  assert(index < dir_.size());
  return dir_[index]->Remove(key);
}

template <typename K, typename V>
void ExtendibleHashTable<K, V>::Insert(const K &key, const V &value) {
  int index = IndexOf(key);
  auto bucket = dir_[index];
  bool inserted = bucket->Insert(key, value);

  int mask = (1 << global_depth_) - 1;
  int hash = std::hash<K>()(key);

  num_inserts_++;
  // assert(num_inserts_ < 15); // TODO: Remove after finish debugging
  LOG_INFO("\n\n[%d] hash %d, mask %d, index %d, inserted %i", num_inserts_, hash, mask, index, inserted);
  if (inserted) { return; };

  /**
   * If the bucket is full and can't be inserted, do the following steps before retrying:
   *    1. If the local depth of the bucket is equal to the global depth,
   *        increment the global depth and double the size of the directory.
   *    2. Increment the local depth of the bucket.
   *    3. Split the bucket and redistribute directory pointers & the kv pairs in the bucket.
   */
  assert(bucket->IsFull());
  // Assert dir_ size is consistent with global depth before double size of dir_
  assert(dir_.size() == (1 << global_depth_));

  if (bucket->GetDepth() == global_depth_) {
    // Redistribute directory pointers
    for (int i = 0, n = dir_.size(); i < n; i++) {
      // tham chiếu tới bucket đang có
      dir_.push_back(dir_[i]);
      assert(dir_[i + n] == dir_[i]);
    }

    // Assert dir_ size is consistent with global depth after double size of dir_
    global_depth_++;
    assert(dir_.size() == (1 << global_depth_));

    LOG_INFO("Global: global_depth %d", global_depth_);
    RedistributeBucket(bucket);

  } else {
    // Case2: In case the local depth is less than the global depth, only Bucket Split takes place. 
    // Then increment only the local depth value by 1. And, assign appropriate pointers.
    LOG_INFO("Local: local_depth %d", bucket->GetDepth());
    RedistributeBucket(bucket);
  }
  // Try to insert again
  Insert(key, value);
}


template <typename K, typename V>
void ExtendibleHashTable<K, V>::RedistributeBucket(std::shared_ptr<Bucket> bucket) {

  // Split the bucket
  bucket->IncrementDepth();
  auto new_bucket = std::make_shared<Bucket>(Bucket(bucket_size_, bucket->GetDepth()));
  num_buckets_++;

  // Tìm vị trí để insert new bucket
  std::list<int> old_indexes;
  std::list<int> new_indexes;
  int new_bit_mask = 1 << (bucket->GetDepth() - 1);
  for (int i = 0, n = dir_.size(); i < n; i++) {
    if (dir_[i] == bucket) { // tìm thấy vị trí của bucket khi chưa split
      LOG_INFO(">>> i %d & %d => %d", i, new_bit_mask, i & new_bit_mask);
      if ((i & new_bit_mask) > 0) {
        LOG_INFO("!!! new_bucket at %d", i);
        dir_[i] = new_bucket;
        new_indexes.push_back(i);
      } else {
        old_indexes.push_back(i);
      }
    }
  }

  // Redistribute 
  auto list = &bucket->GetItems();
  auto it = list->begin();

  int n = list->size(), m = bucket_size_;
  LOG_INFO("Redis: bucket capacity %d/%d", n, m);  

  while (it != list->end()) {
    int index = IndexOf(it->first);
    auto move_to_bucket = dir_[index];
    int hash = std::hash<K>()(it->first);
    LOG_INFO(">>> hash %d, index %d, move? %d", hash, index, move_to_bucket != bucket);
    if (move_to_bucket != bucket) {
      assert(move_to_bucket == new_bucket);
      LOG_INFO("Redistribute to index %d", index);
      assert(move_to_bucket->Insert(it->first, it->second));
      it = list->erase(it);
    } else {
      ++it;
    }
  }
  n = list->size();
  LOG_INFO("Redis: bucket capacity %d/%d", n, m);
  assert(!bucket->IsFull());
}

//===--------------------------------------------------------------------===//
// Bucket
//===--------------------------------------------------------------------===//
template <typename K, typename V>
ExtendibleHashTable<K, V>::Bucket::Bucket(size_t array_size, int depth) : size_(array_size), depth_(depth) {
  assert(list_.size() == 0);
}

template <typename K, typename V>
auto ExtendibleHashTable<K, V>::Bucket::Find(const K &key, V &value) -> bool {
  for (auto kv : list_) {
    if (kv.first == key) {
      value = kv.second;
      return true;
    }
  }
  return false;
}

template <typename K, typename V>
auto ExtendibleHashTable<K, V>::Bucket::Remove(const K &key) -> bool {
  for (auto it = list_.begin(); it != list_.end(); ++it) {
    if (it->first == key) {
      list_.erase(it);
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

  for (auto it = list_.begin(); it != list_.end(); ++it) {
    if (it->first == key) {  // if key exists,
      it->second = value;    // update value
      return true;
    }
  }

  list_.push_back(std::pair<K, V>(key, value));
  return true;
}

template class ExtendibleHashTable<page_id_t, Page *>;
template class ExtendibleHashTable<Page *, std::list<Page *>::iterator>;
template class ExtendibleHashTable<int, int>;
// test purpose
template class ExtendibleHashTable<int, std::string>;
template class ExtendibleHashTable<int, std::list<int>::iterator>;

}  // namespace bustub
