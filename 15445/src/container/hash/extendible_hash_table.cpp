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
    : global_depth_(0), bucket_size_(bucket_size), num_buckets_(1) {
  dir_.push_back(std::make_shared<Bucket>(Bucket(bucket_size, 0)));
}

template <typename K, typename V>
auto ExtendibleHashTable<K, V>::IndexOf(const K &key) -> size_t {
  int mask = (1 << global_depth_) - 1;
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
  if (dir_.empty()) {
    return false;
  }
  return dir_[IndexOf(key)]->Find(key, value);
}

template <typename K, typename V>
auto ExtendibleHashTable<K, V>::Remove(const K &key) -> bool {
  if (dir_.empty()) {
    return false;
  }
  return dir_[IndexOf(key)]->Remove(key);
}

template <typename K, typename V>
void ExtendibleHashTable<K, V>::Insert(const K &key, const V &value) {
  int index = IndexOf(key);
  auto bucket = dir_[index];
  bool inserted = bucket->Insert(key, value);
  int n = dir_.size();

  LOG_INFO("\n\ndir %d, buckets %d, index %d, inserted %i", n, num_buckets_, index, inserted);

  if (!inserted) {
    /**
     * If the bucket is full and can't be inserted, do the following steps before retrying:
     *    1. If the local depth of the bucket is equal to the global depth,
     *        increment the global depth and double the size of the directory.
     *    2. Increment the local depth of the bucket.
     *    3. Split the bucket and redistribute directory pointers & the kv pairs in the bucket.
     */
    if (bucket->GetDepth() == GetGlobalDepth()) {
      assert(dir_.size() == (1 << global_depth_)); // before double size of dir_
      global_depth_++;
      bucket->IncrementDepth();

      // redistribute directory pointers
      for (int i = 0, n = dir_.size(); i < n; i++) {
        if (i == index) {
          // tạo bucket mới (split the bucket)
          auto new_bucket = std::make_shared<Bucket>(Bucket(bucket_size_, global_depth_));
          dir_.push_back(new_bucket);
          num_buckets_++;
          assert(dir_[i + n] == new_bucket);
        } else {
          // tham chiếu tới bucket đang có
          dir_.push_back(dir_[i]);
          assert(dir_[i + n] == dir_[i]);
        }
      }
      assert(dir_.size() == 1 << global_depth_); // after double size of dir_

      // redistribute the kv pairs in the bucket
      LOG_INFO("Global: global_depth_ %d", global_depth_);
      RedistributeBucket(bucket);

    } else {
      int n = dir_.size();
      int i = 0; int k = 0;
      for (; i < n; i++) {
        if (dir_[i] == bucket) {
          k += 1;
          if (k == 2) { break; }
        }
      }
      bucket->IncrementDepth();
      auto new_bucket = std::make_shared<Bucket>(Bucket(bucket_size_, bucket->GetDepth()));
      dir_[i] = new_bucket;

      // redistribute the kv pairs in the bucket
      LOG_INFO("Local: new index %d", i);
      RedistributeBucket(bucket);
    }
    // Try to insert again
    Insert(key, value);
  }
}

template <typename K, typename V>
void ExtendibleHashTable<K, V>::RedistributeBucket(std::shared_ptr<Bucket> bucket) {
  auto list_ = bucket->GetItems();
  auto it = list_.begin();
  while (it != list_.end()) {
    int index = IndexOf(it->first);
    auto new_bucket = dir_[index];
    if (new_bucket != bucket) {
      LOG_INFO("Move to index %d", index);
      // insert trước khi xóa
      if (new_bucket->IsFull()) {

      } else {
        new_bucket->Insert(it->first, it->second);
        it = list_.erase(it);
      }
    } else {
      ++it;
    }
  }
}

//===--------------------------------------------------------------------===//
// Bucket
//===--------------------------------------------------------------------===//
template <typename K, typename V>
ExtendibleHashTable<K, V>::Bucket::Bucket(size_t array_size, int depth) : size_(array_size), depth_(depth) {}

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
