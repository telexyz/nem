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
  bool bucket_found = (bucket == nullptr);
  bool bucket_full = bucket->IsFull();
  bool inserted = bucket->Insert(key, value);

  LOG_INFO("\n\nnum_buckets = `%d`, index = `%d`, bucket_found `%i`, bucket full `%i`, inserted `%i`", num_buckets_, index, bucket_found, bucket_full,
           inserted);

  if (bucket_found or !inserted) {
    /**
     * If the bucket is full and can't be inserted, do the following steps before retrying:
     *    1. If the local depth of the bucket is equal to the global depth,
     *        increment the global depth and double the size of the directory.
     *    2. Increment the local depth of the bucket.
     *    3. Split the bucket and redistribute directory pointers & the kv pairs in the bucket.
     */
    if (bucket->GetDepth() == GetGlobalDepth()) {
      global_depth_++;
      bucket->IncrementDepth();

      // redistribute directory pointers
      int n = dir_.size();
      for (int i = 0; i < n; i++) {
        if (i == index) {
          // tạo bucket mới (split the bucket)
          auto new_bucket = std::make_shared<Bucket>(Bucket(bucket_size_, global_depth_));
          dir_.push_back(new_bucket);
          num_buckets_++;

          // redistribute the kv pairs in the bucket
          auto list_ = bucket->GetItems();
          auto it = list_.begin();
          while (it != list_.end()) {
            int new_index = IndexOf(it->first);
            LOG_INFO("Global %d: new_index = `%d`, index `%i`", global_depth_, new_index, index);
            if (new_index != index) {
              new_bucket->Insert(it->first, it->second); // insert trước khi xóa
              it = list_.erase(it);
            } else { ++it; }
          }
        } else {
          // tham chiếu tới bucket đang có
          dir_.push_back(dir_[i]);
        }
      }

    } else {
      int offset = 1 << bucket->GetDepth();
      bucket->IncrementDepth();
      auto new_bucket = std::make_shared<Bucket>(Bucket(bucket_size_, bucket->GetDepth()));
      dir_[offset + index] = new_bucket;

      // redistribute the kv pairs in the bucket
      auto list_ = bucket->GetItems();
      auto it = list_.begin();
      while (it != list_.end()) {
        int new_index = IndexOf(it->first);
        LOG_INFO("Local: new_index = `%d`, index `%i`", new_index, index);
        if (new_index != index) {
          new_bucket->Insert(it->first, it->second); // insert trước khi xóa
          it = list_.erase(it);
        } else { ++it; }
      }
    }
    // Try to insert again
    Insert(key, value);
  }
}

// template <typename K, typename V>
// void ExtendibleHashTable<K, V>::RedistributeBucket(std::shared_ptr<Bucket> bucket) {
// }

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
