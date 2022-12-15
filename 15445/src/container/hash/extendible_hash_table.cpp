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
  int dir_size = dir_.size();  // convert thành int trước để có thể so sánh với dir_index
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
  latch_.lock();
  auto index = IndexOf(key);
  assert(index < dir_.size());
  bool result = dir_[index]->Find(key, value);
  latch_.unlock();
  return result;
}

template <typename K, typename V>
auto ExtendibleHashTable<K, V>::Remove(const K &key) -> bool {
  latch_.lock();
  auto index = IndexOf(key);
  assert(index < dir_.size());
  bool result = dir_[index]->Remove(key);
  latch_.unlock();
  return result;
}

template <typename K, typename V>
void ExtendibleHashTable<K, V>::Insert(const K &key, const V &value) {
  std::scoped_lock<std::mutex> lock(latch_);
  InsertInternal(key, value);
}

template <typename K, typename V>
void ExtendibleHashTable<K, V>::InsertInternal(const K &key, const V &value) {
  int index = IndexOf(key);
  auto bucket = dir_[index];
  bool inserted = bucket->Insert(key, value);

  int mask = (1 << global_depth_) - 1;
  int hash = std::hash<K>()(key);

  num_inserts_++;
  LOG_INFO("\n\n[%d] hash %d, mask %d, index %d, inserted %i", num_inserts_, hash, mask, index, inserted);

  if (inserted) {
    // in ra kết quả
    std::vector<std::shared_ptr<ExtendibleHashTable<K, V>::Bucket>> buckets;
    for (int i = 0, n = dir_.size(); i < n; i++) {
      auto bucket = dir_[i];
      int k = 0;
      int m = buckets.size();
      for (; k < m; k++) {
        if (buckets[k] == bucket) {
          break;
        }
      }
      std::cout << "Directory Slot #" << i << " => Bucket #" << k;
      if (k == m) {
        std::cout  << ", depth " << bucket->GetDepth() << " ( ";
        buckets.push_back(bucket);
        for (auto const &item : bucket->GetItems()) {
          std::cout << item.first << ", ";
        }
        std::cout << ")";
      }
      std::cout << "\n";
    }
    std::cout << "\nglobal_depth " << global_depth_ << "\n\n";

  } else {
    assert(bucket->IsFull());
    // Assert dir_ size is consistent with global depth before double size of dir_
    // assert(dir_.size() == (1 << global_depth_));

    if (bucket->GetDepth() == global_depth_) {
      // Redistribute directory pointers
      for (int i = 0, n = dir_.size(); i < n; i++) {
        // tham chiếu tới bucket đang có
        dir_.push_back(dir_[i]);
        assert(dir_[i + n] == dir_[i]);
      }

      global_depth_++;
      LOG_INFO("!!! global_depth changed %d", global_depth_);
      // Assert dir_ size is consistent with global depth after double size of dir_
      // assert(dir_.size() == (1 << global_depth_));
    }

    RedistributeBucket(bucket);
    InsertInternal(key, value);
  }
}

template <typename K, typename V>
void ExtendibleHashTable<K, V>::RedistributeBucket(std::shared_ptr<Bucket> bucket) {
  // Split the bucket
  bucket->IncrementDepth();
  auto new_bucket = std::make_shared<Bucket>(Bucket(bucket_size_, bucket->GetDepth()));
  num_buckets_++;

  // Bit mới được mở ra do tăng local depth
  int unlock_bit = 1 << (bucket->GetDepth() - 1);
  // Tìm vị trí để insert new bucket
  for (int i = 0, n = dir_.size(); i < n; i++) {
    if (dir_[i] == bucket) {  // tìm thấy vị trí của bucket khi chưa split
      LOG_INFO(">>> i %d & %d => %d", i, unlock_bit, i & unlock_bit);
      if ((i & unlock_bit) > 0) {
        LOG_INFO("!!! new_bucket at %d", i);
        dir_[i] = new_bucket;
      } else {
      }
    }
  }

  // Redistribute
  auto list = &bucket->GetItems();
  auto it = list->begin();

  int l = new_bucket->GetItems().size();
  int n = list->size();
  int m = bucket_size_;
  LOG_INFO("Redis: bucket %d/%d, new_bucket %d/%d", n, m, l, m);

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

  l = new_bucket->GetItems().size();
  n = list->size();
  LOG_INFO("Redis: bucket %d/%d, new_bucket %d/%d", n, m, l, m);
  assert(!bucket->IsFull() || !new_bucket->IsFull());
}

//===--------------------------------------------------------------------===//
// Bucket
//===--------------------------------------------------------------------===//
template <typename K, typename V>
ExtendibleHashTable<K, V>::Bucket::Bucket(size_t array_size, int depth) : size_(array_size), depth_(depth) {
  assert(list_.empty());
}

template <typename K, typename V>
auto ExtendibleHashTable<K, V>::Bucket::Find(const K &key, V &value) -> bool {
  for (auto &kv : list_) {
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
