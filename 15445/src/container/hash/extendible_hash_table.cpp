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
  dir_[0]->id_ = 0;
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

void Bin(int n, int k) {
  // https://www.geeksforgeeks.org/binary-representation-of-a-given-number/
  for (int i = 1 << (k - 1); i > 0; i = i >> 1) {
    std::cout << ((n & i) != 0 ? 1 : 0);
  }
}

template <typename K, typename V>
void ExtendibleHashTable<K, V>::InsertInternal(const K &key, const V &value) {
  int index = IndexOf(key);
  auto insert_bucket = dir_[index];
  auto before_insert_bucket = *insert_bucket;
  bool inserted = insert_bucket->Insert(key, value);

  num_inserts_++;
  std::cout << "\n\n- - - - - - - - - - - - - - - - - - - - - - - - - - - -\n\n";
  std::cout << "[" << num_inserts_ << "] Insert key " << key << " (";
  Bin(std::hash<K>()(key), global_depth_);
  std::cout << ")"
            << " to bucket #" << insert_bucket->id_ << (inserted ? " success!" : " fail.") << "\n";

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
    std::cout << "Directory index " << i << " (";
    Bin(i, global_depth_);
    std::cout << ") -> bucket #" << bucket->id_ << ", ";
    if (k == m) {
      buckets.push_back(bucket);
      if (inserted && bucket == insert_bucket) {
        std::cout << "depth " << before_insert_bucket.GetDepth() << " ( ";
        for (auto const &item : before_insert_bucket.GetItems()) {
          std::cout << item.first << ", ";
        }
        std::cout << ") => ";
      }
      std::cout << "depth " << bucket->GetDepth() << " ( ";
      for (auto const &item : bucket->GetItems()) {
        std::cout << item.first << ", ";
      }
      std::cout << ")";
    }
    std::cout << "\n";
  }
  std::cout << "global_depth " << global_depth_ << ", bucket size " << bucket_size_ << std::endl;

  if (!inserted) {
    assert(insert_bucket->IsFull());
    // Assert dir_ size is consistent with global depth before double size of dir_
    assert(dir_.size() == (1 << global_depth_));

    if (insert_bucket->GetDepth() == global_depth_) {
      // Redistribute directory pointers
      for (int i = 0, n = dir_.size(); i < n; i++) {
        // tham chiếu tới bucket đang có
        dir_.push_back(dir_[i]);
        assert(dir_[i + n] == dir_[i]);
      }

      global_depth_++;
      std::cout << "!!! double the size of the directory, global_depth change to " << global_depth_ << std::endl;
      // Assert dir_ size is consistent with global depth after double size of dir_
      assert(dir_.size() == (1 << global_depth_));
    }

    RedistributeBucket(insert_bucket);
    InsertInternal(key, value);
  }
}

template <typename K, typename V>
void ExtendibleHashTable<K, V>::RedistributeBucket(std::shared_ptr<Bucket> bucket) {
  // Split the bucket
  bucket->IncrementDepth();
  auto new_bucket = std::make_shared<Bucket>(Bucket(bucket_size_, bucket->GetDepth()));
  new_bucket->id_ = num_buckets_;
  num_buckets_++;

  std::cout << "\nSplit bucket #" << bucket->id_ << " to new bucket #" << new_bucket->id_
            << ", and remap directory pointers:\n";

  // Bit mới được mở ra do tăng local depth
  int unlock_bit = 1 << (bucket->GetDepth() - 1);
  // Tìm vị trí để insert new bucket
  for (int i = 0, n = dir_.size(); i < n; i++) {
    if (dir_[i] == bucket) {  // tìm thấy vị trí của bucket khi chưa split
      std::cout << ">>> index " << i << " (";
      Bin(i, global_depth_);
      std::cout << "), unlock_bit ";
      Bin(unlock_bit, bucket->GetDepth());

      if ((i & unlock_bit) > 0) {
        std::cout << " => point to new bucket #" << new_bucket->id_;
        dir_[i] = new_bucket;
      }
      std::cout << std::endl;
    }
  }

  // Redistribute
  auto list = &bucket->GetItems();
  auto it = list->begin();

  int l = new_bucket->GetItems().size();
  int n = list->size();
  int m = bucket_size_;
  std::cout << "\nRedistribute bucket #" << bucket->id_ << " items:\n";
  std::cout << "(( bucket " << n << "/" << m << ", new_bucket " << l << "/" << m << " ))\n";

  while (it != list->end()) {
    auto key = it->first;
    int index = IndexOf(key);
    auto move_to_bucket = dir_[index];
    bool move = (move_to_bucket != bucket);

    std::cout << ">>> key " << key << " (";
    Bin(std::hash<K>()(key), global_depth_);

    if (move) {
      assert(move_to_bucket == new_bucket);
      std::cout << "), move to index " << index;
      assert(move_to_bucket->Insert(key, it->second));
      it = list->erase(it);
    } else {
      std::cout << "), stay at index " << index;
      ++it;
    }
    std::cout << std::endl;
  }

  l = new_bucket->GetItems().size();
  n = list->size();
  std::cout << "(( bucket " << n << "/" << m << ", new_bucket " << l << "/" << m << " ))\n";
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
