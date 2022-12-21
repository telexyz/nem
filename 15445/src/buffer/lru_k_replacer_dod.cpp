//===----------------------------------------------------------------------===//
//
//                         BusTub
//
// lru_k_replacer.cpp
//
// Identification: src/buffer/lru_k_replacer.cpp
//
// Copyright (c) 2015-2022, Carnegie Mellon University Database Group
//
//===----------------------------------------------------------------------===//

#include "buffer/lru_k_replacer.h"
#include <algorithm>
#include <iostream>
#include "common/logger.h"

namespace bustub {

LRUKReplacer::LRUKReplacer(size_t num_frames, size_t k) : replacer_size_(num_frames), k_(k) {
  // frame_entries_ = new FrameEntry[num_frames];
  evictable_.flip();
  assert(evictable_.size() >= num_frames);
  // assert(evictable_.all()); // đảm bảo tất cả là true
  // assert(is_active_.none()); // đảm bảo tất cả là false
  hits_count_ = new uint16_t[num_frames];
  for (size_t i = 0; i < num_frames; ++i) {
    hits_count_[i] = 0;
  }
  pos_ = new std::list<frame_id_t>::iterator[num_frames];
}

void LRUKReplacer::RecordAccess(frame_id_t frame_id) {
  std::scoped_lock<std::mutex> lock(latch_);
  // * You can also use BUSTUB_ASSERT to abort the process if frame id is invalid.
  BUSTUB_ASSERT(frame_id < static_cast<frame_id_t>(replacer_size_), "Invalid frame_id");

  is_active_.set(frame_id, true);
  size_t updated_hits_count = ++hits_count_[frame_id];

  if (updated_hits_count == 1) {  // first time
    history_list_.emplace_front(frame_id);
    pos_[frame_id] = history_list_.begin();
    if (evictable_[frame_id]) {
      curr_size_++;
      curr_history_size_++;
    }

  } else if (updated_hits_count == k_) {
    // move from history_list_ to cache_list_
    history_list_.erase(pos_[frame_id]);
    if (evictable_[frame_id]) {
      curr_history_size_--;
    }

    // add to begin of the cache_list_
    cache_list_.emplace_front(frame_id);
    pos_[frame_id] = cache_list_.begin();

  } else if (updated_hits_count > k_) {
    if (pos_[frame_id] != cache_list_.begin()) {
      // switch to begin of the cache_list_
      cache_list_.erase(pos_[frame_id]);
      cache_list_.emplace_front(frame_id);
      pos_[frame_id] = cache_list_.begin();
    }
  }
}

void LRUKReplacer::SetEvictable(frame_id_t frame_id, bool set_evictable) {
  std::scoped_lock<std::mutex> lock(latch_);
  BUSTUB_ASSERT(frame_id < static_cast<frame_id_t>(replacer_size_), "Invalid frame_id");

  bool hit_before = (is_active_[frame_id] && hits_count_[frame_id] > 0);
  auto previously_is_evictable = hit_before && evictable_[frame_id];
  auto currently_is_evictable = hit_before && set_evictable;
  evictable_.set(frame_id, set_evictable);

  if (!previously_is_evictable && currently_is_evictable) {
    curr_size_++;
    if (hits_count_[frame_id] < k_) {  // in history_list_
      curr_history_size_++;
    }
  } else if (previously_is_evictable && !currently_is_evictable) {
    curr_size_--;
    if (hits_count_[frame_id] < k_) {  // in history_list_
      curr_history_size_--;
    }
  }
}

auto LRUKReplacer::Evict(frame_id_t *frame_id) -> bool {
  std::scoped_lock<std::mutex> lock(latch_);
  return EvictInternal(frame_id);
}
auto LRUKReplacer::EvictInternal(frame_id_t *frame_id) -> bool {
  if (curr_size_ == 0) {
    return false;
  }

  bool evict_from_history = (curr_history_size_ > 0);
  auto &list = evict_from_history ? history_list_ : cache_list_;
  auto rit = list.rbegin();
  while (!evictable_[*rit]) {
    rit++;
  }

  *frame_id = *rit;
  is_active_.set(*frame_id, false);
  evictable_.set(*frame_id, true);
  hits_count_[*frame_id] = 0;

  // https://stackoverflow.com/questions/1830158/how-to-call-erase-with-a-reverse-iterator
  list.erase(std::next(rit).base());
  curr_history_size_ -= evict_from_history;  // NOLINT
  curr_size_--;
  return true;
}

void LRUKReplacer::Remove(frame_id_t frame_id) {
  std::scoped_lock<std::mutex> lock(latch_);

  if (is_active_[frame_id]) {
    BUSTUB_ASSERT(evictable_[frame_id], "Can't remove an inevictable frame.");
    if (hits_count_[frame_id] > 0 && hits_count_[frame_id] < k_) {  // in history_list_
      history_list_.erase(pos_[frame_id]);
      curr_history_size_--;
      curr_size_--;

    } else if (hits_count_[frame_id] >= k_) {  // in cache_list_
      cache_list_.erase(pos_[frame_id]);
      curr_size_--;
    }

    is_active_.set(frame_id, false);
    evictable_.set(frame_id, true);
    hits_count_[frame_id] = 0;
  }
}

auto LRUKReplacer::Size() -> size_t { return curr_size_; }

}  // namespace bustub
