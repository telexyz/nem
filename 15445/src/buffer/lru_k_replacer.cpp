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
  frame_entries_ = new FrameEntry[num_frames];
}

void LRUKReplacer::RecordAccess(frame_id_t frame_id) {
  std::scoped_lock<std::mutex> lock(latch_);
  // * You can also use BUSTUB_ASSERT to abort the process if frame id is invalid.
  BUSTUB_ASSERT(frame_id < static_cast<frame_id_t>(replacer_size_), "Invalid frame_id");

  FrameEntry *frame_entry = &frame_entries_[frame_id];
  frame_entry->is_active_ = true;
  size_t updated_hits_count = ++frame_entry->hits_count_;

  assert(updated_hits_count > 0);
  assert(frame_entries_[frame_id].hits_count_ > 0);

  if (updated_hits_count == 1) {  // first time
    history_list_.emplace_front(frame_id);
    frame_entry->pos_ = history_list_.begin();
    if (frame_entry->evictable_) {
      curr_size_++;
      curr_history_size_++;
    }

    // std::cout << "h+" << history_list_.size() << " ";
    assert(frame_entries_[frame_id].hits_count_ == 1);
    assert(frame_entries_[frame_id].evictable_);

  } else if (updated_hits_count == k_) {
    // move from history_list_ to cache_list_
    history_list_.erase(frame_entry->pos_);
    if (frame_entry->evictable_) {
      curr_history_size_--;
    }

    // add to begin of the cache_list_
    cache_list_.emplace_front(frame_id);
    frame_entry->pos_ = cache_list_.begin();

    // std::cout << "c+" << curr_size_ - cache_list_.size() << " ";
    assert(frame_entries_[frame_id].hits_count_ == k_);
    assert(frame_entries_[frame_id].pos_ == cache_list_.begin());

  } else if (updated_hits_count > k_) {
    if (frame_entry->pos_ != cache_list_.begin()) {
      // switch to begin of the cache_list_
      cache_list_.erase(frame_entry->pos_);
      cache_list_.emplace_front(frame_id);
      frame_entry->pos_ = cache_list_.begin();
    }

    assert(frame_entries_[frame_id].hits_count_ > k_);
    assert(frame_entries_[frame_id].pos_ == cache_list_.begin());

  } else {  // > 1 and < k_ => Do nothing, just verify code
    assert(frame_entries_[frame_id].hits_count_ > 1);
    assert(frame_entries_[frame_id].hits_count_ < k_);
    assert(frame_entries_[frame_id].pos_ == std::find(history_list_.begin(), history_list_.end(), frame_id));
  }
}

void LRUKReplacer::SetEvictable(frame_id_t frame_id, bool set_evictable) {
  std::scoped_lock<std::mutex> lock(latch_);
  BUSTUB_ASSERT(frame_id < static_cast<frame_id_t>(replacer_size_), "Invalid frame_id");

  auto frame = frame_entries_[frame_id];
  bool hit_before = (frame.is_active_ && frame.hits_count_ > 0);
  auto previously_is_evictable = hit_before && frame.evictable_;
  auto currently_is_evictable = hit_before && set_evictable;
  frame_entries_[frame_id].evictable_ = set_evictable;

  if (!previously_is_evictable && currently_is_evictable) {
    curr_size_++;
    if (frame.hits_count_ < k_) {  // in history_list_
      curr_history_size_++;
    }
  } else if (previously_is_evictable && !currently_is_evictable) {
    curr_size_--;
    if (frame.hits_count_ < k_) {  // in history_list_
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
  while (!frame_entries_[*rit].evictable_) {
    rit++;
  }

  *frame_id = *rit;
  frame_entries_[*frame_id].is_active_ = false;
  frame_entries_[*frame_id].hits_count_ = 0;
  frame_entries_[*frame_id].evictable_ = true;

  // https://stackoverflow.com/questions/1830158/how-to-call-erase-with-a-reverse-iterator
  list.erase(std::next(rit).base());
  curr_history_size_ -= evict_from_history;  // NOLINT
  curr_size_--;
  return true;
}

void LRUKReplacer::Remove(frame_id_t frame_id) {
  std::scoped_lock<std::mutex> lock(latch_);

  assert(frame_id < static_cast<frame_id_t>(replacer_size_));
  auto frame_entry = &frame_entries_[frame_id];
  if (frame_entry->is_active_) {
    BUSTUB_ASSERT(frame_entry->evictable_, "Can't remove an inevictable frame.");
    if (frame_entry->hits_count_ > 0 && frame_entry->hits_count_ < k_) {  // in history_list_
      history_list_.erase(frame_entry->pos_);
      curr_history_size_--;
      curr_size_--;

    } else if (frame_entry->hits_count_ >= k_) {  // in cache_list_
      cache_list_.erase(frame_entry->pos_);
      curr_size_--;
    }

    frame_entries_[frame_id].is_active_ = false;
    frame_entries_[frame_id].hits_count_ = 0;
    frame_entries_[frame_id].evictable_ = true;
  }
}

auto LRUKReplacer::Size() -> size_t { return curr_size_; }

}  // namespace bustub
