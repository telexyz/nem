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

#include <algorithm>
#include "buffer/lru_k_replacer.h"
#include "common/logger.h"

namespace bustub {

LRUKReplacer::LRUKReplacer(size_t num_frames, size_t k) : replacer_size_(num_frames), k_(k) {
  frame_entries_ = new FrameEntry[num_frames];
  history_queue_ = new Queue(2048 * num_frames);
  cache_queue_ = new Queue(2048 * num_frames);
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
    frame_entry->pos_ = history_queue_->Insert(frame_id);
    history_queue_->AdjustEvictFirst(frame_entry->pos_, true, frame_entries_);
    if (frame_entry->evictable_) {
      curr_size_++;
      curr_history_size_++;
    }

    assert(frame_entries_[frame_id].hits_count_ == 1);
    assert(frame_entries_[frame_id].evictable_);

  } else if (updated_hits_count == k_) {
    // move from history_queue_ to cache_queue_
    history_queue_->Evict(frame_entry->pos_, frame_entries_);
    if (frame_entry->evictable_) {
      curr_history_size_--;
    }
    // add to begin of the cache_queue_ (first time)
    frame_entry->pos_ = cache_queue_->Insert(frame_id);
    cache_queue_->AdjustEvictFirst(frame_entry->pos_, true, frame_entries_);

  } else if (updated_hits_count > k_) {
    if (frame_entry->pos_ != cache_queue_->end_ - 1) {
      // switch to begin of the cache_queue_
      cache_queue_->Evict(frame_entry->pos_, frame_entries_);
      frame_entry->pos_ = cache_queue_->Insert(frame_id);
    }
  }
}

void LRUKReplacer::SetEvictable(frame_id_t frame_id, bool set_evictable) {
  std::scoped_lock<std::mutex> lock(latch_);
  BUSTUB_ASSERT(frame_id < static_cast<frame_id_t>(replacer_size_), "Invalid frame_id");

  auto &frame = frame_entries_[frame_id];
  bool hit_before = (frame.is_active_ && frame.hits_count_ > 0);
  auto previously_is_evictable = hit_before && frame.evictable_;
  auto currently_is_evictable = hit_before && set_evictable;
  frame.evictable_ = set_evictable;

  if (!previously_is_evictable && currently_is_evictable) {
    curr_size_++;
    if (frame.hits_count_ < k_) {  // in history_queue_
      curr_history_size_++;
      history_queue_->AdjustEvictFirst(frame.pos_, true, frame_entries_);
    } else {  // in cache_queue_
      cache_queue_->AdjustEvictFirst(frame.pos_, true, frame_entries_);
    }
  } else if (previously_is_evictable && !currently_is_evictable) {
    curr_size_--;
    if (frame.hits_count_ < k_) {  // in history_queue_
      curr_history_size_--;
      history_queue_->AdjustEvictFirst(frame.pos_, false, frame_entries_);
    } else {
      cache_queue_->AdjustEvictFirst(frame.pos_, false, frame_entries_);
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
  auto &queue = evict_from_history ? history_queue_ : cache_queue_;
  size_t i = queue->begin_;
  while (queue->Get(i) == -1 || !frame_entries_[queue->Get(i)].evictable_) {
    i++;
  }

  *frame_id = queue->Get(queue->evict_first_);
  frame_entries_[*frame_id].is_active_ = false;
  frame_entries_[*frame_id].hits_count_ = 0;
  frame_entries_[*frame_id].evictable_ = true;

  queue->Evict(queue->evict_first_, frame_entries_);
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
    if (frame_entry->hits_count_ > 0 && frame_entry->hits_count_ < k_) {  // in history_queue_
      history_queue_->Evict(frame_entry->pos_, frame_entries_);
      curr_history_size_--;
      curr_size_--;

    } else if (frame_entry->hits_count_ >= k_) {  // in cache_queue_
      cache_queue_->Evict(frame_entry->pos_, frame_entries_);
      curr_size_--;
    }

    frame_entries_[frame_id].is_active_ = false;
    frame_entries_[frame_id].hits_count_ = 0;
    frame_entries_[frame_id].evictable_ = true;
  }
}

auto LRUKReplacer::Size() -> size_t { return curr_size_; }

}  // namespace bustub
