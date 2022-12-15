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

namespace bustub {

LRUKReplacer::LRUKReplacer(size_t num_frames, size_t k) : replacer_size_(num_frames), k_(k) {}

auto LRUKReplacer::Evict(frame_id_t *frame_id) -> bool { return false; }

void LRUKReplacer::RecordAccess(frame_id_t frame_id) {}

void LRUKReplacer::SetEvictable(frame_id_t frame_id, bool set_evictable) {
  auto frame_it = frame_entries_.find(frame_id);
  bool found = (frame_it != frame_entries_.end());
  auto previously_is_evictable = found && frame_it->second.evictable_;
  auto currently_is_evictable = set_evictable;
  frame_entries_[frame_id].evictable_ = set_evictable;

  if (!previously_is_evictable && currently_is_evictable) {
    curr_size_++;
  } else if (previously_is_evictable && !currently_is_evictable) {
    curr_size_--;
  }
}

void LRUKReplacer::Remove(frame_id_t frame_id) {}

auto LRUKReplacer::Size() -> size_t {
  // đoạn code để verify curr_size_ được tăng giảm đúng ở các phần code khác,
  // comment out khi submit to gradescope
  //* BEGIN verify code
  size_t count = 0;
  for (auto const &frame_entry : frame_entries_) {
    if (frame_entry.second.evictable_) {
      count++;
    }
  }
  assert(count == curr_size_);
  /* END verify code */
  return curr_size_;
}

}  // namespace bustub
