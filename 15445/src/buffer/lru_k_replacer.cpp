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

// const bool DEBUG = false;

namespace bustub {

LRUKReplacer::LRUKReplacer(size_t num_frames, size_t k) : replacer_size_(num_frames), k_(k) {
  // if (DEBUG) {
  //   std::cout << "\n\nLRUKReplacer lru_replacer(" << num_frames << ", " << k_ << ");\n";
  //   std::cout << "int value;\nlru_replacer.Evict(0)\n";
  // }
}

void LRUKReplacer::RecordAccess(frame_id_t frame_id) {
  std::scoped_lock<std::mutex> lock(latch_);
  // if (DEBUG) {
  //   std::cout << "lru_replacer.RecordAccess(" << frame_id << ");\n";
  // }
  // * You can also use BUSTUB_ASSERT to abort the process if frame id is invalid.
  BUSTUB_ASSERT(frame_id <= static_cast<frame_id_t>(replacer_size_), "Invalid frame_id");

  // frame_entries_[frame_id] có thể đã tồn tại từ trước
  // do SetEvictable(frame_id, ..) được gọi trước
  FrameEntry *frame_entry = &frame_entries_[frame_id];
  size_t updated_hits_count = ++frame_entry->hits_count_;

  //* - - - BEGIN verify code - - -
  // assert(updated_hits_count > 0);
  // assert(frame_entries_[frame_id].hits_count_ > 0);
  /* - - - END verify code - - - */

  if (updated_hits_count == 1) {  // first time
    history_list_.emplace_front(frame_id);
    frame_entry->pos_ = history_list_.begin();
    if (frame_entry->evictable_) {
      curr_size_++;
      curr_history_size_++;
    }

    //* - - - BEGIN verify code - - -
    // assert(frame_entries_[frame_id].hits_count_ == 1);
    // assert(frame_entries_[frame_id].evictable_);
    /* - - - END verify code - - - */

  } else if (updated_hits_count == k_) {
    // move from history_list_ to cache_list_
    history_list_.erase(frame_entry->pos_);
    if (frame_entry->evictable_) {
      curr_history_size_--;
    }
    // add to begin of the cache_list_
    cache_list_.emplace_front(frame_id);
    frame_entry->pos_ = cache_list_.begin();

    //* - - - BEGIN verify code - - -
    // assert(frame_entries_[frame_id].hits_count_ == k_);
    // assert(frame_entries_[frame_id].pos_ == cache_list_.begin());
    /* - - - END verify code - - - */
  } else if (updated_hits_count > k_) {
    if (frame_entry->pos_ != cache_list_.begin()) {
      // switch to begin of the cache_list_
      cache_list_.erase(frame_entry->pos_);
      cache_list_.emplace_front(frame_id);
      frame_entry->pos_ = cache_list_.begin();
    }

    //* - - - BEGIN verify code - - -
    // assert(frame_entries_[frame_id].hits_count_ > k_);
    // assert(frame_entries_[frame_id].pos_ == cache_list_.begin());
    /* - - - END verify code - - - */
  }

  //* - - - BEGIN verify code - - -
  // else {  // > 1 and < k_ => Do nothing, just verify code
  //   assert(frame_entries_[frame_id].hits_count_ > 1);
  //   assert(frame_entries_[frame_id].hits_count_ < k_);
  //   assert(frame_entries_[frame_id].pos_ == std::find(history_list_.begin(), history_list_.end(), frame_id));
  // }
  /* - - - END verify code - - - */
}

void LRUKReplacer::SetEvictable(frame_id_t frame_id, bool set_evictable) {
  std::scoped_lock<std::mutex> lock(latch_);
  // if (DEBUG) {
  //   std::cout << "lru_replacer.SetEvictable(" << frame_id << ", " << (set_evictable ? "true" : "false") << ");\n";
  // }
  BUSTUB_ASSERT(frame_id <= static_cast<frame_id_t>(replacer_size_), "Invalid frame_id");

  auto frame_it = frame_entries_.find(frame_id);
  bool found = (frame_it != frame_entries_.end());
  bool hit_before = (found && frame_it->second.hits_count_ > 0);
  auto previously_is_evictable = hit_before && frame_it->second.evictable_;
  auto currently_is_evictable = hit_before && set_evictable;
  frame_entries_[frame_id].evictable_ = set_evictable;

  if (!previously_is_evictable && currently_is_evictable) {
    curr_size_++;
    if (frame_it->second.hits_count_ < k_) {  // in history_list_
      curr_history_size_++;
    }
  } else if (previously_is_evictable && !currently_is_evictable) {
    curr_size_--;
    if (frame_it->second.hits_count_ < k_) {  // in history_list_
      curr_history_size_--;
    }
  }
}

auto LRUKReplacer::Evict(frame_id_t *frame_id) -> bool {
  std::scoped_lock<std::mutex> lock(latch_);
  // if (DEBUG) {
  //   std::cout << "lru_replacer.Evict(&value);\n";
  // }
  return EvictInternal(frame_id);
}
auto LRUKReplacer::EvictInternal(frame_id_t *frame_id) -> bool {
  //* - - - BEGIN verify code - - -
  // size_t count = 0;
  // for (auto frame_id : history_list_) {
  //   if (frame_entries_[frame_id].evictable_) {
  //     count++;
  //   }
  // }
  // LOG_INFO(">>> %d should equal %d", static_cast<int>(curr_history_size_), static_cast<int>(count));
  // assert(count == curr_history_size_);
  /* - - - END verify code - - - */

  if (curr_history_size_ > 0) {  // should evict from history_list_
    auto rit = history_list_.rbegin();
    while (!frame_entries_[*rit].evictable_) {
      rit++;
    }
    *frame_id = *rit;
    frame_entries_.erase(*frame_id);
    // https://stackoverflow.com/questions/1830158/how-to-call-erase-with-a-reverse-iterator
    history_list_.erase(std::next(rit).base());
    curr_history_size_--;
    curr_size_--;
    return true;
  }
  if (curr_size_ > 0) {
    auto rit = cache_list_.rbegin();
    while (!frame_entries_[*rit].evictable_) {
      rit++;
    }
    *frame_id = *rit;
    frame_entries_.erase(*frame_id);
    // https://stackoverflow.com/questions/1830158/how-to-call-erase-with-a-reverse-iterator
    cache_list_.erase(std::next(rit).base());
    curr_size_--;
    return true;
  }
  return false;
}

void LRUKReplacer::Remove(frame_id_t frame_id) {
  std::scoped_lock<std::mutex> lock(latch_);
  // if (DEBUG) {
  //   std::cout << "lru_replacer.Remove(" << frame_id << ");\n";
  // }
  if (frame_id <= static_cast<frame_id_t>(replacer_size_)) {  // validate frame_id
    auto it = frame_entries_.find(frame_id);
    if (it != frame_entries_.end()) {
      auto frame_entry = &it->second;
      BUSTUB_ASSERT(frame_entry->evictable_, "Can't remove an inevictable frame.");
      if (frame_entry->hits_count_ > 0 && frame_entry->hits_count_ < k_) {  // in history_list_
        history_list_.erase(frame_entry->pos_);
        curr_history_size_--;
      } else if (frame_entry->hits_count_ >= k_) {  // in cache_list_
        cache_list_.erase(frame_entry->pos_);
      }
      if (frame_entry->hits_count_ > 0) {
        curr_size_--;
      }
      frame_entries_.erase(frame_id);
    }
  }
}

auto LRUKReplacer::Size() -> size_t {
  // if (DEBUG) {
  //   std::cout << "lru_replacer.Size();\n";
  // }
  std::scoped_lock<std::mutex> lock(latch_);
  // đoạn code để verify curr_size_ được tăng giảm đúng ở các phần code khác,
  // comment out khi submit to gradescope
  //* - - - BEGIN verify code - - -
  // size_t count = 0;
  // for (auto const &frame_entry : frame_entries_) {
  //   if (frame_entry.second.evictable_ && frame_entry.second.hits_count_ > 0) {
  //     count++;
  //   }
  // }
  // // return count;
  // LOG_INFO(">>> %d should equal %d", static_cast<int>(curr_size_), static_cast<int>(count));
  // assert(curr_size_ == count);
  /* - - - END verify code - - - */
  return curr_size_;
}

}  // namespace bustub
