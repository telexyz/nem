//===----------------------------------------------------------------------===//
//
//                         BusTub
//
// lru_k_replacer.h
//
// Identification: src/include/buffer/lru_k_replacer.h
//
// Copyright (c) 2015-2022, Carnegie Mellon University Database Group
//
//===----------------------------------------------------------------------===//

#pragma once

#include <iostream>
#include <limits>
#include <mutex>  // NOLINT

#include "common/config.h"
#include "common/macros.h"

namespace bustub {

/**
 * LRUKReplacer implements the LRU-k replacement policy.
 *
 * The LRU-k algorithm evicts a frame whose backward k-distance is maximum
 * of all frames. Backward k-distance is computed as the difference in time between
 * current timestamp and the timestamp of kth previous access.
 *
 * A frame with less than k historical references is given
 * +inf as its backward k-distance. When multiple frames have +inf backward k-distance,
 * classical LRU algorithm is used to choose victim.
 */
class LRUKReplacer {
 public:
  /**
   * @brief a new LRUKReplacer.
   * @param num_frames the maximum number of frames the LRUReplacer will be required to store
   */
  explicit LRUKReplacer(size_t num_frames, size_t k);

  DISALLOW_COPY_AND_MOVE(LRUKReplacer);

  /**
   * @brief Destroys the LRUReplacer.
   */
  ~LRUKReplacer() {
    delete[] frame_entries_;
    delete history_queue_;
    delete cache_queue_;
  }

  /**
   * @brief Record the event that the given frame id is accessed at current timestamp.
   * Create a new entry for access history if frame id has not been seen before.
   *
   * If frame id is invalid (ie. larger than replacer_size_), throw an exception. You can
   * also use BUSTUB_ASSERT to abort the process if frame id is invalid.
   *
   * @param frame_id id of frame that received a new access.
   */
  void RecordAccess(frame_id_t frame_id);

  /**
   * @brief Toggle whether a frame is evictable or non-evictable. This function also
   * controls replacer's size. Note that size is equal to number of evictable entries.
   *
   * If a frame was previously evictable and is to be set to non-evictable, then size should
   * decrement. If a frame was previously non-evictable and is to be set to evictable,
   * then size should increment.
   *
   * If frame id is invalid, throw an exception or abort the process.
   *
   * For other scenarios, this function should terminate without modifying anything.
   *
   * @param frame_id id of frame whose 'evictable' status will be modified
   * @param set_evictable whether the given frame is evictable or not
   */
  void SetEvictable(frame_id_t frame_id, bool set_evictable);

  /**
   * @brief Find the frame with largest backward k-distance and evict that frame. Only frames
   * that are marked as 'evictable' are candidates for eviction.
   *
   * A frame with less than k historical references is given +inf as its backward k-distance.
   * If multiple frames have inf backward k-distance, then evict the frame with the earliest
   * timestamp overall.
   *
   * Successful eviction of a frame should decrement the size of replacer and remove the frame's
   * access history.
   *
   * @param[out] frame_id id of frame that is evicted.
   * @return true if a frame is evicted successfully, false if no frames can be evicted.
   */
  auto Evict(frame_id_t *frame_id) -> bool;

  /**
   * @brief Remove an evictable frame from replacer, along with its access history.
   * This function should also decrement replacer's size if removal is successful.
   *
   * Note that this is different from evicting a frame, which always remove the frame
   * with largest backward k-distance. This function removes specified frame id,
   * no matter what its backward k-distance is.
   *
   * If Remove is called on a non-evictable frame, throw an exception or abort the
   * process.
   *
   * If specified frame is not found, directly return from this function.
   *
   * @param frame_id id of frame to be removed
   */
  void Remove(frame_id_t frame_id);

  /**
   * @brief Return replacer's size, which tracks the number of evictable frames.
   *
   * @return size_t
   */
  auto Size() -> size_t;

  struct FrameEntry {
    size_t hits_count_{0};
    bool evictable_{true};
    bool is_active_{false};
    size_t pos_;
  };

  class Queue {
   public:
    explicit Queue(size_t size) : size_(size) {
      queue_ = new frame_id_t[size];
      assert(begin_ == 0 && end_ == 0);
    }

    ~Queue() { delete[] queue_; }

    inline auto Insert(frame_id_t frame_id) -> size_t {
      assert(end_ >= 0 && end_ < size_);
      queue_[end_] = frame_id;
      return end_++;
    }

    inline auto Evict(size_t pos, FrameEntry *frame_entries_) -> void {
      assert(pos >= 0 && pos < size_);
      queue_[pos] = -1;
      AdjustEvictFirst(pos, false, frame_entries_);
    }

    inline auto Get(size_t pos) -> frame_id_t { return queue_[pos]; }

    auto AdjustEvictFirst(size_t pos, bool evictable, FrameEntry *frame_entries_) -> void {
      // std::cout << "(( pos " << pos << ", evictable " << evictable << ")) ";
      // std::cout << "(( queue " << this << ")) ";
      // for (size_t i = begin_; i < end_; i++) {
      //   std::cout << i << ":" << queue_[i] << ":" << (queue_[i] >= 0 && frame_entries_[queue_[i]].evictable_ ? "true"
      //   : "false") << ", ";
      // }
      // std::cout << "\n";

      if (evictable && pos < evict_first_) {
        evict_first_ = pos;
        // std::cout << ">>> " << this << ": evict_first_ " << pos << ", evictable " << evictable << std::endl;
      }
      if (!evictable && pos == evict_first_) {
        while (evict_first_ < end_) {
          auto frame_id = queue_[evict_first_];
          // std::cout << "!!! evict_first_ " << evict_first_ << ", end_ " << end_;
          // std::cout << ", frame_id " << frame_id << ", evictable_ ";
          // std::cout << (frame_id >= 0 && frame_entries_[frame_id].evictable_) << std::endl;
          if (frame_id >= 0 && frame_entries_[frame_id].evictable_) {
            break;
          }
          evict_first_++;
        }
        // std::cout << ">>> " << this << ": evict_first_ " << pos << ", evictable " << evictable << std::endl;
      }
    }

    size_t size_;
    size_t begin_{0};
    size_t end_{0};
    size_t evict_first_{999999999};
    frame_id_t *queue_;
  };

 private:
  auto EvictInternal(frame_id_t *frame_id) -> bool;
  // std::unordered_map<frame_id_t, FrameEntry> frame_entries_; // thay bằng frame_entries_ array
  FrameEntry *frame_entries_;  // frame_id chính là index
  Queue *history_queue_;
  Queue *cache_queue_;

  size_t curr_history_size_{0};
  size_t curr_size_{0};
  size_t replacer_size_;  // = num_frames, số lượng tối đa frames mà buffer pool chứa được
  size_t k_;

  std::mutex latch_;
};

}  // namespace bustub
