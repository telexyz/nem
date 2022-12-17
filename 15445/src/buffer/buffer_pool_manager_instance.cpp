//===----------------------------------------------------------------------===//
//
//                         BusTub
//
// buffer_pool_manager_instance.cpp
//
// Identification: src/buffer/buffer_pool_manager.cpp
//
// Copyright (c) 2015-2021, Carnegie Mellon University Database Group
//
//===----------------------------------------------------------------------===//

#include "buffer/buffer_pool_manager_instance.h"
#include <iostream>
#include "common/logger.h"

#include "common/exception.h"
#include "common/macros.h"

namespace bustub {

BufferPoolManagerInstance::BufferPoolManagerInstance(size_t pool_size, DiskManager *disk_manager, size_t replacer_k,
                                                     LogManager *log_manager)
    : pool_size_(pool_size), disk_manager_(disk_manager), log_manager_(log_manager) {
  // we allocate a consecutive memory space for the buffer pool
  pages_ = new Page[pool_size_];
  page_table_ = new ExtendibleHashTable<page_id_t, frame_id_t>(bucket_size_);
  replacer_ = new LRUKReplacer(pool_size, replacer_k);

  // Initially, every page is in the free list.
  for (size_t i = 0; i < pool_size_; ++i) {
    free_list_.emplace_back(static_cast<int>(i));
  }

  // TODO(students): remove this line after you have implemented the buffer pool manager
  // throw NotImplementedException(
  //     "BufferPoolManager is not implemented yet. If you have finished implementing BPM, please remove the throw "
  //     "exception line in `buffer_pool_manager_instance.cpp`.");
}

BufferPoolManagerInstance::~BufferPoolManagerInstance() {
  delete[] pages_;
  delete page_table_;
  delete replacer_;
}

auto BufferPoolManagerInstance::PrepareFrame() -> frame_id_t {
  frame_id_t frame_id = -1;   // -1 = invalid frame_id
  if (!free_list_.empty()) {  // pick a new frame from free_list_ first
    frame_id = free_list_.front();
    free_list_.pop_front();

  } else if (replacer_->Size() > 0) {     // then use replacer_
    assert(replacer_->Evict(&frame_id));  // chắc chắn phải lấy lại được 1 frame
    // * If the replacement frame has a dirty page, you should write it back to the disk first.
    Page *evict_page = &pages_[frame_id];
    if (evict_page->IsDirty()) {
      page_id_t pid = evict_page->GetPageId();
      char *dat = evict_page->GetData();

      disk_manager_->WritePage(pid, dat);
      bool page_in_table = page_table_->Remove(pid);
      std::cout << "\n   *** PrepareFrame: dirty page " << pid << ", data `" << dat << "`\n";
      if (page_in_table) {
        std::cout << "   *** PrepareFrame: remove page_id " << pid << " from page_table_\n";
      }

      /* DEBUG begin */
      // ResetPage(evict_page);
      // std::cout << "   *** PrepareFrame: reset page " << pid << ", data `" << dat << "`\n";
      // disk_manager_->ReadPage(pid, dat);
      // std::cout << "   *** PrepareFrame: reread from disk page " << pid << ", data `" << dat << "`\n\n";
      /* DEBUG end */
    }
    // * You also need to reset the memory and metadata for the new page.
    ResetPage(evict_page);
  }
  return frame_id;
}

/**
 * @param[out] page_id id of created page
 * @return nullptr if no new pages could be created, otherwise pointer to new page
 */
auto BufferPoolManagerInstance::NewPgImp(page_id_t *page_id) -> Page * {
  std::scoped_lock<std::mutex> lock(latch_);

  // * Create a new page in the buffer pool. Set page_id to the new page's id, or nullptr if all frames
  // * are currently in use and not evictable (in another word, pinned).

  // * You should pick the replacement frame from either the free list or the replacer (always find from the free list
  // * first), and then call the AllocatePage() method to get a new page id.
  std::cout << ">>> NewPgImp: frame_id ";
  frame_id_t frame_id = PrepareFrame();
  std::cout << frame_id;

  if (frame_id == -1) {  // there is neither empty nor evictable frame
    std::cout << std::endl;
    page_id = nullptr;
    return nullptr;
  }

  assert(frame_id >= 0 && frame_id < static_cast<int>(pool_size_));  // ensure valid frame_id

  // * Remember to "Pin" the frame by calling replacer.SetEvictable(frame_id, false)
  // * so that the replacer wouldn't evict the frame before the buffer pool manager "Unpin"s it.
  // * Also, remember to record the access history of the frame in the replacer for the lru-k algorithm to work.
  replacer_->RecordAccess(frame_id);
  replacer_->SetEvictable(frame_id, false);

  *page_id = AllocatePage();
  page_table_->Insert(*page_id, frame_id);

  frame_id_t tmp;
  assert(page_table_->Find(*page_id, tmp));  // đảm bảo insert thành công

  std::cout << ", page_id " << *page_id << std::endl;
  pages_[frame_id].page_id_ = *page_id;
  pages_[frame_id].pin_count_ = 1;

  // if (frame_id > 0) { std::cout << "!!! pages_[0].data_ `" << pages_[0].data_ << "`\n"; }
  // showPages();  // DEBUG
  return &pages_[frame_id];
}

// Cấu trúc dữ liệu:
// `Page` là một đơn vị lưu trữ dữ liệu của database, được persistent trên disk và load vào memory qua `pages_`
// `pages_` mảng bộ nhớ được cấp phát cố định cho buffer pool, được chia ra thành `frames`
// `page_table_` là 1 ExtendibleHash để trỏ page_id (Page) tới frame_id (stt của mảng pages_)
// `replacer_` LRU-K replacer để quản lý xem frame nào có thể evict
// `free_list_` chứa các frames chưa được dùng tới (luôn cấp phát frames từ đây trước)

/**
 * @brief Fetch the requested page from the buffer pool. Return nullptr if page_id needs to be fetched from the disk
 * but all frames are currently in use and not evictable (in another word, pinned).
 * In addition, remember to disable eviction and record the access history of the frame like you did for NewPgImp().
 *
 * @param page_id id of page to be fetched
 * @return nullptr if page_id cannot be fetched, otherwise pointer to the requested page
 */
auto BufferPoolManagerInstance::FetchPgImp(page_id_t page_id) -> Page * {
  std::scoped_lock<std::mutex> lock(latch_);
  std::cout << ">>> FetchPgImp: page_id " << page_id << " => frame_id ";

  frame_id_t frame_id = -1;
  if (page_table_->Find(page_id, frame_id)) {  // in memory page
    assert(frame_id >= 0 && frame_id < static_cast<int>(pool_size_));
    std::cout << frame_id << ", in-memory data `" << pages_[frame_id].data_ << "`\n";
    return &pages_[frame_id];
  }

  frame_id = PrepareFrame();
  std::cout << frame_id;

  if (frame_id == -1) {
    std::cout << ", no frame to load\n";
    return nullptr;
  }

  assert(frame_id >= 0 && frame_id < static_cast<int>(pool_size_));

  replacer_->RecordAccess(frame_id);
  replacer_->SetEvictable(frame_id, false);

  // Load page từ disk vào frame
  disk_manager_->ReadPage(page_id, pages_[frame_id].data_);
  page_table_->Insert(page_id, frame_id);  // map page_id to frame_id

  frame_id_t tmp;
  assert(page_table_->Find(page_id, tmp));  // đảm bảo insert thành công

  std::cout << ", load from disk data `" << pages_[frame_id].data_ << "`\n";
  pages_[frame_id].pin_count_ = 1;
  pages_[frame_id].page_id_ = page_id;
  return &pages_[frame_id];
}

/**
 * @brief Unpin the target page from the buffer pool. If page_id is not in the buffer pool or its pin count is already
 * 0, return false.
 *
 * Decrement the pin count of a page. If the pin count reaches 0, the frame should be evictable by the replacer.
 * Also, set the dirty flag on the page to indicate if the page was modified.
 *
 * @param page_id id of page to be unpinned
 * @param is_dirty true if the page should be marked as dirty, false otherwise
 * @return false if the page is not in the page table or its pin count is <= 0 before this call, true otherwise
 */
auto BufferPoolManagerInstance::UnpinPgImp(page_id_t page_id, bool is_dirty) -> bool {
  std::scoped_lock<std::mutex> lock(latch_);
  std::cout << ">>> UnpinPgImp(page_id " << page_id << ", is_dirty " << is_dirty << "):";

  frame_id_t frame_id = -1;
  if (page_table_->Find(page_id, frame_id)) {  // page in memory
    assert(frame_id >= 0 && frame_id < static_cast<int>(pool_size_));
    Page *unpin_page = &pages_[frame_id];
    std::cout << " frame_id " << frame_id << ", pin_count " << unpin_page->pin_count_;

    if (unpin_page->pin_count_ <= 0) {
      std::cout << " => false: pin_count is 0\n";
      return false;
    }

    int decreased_pin_count = --unpin_page->pin_count_;
    std::cout << ", decreased_pin_count " << decreased_pin_count;

    if (decreased_pin_count <= 0) {
      replacer_->SetEvictable(frame_id, true);
    }

    std::cout << " => true: page in memory\n";
    unpin_page->is_dirty_ = is_dirty;
    return true;
  }  // end page in memory
  std::cout << " => false: page in disk\n";
  return false;
}

auto BufferPoolManagerInstance::FlushPgImp(page_id_t page_id) -> bool { return false; }

void BufferPoolManagerInstance::FlushAllPgsImp() {}

auto BufferPoolManagerInstance::DeletePgImp(page_id_t page_id) -> bool { return false; }

auto BufferPoolManagerInstance::AllocatePage() -> page_id_t { return next_page_id_++; }

}  // namespace bustub
