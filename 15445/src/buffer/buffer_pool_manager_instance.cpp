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

// const bool LOG_INPUT = true;
// const bool LOG_DEBUG = false;

const bool LOG_INPUT = false;
const bool LOG_DEBUG = true;

// const bool LOG_INPUT = false;
// const bool LOG_DEBUG = false;

namespace bustub {

BufferPoolManagerInstance::BufferPoolManagerInstance(size_t pool_size, DiskManager *disk_manager, size_t replacer_k,
                                                     LogManager *log_manager)
    : pool_size_(pool_size), disk_manager_(disk_manager), log_manager_(log_manager) {
  if (LOG_INPUT) {
    std::cout << "\n\n  auto *bpm = new BufferPoolManagerInstance(" << pool_size << ", disk_manager, " << replacer_k
              << ");\n";
  }

  // we allocate a consecutive memory space for the buffer pool
  pages_ = new Page[pool_size_];
  page_table_ = new ExtendibleHashTable<page_id_t, frame_id_t>(bucket_size_);
  replacer_ = new LRUKReplacer(pool_size, replacer_k);

  // Initially, every page is in the free list.
  for (size_t i = 0; i < pool_size_; ++i) {
    free_list_.emplace_back(static_cast<int>(i));
  }
}

BufferPoolManagerInstance::~BufferPoolManagerInstance() {
  delete[] pages_;
  delete page_table_;
  delete replacer_;
}

// helper
auto BufferPoolManagerInstance::PrepareFrame() -> frame_id_t {
  if (LOG_DEBUG) {
    std::cout << "\n*** PrepareFrame: evictable " << replacer_->Size() << ", free " << free_list_.size();
  }
  frame_id_t frame_id = -1;   // -1 = invalid frame_id
  if (!free_list_.empty()) {  // pick a new frame from free_list_ first
    frame_id = free_list_.front();
    free_list_.pop_front();

  } else if (replacer_->Size() > 0) {     // then use replacer_
    assert(replacer_->Evict(&frame_id));  // chắc chắn phải lấy lại được 1 frame
    // * If the replacement frame has a dirty page, you should write it back to the disk first.
    Page *evict_page = &pages_[frame_id];
    page_id_t pid = evict_page->GetPageId();

    bool page_in_table = page_table_->Remove(pid);
    if (page_in_table && LOG_DEBUG) {
      std::cout << "\n    - remove page_id " << pid << " from page_table_";
    }

    if (evict_page->IsDirty()) {
      char *dat = evict_page->GetData();

      disk_manager_->WritePage(pid, dat);
      if (LOG_DEBUG) {
        std::cout << "\n    - write to disk dirty page " << pid << ", data `" << dat;
      }

      /* DEBUG begin: đảm bảo việc ghi data vào disk và đọc trở lại thành công */
      // ResetPage(evict_page);
      // std::cout << "*** PrepareFrame: reset page " << pid << ", data `" << dat << "`\n";
      // disk_manager_->ReadPage(pid, dat);
      // std::cout << "*** PrepareFrame: reread from disk page " << pid << ", data `" << dat << "`\n\n";
      /* DEBUG end */
    }
    // * You also need to reset the memory and metadata for the new page.
    ResetPage(evict_page);
  }
  if (LOG_DEBUG) {
    std::cout << "`\n";
  }
  return frame_id;
}

/**
 * @param[out] page_id id of created page
 * @return nullptr if no new pages could be created, otherwise pointer to new page
 */
auto BufferPoolManagerInstance::NewPgImp(page_id_t *page_id) -> Page * {
  std::scoped_lock<std::mutex> lock(latch_);

  if (LOG_INPUT) {
    std::cout << "  bpm->NewPage(&page_id_temp);\n";
  }

  // * Create a new page in the buffer pool. Set page_id to the new page's id, or nullptr if all frames
  // * are currently in use and not evictable (in another word, pinned).

  // * You should pick the replacement frame from either the free list or the replacer (always find from the free list
  // * first), and then call the AllocatePage() method to get a new page id.
  frame_id_t frame_id = PrepareFrame();
  if (LOG_DEBUG) {
    std::cout << ">>> NewPgImp: frame_id ";
    std::cout << frame_id;
  }

  if (frame_id == -1) {  // there is neither empty nor evictable frame
    if (LOG_DEBUG) {
      std::cout << std::endl;
    }
    page_id = nullptr;
    return nullptr;
  }

  assert(frame_id >= 0 && frame_id < static_cast<int>(pool_size_));  // ensure valid frame_id

  *page_id = AllocatePage();
  pages_[frame_id].page_id_ = *page_id;
  page_table_->Insert(*page_id, frame_id);
  // frame_id_t tmp;
  // assert(page_table_->Find(*page_id, tmp));  // đảm bảo insert thành công
  if (LOG_DEBUG) {
    snprintf(pages_[frame_id].data_, BUSTUB_PAGE_SIZE, "page%d", *page_id);
    std::cout << ", page_id " << *page_id << std::endl;
  }

  // * Remember to "Pin" the frame by calling replacer.SetEvictable(frame_id, false)
  // * so that the replacer wouldn't evict the frame before the buffer pool manager "Unpin"s it.
  // * Also, remember to record the access history of the frame in the replacer for the lru-k algorithm to work.
  PinFrame(frame_id);
  return &pages_[frame_id];
}

// - - - - - - - - -
// Cấu trúc dữ liệu:
// - - - - - - - - -
// `Page` là một đơn vị lưu trữ dữ liệu của database, được persistent trên disk và load vào memory qua `pages_`
// `pages_` mảng bộ nhớ được cấp phát cố định cho buffer pool, được chia ra thành `frames`
// `page_table_` là 1 ExtendibleHash để trỏ page_id (Page) tới frame_id (stt của mảng pages_)
// `replacer_` LRU-K replacer để quản lý xem frame nào có thể evict
// `free_list_` chứa các frames chưa được dùng tới (luôn cấp phát frames từ đây trước)
// - - -
// Chú ý:
// - - -
// 2 thao tác cung cấp page cho layer cao hơn của DB là NewPgImp() và FetchPgImp()
// cần tăng page.pin_count_ lên 1
// - - - - - - - - -

/**
 * @brief Fetch the requested page from the buffer pool. Return nullptr if page_id needs to be fetched from the disk
 * but all frames are currently in use and not evictable (in another word, pinned).
 *
 * First search for page_id in the buffer pool. If not found, pick a replacement frame from either the free list or
 * the replacer (always find from the free list first), read the page from disk by calling disk_manager_->ReadPage(),
 * and replace the old page in the frame. Similar to NewPgImp(), if the old page is dirty, you need to write it back
 * to disk and update the metadata of the new page
 *
 * In addition, remember to disable eviction and record the access history of the frame like you did for NewPgImp().
 *
 * @param page_id id of page to be fetched
 * @return nullptr if page_id cannot be fetched, otherwise pointer to the requested page
 */
auto BufferPoolManagerInstance::FetchPgImp(page_id_t page_id) -> Page * {
  std::scoped_lock<std::mutex> lock(latch_);
  if (LOG_INPUT) {
    std::cout << "  bpm->FetchPage(" << page_id << ");\n";
  }

  frame_id_t frame_id = -1;
  if (page_table_->Find(page_id, frame_id)) {  // in memory page
    assert(frame_id >= 0 && frame_id < static_cast<int>(pool_size_));
    if (LOG_DEBUG) {
      std::cout << ">>> FetchPgImp: page_id " << page_id << " => frame_id ";
      std::cout << frame_id << ", in-memory data `" << pages_[frame_id].data_ << "`\n";
    }

  } else {
    frame_id = PrepareFrame();
    if (LOG_DEBUG) {
      std::cout << ">>> FetchPgImp: page_id " << page_id << " => frame_id " << frame_id;
    }

    if (frame_id == -1) {
      if (LOG_DEBUG) {
        std::cout << ", no frame to load\n";
      }
      return nullptr;
    }

    assert(frame_id >= 0 && frame_id < static_cast<int>(pool_size_));

    // Load page từ disk vào frame
    disk_manager_->ReadPage(page_id, pages_[frame_id].data_);
    pages_[frame_id].page_id_ = page_id;
    page_table_->Insert(page_id, frame_id);  // map page_id to frame_id

    if (LOG_DEBUG) {
      std::cout << ", load from disk data `" << pages_[frame_id].data_ << "`\n";
    }
  }

  PinFrame(frame_id);
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
  if (LOG_INPUT) {
    std::cout << "  bpm->UnpinPage(" << page_id << ", " << (is_dirty ? "true" : "false") << ");\n";
  }
  std::scoped_lock<std::mutex> lock(latch_);
  if (LOG_DEBUG) {
    std::cout << ">>> UnpinPgImp(page_id " << page_id << ", is_dirty " << is_dirty << "):";
  }

  frame_id_t frame_id = -1;
  if (page_table_->Find(page_id, frame_id)) {  // page in memory
    assert(frame_id >= 0 && frame_id < static_cast<int>(pool_size_));
    Page *unpin_page = &pages_[frame_id];
    if (LOG_DEBUG) {
      std::cout << " frame_id " << frame_id << ", pin_count " << unpin_page->pin_count_;
    }

    if (unpin_page->pin_count_ <= 0) {
      if (LOG_DEBUG) {
        std::cout << " => false: pin_count is 0\n";
      }
      return false;
    }

    int decreased_pin_count = --unpin_page->pin_count_;
    if (LOG_DEBUG) {
      std::cout << ", decreased_pin_count " << decreased_pin_count;
    }

    if (decreased_pin_count <= 0) {
      replacer_->SetEvictable(frame_id, true);
    }

    if (LOG_DEBUG) {
      std::cout << " => true: page in memory\n";
    }

    if (is_dirty) {
      unpin_page->is_dirty_ = true;
    }
    return true;
  }  // end page in memory

  if (LOG_DEBUG) {
    std::cout << " => false: page in disk\n";
  }
  return false;
}

/**
 * @brief Flush the target page to disk.
 *
 * Use the DiskManager::WritePage() method to flush a page to disk, REGARDLESS of the dirty flag.
 * Unset the dirty flag of the page after flushing.
 *
 * @param page_id id of page to be flushed, cannot be INVALID_PAGE_ID
 * @return false if the page could not be found in the page table, true otherwise
 */
auto BufferPoolManagerInstance::FlushPgImp(page_id_t page_id) -> bool {
  if (LOG_INPUT) {
    std::cout << "  bpm->FlushPage(" << page_id << ");\n";
  }
  std::scoped_lock<std::mutex> lock(latch_);
  frame_id_t frame_id = -1;
  if (page_table_->Find(page_id, frame_id)) {
    assert(page_id != INVALID_PAGE_ID);
    disk_manager_->WritePage(page_id, pages_[frame_id].data_);
    pages_[frame_id].is_dirty_ = false;  // unset dirty flag
  }
  return false;
}

/**
 * @brief Flush all the pages in the buffer pool to disk.
 */
void BufferPoolManagerInstance::FlushAllPgsImp() {
  if (LOG_INPUT) {
    std::cout << "  bpm->FlushAllPages();\n";
  }

  std::scoped_lock<std::mutex> lock(latch_);
  for (size_t i = 0; i < pool_size_; i++) {
    Page *page = &pages_[i];
    if (page->page_id_ != INVALID_PAGE_ID) {
      disk_manager_->WritePage(page->page_id_, page->data_);
      page->is_dirty_ = false;  // unset dirty flag
    }
  }
}

/**
 * @brief Delete a page from the buffer pool. If page_id is not in the buffer pool, do nothing and return true. If the
 * page is pinned and cannot be deleted, return false immediately.
 *
 * After deleting the page from the page table, stop tracking the frame in the replacer and add the frame
 * back to the free list. Also, reset the page's memory and metadata. Finally, you should call DeallocatePage() to
 * imitate freeing the page on the disk.
 *
 * @param page_id id of page to be deleted
 * @return false if the page exists but could not be deleted, true if the page didn't exist or deletion succeeded
 */
auto BufferPoolManagerInstance::DeletePgImp(page_id_t page_id) -> bool {
  if (LOG_INPUT) {
    std::cout << "  bpm->DeletePage(" << page_id << ");\n";
  }
  std::scoped_lock<std::mutex> lock(latch_);
  frame_id_t frame_id = -1;
  if (page_table_->Find(page_id, frame_id)) {
    assert(page_id != INVALID_PAGE_ID);
    if (pages_[frame_id].pin_count_ > 0) {  // pinned
      return false;
    }
    page_table_->Remove(page_id);
    DeallocatePage(page_id);
    free_list_.emplace_back(frame_id);
    ResetPage(&pages_[frame_id]);
  }
  return true;  // if page_id not in memory
}

auto BufferPoolManagerInstance::AllocatePage() -> page_id_t { return next_page_id_++; }

}  // namespace bustub
