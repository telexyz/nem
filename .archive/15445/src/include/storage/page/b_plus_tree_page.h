//===----------------------------------------------------------------------===//
//
//                         CMU-DB Project (15-445/645)
//                         ***DO NO SHARE PUBLICLY***
//
// Identification: src/include/page/b_plus_tree_page.h
//
// Copyright (c) 2018, Carnegie Mellon University Database Group
//
//===----------------------------------------------------------------------===//

// B+Tree Parent Page
// This is the parent class that both the Internal Page and Leaf Page inherit from.
// The Parent Page only contains information that both child classes share.

#pragma once

#include <cassert>
#include <climits>
#include <cstdlib>
#include <string>

#include "buffer/buffer_pool_manager.h"
#include "storage/index/generic_key.h"

namespace bustub {

#define MappingType std::pair<KeyType, ValueType>

#define INDEX_TEMPLATE_ARGUMENTS template <typename KeyType, typename ValueType, typename KeyComparator>

// define page type enum
enum class IndexPageType { INVALID_INDEX_PAGE = 0, LEAF_PAGE, INTERNAL_PAGE };

/**
 * Both internal and leaf page are inherited from this page.
 *
 * It actually serves as a header part for each B+ tree page and
 * contains information shared by both leaf page and internal page.
 *
 * Header format (size in byte, 24 bytes in total):
 * ----------------------------------------------------------------------------
 * | PageType (4) | LSN (4) | CurrentSize (4) | MaxSize (4) |
 * ----------------------------------------------------------------------------
 * | ParentPageId (4) | PageId(4) |
 * ----------------------------------------------------------------------------
 */
class BPlusTreePage {
 public:

  inline auto IsLeafPage() const -> bool { return page_type_ == IndexPageType::LEAF_PAGE; }
  inline auto IsRootPage() const -> bool { return parent_page_id_ == INVALID_PAGE_ID; }
  inline void SetPageType(IndexPageType page_type) { page_type_ = page_type; }

  inline auto GetSize() const -> int { return size_; }
  inline void SetSize(int size) { size_ = size; }
  inline void IncreaseSize(int amount) { size_ += amount; }

  auto GetMaxSize() const -> int { return max_size_; }
  void SetMaxSize(int max_size) { max_size_ = max_size; }
/*
 * Helper method to get min page size
 * Generally, min page size == max page size / 2
 */
  auto GetMinSize() const -> int { return max_size_ / 2; }

  auto GetParentPageId() const -> page_id_t { return parent_page_id_; }
  void SetParentPageId(page_id_t parent_page_id) { parent_page_id_ = parent_page_id; }

  auto GetPageId() const -> page_id_t { return page_id_; }
  void SetPageId(page_id_t page_id) { page_id_ = page_id; }

  void SetLSN(lsn_t lsn = INVALID_LSN) { lsn_ = lsn; }

 private:
  // member variable, attributes that both internal and leaf page share
  IndexPageType page_type_ __attribute__((__unused__));
  lsn_t lsn_ __attribute__((__unused__));
  int size_ __attribute__((__unused__));
  int max_size_ __attribute__((__unused__));
  page_id_t parent_page_id_ __attribute__((__unused__));
  page_id_t page_id_ __attribute__((__unused__));
};

}  // namespace bustub
