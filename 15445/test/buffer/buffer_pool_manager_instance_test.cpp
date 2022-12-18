//===----------------------------------------------------------------------===//
//
//                         BusTub
//
// buffer_pool_manager_instance_test.cpp
//
// Identification: test/buffer/buffer_pool_manager_test.cpp
//
// Copyright (c) 2015-2021, Carnegie Mellon University Database Group
//
//===----------------------------------------------------------------------===//

#include "buffer/buffer_pool_manager_instance.h"

#include <cstdio>
#include <random>
#include <string>

#include "buffer/buffer_pool_manager.h"
#include "gtest/gtest.h"

namespace bustub {

TEST(BufferPoolManagerInstanceTest, BuildReleaseTest) {
  const std::string db_name = "test.db";
  const size_t buffer_pool_size = 3;
  const size_t k = 5;
  page_id_t page_id_temp;

  auto *disk_manager = new DiskManager(db_name);
  auto *bpm = new BufferPoolManagerInstance(buffer_pool_size, disk_manager, k);

  // Scenario: We should be able to create new pages until we fill up the buffer pool.
  for (size_t i = 0; i < buffer_pool_size; ++i) {
    EXPECT_NE(nullptr, bpm->NewPage(&page_id_temp));
  }

  // Scenario: Once the buffer pool is full, we should not be able to create any new pages.
  EXPECT_EQ(nullptr, bpm->NewPage(&page_id_temp));

  // Scenario: After unpinning pages {0, 1} we should be able to create 2 new pages
  for (int i = 0; i < 2; ++i) {
    EXPECT_EQ(true, bpm->UnpinPage(i, true));
    bpm->FlushPage(i);
  }
  for (int i = 0; i < 2; ++i) {
    EXPECT_NE(nullptr, bpm->NewPage(&page_id_temp));
    bpm->UnpinPage(page_id_temp, false);
  }
}

// NOLINTNEXTLINE
// Check whether pages containing terminal characters can be recovered
TEST(BufferPoolManagerInstanceTest, BinaryDataTest) {
  const std::string db_name = "test.db";
  const size_t buffer_pool_size = 10;
  const size_t k = 5;

  std::random_device r;
  std::default_random_engine rng(r());
  std::uniform_int_distribution<char> uniform_dist(0);

  auto *disk_manager = new DiskManager(db_name);
  auto *bpm = new BufferPoolManagerInstance(buffer_pool_size, disk_manager, k);

  page_id_t page_id_temp;
  auto *page0 = bpm->NewPage(&page_id_temp);

  // Scenario: The buffer pool is empty. We should be able to create a new page.
  ASSERT_NE(nullptr, page0);
  EXPECT_EQ(0, page_id_temp);

  char random_binary_data[BUSTUB_PAGE_SIZE];
  // Generate random binary data
  for (char &i : random_binary_data) {
    i = uniform_dist(rng);
  }

  // Insert terminal characters both in the middle and at end
  random_binary_data[BUSTUB_PAGE_SIZE / 2] = '\0';
  random_binary_data[BUSTUB_PAGE_SIZE - 1] = '\0';

  // Scenario: Once we have a page, we should be able to read and write content.
  std::memcpy(page0->GetData(), random_binary_data, BUSTUB_PAGE_SIZE);
  EXPECT_EQ(0, std::memcmp(page0->GetData(), random_binary_data, BUSTUB_PAGE_SIZE));

  // Scenario: We should be able to create new pages until we fill up the buffer pool.
  for (size_t i = 1; i < buffer_pool_size; ++i) {
    EXPECT_NE(nullptr, bpm->NewPage(&page_id_temp));
  }

  // Scenario: Once the buffer pool is full, we should not be able to create any new pages.
  for (size_t i = buffer_pool_size; i < buffer_pool_size * 2; ++i) {
    EXPECT_EQ(nullptr, bpm->NewPage(&page_id_temp));
  }

  // Scenario: After unpinning pages {0, 1, 2, 3, 4} we should be able to create 5 new pages
  for (int i = 0; i < 5; ++i) {
    EXPECT_EQ(true, bpm->UnpinPage(i, true));
    bpm->FlushPage(i);
  }
  for (int i = 0; i < 5; ++i) {
    EXPECT_NE(nullptr, bpm->NewPage(&page_id_temp));
    bpm->UnpinPage(page_id_temp, false);
  }
  // Scenario: We should be able to fetch the data we wrote a while ago.
  page0 = bpm->FetchPage(0);
  EXPECT_EQ(0, memcmp(page0->GetData(), random_binary_data, BUSTUB_PAGE_SIZE));
  EXPECT_EQ(true, bpm->UnpinPage(0, true));

  // Shutdown the disk manager and remove the temporary file we created.
  disk_manager->ShutDown();
  remove("test.db");

  delete bpm;
  delete disk_manager;
}

// NOLINTNEXTLINE
TEST(BufferPoolManagerInstanceTest, SampleTest) {
  const std::string db_name = "test.db";
  const size_t buffer_pool_size = 10;
  const size_t k = 5;

  auto *disk_manager = new DiskManager(db_name);
  auto *bpm = new BufferPoolManagerInstance(buffer_pool_size, disk_manager, k);

  page_id_t page_id_temp;
  auto *page0 = bpm->NewPage(&page_id_temp);
  // std::cout << "!!! page_id_temp " << page_id_temp << "\n";

  // Scenario: The buffer pool is empty. We should be able to create a new page.
  ASSERT_NE(nullptr, page0);
  EXPECT_EQ(0, page_id_temp);

  // Scenario: Once we have a page, we should be able to read and write content.
  snprintf(page0->GetData(), BUSTUB_PAGE_SIZE, "Hello");
  EXPECT_EQ(0, strcmp(page0->GetData(), "Hello"));

  // Scenario: We should be able to create new pages until we fill up the buffer pool.
  for (size_t i = 1; i < buffer_pool_size; ++i) {
    EXPECT_NE(nullptr, bpm->NewPage(&page_id_temp));
  }

  // Scenario: Once the buffer pool is full, we should not be able to create any new pages.
  for (size_t i = buffer_pool_size; i < buffer_pool_size * 2; ++i) {
    EXPECT_EQ(nullptr, bpm->NewPage(&page_id_temp));
  }

  // Scenario: After unpinning pages {0, 1, 2, 3, 4} and pinning another 4 new pages,
  // there would still be one buffer page left for reading page 0.
  for (int i = 0; i < 5; ++i) {
    EXPECT_EQ(true, bpm->UnpinPage(i, true));
  }
  for (int i = 0; i < 4; ++i) {
    EXPECT_NE(nullptr, bpm->NewPage(&page_id_temp));
  }

  // Scenario: We should be able to fetch the data we wrote a while ago.
  page0 = bpm->FetchPage(0);
  EXPECT_EQ(0, strcmp(page0->GetData(), "Hello"));

  // Scenario: If we unpin page 0 and then make a new page, all the buffer pages should
  // now be pinned. Fetching page 0 should fail.
  EXPECT_EQ(true, bpm->UnpinPage(0, true));
  EXPECT_NE(nullptr, bpm->NewPage(&page_id_temp));
  EXPECT_EQ(nullptr, bpm->FetchPage(0));

  // Shutdown the disk manager and remove the temporary file we created.
  disk_manager->ShutDown();
  remove("test.db");

  delete bpm;
  delete disk_manager;
}

TEST(BufferPoolManagerInstanceTest, gradescope7) {
  const std::string db_name = "test.db";
  page_id_t page_id_temp;

  auto *disk_manager = new DiskManager(db_name);
  auto *bpm = new BufferPoolManagerInstance(2, disk_manager, 5);
  auto page0 = bpm->NewPage(&page_id_temp);
  // Scenario: Once we have a page, we should be able to read and write content.
  snprintf(page0->GetData(), BUSTUB_PAGE_SIZE, "page0");
  EXPECT_EQ(0, strcmp(page0->GetData(), "page0"));

  bpm->NewPage(&page_id_temp);
  bpm->UnpinPage(0, true);
  bpm->UnpinPage(1, true);
  bpm->NewPage(&page_id_temp);
  bpm->UnpinPage(2, true);
  bpm->NewPage(&page_id_temp);
  bpm->UnpinPage(3, true);
  bpm->FetchPage(0);
  bpm->FetchPage(1);
  bpm->UnpinPage(0, false);
  bpm->UnpinPage(1, true);
  bpm->NewPage(&page_id_temp);
  bpm->UnpinPage(4, true);
  bpm->NewPage(&page_id_temp);
  bpm->UnpinPage(5, true);
  page0 = bpm->FetchPage(0);
  EXPECT_EQ(0, strcmp(page0->GetData(), "page0"));
  // /autograder/source/bustub/test/buffer/grading_buffer_pool_manager_instance_test.cpp:24Failure
  // Expected equality of these values:
  // 0
  // strcmp(page->GetData(), "page0")
  //   Which is: -1
  disk_manager->ShutDown();
  remove("test.db");

  delete bpm;
  delete disk_manager;
}

TEST(BufferPoolManagerInstanceTest, gradescope8) {
  const std::string db_name = "test.db";
  page_id_t page_id_temp;

  auto *disk_manager = new DiskManager(db_name);
  auto *bpm = new BufferPoolManagerInstance(10, disk_manager, 5);
  bpm->NewPage(&page_id_temp);
  bpm->NewPage(&page_id_temp);
  bpm->NewPage(&page_id_temp);
  bpm->NewPage(&page_id_temp);
  bpm->NewPage(&page_id_temp);
  bpm->NewPage(&page_id_temp);
  bpm->NewPage(&page_id_temp);
  bpm->NewPage(&page_id_temp);
  bpm->NewPage(&page_id_temp);
  bpm->NewPage(&page_id_temp);

  for (page_id_t i = 0; i < 10; i++) {
    bpm->FetchPage(i);
    EXPECT_EQ(true, bpm->UnpinPage(i, true));
    EXPECT_EQ(true, bpm->UnpinPage(i, true));
    bpm->FlushPage(0);
  }

  bpm->NewPage(&page_id_temp);
  bpm->UnpinPage(10, true);
  bpm->NewPage(&page_id_temp);
  bpm->UnpinPage(11, true);
  bpm->NewPage(&page_id_temp);
  bpm->UnpinPage(12, true);
  bpm->NewPage(&page_id_temp);
  bpm->UnpinPage(13, true);
  bpm->NewPage(&page_id_temp);
  bpm->UnpinPage(14, true);
  bpm->NewPage(&page_id_temp);
  bpm->UnpinPage(15, true);
  bpm->NewPage(&page_id_temp);
  bpm->UnpinPage(16, true);
  bpm->NewPage(&page_id_temp);
  bpm->UnpinPage(17, true);
  bpm->NewPage(&page_id_temp);
  bpm->UnpinPage(18, true);
  bpm->NewPage(&page_id_temp);
  bpm->UnpinPage(19, true);
  bpm->FetchPage(0);
  bpm->FetchPage(1);
  bpm->FetchPage(2);
  bpm->FetchPage(3);
  bpm->FetchPage(4);
  bpm->FetchPage(5);
  bpm->FetchPage(6);
  bpm->FetchPage(7);
  bpm->FetchPage(8);
  bpm->FetchPage(9);
  bpm->UnpinPage(4, true);
  bpm->NewPage(&page_id_temp);
  bpm->FetchPage(4);
  bpm->FetchPage(5);
  bpm->FetchPage(6);
  bpm->FetchPage(7);
  bpm->UnpinPage(5, false);
  bpm->UnpinPage(6, false);
  bpm->UnpinPage(7, false);
  bpm->UnpinPage(5, false);
  bpm->UnpinPage(6, false);
  bpm->UnpinPage(7, false);
  bpm->NewPage(&page_id_temp);
  bpm->FetchPage(5);
  bpm->FetchPage(7);
  // bpm->ShowPages();
  EXPECT_EQ(nullptr, bpm->FetchPage(6));

  // /autograder/source/bustub/test/buffer/grading_buffer_pool_manager_instance_test.cpp:334: Failure
  // Expected equality of these values:
  //   nullptr
  //     Which is: NULL
  //   bpm->FetchPage(page_ids[6])
  //     Which is: 0x62e0000075f8

  disk_manager->ShutDown();
  remove("test.db");
  delete bpm;
  delete disk_manager;
}

}  // namespace bustub
