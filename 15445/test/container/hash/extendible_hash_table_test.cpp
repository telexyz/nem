/**
 * extendible_hash_test.cpp
 */

#include <memory>
#include <thread>  // NOLINT

#include "container/hash/extendible_hash_table.h"
#include "gtest/gtest.h"

namespace bustub {

TEST(ExtendibleHashTableTest, SampleTest) {
  auto table = std::make_unique<ExtendibleHashTable<int, std::string>>(2);
  // => global_depth_ = 0, bucket_size_ = 2, num_buckets_ = 1
  std::string result;

  table->Insert(1, "a");
  EXPECT_TRUE(table->Find(1, result));
  table->Insert(2, "b");
  EXPECT_TRUE(table->Find(2, result));
  table->Insert(3, "c");
  EXPECT_TRUE(table->Find(3, result));
  table->Insert(4, "d");
  EXPECT_TRUE(table->Find(4, result));
  table->Insert(5, "e");
  EXPECT_TRUE(table->Find(5, result));
  table->Insert(6, "f");
  EXPECT_TRUE(table->Find(6, result));
  table->Insert(7, "g");
  EXPECT_TRUE(table->Find(7, result));
  table->Insert(8, "h");
  EXPECT_TRUE(table->Find(8, result));
  table->Insert(9, "i");
  EXPECT_TRUE(table->Find(9, result));

  EXPECT_EQ(2, table->GetLocalDepth(0));
  EXPECT_EQ(3, table->GetLocalDepth(1));
  EXPECT_EQ(2, table->GetLocalDepth(2));
  EXPECT_EQ(2, table->GetLocalDepth(3));

  table->Find(9, result);
  EXPECT_EQ("i", result);
  table->Find(8, result);
  EXPECT_EQ("h", result);
  table->Find(2, result);
  EXPECT_EQ("b", result);
  EXPECT_FALSE(table->Find(10, result));

  EXPECT_TRUE(table->Remove(8));
  EXPECT_TRUE(table->Remove(4));
  EXPECT_TRUE(table->Remove(1));
  EXPECT_FALSE(table->Remove(20));
}

TEST(ExtendibleHashTableTest, InsertMultipleSplit) {
  auto table = std::make_unique<ExtendibleHashTable<int, int>>(2);
  table->Insert(0, 0);
  table->Insert(1024, 1024);
  table->Insert(4, 4);
}

TEST(ExtendibleHashTableTest, ConcurrentInsertTest) {
  const int num_runs = 50;
  const int num_threads = 3;

  // Run concurrent test multiple times to guarantee correctness.
  for (int run = 0; run < num_runs; run++) {
    auto table = std::make_unique<ExtendibleHashTable<int, int>>(2);
    std::vector<std::thread> threads;
    threads.reserve(num_threads);

    for (int tid = 0; tid < num_threads; tid++) {
      threads.emplace_back([tid, &table]() { table->Insert(tid, tid); });
      // table->Insert(tid, tid);
    }
    for (int i = 0; i < num_threads; i++) {
      threads[i].join();
    }

    EXPECT_EQ(table->GetGlobalDepth(), 1);
    for (int i = 0; i < num_threads; i++) {
      int val;
      EXPECT_TRUE(table->Find(i, val));
      EXPECT_EQ(i, val);
    }
  }
}

TEST(ExtendibleHashTableTest, GetNumBuckets) {
  auto table = std::make_unique<ExtendibleHashTable<int, int>>(2);
  table->Insert(4, 0);
  table->Insert(12, 0);
  table->Insert(16, 0);
  table->Insert(64, 0);
  table->Insert(31, 0);
  table->Insert(10, 0);

  table->Insert(51, 0);
  table->Insert(15, 0);
  table->Insert(18, 0);
  table->Insert(20, 0);

  table->Insert(7, 0);
  table->Insert(23, 0);
  table->Insert(11, 0);
  table->Insert(19, 0);
}

TEST(ExtendibleHashTableTest, GRADER_GetNumBuckets) {
  auto table = std::make_unique<ExtendibleHashTable<int, std::string>>(4);

  table->Insert(4, "a");
  table->Insert(12, "a");
  table->Insert(16, "a");
  table->Insert(64, "a");
  table->Insert(31, "a");
  table->Insert(10, "a");
  table->Insert(51, "a");
  table->Insert(15, "a");
  table->Insert(18, "a");
  table->Insert(20, "a");
  table->Insert(7, "a");
  table->Insert(23, "a");

  // [4,12,16,64,31,10,51,15,18,20,7,23].map((f) => f.toString(2))
  // [ 4: '100', 12: '1100', 16: '10000', 64: '1000000', 31: '11111', 10: '1010', 51: '110011', 15: '1111', 18: '10010',
  // 20: '10100', 7: '111', 23: '10111']

  EXPECT_EQ(6, table->GetNumBuckets());
}

TEST(ExtendibleHashTableTest, GRADER_LocalDepth) {
  auto table = std::make_unique<ExtendibleHashTable<int, std::string>>(4);

  table->Insert(4, "a");
  table->Insert(12, "a");
  table->Insert(16, "a");
  table->Insert(64, "a");
  table->Insert(5, "a");
  table->Insert(10, "a");
  table->Insert(51, "a");
  table->Insert(15, "a");
  table->Insert(18, "a");
  table->Insert(20, "a");
  table->Insert(7, "a");
  table->Insert(21, "a");

  EXPECT_EQ(2, table->GetLocalDepth(5));
}

}  // namespace bustub
