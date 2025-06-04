#include <gtest/gtest.h>
#include "lar/core/spacial/region_tree.h"

using namespace lar;

TEST(RegionTreeTest, Print) {
  // Given
  std::ostringstream output;
  RegionTree<size_t> tree;
  tree.insert(1, Rect(1, 2, 3, 4), 1);
  // When
  tree.print(output);
  // Then
  EXPECT_EQ(output.str(),
    "( 1, 2 ) - ( 3, 4 )\n"
    "\t( 1, 2 ) - ( 3, 4 ) - #1\n");
}

TEST(RegionTreeTest, Insert) {
  // Given
  std::ostringstream output;
  RegionTree<size_t> tree;
  // When
  tree.insert(1, lar::Rect(12, 4, 24, 15), 1);
  tree.insert(2, lar::Rect(23, 24, 26, 26), 2);
  tree.insert(3, lar::Rect(12, 22, 16, 26), 3);
  tree.insert(4, lar::Rect(2, 16, 23, 17), 4);
  tree.insert(5, lar::Rect(1, 1, 2, 4), 5);
  tree.insert(6, lar::Rect(9, 14, 13, 18), 6);
  tree.insert(7, lar::Rect(5, 19, 8, 23), 7);
  tree.insert(8, lar::Rect(20, 14, 26, 22), 8);
  // Then
  EXPECT_EQ(tree[1], 1);
  EXPECT_EQ(tree[2], 2);
  EXPECT_EQ(tree[3], 3);
  EXPECT_EQ(tree[4], 4);
  EXPECT_EQ(tree[5], 5);
  EXPECT_EQ(tree[6], 6);
  EXPECT_EQ(tree[7], 7);
  EXPECT_EQ(tree[8], 8);
}

TEST(RegionTreeTest, Erase) {
  // Given
  std::ostringstream output;
  RegionTree<size_t> tree;
  // When
  tree.insert(1, lar::Rect(12, 4, 24, 15), 1);
  tree.insert(2, lar::Rect(23, 24, 26, 26), 2);
  tree.insert(3, lar::Rect(12, 22, 16, 26), 3);
  tree.insert(4, lar::Rect(2, 16, 23, 17), 4);
  tree.insert(5, lar::Rect(1, 1, 2, 4), 5);
  tree.insert(6, lar::Rect(9, 14, 13, 18), 6);
  tree.insert(7, lar::Rect(5, 19, 8, 23), 7);
  tree.insert(8, lar::Rect(20, 14, 26, 22), 8);
  tree.erase(1);
  tree.erase(2);
  tree.erase(3);
  tree.erase(4);
  tree.erase(5);
  tree.erase(6);
  tree.erase(7);
  tree.erase(8);
  // Then
  EXPECT_EQ(tree.size(), 0);
}

TEST(RegionTreeTest, Find) {
  // Given
  std::ostringstream output;
  RegionTree<size_t> tree;
  // When
  tree.insert(1, lar::Rect(12, 4, 24, 15), 1);
  tree.insert(2, lar::Rect(23, 24, 26, 26), 2);
  tree.insert(3, lar::Rect(12, 22, 16, 26), 3);
  tree.insert(4, lar::Rect(2, 16, 23, 17), 4);
  tree.insert(5, lar::Rect(1, 1, 2, 4), 5);
  tree.insert(6, lar::Rect(9, 14, 13, 18), 6);
  tree.insert(7, lar::Rect(5, 19, 8, 23), 7);
  tree.insert(8, lar::Rect(20, 14, 26, 22), 8);
  std::vector<size_t*> result = tree.find(Rect(1, 1, 24, 15));
  // Then
  // EXPECT_EQ(std::count(result.begin(), result.end(), 1), 1);
  // EXPECT_EQ(std::count(result.begin(), result.end(), 5), 1);
  // EXPECT_EQ(std::count(result.begin(), result.end(), 6), 1);
  // EXPECT_EQ(std::count(result.begin(), result.end(), 8), 1);
}
