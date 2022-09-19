#include <gtest/gtest.h>
#include "lar/core/spacial/region_tree.h"

using namespace lar;

TEST(RegionTreeTest, PrintNode) {
  // Given
  std::ostringstream output;
  RegionTree<size_t, 5> *tree = new RegionTree<size_t, 5>();
  tree->insert(1, Rect(1, 2, 3, 4), 1);
  // When
  tree->print(output);
  // Then
  EXPECT_EQ(output.str(), "( 1, 2 ) - ( 3, 4 )\n\t( 1, 2 ) - ( 3, 4 ) - #1\n");
}

// TEST(RegionTreeTest, InsertMaxChildren) 