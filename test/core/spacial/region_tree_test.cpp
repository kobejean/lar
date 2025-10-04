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
  std::vector<size_t*> result;
  tree.find(Rect(1, 1, 24, 15), result);
  // Then
  EXPECT_EQ(std::count_if(result.begin(), result.end(), [](size_t* p) { return *p == 1; }), 1);
  EXPECT_EQ(std::count_if(result.begin(), result.end(), [](size_t* p) { return *p == 5; }), 1);
  EXPECT_EQ(std::count_if(result.begin(), result.end(), [](size_t* p) { return *p == 6; }), 1);
  EXPECT_EQ(std::count_if(result.begin(), result.end(), [](size_t* p) { return *p == 8; }), 1);
}

TEST(RegionTreeTest, UpdateBoundsPreservesPointers) {
  // Given
  RegionTree<size_t> tree;
  size_t* ptr1 = tree.insert(100, Rect(10, 10, 20, 20), 1);
  size_t* ptr2 = tree.insert(200, Rect(30, 30, 40, 40), 2);
  size_t* ptr3 = tree.insert(300, Rect(50, 50, 60, 60), 3);

  // When - update bounds to trigger tree reorganization
  tree.updateBounds(1, Rect(100, 100, 110, 110)); // Move far away
  tree.updateBounds(2, Rect(5, 5, 15, 15));       // Move to different region

  // Then - pointers should still be valid and point to same values
  EXPECT_EQ(*ptr1, 100) << "Pointer to item 1 should still be valid";
  EXPECT_EQ(*ptr2, 200) << "Pointer to item 2 should still be valid";
  EXPECT_EQ(*ptr3, 300) << "Pointer to item 3 should still be valid";

  // Verify we can still find items in new locations
  std::vector<size_t*> result;
  tree.find(Rect(95, 95, 115, 115), result);
  EXPECT_EQ(std::count_if(result.begin(), result.end(), [](size_t* p) { return *p == 100; }), 1);
}

TEST(RegionTreeTest, UpdateBoundsOnce) {
  // Given
  RegionTree<size_t> tree;
  size_t* ptr = tree.insert(42, Rect(0, 0, 10, 10), 1);

  // When - update bounds once
  tree.updateBounds(1, Rect(20, 20, 30, 30));

  // Then
  EXPECT_EQ(*ptr, 42) << "Pointer should remain valid after update";
}

TEST(RegionTreeTest, UpdateBoundsMultipleTimes) {
  // Given
  RegionTree<size_t> tree;
  size_t* ptr = tree.insert(42, Rect(0, 0, 10, 10), 1);

  // When - update bounds multiple times
  tree.updateBounds(1, Rect(20, 20, 30, 30));
  EXPECT_EQ(*ptr, 42) << "Pointer should remain valid after first update";

  tree.updateBounds(1, Rect(40, 40, 50, 50));
  EXPECT_EQ(*ptr, 42) << "Pointer should remain valid after second update";

  tree.updateBounds(1, Rect(60, 60, 70, 70));
  EXPECT_EQ(*ptr, 42) << "Pointer should remain valid after third update";

  // Then - verify final location
  std::vector<size_t*> result;
  tree.find(Rect(55, 55, 75, 75), result);
  EXPECT_EQ(result.size(), 1);
  EXPECT_EQ(*result[0], 42);
}

// ============================================================================
// Comprehensive Underflow/Overflow Tests
// ============================================================================
// Note: MAX_CHILDREN = 4 for RegionTree<size_t>
// Underflow: < 2 children, Overflow: > 4 children

TEST(RegionTreeTest, CascadingUnderflow) {
  // Given - Build tree with 3 levels
  // Root -> 2 internal nodes -> 4 leaves each
  RegionTree<size_t> tree;

  // Insert 8 items to create multi-level tree
  // Group 1: ids 1-4 in region (0-50, 0-50)
  size_t* ptr1 = tree.insert(1, Rect(0, 0, 10, 10), 1);
  size_t* ptr2 = tree.insert(2, Rect(10, 10, 20, 20), 2);
  size_t* ptr3 = tree.insert(3, Rect(20, 20, 30, 30), 3);
  size_t* ptr4 = tree.insert(4, Rect(30, 30, 40, 40), 4);

  // Group 2: ids 5-8 in region (100-150, 100-150)
  size_t* ptr5 = tree.insert(5, Rect(100, 100, 110, 110), 5);
  size_t* ptr6 = tree.insert(6, Rect(110, 110, 120, 120), 6);
  size_t* ptr7 = tree.insert(7, Rect(120, 120, 130, 130), 7);
  size_t* ptr8 = tree.insert(8, Rect(130, 130, 140, 140), 8);

  // When - Erase items to trigger cascading underflow
  tree.erase(1);
  tree.erase(2);
  tree.erase(3); // Should trigger underflow in first group

  // Then - All remaining pointers should still be valid
  EXPECT_EQ(*ptr4, 4);
  EXPECT_EQ(*ptr5, 5);
  EXPECT_EQ(*ptr6, 6);
  EXPECT_EQ(*ptr7, 7);
  EXPECT_EQ(*ptr8, 8);
  EXPECT_EQ(tree.size(), 5);
}

TEST(RegionTreeTest, UnderflowPropagationToRoot) {
  // Given - Create minimal tree that will underflow all the way to root
  RegionTree<size_t> tree;

  size_t* ptr1 = tree.insert(1, Rect(0, 0, 10, 10), 1);
  size_t* ptr2 = tree.insert(2, Rect(20, 20, 30, 30), 2);
  size_t* ptr3 = tree.insert(3, Rect(40, 40, 50, 50), 3);

  // When - Erase to trigger underflow
  tree.erase(2);

  // Then - Remaining pointers valid, tree still functional
  EXPECT_EQ(*ptr1, 1);
  EXPECT_EQ(*ptr3, 3);
  EXPECT_EQ(tree.size(), 2);

  std::vector<size_t*> result;
  tree.find(Rect(0, 0, 50, 50), result);
  EXPECT_EQ(result.size(), 2);
}

TEST(RegionTreeTest, UpdateBoundsDuringUnderflow) {
  // Given - Tree with items that will trigger underflow
  RegionTree<size_t> tree;

  size_t* ptr1 = tree.insert(1, Rect(0, 0, 10, 10), 1);
  size_t* ptr2 = tree.insert(2, Rect(10, 10, 20, 20), 2);
  size_t* ptr3 = tree.insert(3, Rect(20, 20, 30, 30), 3);
  size_t* ptr4 = tree.insert(4, Rect(30, 30, 40, 40), 4);
  size_t* ptr5 = tree.insert(5, Rect(40, 40, 50, 50), 5);

  // When - Erase to cause underflow, then update bounds
  tree.erase(2);
  tree.erase(3);
  tree.updateBounds(1, Rect(100, 100, 110, 110)); // Move far away

  // Then - All pointers still valid
  EXPECT_EQ(*ptr1, 1);
  EXPECT_EQ(*ptr4, 4);
  EXPECT_EQ(*ptr5, 5);

  std::vector<size_t*> result;
  tree.find(Rect(95, 95, 115, 115), result);
  EXPECT_EQ(std::count_if(result.begin(), result.end(), [](size_t* p) { return *p == 1; }), 1);
}

TEST(RegionTreeTest, AlternatingInsertErase) {
  // Given - Empty tree
  RegionTree<size_t> tree;
  std::vector<size_t*> pointers;

  // When - Alternate insert and erase to stress tree rebalancing
  for (int i = 0; i < 20; i++) {
    size_t id = i + 1;
    size_t* ptr = tree.insert(id, Rect(i * 10, i * 10, i * 10 + 5, i * 10 + 5), id);
    pointers.push_back(ptr);

    if (i % 3 == 0 && i > 0) {
      tree.erase(i / 2); // Erase earlier items
    }
  }

  // Then - Remaining pointers should be valid
  for (size_t* ptr : pointers) {
    if (ptr != nullptr) {
      EXPECT_GT(*ptr, 0);
      EXPECT_LE(*ptr, 20);
    }
  }
}

TEST(RegionTreeTest, UpdateBoundsWithComplexRebalancing) {
  // Given - Build complex tree
  RegionTree<size_t> tree;
  std::vector<size_t*> pointers;

  // Insert 12 items to create multi-level tree
  for (int i = 0; i < 12; i++) {
    size_t id = i + 1;
    size_t* ptr = tree.insert(id, Rect(i * 5, i * 5, i * 5 + 3, i * 5 + 3), id);
    pointers.push_back(ptr);
  }

  // When - Update bounds of multiple items to trigger rebalancing
  tree.updateBounds(1, Rect(200, 200, 203, 203));
  tree.updateBounds(5, Rect(210, 210, 213, 213));
  tree.updateBounds(9, Rect(220, 220, 223, 223));

  // Then - All original pointers still valid
  for (size_t i = 0; i < pointers.size(); i++) {
    EXPECT_EQ(*pointers[i], i + 1) << "Pointer " << i << " should still be valid";
  }
  EXPECT_EQ(tree.size(), 12);
}

TEST(RegionTreeTest, StressTestMassiveRebalancing) {
  // Given - Large tree
  RegionTree<size_t> tree;
  std::map<size_t, size_t*> id_to_ptr;

  // Insert many items
  for (int i = 0; i < 50; i++) {
    size_t id = i + 1;
    size_t* ptr = tree.insert(id, Rect(i * 2, i * 2, i * 2 + 1, i * 2 + 1), id);
    id_to_ptr[id] = ptr;
  }

  // When - Erase half the items (triggers many underflows)
  for (int i = 0; i < 25; i++) {
    tree.erase(i * 2 + 1);
    id_to_ptr.erase(i * 2 + 1);
  }

  // Then - All remaining pointers should be valid
  for (const auto& [id, ptr] : id_to_ptr) {
    EXPECT_EQ(*ptr, id) << "ID " << id << " pointer should still be valid";
  }
  EXPECT_EQ(tree.size(), 25);
}

TEST(RegionTreeTest, PointerStabilityThroughComplexOperations) {
  // Given
  RegionTree<size_t> tree;

  size_t* p1 = tree.insert(100, Rect(0, 0, 5, 5), 1);
  size_t* p2 = tree.insert(200, Rect(10, 10, 15, 15), 2);
  size_t* p3 = tree.insert(300, Rect(20, 20, 25, 25), 3);
  size_t* p4 = tree.insert(400, Rect(30, 30, 35, 35), 4);
  size_t* p5 = tree.insert(500, Rect(40, 40, 45, 45), 5);

  // When - Complex sequence of operations
  tree.updateBounds(1, Rect(100, 100, 105, 105)); // Move
  tree.erase(3);                                   // Delete
  tree.updateBounds(2, Rect(110, 110, 115, 115)); // Move
  tree.updateBounds(5, Rect(50, 50, 55, 55));     // Move

  // Then - Remaining pointers stay valid
  EXPECT_EQ(*p1, 100);
  EXPECT_EQ(*p2, 200);
  EXPECT_EQ(*p4, 400);
  EXPECT_EQ(*p5, 500);

  // Verify values are findable
  std::vector<size_t*> result;
  tree.find(Rect(0, 0, 200, 200), result);
  EXPECT_GE(result.size(), 1);
}

TEST(RegionTreeTest, UpdateBoundsWithOverflow) {
  // Given - Build tree with 8 items (MAX_CHILDREN=4, so this creates splits)
  RegionTree<size_t> tree;
  std::vector<size_t*> ptrs;

  for (size_t i = 1; i <= 8; i++) {
    ptrs.push_back(tree.insert(i * 100, Rect(i*10, i*10, i*10+5, i*10+5), i));
  }

  // When - Update bounds to force tree reorganization with overflow
  tree.updateBounds(1, Rect(100, 100, 105, 105)); // Move to far corner
  tree.updateBounds(5, Rect(50, 50, 55, 55));     // Move to middle

  // Then - All pointers should remain valid
  for (size_t i = 0; i < 8; i++) {
    EXPECT_EQ(*ptrs[i], (i+1) * 100) << "Pointer " << i << " should remain valid";
  }

  // Verify tree structure is still valid
  EXPECT_EQ(tree.size(), 8);
}

TEST(RegionTreeTest, UpdateBoundsTriggersUnderflow) {
  // Given - Build tree with exactly 5 items to create specific structure
  // (With MAX_CHILDREN=4, this forces one split)
  RegionTree<size_t> tree;
  size_t* ptr1 = tree.insert(100, Rect(0, 0, 10, 10), 1);
  size_t* ptr2 = tree.insert(200, Rect(20, 0, 30, 10), 2);
  size_t* ptr3 = tree.insert(300, Rect(40, 0, 50, 10), 3);
  size_t* ptr4 = tree.insert(400, Rect(60, 0, 70, 10), 4);
  size_t* ptr5 = tree.insert(500, Rect(80, 0, 90, 10), 5);

  // When - Erase some to get close to underflow, then updateBounds
  tree.erase(4);
  tree.erase(5);
  // Now update bounds on remaining items to trigger potential underflow
  tree.updateBounds(1, Rect(100, 100, 110, 110));

  // Then - Pointers should still be valid despite underflow handling
  EXPECT_EQ(*ptr1, 100) << "ptr1 should remain valid after underflow";
  EXPECT_EQ(*ptr2, 200) << "ptr2 should remain valid";
  EXPECT_EQ(*ptr3, 300) << "ptr3 should remain valid";
  EXPECT_EQ(tree.size(), 3);
}

// ============================================================================
// Move Semantics and Safety Tests
// ============================================================================

TEST(RegionTreeTest, MoveConstructorTransfersOwnership) {
  // Given - Tree with data
  RegionTree<size_t> tree1;
  tree1.insert(42, Rect(0, 0, 1, 1), 1);
  tree1.insert(99, Rect(2, 2, 3, 3), 2);

  EXPECT_EQ(tree1.size(), 2);
  EXPECT_EQ(tree1[1], 42);

  // When - Move construct tree2 from tree1
  RegionTree<size_t> tree2(std::move(tree1));

  // Then - tree2 should have the data
  EXPECT_EQ(tree2.size(), 2);
  EXPECT_EQ(tree2[1], 42);
  EXPECT_EQ(tree2[2], 99);

  // tree1 should be empty (moved-from state)
  EXPECT_EQ(tree1.size(), 0);
}

TEST(RegionTreeTest, MoveAssignmentTransfersOwnership) {
  // Given - Two trees
  RegionTree<size_t> tree1;
  tree1.insert(42, Rect(0, 0, 1, 1), 1);
  tree1.insert(99, Rect(2, 2, 3, 3), 2);

  RegionTree<size_t> tree2;
  tree2.insert(100, Rect(5, 5, 6, 6), 10);

  EXPECT_EQ(tree1.size(), 2);
  EXPECT_EQ(tree2.size(), 1);

  // When - Move assign tree1 to tree2
  tree2 = std::move(tree1);

  // Then - tree2 should have tree1's data
  EXPECT_EQ(tree2.size(), 2);
  EXPECT_EQ(tree2[1], 42);
  EXPECT_EQ(tree2[2], 99);

  // tree1 should be empty (moved-from state)
  EXPECT_EQ(tree1.size(), 0);
}

TEST(RegionTreeTest, MovedFromTreeState) {
  // Given - Tree with data
  RegionTree<size_t> tree1;
  tree1.insert(42, Rect(0, 0, 1, 1), 1);

  // When - Move construct tree2 from tree1
  RegionTree<size_t> tree2(std::move(tree1));

  // Then - tree2 has the data, tree1 is in valid but unspecified state
  EXPECT_EQ(tree2.size(), 1);
  EXPECT_EQ(tree2[1], 42);

  // Moved-from tree should be empty
  EXPECT_EQ(tree1.size(), 0);

  // Note: Using moved-from tree beyond checking size() is undefined behavior
  // We do NOT test insertion into moved-from tree as it's not guaranteed to work
}

// ============================================================================
// Root Sentinel Pattern Tests
// ============================================================================

TEST(RegionTreeTest, RootNeverBecomesLeaf) {
  // This test verifies that the root maintains the sentinel pattern
  // The concern: After certain erase operations, root might become a leaf node

  // Given - Tree that will trigger root replacement
  RegionTree<size_t> tree;

  // Insert items to create a 3-level tree
  // Root (height=2) -> Internal nodes (height=1) -> Leaves (height=0)
  tree.insert(1, Rect(0, 0, 5, 5), 1);
  tree.insert(2, Rect(10, 10, 15, 15), 2);
  tree.insert(3, Rect(20, 20, 25, 25), 3);
  tree.insert(4, Rect(30, 30, 35, 35), 4);
  tree.insert(5, Rect(40, 40, 45, 45), 5);
  tree.insert(6, Rect(50, 50, 55, 55), 6);

  // When - Erase items until only one remains
  tree.erase(2);
  tree.erase(3);
  tree.erase(4);
  tree.erase(5);
  tree.erase(6);

  // Then - Root should never be a leaf (height should be > 0)
  // This insertion should work without special cases
  tree.insert(10, Rect(100, 100, 105, 105), 10);

  EXPECT_EQ(tree.size(), 2);
  EXPECT_EQ(tree[1], 1);
  EXPECT_EQ(tree[10], 10);
}
