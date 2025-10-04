#include <gtest/gtest.h>
#include "lar/core/spatial/rect.h"

using namespace lar;

TEST(RectTest, Print) {
  // Given
  std::ostringstream output;
  lar::Rect rect(1, 2, 3, 4);
  // When
  rect.print(output);
  // Then
  EXPECT_EQ(output.str(), "( 1, 2 ) - ( 3, 4 )" );
}

TEST(RectTest, IntersectsWithIdenticalRects) {
  // Given
  std::ostringstream output;
  lar::Rect rect1( 1,  2,  3,  4);
  lar::Rect rect2(-4, -3, -2, -1);
  lar::Rect rect3( 1,  1,  1,  1);
  lar::Rect rect4( 0,  0,  0,  0);
  lar::Rect rect5(-1, -1, -1, -1);
  // Then
  EXPECT_TRUE(rect1.intersectsWith(rect1));
  EXPECT_TRUE(rect2.intersectsWith(rect2));
  EXPECT_TRUE(rect3.intersectsWith(rect3));
  EXPECT_TRUE(rect4.intersectsWith(rect4));
  EXPECT_TRUE(rect5.intersectsWith(rect5));
}

TEST(RectTest, IntersectsWithIntersectingRects) {
  // Given
  lar::Rect rect_center(-1, -1, 1, 1);
  lar::Rect rect_above(-0.9, 1, 0.9, 2);
  lar::Rect rect_below(-0.9, -2, 0.9, -1);
  lar::Rect rect_right(1, -0.9, 2, 0.9);
  lar::Rect rect_left(-2, -0.9, 1, 0.9);
  lar::Rect rect_inside(-0.9, -0.9, 0.9, 0.9);
  // Then
  EXPECT_TRUE(rect_center.intersectsWith(rect_above));
  EXPECT_TRUE(rect_center.intersectsWith(rect_below));
  EXPECT_TRUE(rect_center.intersectsWith(rect_right));
  EXPECT_TRUE(rect_center.intersectsWith(rect_left));
  EXPECT_TRUE(rect_center.intersectsWith(rect_inside));
  EXPECT_TRUE(rect_above.intersectsWith(rect_center));
  EXPECT_TRUE(rect_below.intersectsWith(rect_center));
  EXPECT_TRUE(rect_right.intersectsWith(rect_center));
  EXPECT_TRUE(rect_left.intersectsWith(rect_center));
  EXPECT_TRUE(rect_inside.intersectsWith(rect_center));
}

TEST(RectTest, IntersectsWithDisjoinedRects) {
  // Given
  lar::Rect rect_center(-1, -1, 1, 1);
  lar::Rect rect_above(-2, 1.1, 2, 2);
  lar::Rect rect_below(-2, -2, 2, -1.1);
  lar::Rect rect_right(1.1, -2, 2, 2);
  lar::Rect rect_left(-2, -2, -1.1, 2);
  // Then
  EXPECT_FALSE(rect_center.intersectsWith(rect_above));
  EXPECT_FALSE(rect_center.intersectsWith(rect_below));
  EXPECT_FALSE(rect_center.intersectsWith(rect_right));
  EXPECT_FALSE(rect_center.intersectsWith(rect_left));
  EXPECT_FALSE(rect_above.intersectsWith(rect_center));
  EXPECT_FALSE(rect_below.intersectsWith(rect_center));
  EXPECT_FALSE(rect_right.intersectsWith(rect_center));
  EXPECT_FALSE(rect_left.intersectsWith(rect_center));
}
