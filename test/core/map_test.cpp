#include <gtest/gtest.h>
#include "lar/core/map.h"

using namespace lar;

TEST(MapTest, GlobalPointNotReady) {
  // Given
  Map map;
  Eigen::Vector3d relative{ 0, 4, 0 };
  // When
  Eigen::Vector3d global;
  bool success = map.globalPointFrom(relative, global);
  // Then
  EXPECT_FALSE(success);
}

TEST(MapTest, GlobalPointFromExample1) {
  // Given
  Map map;
  Eigen::Matrix4d origin;
  origin <<  -8.34987069574182e-06, 0, -3.3834374392020437e-06,  37.5236728,
            4.2473697485546255e-06, 0, -1.0481939989941319e-05, 139.9380725,
                                 0, 1,                       0,         208,
                                 0, 0,                       0,           1;
  map.origin = Eigen::Transform<double, 3, Eigen::Affine>(origin);
  Eigen::Vector3d relative{ 0, 4, 0 };
  // When
  Eigen::Vector3d global;
  bool success = map.globalPointFrom(relative, global);
  // Then
  EXPECT_TRUE(success);
  EXPECT_NEAR(global.x(), 37.5236728, 1e-5);
  EXPECT_NEAR(global.y(), 139.9380725, 1e-5);
  EXPECT_NEAR(global.z(), 212, 1e-5);
}

TEST(MapTest, GlobalPointFromExample2) {
  // Given
  Map map;
  Eigen::Matrix4d origin;
  origin <<  -8.34987069574182e-06, 0, -3.3834374392020437e-06,  37.5236728,
            4.2473697485546255e-06, 0, -1.0481939989941319e-05, 139.9380725,
                                 0, 1,                       0,         208,
                                 0, 0,                       0,           1;
  map.origin = Eigen::Transform<double, 3, Eigen::Affine>(origin);
  Eigen::Vector3d relative{ 1289.6952137156, 0, 1289.6952137156 };
  // When
  Eigen::Vector3d global;
  bool success = map.globalPointFrom(relative, global);
  // Then
  EXPECT_TRUE(success);
  EXPECT_NEAR(global.x(), 37.5085404, 1e-5);
  EXPECT_NEAR(global.y(), 139.9300318, 1e-5);
  EXPECT_NEAR(global.z(), 208, 1e-5);
}

TEST(MapTest, RelativePointNotReady) {
  // Given
  Map map;
  Eigen::Vector3d global{ 37.5236728, 139.9380725, 212 };
  // When
  Eigen::Vector3d relative;
  bool success = map.relativePointFrom(relative, global);
  // Then
  EXPECT_FALSE(success);
}

TEST(MapTest, RelativePointFromExample1) {
  // Given
  Map map;
  Eigen::Matrix4d origin;
  origin <<  -8.34987069574182e-06, 0, -3.3834374392020437e-06,  37.5236728,
            4.2473697485546255e-06, 0, -1.0481939989941319e-05, 139.9380725,
                                 0, 1,                       0,         208,
                                 0, 0,                       0,           1;
  map.origin = Eigen::Transform<double, 3, Eigen::Affine>(origin);
  Eigen::Vector3d global{ 37.5236728, 139.9380725, 212 };
  // When
  Eigen::Vector3d relative;
  bool success = map.relativePointFrom(global, relative);
  // Then
  EXPECT_TRUE(success);
  EXPECT_NEAR(relative.x(), 0, 1e-2);
  EXPECT_NEAR(relative.y(), 4, 1e-2);
  EXPECT_NEAR(relative.z(), 0, 1e-2);
}

TEST(MapTest, RelativePointFromExample2) {
  // Given
  Map map;
  Eigen::Matrix4d origin;
  origin <<  -8.34987069574182e-06, 0, -3.3834374392020437e-06,  37.5236728,
            4.2473697485546255e-06, 0, -1.0481939989941319e-05, 139.9380725,
                                 0, 1,                       0,         208,
                                 0, 0,                       0,           1;
  map.origin = Eigen::Transform<double, 3, Eigen::Affine>(origin);
  Eigen::Vector3d global{ 37.5085404, 139.9300318, 208 };
  // When
  Eigen::Vector3d relative;
  bool success = map.relativePointFrom(global, relative);
  // Then
  EXPECT_TRUE(success);
  EXPECT_NEAR(relative.x(), 1289.6959515561, 1e-5);
  EXPECT_NEAR(relative.y(), 0, 1e-2);
  EXPECT_NEAR(relative.z(), 1289.6959515561, 1e-5);
}