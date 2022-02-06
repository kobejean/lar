#include <gtest/gtest.h>
#include "lar/core/utils/json.h"
#include "lar/core/map.h"

using namespace Eigen;
using json = nlohmann::json;
using namespace lar;


TEST(JSONTest, MatrixSerialization) {
  // Given
  Eigen::Matrix3d mat;
  mat << 0.1,0.4,0.7,
         0.2,0.5,0.8,
         0.3,0.6,0.9;
  // When
  json mat_json = mat;
  // Then
  EXPECT_EQ(mat_json.dump(), "[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]");
}

TEST(JSONTest, MatrixDeserialization) {
  // Given
  json mat_json = json::parse("[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]");
  // When
  Eigen::Matrix3d mat = mat_json;
  // Then
  EXPECT_NEAR(mat(0,0), 0.1, 1e-10);
  EXPECT_NEAR(mat(1,0), 0.2, 1e-10);
  EXPECT_NEAR(mat(2,0), 0.3, 1e-10);
  EXPECT_NEAR(mat(0,1), 0.4, 1e-10);
  EXPECT_NEAR(mat(1,1), 0.5, 1e-10);
  EXPECT_NEAR(mat(2,1), 0.6, 1e-10);
  EXPECT_NEAR(mat(0,2), 0.7, 1e-10);
  EXPECT_NEAR(mat(1,2), 0.8, 1e-10);
  EXPECT_NEAR(mat(2,2), 0.9, 1e-10);
}
