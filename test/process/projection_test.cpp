#include <gtest/gtest.h>
#include "geoar/process/projection.h"

using namespace Eigen;
using json = nlohmann::json;
using namespace geoar;

std::string json_string1 = "{\"id\":0,\"timestamp\":136463.350228125,\"intrinsics\":{\"focalLength\":1594.7247314453125,\"principlePoint\":{\"x\":952.8605346679688,\"y\":714.1676025390625}},\"transform\":[[0.5164972543716431,-0.021958883851766586,0.8560072779655457,0],[0.09486261755228043,0.9949849843978882,-0.03171413019299507,0],[-0.8510180115699768,0.0975833460688591,0.515990138053894,0],[-0.04407598823308945,0.009304665960371494,-0.17797423899173737,0.9999998807907104]]}";
json frame_data1 = json::parse(json_string1);

TEST(ProjectionTest, ProjectToWorld1) {
  // Given
  Projection projection(frame_data1);
  cv::Point2f pt(300.0f, 600.0f);
  float depth = 3.0;
  // When
  Vector3d result = projection.projectToWorld(pt, depth);
  
  EXPECT_NEAR(result[0], 1.8950091973238727, 1e-5);
  EXPECT_NEAR(result[1], -0.04278100611330371, 1e-5);
  EXPECT_NEAR(result[2], -2.7840722741000694, 1e-5);
}

TEST(ProjectionTest, ProjectToWorld2) {
  // Given
  Projection projection(frame_data1);
  cv::Point2f pt(368.6075439453125f, 612.209228515625f);
  float depth = 3.9;
  // When
  Vector3d result = projection.projectToWorld(pt, depth);
  
  EXPECT_NEAR(result[0], 2.5605623834723801, 1e-5);
  EXPECT_NEAR(result[1], -0.091799759847312201, 1e-5);
  EXPECT_NEAR(result[2], -3.4213304361981178, 1e-5);
}

TEST(ProjectionTest, ProjectToImage1) {
  // Given
  Projection projection(frame_data1);
  Vector3d pt(2.2181900615833667, -0.0514619514589163, -3.2184219466181245);
  // When
  cv::Point2f result = projection.projectToImage(pt);
  
  EXPECT_NEAR(result.x, 300.0f, 1e-2);
  EXPECT_NEAR(result.y, 600.0f, 1e-2);
}

TEST(ProjectionTest, ProjectToImage2) {
  // Given
  Projection projection(frame_data1);
  Vector3d pt(3.0, 0.5, -3.0);
  // When
  cv::Point2f result = projection.projectToImage(pt);
  
  EXPECT_NEAR(result.x, 612.209228515625f, 1e-2);
  EXPECT_NEAR(result.y, 368.6075439453125f, 1e-2);
}