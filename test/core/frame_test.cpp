#include <gtest/gtest.h>
#include "geoar/core/frame.h"

using namespace Eigen;
using json = nlohmann::json;
using namespace geoar;


TEST(FrameTest, Initialization) {
  // Given
  json frame_data = json::parse("{\"id\":10,\"timestamp\":136463.350228125,\"intrinsics\":{\"focalLength\":1594.7247314453125,\"principlePoint\":{\"x\":952.8605346679688,\"y\":714.1676025390625}},\"transform\":[[0.5164972543716431,-0.021958883851766586,0.8560072779655457,0],[0.09486261755228043,0.9949849843978882,-0.03171413019299507,0],[-0.8510180115699768,0.0975833460688591,0.515990138053894,0],[-0.04407598823308945,0.009304665960371494,-0.17797423899173737,0.9999998807907104]]}");
  // When
  Frame frame(frame_data);
  // Then
  EXPECT_EQ(frame.id, 10);
  EXPECT_NEAR(frame.intrinsics["focalLength"], 1594.7247314453125, 1e-10);
}


TEST(FrameTest, PoseFromTransform) {
  // Given
  json transform = json::parse("[[0.5164972543716431,-0.021958883851766586,0.8560072779655457,0],[0.09486261755228043,0.9949849843978882,-0.03171413019299507,0],[-0.8510180115699768,0.0975833460688591,0.515990138053894,0],[-0.04407598823308945,0.009304665960371494,-0.17797423899173737,0.9999998807907104]]");
  // When
  g2o::SE3Quat result = Frame::poseFromTransform(transform);
  // Then
  g2o::Vector6 vector = result.toMinimalVector();
  EXPECT_NEAR(vector[0],  0.17531668611470036 , 1e-10);
  EXPECT_NEAR(vector[1],  0.010721137413063664, 1e-10);
  EXPECT_NEAR(vector[2], -0.053415504063665187, 1e-10);
  EXPECT_NEAR(vector[3], -0.86998166584441261 , 1e-10);
  EXPECT_NEAR(vector[4],  0.033570104611351119, 1e-10);
  EXPECT_NEAR(vector[5], -0.49053484893787047 , 1e-10);
}

TEST(FrameTest, TransformFromPose) {
  // Given
  Eigen::Matrix3d rot;
  rot <<  0.516497 , -0.0219589,  0.856007 ,
         -0.0948626, -0.994985 ,  0.0317141,
          0.851018 , -0.0975834, -0.51599  ;
  Eigen::Vector3d position(0.175317, 0.0107211, -0.0534155);
  Eigen::Quaterniond orientation(rot);
  g2o::SE3Quat pose(orientation, position);
  // When
  json result = Frame::transformFromPose(pose);
  // Then
  EXPECT_NEAR(result[0][0], 0.5164972543716431, 1e-4);
  EXPECT_NEAR(result[0][1], -0.021958883851766586, 1e-4);
  EXPECT_NEAR(result[0][2], 0.8560072779655457, 1e-4);
  EXPECT_NEAR(result[0][3], 0., 1e-4);
  EXPECT_NEAR(result[1][0], 0.09486261755228043, 1e-4);
  EXPECT_NEAR(result[1][1], 0.9949849843978882, 1e-4);
  EXPECT_NEAR(result[1][2], -0.03171413019299507, 1e-4);
  EXPECT_NEAR(result[1][3], 0., 1e-4);
  EXPECT_NEAR(result[2][0], -0.8510180115699768, 1e-4);
  EXPECT_NEAR(result[2][1], 0.0975833460688591, 1e-4);
  EXPECT_NEAR(result[2][2], 0.515990138053894, 1e-4);
  EXPECT_NEAR(result[2][3], 0., 1e-4);
  EXPECT_NEAR(result[3][0], -0.04407598823308945, 1e-4);
  EXPECT_NEAR(result[3][1], 0.009304665960371494, 1e-4);
  EXPECT_NEAR(result[3][2], -0.17797423899173737, 1e-4);
  EXPECT_NEAR(result[3][3], 1., 1e-4);
}
