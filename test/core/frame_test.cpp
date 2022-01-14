#include <gtest/gtest.h>
#include "geoar/core/frame.h"

using namespace Eigen;
using json = nlohmann::json;
using namespace geoar;


TEST(FrameTest, Initialization) {
  // Given
  std::string json_string1 = "{\"id\":0,\"timestamp\":136463.350228125,\"intrinsics\":{\"focalLength\":1594.7247314453125,\"principlePoint\":{\"x\":952.8605346679688,\"y\":714.1676025390625}},\"transform\":[[0.5164972543716431,-0.021958883851766586,0.8560072779655457,0],[0.09486261755228043,0.9949849843978882,-0.03171413019299507,0],[-0.8510180115699768,0.0975833460688591,0.515990138053894,0],[-0.04407598823308945,0.009304665960371494,-0.17797423899173737,0.9999998807907104]]}";
  json frame_data1 = json::parse(json_string1);
  Frame frame(frame_data1, 0);
  // Then
  EXPECT_NEAR(frame.intrinsics["focalLength"], 1594.7247314453125, 1e-10);
}
