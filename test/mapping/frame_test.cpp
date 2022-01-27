#include <gtest/gtest.h>
#include "geoar/mapping/frame.h"

using namespace Eigen;
using json = nlohmann::json;
using namespace geoar;


TEST(FrameTest, Initialization) {
  // Given
  json frame_data = json::parse("{\"extrinsics\":[0.5164972543716431,-0.021958883851766586,0.8560072779655457,0.0,0.09486261755228043,0.9949849843978882,-0.03171413019299507,0.0,-0.8510180115699768,0.0975833460688591,0.515990138053894,0.0,-0.04407598823308945,0.009304665960371494,-0.17797423899173737,0.9999998807907104],\"id\":10,\"intrinsics\":[1594.7247314453125,0.0,0.0,0.0,1594.7247314453125,0.0,952.8605346679688,714.1676025390625,1.0],\"timestamp\":136463350}");
  // When
  Frame frame = frame_data;
  // Then
  EXPECT_EQ(frame.id, 10);
  EXPECT_NEAR(frame.intrinsics(0,0), 1594.7247314453125, 1e-10);
}
