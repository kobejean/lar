#include <gtest/gtest.h>
#include "geoar/core/utils/json.h"
#include "geoar/core/map.h"

using namespace Eigen;
using json = nlohmann::json;
using namespace geoar;


TEST(JsonTest, MapDeserialization) {
  // Given
  json map_json = json::parse("{\"landmarks\":[{\"desc\":\"IL5sAIARAME/AICBxTH+xx0BACAABnAEAAAAPIYASPcfAPj/QhACAP47XAcANwAAAMjxGOP//8dDYHj/AA==\",\"id\":19,\"position\":[28.978420115684386,9.0347303998687,-17.00002901344248]}]}");
  // When
  geoar::Map map = map_json;
  // Then
  std::vector<uint8_t> expected_desc = {32, 190, 108,   0, 128,  17,   0, 193,  63,   0, 128, 129, 197,  49, 254, 199,  29,   1,   0,  32,   0,   6, 112,   4,   0,   0,   0,  60, 134,   0,  72, 247,  31,   0, 248, 255,  66,  16,   2,   0, 254,  59,  92,   7,   0,  55,   0,   0,   0, 200, 241,  24, 227, 255, 255, 199,  67,  96, 120, 255,   0};
  cv::Mat actual_desc = map.landmarks[0].desc;

  // Check `desc` values
  for (int i = 0; i < 61; ++i) {
    EXPECT_EQ(actual_desc.at<uint8_t>(0,i), expected_desc[i]) << " index:" << i;
  }
  EXPECT_EQ(map.landmarks[0].id, 19);
  EXPECT_NEAR(map.landmarks[0].position.x(), 28.978420115684386, 1e-10);
  EXPECT_NEAR(map.landmarks[0].position.y(), 9.0347303998687, 1e-10);
  EXPECT_NEAR(map.landmarks[0].position.z(), -17.00002901344248, 1e-10);
}
