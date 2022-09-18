#include <gtest/gtest.h>
#include <fstream>
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

TEST(MapTest, JSONSerialization) {
  // Given
  std::ifstream ifs("./test/_fixture/processed_map_data/map.json");
  std::string json_string((std::istreambuf_iterator<char>(ifs)), std::istreambuf_iterator<char>());
  lar::Map map = nlohmann::json::parse(json_string);
  // When
  std::cout << "Serializing to JSON string " << std::endl;
  nlohmann::json map_json = map;
  // Then
  EXPECT_EQ(map_json.dump(2).substr(0,1000), json_string.substr(0,1000));
}

TEST(MapTest, JSONDeserialization) {
  // Given
  std::ifstream ifs("./test/_fixture/processed_map_data/map.json");
  std::string json_string((std::istreambuf_iterator<char>(ifs)), std::istreambuf_iterator<char>());
  nlohmann::json map_json = nlohmann::json::parse(json_string);
  // When
  lar::Map map = map_json;
  // Then
  std::vector<uint8_t> expected_desc = {32, 190, 108,   0, 128,  17,   0, 193,  63,   0, 128, 129, 197,  49, 254, 199,  29,   1,   0,  32,   0,   6, 112,   4,   0,   0,   0,  60, 134,   0,  72, 247,  31,   0, 248, 255,  66,  16,   2,   0, 254,  59,  92,   7,   0,  55,   0,   0,   0, 200, 241,  24, 227, 255, 255, 199,  67,  96, 120, 255,   0};
  cv::Mat actual_desc = map.landmarks[19].desc;

  // Check `desc` values
  for (int i = 0; i < 61; ++i) {
    EXPECT_EQ(actual_desc.at<uint8_t>(0,i), expected_desc[i]) << " index:" << i;
  }
  EXPECT_EQ(actual_desc.rows, 1);
  EXPECT_EQ(actual_desc.cols, expected_desc.size());

  EXPECT_EQ(map.landmarks[19].id, 19);
  EXPECT_NEAR(map.landmarks[19].position.x(), 28.93247239591809, 1e-10);
  EXPECT_NEAR(map.landmarks[19].position.y(), 8.7041057413508671, 1e-10);
  EXPECT_NEAR(map.landmarks[19].position.z(), -17.020310022134733, 1e-10);
}