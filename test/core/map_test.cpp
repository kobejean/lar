#include <gtest/gtest.h>
#include <fstream>
#include "lar/core/map.h"

using namespace lar;

// TEST(MapTest, GlobalPointNotReady) {
//   // Given
//   Map map;
//   Eigen::Vector3d relative{ 0, 4, 0 };
//   // When
//   Eigen::Vector3d global;
//   bool success = map.globalPointFrom(relative, global);
//   // Then
//   EXPECT_FALSE(success);
// }

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
  map.globalPointFrom(relative, global);
  // Then
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
  map.globalPointFrom(relative, global);
  // Then
  EXPECT_NEAR(global.x(), 37.5085404, 1e-5);
  EXPECT_NEAR(global.y(), 139.9300318, 1e-5);
  EXPECT_NEAR(global.z(), 208, 1e-5);
}

// TEST(MapTest, RelativePointNotReady) {
//   // Given
//   Map map;
//   Eigen::Vector3d global{ 37.5236728, 139.9380725, 212 };
//   // When
//   Eigen::Vector3d relative;
//   map.relativePointFrom(relative, global);
//   // Then
//   EXPECT_FALSE(success);
// }

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
  map.relativePointFrom(global, relative);
  // Then
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
  map.relativePointFrom(global, relative);
  // Then
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
  // size_t id = 45347;
  // std::vector<uint8_t> expected_desc = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 64, 64, 128, 63, 64, 64, 64, 64, 128, 63};
  // cv::Mat actual_desc = map.landmarks[710].desc;

  // // Check `desc` values
  // for (int i = 0; i < actual_desc.cols; ++i) {
  //   EXPECT_EQ(actual_desc.at<uint8_t>(0,i), expected_desc[i]) << " index:" << i;
  // }
  // EXPECT_EQ(actual_desc.rows, 1);
  // EXPECT_EQ(actual_desc.cols, expected_desc.size());

  // EXPECT_EQ(map.landmarks[710].id, 710);
  // EXPECT_NEAR(map.landmarks[710].position.x(), 28.99345023444182, 1e-10);
  // EXPECT_NEAR(map.landmarks[710].position.y(), 8.76631960321935, 1e-10);
  // EXPECT_NEAR(map.landmarks[710].position.z(), -17.06995683511346, 1e-10);
}