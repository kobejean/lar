#include <gtest/gtest.h>
#include <fstream>
#include <opencv2/opencv.hpp>

#include "geoar/core/utils/json.h"
#include "geoar/tracking/tracker.h"

using namespace geoar;

TEST(TrackerTest, LocalizeWithTransform) {
  // Given
  std::ifstream map_data_ifs("./test/_fixture/processed_map_data/map.json");
  geoar::Map map = nlohmann::json::parse(map_data_ifs);
  cv::Mat image = cv::imread("./test/_fixture/raw_map_data/00000004_image.jpeg", cv::IMREAD_GRAYSCALE);
  cv::Mat intrinsics(3, 3, CV_32FC1);
  intrinsics.at<float>(0,0) = 1594.2728271484375;
  intrinsics.at<float>(1,1) = 1594.2728271484375;
  intrinsics.at<float>(0,2) = 952.8714599609375;
  intrinsics.at<float>(1,2) = 714.1612548828125;
  intrinsics.at<float>(2,2) = 1.;
  Tracker tracker(map);
  // When
  cv::Mat transform;
  tracker.localize(image, intrinsics, transform);
  // Then
  EXPECT_NEAR(transform.at<double>(0,0), -0.27073314785957336, 1e-2);
  EXPECT_NEAR(transform.at<double>(0,1), 0.05986318364739418, 1e-2);
  EXPECT_NEAR(transform.at<double>(0,2), -0.96079128980636597, 1e-2);
  EXPECT_NEAR(transform.at<double>(0,3), -2.8749935626983643, 1e+0);
  EXPECT_NEAR(transform.at<double>(1,0), -0.021625135093927383, 1e-2);
  EXPECT_NEAR(transform.at<double>(1,1), 0.99743443727493286, 1e-2);
  EXPECT_NEAR(transform.at<double>(1,2), 0.068239845335483551, 1e-2);
  EXPECT_NEAR(transform.at<double>(1,3), 0.084447085857391357, 1e+0);
  EXPECT_NEAR(transform.at<double>(2,0), 0.96241146326065063, 1e-2);
  EXPECT_NEAR(transform.at<double>(2,1), 0.03925202414393425, 1e-2);
  EXPECT_NEAR(transform.at<double>(2,2), -0.26874402165412903, 1e-2);
  EXPECT_NEAR(transform.at<double>(2,3), -27.358242034912109, 1e+0);
  EXPECT_NEAR(transform.at<double>(3,0), 0., 1e-5);
  EXPECT_NEAR(transform.at<double>(3,1), 0., 1e-5);
  EXPECT_NEAR(transform.at<double>(3,2), 0., 1e-5);
  EXPECT_NEAR(transform.at<double>(3,3), 1., 1e-5);
}