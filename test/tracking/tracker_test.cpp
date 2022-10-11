#include <gtest/gtest.h>
#include <fstream>
#include <opencv2/opencv.hpp>

#include "lar/core/utils/json.h"
#include "lar/tracking/tracker.h"

using namespace lar;

// TEST(TrackerTest, LocalizeWithTransform) {
//   // Given
//   std::ifstream map_data_ifs("./test/_fixture/processed_map_data/map.json");
//   lar::Map map = nlohmann::json::parse(map_data_ifs);
//   cv::Mat image = cv::imread("./test/_fixture/raw_map_data/00000004_image.jpeg", cv::IMREAD_GRAYSCALE);
//   cv::Mat intrinsics = (cv::Mat_<float>(3,3) << 1594.2728271484375, 0., 952.8714599609375,
//                                                 0., 1594.2728271484375, 714.1612548828125, 
//                                                 0., 0., 1.);
//   cv::Mat transform = (cv::Mat_<double>(4,4) << 1., 0., 0., -3.,
//                                                 0., 1., 0., -0.,
//                                                 0., 0., 1., -27.,
//                                                 0., 0., 0., 1.);
//   Tracker tracker(map);
//   // When
//   tracker.localize(image, intrinsics, transform);
//   // Then
//   EXPECT_NEAR(transform.at<double>(0,0), -0.27073314785957336, 1e-2);
//   EXPECT_NEAR(transform.at<double>(0,1), 0.05986318364739418, 1e-2);
//   EXPECT_NEAR(transform.at<double>(0,2), -0.960791289806366, 1e-2);
//   EXPECT_NEAR(transform.at<double>(0,3), -2.8749935626983643, 1e+0);
//   EXPECT_NEAR(transform.at<double>(1,0), -0.021625135093927383, 1e-2);
//   EXPECT_NEAR(transform.at<double>(1,1), 0.9974344372749329, 1e-2);
//   EXPECT_NEAR(transform.at<double>(1,2), 0.06823984533548355, 1e-2);
//   EXPECT_NEAR(transform.at<double>(1,3), 0.08444708585739136, 1e+0);
//   EXPECT_NEAR(transform.at<double>(2,0), 0.9624114632606506, 1e-2);
//   EXPECT_NEAR(transform.at<double>(2,1), 0.03925202414393425, 1e-2);
//   EXPECT_NEAR(transform.at<double>(2,2), -0.26874402165412903, 1e-2);
//   EXPECT_NEAR(transform.at<double>(2,3), -27.35824203491211, 1e+0);
//   EXPECT_NEAR(transform.at<double>(3,0), 0., 1e-5);
//   EXPECT_NEAR(transform.at<double>(3,1), 0., 1e-5);
//   EXPECT_NEAR(transform.at<double>(3,2), 0., 1e-5);
//   EXPECT_NEAR(transform.at<double>(3,3), 1., 1e-5);

//   // -0.27073314785957336,-0.021625135093927383,0.9624114632606506,0.0,
//   // 0.05986318364739418,0.9974344372749329,0.03925202414393425,0.0,
//   // -0.960791289806366,0.06823984533548355,-0.26874402165412903,0.0,
//   // -2.8749935626983643,0.08444708585739136,-27.35824203491211,0.9999999403953552
// }