#include <gtest/gtest.h>
#include <fstream>
#include <opencv2/opencv.hpp>
#include <Eigen/Core>

#include "lar/core/utils/json.h"
#include "lar/mapping/mapper.h"

using namespace lar;

TEST(MapperTest, WriteMetadata) {
  // Given
  cv::Mat image = cv::imread("./test/_fixture/raw_map_data/00000004_image.jpeg", cv::IMREAD_GRAYSCALE);
  cv::Mat depth = cv::imread("./test/_fixture/raw_map_data/00000004_depth.pfm", cv::IMREAD_UNCHANGED);
  cv::Mat confidence = cv::imread("./test/_fixture/raw_map_data/00000004_confidence.pfm", cv::IMREAD_UNCHANGED);
  Eigen::Matrix3d intrinsics;
  intrinsics << 1,4,7,
                2,5,8,
                3,6,9;
  Eigen::Matrix4d extrinsics;
  extrinsics <<  .1, .2, .3, .4,
                 .5, .6, .7, .8,
                 .9,1.0,1.1,1.2,
                1.3,1.4,1.5,1.6;
  Frame frame;
  frame.timestamp=123456789123456789;
  frame.intrinsics=intrinsics;
  frame.extrinsics=extrinsics;
  Mapper mapper("./test/_fixture/output");
  Eigen::Vector3d relative(0.1,0.2,0.3);
  Eigen::Vector3d global(100.0,-50.0,500.0);
  Eigen::Vector3d accuracy(10.0,10.0,30.0);
  GPSObservation observation{
    .timestamp=987654321987654321,
    .relative=relative,
    .global=global,
    .accuracy=accuracy
  };
  mapper.data.gps_obs.push_back(observation);
  mapper.addFrame(frame, image, depth, confidence);
  // When
  mapper.writeMetadata();
  // Then
  std::ifstream frames_ifs("./test/_fixture/output/frames.json");
  std::vector<Frame> frames = nlohmann::json::parse(frames_ifs);
  EXPECT_EQ(frames[0].timestamp, 123456789123456789);
  EXPECT_NEAR(frames[0].intrinsics(0,0), 1, 1e-10);
  EXPECT_NEAR(frames[0].intrinsics(1,0), 2, 1e-10);
  EXPECT_NEAR(frames[0].intrinsics(2,0), 3, 1e-10);
  EXPECT_NEAR(frames[0].intrinsics(0,1), 4, 1e-10);
  EXPECT_NEAR(frames[0].intrinsics(1,1), 5, 1e-10);
  EXPECT_NEAR(frames[0].intrinsics(2,1), 6, 1e-10);
  EXPECT_NEAR(frames[0].intrinsics(0,2), 7, 1e-10);
  EXPECT_NEAR(frames[0].intrinsics(1,2), 8, 1e-10);
  EXPECT_NEAR(frames[0].intrinsics(2,2), 9, 1e-10);
  EXPECT_NEAR(frames[0].extrinsics(0,0), 0.1, 1e-10);
  EXPECT_NEAR(frames[0].extrinsics(0,1), 0.2, 1e-10);
  EXPECT_NEAR(frames[0].extrinsics(0,2), 0.3, 1e-10);
  EXPECT_NEAR(frames[0].extrinsics(0,3), 0.4, 1e-10);
  EXPECT_NEAR(frames[0].extrinsics(1,0), 0.5, 1e-10);
  EXPECT_NEAR(frames[0].extrinsics(1,1), 0.6, 1e-10);
  EXPECT_NEAR(frames[0].extrinsics(1,2), 0.7, 1e-10);
  EXPECT_NEAR(frames[0].extrinsics(1,3), 0.8, 1e-10);
  EXPECT_NEAR(frames[0].extrinsics(2,0), 0.9, 1e-10);
  EXPECT_NEAR(frames[0].extrinsics(2,1), 1.0, 1e-10);
  EXPECT_NEAR(frames[0].extrinsics(2,2), 1.1, 1e-10);
  EXPECT_NEAR(frames[0].extrinsics(2,3), 1.2, 1e-10);
  EXPECT_NEAR(frames[0].extrinsics(3,0), 1.3, 1e-10);
  EXPECT_NEAR(frames[0].extrinsics(3,1), 1.4, 1e-10);
  EXPECT_NEAR(frames[0].extrinsics(3,2), 1.5, 1e-10);
  EXPECT_NEAR(frames[0].extrinsics(3,3), 1.6, 1e-10);

  std::ifstream gps_ifs("./test/_fixture/output/gps.json");
  std::vector<GPSObservation> gps_obs = nlohmann::json::parse(gps_ifs);
  EXPECT_EQ(gps_obs[0].timestamp, 987654321987654321);
  EXPECT_NEAR(gps_obs[0].relative.x(), 0.1, 1e-10);
  EXPECT_NEAR(gps_obs[0].relative.y(), 0.2, 1e-10);
  EXPECT_NEAR(gps_obs[0].relative.z(), 0.3, 1e-10);
  EXPECT_NEAR(gps_obs[0].global.x(), 100.0, 1e-10);
  EXPECT_NEAR(gps_obs[0].global.y(), -50.0, 1e-10);
  EXPECT_NEAR(gps_obs[0].global.z(), 500.0, 1e-10);
  EXPECT_NEAR(gps_obs[0].accuracy.x(), 10.0, 1e-10);
  EXPECT_NEAR(gps_obs[0].accuracy.y(), 10.0, 1e-10);
  EXPECT_NEAR(gps_obs[0].accuracy.z(), 30.0, 1e-10);
}

TEST(MapperTest, ReadMetadata) {
  // Given
  Mapper mapper("./test/_fixture/raw_map_data");
  // When
  mapper.readMetadata();
  // Then
  EXPECT_EQ(mapper.data.frames[0].timestamp, 136463350);
  EXPECT_NEAR(mapper.data.frames[0].intrinsics(0,0), 1594.7247314453125, 1e-10);
  EXPECT_NEAR(mapper.data.frames[0].intrinsics(0,1), 0, 1e-10);
  EXPECT_NEAR(mapper.data.frames[0].intrinsics(0,2), 952.86053466796875, 1e-10);
  EXPECT_NEAR(mapper.data.frames[0].intrinsics(1,0), 0, 1e-10);
  EXPECT_NEAR(mapper.data.frames[0].intrinsics(1,1), 1594.7247314453125, 1e-10);
  EXPECT_NEAR(mapper.data.frames[0].intrinsics(1,2), 714.1676025390625, 1e-10);
  EXPECT_NEAR(mapper.data.frames[0].intrinsics(2,0), 0, 1e-10);
  EXPECT_NEAR(mapper.data.frames[0].intrinsics(2,1), 0, 1e-10);
  EXPECT_NEAR(mapper.data.frames[0].intrinsics(2,2), 1, 1e-10);
}