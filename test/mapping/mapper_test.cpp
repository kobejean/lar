#include <gtest/gtest.h>
#include <fstream>
#include <opencv2/opencv.hpp>
#include <Eigen/Core>

#include "geoar/core/utils/json.h"
#include "geoar/mapping/mapper.h"

using namespace geoar;

TEST(MapperTest, AddFrame) {
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
  Mapper::FrameMetadata metadata{
    .timestamp=123456789123456789,
    .intrinsics=intrinsics,
    .extrinsics=extrinsics
  };
  Mapper mapper("./test/_fixture/output");
  // When
  mapper.addFrame(image, depth, confidence, metadata);
  mapper.writeMetadata();
  // Then
  std::ifstream ifs("./test/_fixture/output/frames.json");
  std::vector<Mapper::FrameMetadata> frames = nlohmann::json::parse(ifs);
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
}

TEST(MapperTest, AddGPSObservation) {
  // Given
  Eigen::Vector3d relative(0.1,0.2,0.3);
  Eigen::Vector3d global(100.0,-50.0,500.0);
  Eigen::Vector3d accuracy(10.0,10.0,30.0);
  Mapper::GPSObservation observation{
    .timestamp=987654321987654321,
    .relative=relative,
    .global=global,
    .accuracy=accuracy
  };
  Mapper mapper("./test/_fixture/output");
  // When
  mapper.addGPSObservation(observation);
  mapper.writeMetadata();
  // Then
  std::ifstream ifs("./test/_fixture/output/gps_observations.json");
  std::vector<Mapper::GPSObservation> gps_observations = nlohmann::json::parse(ifs);
  EXPECT_EQ(gps_observations[0].timestamp, 987654321987654321);
  EXPECT_NEAR(gps_observations[0].relative.x(), 0.1, 1e-10);
  EXPECT_NEAR(gps_observations[0].relative.y(), 0.2, 1e-10);
  EXPECT_NEAR(gps_observations[0].relative.z(), 0.3, 1e-10);
  EXPECT_NEAR(gps_observations[0].global.x(), 100.0, 1e-10);
  EXPECT_NEAR(gps_observations[0].global.y(), -50.0, 1e-10);
  EXPECT_NEAR(gps_observations[0].global.z(), 500.0, 1e-10);
  EXPECT_NEAR(gps_observations[0].accuracy.x(), 10.0, 1e-10);
  EXPECT_NEAR(gps_observations[0].accuracy.y(), 10.0, 1e-10);
  EXPECT_NEAR(gps_observations[0].accuracy.z(), 30.0, 1e-10);
}