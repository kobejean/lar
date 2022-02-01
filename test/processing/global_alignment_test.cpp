#include <gtest/gtest.h>
#include "lar/core/utils/wgs84.h"
#include "lar/processing/global_alignment.h"

using namespace Eigen;
using json = nlohmann::json;
using namespace lar;

TEST(GlobalAlignmentTest, Centroids) {
  // Given
  Mapper::Data data;
  GPSObservation observation1;
  observation1.relative = { 0, -6,4 };
  observation1.global = { 37.5236728, 139.9380725, 212 };
  observation1.accuracy = { 10, 10, 30};
  data.gps_obs.push_back(observation1);

  GPSObservation observation2;
  observation2.relative = { 1820, -1, 0};
  observation2.global = { 37.5085404, 139.9300318, 208};
  observation2.accuracy = { 5, 5, 15};
  data.gps_obs.push_back(observation2);

  GlobalAlignment aligner(data);
  // When
  Vector3d rc, gc;
  aligner.centroids(rc, gc);
  // Then
  EXPECT_NEAR(rc.x(), 1456, 1e-5);
  EXPECT_NEAR(rc.y(), -2, 1e-5);
  EXPECT_NEAR(rc.z(), 0.8, 1e-5);
  EXPECT_NEAR(gc.x(), 37.5115669, 1e-5);
  EXPECT_NEAR(gc.y(), 139.9316399, 1e-5);
  EXPECT_NEAR(gc.z(), 208.8, 1e-5);
}

TEST(GlobalAlignmentTest, CrossCovariance) {
  // Given
  Mapper::Data data;
  GPSObservation observation1;
  observation1.relative = { 0, 4, 0 };
  observation1.global = { 37.5236728, 139.9380725, 212 };
  observation1.accuracy = { 10, 0, 0};
  data.gps_obs.push_back(observation1);

  GPSObservation observation2;
  observation2.relative = { 1287, 0, 1287 };
  observation2.global = { 37.5085404, 139.9300318, 208 };  // -0.0151324, -0.0080407, 62
  observation2.accuracy = { 5, 5, 15};
  data.gps_obs.push_back(observation2);

  GlobalAlignment aligner(data);
  Vector3d rc, gc;
  aligner.centroids(rc, gc);
  auto D = wgs84::wgs84_scaling(gc);
  // std::cout << "rc:" << std::endl << rc << std::endl;
  // std::cout << "gc:" << std::endl << gc << std::endl;
  Eigen::Matrix3d cc = aligner.crossCovariance(rc, gc, D);
  // When
  // std::cout << cc << std::endl;
  // Then
  // EXPECT_NEAR(rc.x(), 1456, 1e-5);
  // EXPECT_NEAR(rc.y(), -2, 1e-5);
  // EXPECT_NEAR(rc.z(), 0.8, 1e-5);
  // EXPECT_NEAR(gc.x(), 37.5115669, 1e-5);
  // EXPECT_NEAR(gc.y(), 139.9316399, 1e-5);
  // EXPECT_NEAR(gc.z(), 208.8, 1e-5);
}

TEST(GlobalAlignmentTest, UpdateAlignment) {
  // Given
  Mapper::Data data;
  GPSObservation observation1;
  observation1.relative = { 0, 4, 0 };
  observation1.global = { 37.5236728, 139.9380725, 212 };
  observation1.accuracy = { 10, 0, 0};
  data.gps_obs.push_back(observation1);

  GPSObservation observation2;
  observation2.relative = { 1287, 0, 1287 };
  observation2.global = { 37.5085404, 139.9300318, 208 };
  observation2.accuracy = { 5, 5, 15};
  data.gps_obs.push_back(observation2);

  GlobalAlignment aligner(data);
  // When
  aligner.updateAlignment();
  // Then
  // EXPECT_NEAR(rc.x(), 1456, 1e-5);
  // EXPECT_NEAR(rc.y(), -2, 1e-5);
  // EXPECT_NEAR(rc.z(), 0.8, 1e-5);
  // EXPECT_NEAR(gc.x(), 37.5115669, 1e-5);
  // EXPECT_NEAR(gc.y(), 139.9316399, 1e-5);
  // EXPECT_NEAR(gc.z(), 208.8, 1e-5);
}