#include <gtest/gtest.h>
#include "lar/core/utils/wgs84.h"
#include "lar/processing/global_alignment.h"

using namespace Eigen;
using json = nlohmann::json;
using namespace lar;

TEST(GlobalAlignmentTest, Centroids) {
  // Given
  std::shared_ptr<Mapper::Data> data = std::make_shared<Mapper::Data>();
  GPSObservation observation1;
  observation1.relative = { 0, -6,4 };
  observation1.global = { 37.5236728, 139.9380725, 212 };
  observation1.accuracy = { 10, 10, 30};
  data->gps_obs.push_back(observation1);

  GPSObservation observation2;
  observation2.relative = { 1823.9011198309, -1, 0};
  observation2.global = { 37.5085404, 139.9300318, 208};
  observation2.accuracy = { 5, 5, 15 };
  data->gps_obs.push_back(observation2);

  GlobalAlignment aligner(data);
  // When
  Vector3d rc, gc;
  aligner.centroids(rc, gc);
  // Then
  EXPECT_NEAR(rc.x(), 1459.1208958647201, 1e-10);
  EXPECT_NEAR(rc.y(), -2, 1e-10);
  EXPECT_NEAR(rc.z(), 0.8, 1e-10);
  EXPECT_NEAR(gc.x(), 37.51156688, 1e-10);
  EXPECT_NEAR(gc.y(), 139.93163994, 1e-10);
  EXPECT_NEAR(gc.z(), 208.8, 1e-10);
}

TEST(GlobalAlignmentTest, CrossCovariance) {
  // Given
  std::shared_ptr<Mapper::Data> data = std::make_shared<Mapper::Data>();
  GPSObservation observation1;
  observation1.relative = { 0, 4, 0 };
  observation1.global = { 37.5236728, 139.9380725, 212 };
  observation1.accuracy = { 10, 0, 0 };
  data->gps_obs.push_back(observation1);

  GPSObservation observation2;
  observation2.relative = { 1289.6959515561, 0, 1289.6959515561 };
  observation2.global = { 37.5085404, 139.9300318, 208 };  // -0.0151324, -0.0080407, 62
  observation2.accuracy = { 5, 5, 15 };
  data->gps_obs.push_back(observation2);

  GlobalAlignment aligner(data);
  Vector3d rc, gc;
  aligner.centroids(rc, gc);
  auto D = wgs84::wgs84_scaling(gc);
  // When
  Eigen::Matrix3d CC = aligner.crossCovariance(rc, gc, D);
  // Then - Updated values after fixing WGS84 precision from 1e-4 to 1e-10
  EXPECT_NEAR(CC(0,0), -17325.921528590075, 1e-10);
  EXPECT_NEAR(CC(0,1), 0, 1e-10);
  EXPECT_NEAR(CC(0,2), -17325.921528590075, 1e-10);
  EXPECT_NEAR(CC(1,0), -7333.9860039396517, 1e-10);
  EXPECT_NEAR(CC(1,1), 0, 1e-10);
  EXPECT_NEAR(CC(1,2), -7333.9860039396517, 1e-10);
  EXPECT_NEAR(CC(2,0), 0, 1e-10);
  EXPECT_NEAR(CC(2,1), 0.128, 1e-10);
  EXPECT_NEAR(CC(2,2), 0, 1e-10);
}

TEST(GlobalAlignmentTest, UpdateAlignment) {
  // Given
  std::shared_ptr<Mapper::Data> data = std::make_shared<Mapper::Data>();
  GPSObservation observation1;
  observation1.relative = { 0, 4, 0 };
  observation1.global = { 37.5236728, 139.9380725, 212 };
  observation1.accuracy = { 10, 0, 0 };
  data->gps_obs.push_back(observation1);

  GPSObservation observation2;
  observation2.relative = { 1289.6959515561, 0, 1289.6959515561 };
  observation2.global = { 37.5085404, 139.9300318, 208 };
  observation2.accuracy = { 5, 5, 15};
  data->gps_obs.push_back(observation2);

  GlobalAlignment aligner(data);
  // When
  aligner.updateAlignment();
  // Then - Updated values after fixing WGS84 precision from 1e-4 to 1e-10
  auto origin = data->map.origin.matrix();
  EXPECT_NEAR(origin(0,0), -3.3840531835428795e-06, 1e-15);
  EXPECT_NEAR(origin(0,1), 0, 1e-10);
  EXPECT_NEAR(origin(0,2), -8.3517791308255934e-06, 1e-15);
  EXPECT_NEAR(origin(0,3), 37.52367540433918, 1e-10);
  EXPECT_NEAR(origin(1,0), -1.0483862418883284e-05, 1e-15);
  EXPECT_NEAR(origin(1,1), 0, 1e-10);
  EXPECT_NEAR(origin(1,2), 4.2479509382021304e-06, 1e-15);
  EXPECT_NEAR(origin(1,3), 139.93807388383271, 1e-10);
  EXPECT_NEAR(origin(2,0), 0, 1e-10);
  EXPECT_NEAR(origin(2,1), 1, 1e-10);
  EXPECT_NEAR(origin(2,2), 0, 1e-10);
  EXPECT_NEAR(origin(2,3), 208, 1e-10);
  EXPECT_NEAR(origin(3,0), 0, 1e-10);
  EXPECT_NEAR(origin(3,1), 0, 1e-10);
  EXPECT_NEAR(origin(3,2), 0, 1e-10);
  EXPECT_NEAR(origin(3,3), 1, 1e-8);
}