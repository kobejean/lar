#include <gtest/gtest.h>
#include <Eigen/Core>

#include "geoar/mapping/location_matcher.h"

using namespace geoar;

TEST(LocationMatcherTest, NormalMatch) {
  // Given
  LocationMatcher location_matcher;
  // When
  location_matcher.recordLocation(2, Eigen::Vector3d(128,50,650) , Eigen::Vector3d(10,10,30));
  location_matcher.recordPosition(1, Eigen::Vector3d(1,2,3));
  location_matcher.recordPosition(5, Eigen::Vector3d(10,20,30));
  // Then
  EXPECT_EQ(location_matcher.matches.size(), 1);
  EXPECT_EQ(location_matcher.matches[0].timestamp, 2);
  EXPECT_NEAR(location_matcher.matches[0].relative.x(), 3.25, 1e-10);
  EXPECT_NEAR(location_matcher.matches[0].relative.y(), 6.5, 1e-10);
  EXPECT_NEAR(location_matcher.matches[0].relative.z(), 9.75, 1e-10);
  EXPECT_NEAR(location_matcher.matches[0].accuracy.x(), 10, 1e-10);
  EXPECT_NEAR(location_matcher.matches[0].accuracy.y(), 10, 1e-10);
  EXPECT_NEAR(location_matcher.matches[0].accuracy.z(), 30, 1e-10);
}

TEST(LocationMatcherTest, EarlyLocation) {
  // Given
  LocationMatcher location_matcher;
  // When
  location_matcher.recordLocation(2, Eigen::Vector3d(128,50,650) , Eigen::Vector3d(10,10,30));
  location_matcher.recordPosition(10, Eigen::Vector3d(1,2,3));
  location_matcher.recordPosition(14, Eigen::Vector3d(10,20,30));
  location_matcher.recordLocation(11, Eigen::Vector3d(128,50,650) , Eigen::Vector3d(5,5,30));
  // Then
  EXPECT_EQ(location_matcher.matches.size(), 1);
  EXPECT_EQ(location_matcher.matches[0].timestamp, 11);
  EXPECT_NEAR(location_matcher.matches[0].relative.x(), 3.25, 1e-10);
  EXPECT_NEAR(location_matcher.matches[0].accuracy.x(), 5, 1e-10);
}

TEST(LocationMatcherTest, LateLocation) {
  // Given
  LocationMatcher location_matcher;
  // When
  location_matcher.recordPosition(0, Eigen::Vector3d(-1,-2,-3));
  location_matcher.recordPosition(10, Eigen::Vector3d(1,2,3));
  location_matcher.recordLocation(11, Eigen::Vector3d(128,50,650) , Eigen::Vector3d(10,10,30));
  location_matcher.recordPosition(14, Eigen::Vector3d(10,20,30));
  // Then
  EXPECT_EQ(location_matcher.matches.size(), 1);
  EXPECT_EQ(location_matcher.matches[0].timestamp, 11);
  EXPECT_NEAR(location_matcher.matches[0].relative.x(), 3.25, 1e-10);
  EXPECT_NEAR(location_matcher.matches[0].accuracy.x(), 10, 1e-10);
}

TEST(LocationMatcherTest, DoubleLocation) {
  // Given
  LocationMatcher location_matcher;
  // When
  location_matcher.recordLocation(11, Eigen::Vector3d(128,50,650) , Eigen::Vector3d(10,10,30));
  location_matcher.recordPosition(0, Eigen::Vector3d(-1,-2,-3));
  location_matcher.recordPosition(10, Eigen::Vector3d(1,2,3));
  location_matcher.recordLocation(13, Eigen::Vector3d(128,50,650) , Eigen::Vector3d(5,5,30));
  location_matcher.recordPosition(14, Eigen::Vector3d(10,20,30));
  // Then
  EXPECT_EQ(location_matcher.matches.size(), 2);
  EXPECT_EQ(location_matcher.matches[0].timestamp, 11);
  EXPECT_NEAR(location_matcher.matches[0].relative.x(), 3.25, 1e-10);
  EXPECT_NEAR(location_matcher.matches[0].accuracy.x(), 10, 1e-10);
  EXPECT_EQ(location_matcher.matches[1].timestamp, 13);
  EXPECT_NEAR(location_matcher.matches[1].relative.x(), 7.75, 1e-10);
  EXPECT_NEAR(location_matcher.matches[1].accuracy.x(), 5, 1e-10);
}

TEST(LocationMatcherTest, UnmatchedLocations) {
  // Given
  LocationMatcher location_matcher;
  // When
  location_matcher.recordLocation(11, Eigen::Vector3d(128,50,650) , Eigen::Vector3d(10,10,30));
  location_matcher.recordPosition(0, Eigen::Vector3d(-1,-2,-3));
  location_matcher.recordPosition(10, Eigen::Vector3d(1,2,3));
  // Then
  EXPECT_EQ(location_matcher.matches.size(), 0);
}


TEST(LocationMatcherTest, ZeroDuration) {
  // Given
  LocationMatcher location_matcher;
  // When
  location_matcher.recordLocation(1, Eigen::Vector3d(128,50,650) , Eigen::Vector3d(10,10,30));
  location_matcher.recordPosition(1, Eigen::Vector3d(1,2,3));
  location_matcher.recordPosition(1, Eigen::Vector3d(10,20,30));
  // Then
  EXPECT_EQ(location_matcher.matches.size(), 1);
  EXPECT_EQ(location_matcher.matches[0].timestamp, 1);
  EXPECT_NEAR(location_matcher.matches[0].relative.x(), 5.5, 1e-10);
}

TEST(LocationMatcherTest, UnmatchedEarlyLocationZeroDuration) {
  // Given
  LocationMatcher location_matcher;
  // When
  location_matcher.recordLocation(0, Eigen::Vector3d(128,50,650) , Eigen::Vector3d(10,10,30));
  location_matcher.recordPosition(1, Eigen::Vector3d(1,2,3));
  location_matcher.recordPosition(1, Eigen::Vector3d(10,20,30));
  // Then
  EXPECT_EQ(location_matcher.matches.size(), 0);
}

TEST(LocationMatcherTest, UnmatchedLateLocationZeroDuration) {
  // Given
  LocationMatcher location_matcher;
  // When
  location_matcher.recordLocation(2, Eigen::Vector3d(128,50,650) , Eigen::Vector3d(10,10,30));
  location_matcher.recordPosition(1, Eigen::Vector3d(1,2,3));
  location_matcher.recordPosition(1, Eigen::Vector3d(10,20,30));
  // Then
  EXPECT_EQ(location_matcher.matches.size(), 0);
}