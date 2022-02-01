/*
 * MIT License
 *
 * Copyright (c) 2018  Christian Berger
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

#include <gtest/gtest.h>
#include <Eigen/Core>
#include "lar/core/utils/wgs84.h"

using namespace lar;

TEST(WGS84Test, ToCartesianIdenticalPosition) {
  // Given
  const Eigen::Vector2d wgs84_reference{52.247041, 10.575830};
  // When
  Eigen::Vector2d result = wgs84::to_cartesian(wgs84_reference, wgs84_reference);
  // Then
  EXPECT_NEAR(result.x(), 0, 1e-10);
  EXPECT_NEAR(result.y(), 0, 1e-10);
}

TEST(WGS84Test, ToCartesianExample1) {
  // Given
  const Eigen::Vector2d wgs84_reference{52.247041, 10.575830};
  const Eigen::Vector2d wgs84_position{52.248091, 10.57417};
  // When
  Eigen::Vector2d result = wgs84::to_cartesian(wgs84_reference, wgs84_position);
  // Then
  EXPECT_NEAR(result.x(), -113.3742031902, 1e-10);
  EXPECT_NEAR(result.y(), 116.8369533306, 1e-10);
}

TEST(WGS84Test, ToCartesianExample2) {
  // Given
  const Eigen::Vector2d wgs84_reference{52.247041, 10.575830};
  const Eigen::Vector2d wgs84_position{52.251011, 10.573568};
  // When
  Eigen::Vector2d result = wgs84::to_cartesian(wgs84_reference, wgs84_position);
  // Then
  EXPECT_NEAR(result.x(), -154.4792838235449, 1e-10);
  EXPECT_NEAR(result.y(), 441.75256808963206, 1e-10);
}

TEST(WGS84Test, FromCartesianIdenticalPosition) {
  // Given
  const Eigen::Vector2d wgs84_reference{52.247041, 10.575830};
  const Eigen::Vector2d cartesian_position{0.0, 0.0};
  // When
  Eigen::Vector2d result = wgs84::from_cartesian(wgs84_reference, cartesian_position);
  // Then
  EXPECT_NEAR(result.x(), 52.247041, 1e-10);
  EXPECT_NEAR(result.y(), 10.575830, 1e-10);
}

TEST(WGS84Test, FromCartesianExample1) {
  // Given
  const Eigen::Vector2d wgs84_reference{52.247041, 10.575830};
  const Eigen::Vector2d cartesian_position{-154.48, 441.75};
  // When
  Eigen::Vector2d result = wgs84::from_cartesian(wgs84_reference, cartesian_position);
  // Then
  EXPECT_NEAR(result.x(), 52.2510109984, 1e-10);
  EXPECT_NEAR(result.y(), 10.5735679901, 1e-10);
}

TEST(WGS84Test, FromCartesianExample2) {
  // Given
  const Eigen::Vector2d wgs84_reference{52.247041, 10.575830};
  const Eigen::Vector2d cartesian_position{-208.57, 431.07};
  // When
  Eigen::Vector2d result = wgs84::from_cartesian(wgs84_reference, cartesian_position);
  // Then
  EXPECT_NEAR(result.x(), 52.2509150177, 1e-10);
  EXPECT_NEAR(result.y(), 10.5727759711, 1e-10);
}


TEST(WGS84Test, WGS84Scaling) {
  // Given
  const Eigen::Vector3d wgs84_reference{37.5115669, 139.9316399, 212};
  // When
  auto result = wgs84::wgs84_scaling(wgs84_reference);
  // Then
  auto diagonal = result.diagonal();
  EXPECT_NEAR(diagonal.x(), 110996.07085693209, 1e-10);
  EXPECT_NEAR(diagonal.y(), 88419.017880293017, 1e-10);
  EXPECT_NEAR(diagonal.z(), 1, 1e-10);
}