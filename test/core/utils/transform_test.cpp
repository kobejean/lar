/*
 * MIT License
 *
 * Copyright (c) 2024 LAR Team
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
#include <Eigen/Geometry>
#include <cmath>
#include "lar/core/utils/transform.h"

using namespace lar;

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

class TransformUtilsTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Tolerance for floating point comparisons
        tolerance_ = 1e-10;

        // Common test data
        identity_transform_ = Eigen::Matrix4d::Identity();

        // Simple translation transform
        translation_transform_ = Eigen::Matrix4d::Identity();
        translation_transform_(0, 3) = 1.0; // X translation
        translation_transform_(1, 3) = 2.0; // Y translation
        translation_transform_(2, 3) = 3.0; // Z translation

        // Simple rotation transform (90° around Z-axis)
        rotation_transform_ = Eigen::Matrix4d::Identity();
        rotation_transform_.block<3,3>(0,0) = Eigen::AngleAxisd(M_PI/2, Eigen::Vector3d::UnitZ()).toRotationMatrix();

        // Combined translation and rotation transform
        combined_transform_ = Eigen::Matrix4d::Identity();
        combined_transform_.block<3,3>(0,0) = Eigen::AngleAxisd(M_PI/4, Eigen::Vector3d::UnitY()).toRotationMatrix();
        combined_transform_(0, 3) = 5.0;
        combined_transform_(1, 3) = -2.0;
        combined_transform_(2, 3) = 1.5;
    }

    double tolerance_;
    Eigen::Matrix4d identity_transform_;
    Eigen::Matrix4d translation_transform_;
    Eigen::Matrix4d rotation_transform_;
    Eigen::Matrix4d combined_transform_;
};

// ============================================================================
// Rotation Matrix ↔ Axis-Angle Conversion Tests
// ============================================================================

TEST_F(TransformUtilsTest, RotationMatrixToAxisAngleIdentity) {
    // Given
    Eigen::Matrix3d identity = Eigen::Matrix3d::Identity();

    // When
    Eigen::Vector3d axis_angle = utils::TransformUtils::rotationMatrixToAxisAngle(identity);

    // Then
    EXPECT_NEAR(axis_angle.norm(), 0.0, tolerance_);
}

TEST_F(TransformUtilsTest, AxisAngleToRotationMatrixZeroRotation) {
    // Given
    Eigen::Vector3d zero_rotation = Eigen::Vector3d::Zero();

    // When
    Eigen::Matrix3d result = utils::TransformUtils::axisAngleToRotationMatrix(zero_rotation);

    // Then
    EXPECT_TRUE(result.isApprox(Eigen::Matrix3d::Identity(), tolerance_));
}

TEST_F(TransformUtilsTest, RotationConversionRoundTrip90DegreesZ) {
    // Given - 90 degree rotation around Z axis
    Eigen::Vector3d original_axis_angle(0, 0, M_PI/2);

    // When - Convert to matrix and back
    Eigen::Matrix3d rotation_matrix = utils::TransformUtils::axisAngleToRotationMatrix(original_axis_angle);
    Eigen::Vector3d recovered_axis_angle = utils::TransformUtils::rotationMatrixToAxisAngle(rotation_matrix);

    // Then
    EXPECT_NEAR(recovered_axis_angle(0), original_axis_angle(0), tolerance_);
    EXPECT_NEAR(recovered_axis_angle(1), original_axis_angle(1), tolerance_);
    EXPECT_NEAR(recovered_axis_angle(2), original_axis_angle(2), tolerance_);
}

TEST_F(TransformUtilsTest, RotationConversionRoundTrip45DegreesY) {
    // Given - 45 degree rotation around Y axis
    Eigen::Vector3d original_axis_angle(0, M_PI/4, 0);

    // When - Convert to matrix and back
    Eigen::Matrix3d rotation_matrix = utils::TransformUtils::axisAngleToRotationMatrix(original_axis_angle);
    Eigen::Vector3d recovered_axis_angle = utils::TransformUtils::rotationMatrixToAxisAngle(rotation_matrix);

    // Then
    EXPECT_NEAR(recovered_axis_angle(0), original_axis_angle(0), tolerance_);
    EXPECT_NEAR(recovered_axis_angle(1), original_axis_angle(1), tolerance_);
    EXPECT_NEAR(recovered_axis_angle(2), original_axis_angle(2), tolerance_);
}

TEST_F(TransformUtilsTest, RotationConversionArbitraryAxis) {
    // Given - Rotation around arbitrary axis
    Eigen::Vector3d axis(1, 2, 3);
    axis.normalize();
    double angle = M_PI/3; // 60 degrees
    Eigen::Vector3d original_axis_angle = axis * angle;

    // When - Convert to matrix and back
    Eigen::Matrix3d rotation_matrix = utils::TransformUtils::axisAngleToRotationMatrix(original_axis_angle);
    Eigen::Vector3d recovered_axis_angle = utils::TransformUtils::rotationMatrixToAxisAngle(rotation_matrix);

    // Then
    EXPECT_NEAR(recovered_axis_angle(0), original_axis_angle(0), tolerance_);
    EXPECT_NEAR(recovered_axis_angle(1), original_axis_angle(1), tolerance_);
    EXPECT_NEAR(recovered_axis_angle(2), original_axis_angle(2), tolerance_);
}

// ============================================================================
// Transform Validation Tests
// ============================================================================

TEST_F(TransformUtilsTest, ValidateTransformMatrixIdentity) {
    // When/Then
    EXPECT_TRUE(utils::TransformUtils::validateTransformMatrix(identity_transform_, "Identity"));
}

TEST_F(TransformUtilsTest, ValidateTransformMatrixValidTranslation) {
    // When/Then
    EXPECT_TRUE(utils::TransformUtils::validateTransformMatrix(translation_transform_, "Translation"));
}

TEST_F(TransformUtilsTest, ValidateTransformMatrixValidRotation) {
    // When/Then
    EXPECT_TRUE(utils::TransformUtils::validateTransformMatrix(rotation_transform_, "Rotation"));
}

TEST_F(TransformUtilsTest, ValidateTransformMatrixValidCombined) {
    // When/Then
    EXPECT_TRUE(utils::TransformUtils::validateTransformMatrix(combined_transform_, "Combined"));
}

TEST_F(TransformUtilsTest, ValidateTransformMatrixInvalidBottomRow) {
    // Given - Invalid bottom row
    Eigen::Matrix4d invalid_transform = identity_transform_;
    invalid_transform(3, 0) = 0.5; // Should be 0

    // When/Then
    EXPECT_FALSE(utils::TransformUtils::validateTransformMatrix(invalid_transform, "Invalid"));
}

TEST_F(TransformUtilsTest, ValidateTransformMatrixNonOrthogonalRotation) {
    // Given - Non-orthogonal rotation matrix
    Eigen::Matrix4d invalid_transform = identity_transform_;
    invalid_transform(0, 0) = 2.0; // Scale instead of rotation

    // When/Then
    EXPECT_FALSE(utils::TransformUtils::validateTransformMatrix(invalid_transform, "NonOrthogonal"));
}

// ============================================================================
// Extract/Create Transform Component Tests
// ============================================================================

TEST_F(TransformUtilsTest, ExtractPositionFromTranslation) {
    // When
    Eigen::Vector3d position = utils::TransformUtils::extractPosition(translation_transform_);

    // Then
    EXPECT_NEAR(position(0), 1.0, tolerance_);
    EXPECT_NEAR(position(1), 2.0, tolerance_);
    EXPECT_NEAR(position(2), 3.0, tolerance_);
}

TEST_F(TransformUtilsTest, ExtractRotationFromRotation) {
    // When
    Eigen::Matrix3d rotation = utils::TransformUtils::extractRotation(rotation_transform_);

    // Then - Should be 90 degree rotation around Z
    Eigen::Vector3d x_axis_rotated = rotation * Eigen::Vector3d::UnitX();
    EXPECT_NEAR(x_axis_rotated(0), 0.0, tolerance_);
    EXPECT_NEAR(x_axis_rotated(1), 1.0, tolerance_);
    EXPECT_NEAR(x_axis_rotated(2), 0.0, tolerance_);
}

TEST_F(TransformUtilsTest, CreateTransformFromComponents) {
    // Given
    Eigen::Vector3d position(5.0, -3.0, 2.0);
    Eigen::Matrix3d rotation = Eigen::AngleAxisd(M_PI/6, Eigen::Vector3d::UnitX()).toRotationMatrix();

    // When
    Eigen::Matrix4d result = utils::TransformUtils::createTransform(position, rotation);

    // Then
    EXPECT_TRUE(utils::TransformUtils::validateTransformMatrix(result, "Created"));
    EXPECT_TRUE(utils::TransformUtils::extractPosition(result).isApprox(position, tolerance_));
    EXPECT_TRUE(utils::TransformUtils::extractRotation(result).isApprox(rotation, tolerance_));
}

TEST_F(TransformUtilsTest, CreateTransformFromPositionAndAxisAngle) {
    // Given
    Eigen::Vector3d position(1.0, 2.0, 3.0);
    Eigen::Vector3d orientation(M_PI/4, 0, 0); // 45 degrees around X

    // When
    Eigen::Matrix4d result = utils::TransformUtils::createTransform(position, orientation);

    // Then
    EXPECT_TRUE(utils::TransformUtils::validateTransformMatrix(result, "CreatedFromAxisAngle"));
    EXPECT_TRUE(utils::TransformUtils::extractPosition(result).isApprox(position, tolerance_));

    // Verify rotation by checking axis-angle conversion
    Eigen::Matrix3d extracted_rotation = utils::TransformUtils::extractRotation(result);
    Eigen::Vector3d recovered_orientation = utils::TransformUtils::rotationMatrixToAxisAngle(extracted_rotation);
    EXPECT_TRUE(recovered_orientation.isApprox(orientation, tolerance_));
}

// ============================================================================
// Transform Comparison Tests
// ============================================================================

TEST_F(TransformUtilsTest, IsApproximatelyEqualIdentical) {
    // When/Then
    EXPECT_TRUE(utils::TransformUtils::isApproximatelyEqual(identity_transform_, identity_transform_));
}

TEST_F(TransformUtilsTest, IsApproximatelyEqualWithinTolerance) {
    // Given
    Eigen::Matrix4d slightly_different = identity_transform_;
    slightly_different(0, 3) = 1e-8; // Very small difference

    // When/Then
    EXPECT_TRUE(utils::TransformUtils::isApproximatelyEqual(identity_transform_, slightly_different, 1e-6));
    EXPECT_FALSE(utils::TransformUtils::isApproximatelyEqual(identity_transform_, slightly_different, 1e-10));
}

TEST_F(TransformUtilsTest, IsApproximatelyEqualDifferent) {
    // When/Then
    EXPECT_FALSE(utils::TransformUtils::isApproximatelyEqual(identity_transform_, translation_transform_));
}

// ============================================================================
// Real-World Camera Pose Scenarios
// ============================================================================

TEST_F(TransformUtilsTest, CameraPoseScenario) {
    // Given - Realistic camera pose (position + orientation)
    Eigen::Vector3d camera_position(2.5, -1.2, 0.8);
    Eigen::Vector3d camera_orientation(0.1, -0.3, 0.05); // Small rotations typical for camera

    // When - Create transform and extract components back
    Eigen::Matrix4d camera_transform = utils::TransformUtils::createTransform(camera_position, camera_orientation);
    Eigen::Vector3d extracted_position = utils::TransformUtils::extractPosition(camera_transform);
    Eigen::Matrix3d extracted_rotation = utils::TransformUtils::extractRotation(camera_transform);
    Eigen::Vector3d extracted_orientation = utils::TransformUtils::rotationMatrixToAxisAngle(extracted_rotation);

    // Then
    EXPECT_TRUE(utils::TransformUtils::validateTransformMatrix(camera_transform, "CameraPose"));
    EXPECT_TRUE(extracted_position.isApprox(camera_position, tolerance_));
    EXPECT_TRUE(extracted_orientation.isApprox(camera_orientation, tolerance_));
}

TEST_F(TransformUtilsTest, VIOToLARCoordinateTransform) {
    // Given - Simulate VIO to LAR coordinate transform
    Eigen::Vector3d vio_position(10.0, 5.0, 2.0);
    Eigen::Vector3d vio_orientation(0.0, 0.0, M_PI); // 180 degree rotation around Z

    Eigen::Matrix4d T_vio_from_camera = utils::TransformUtils::createTransform(vio_position, vio_orientation);

    // Create LAR camera pose (different coordinate frame)
    Eigen::Vector3d lar_position(-8.0, 12.0, 1.8);
    Eigen::Vector3d lar_orientation(0.1, -0.2, M_PI + 0.1); // Slightly different rotation

    Eigen::Matrix4d T_lar_from_camera = utils::TransformUtils::createTransform(lar_position, lar_orientation);

    // When - Compute coordinate transform
    Eigen::Matrix4d T_vio_from_lar = T_vio_from_camera * T_lar_from_camera.inverse();

    // Then
    EXPECT_TRUE(utils::TransformUtils::validateTransformMatrix(T_vio_from_lar, "VIOToLAR"));

    // Verify the transform makes sense by applying it
    Eigen::Vector4d lar_point(1.0, 0.0, 0.0, 1.0);
    Eigen::Vector4d vio_point = T_vio_from_lar * lar_point;

    // Should be a valid 3D point in homogeneous coordinates
    EXPECT_NEAR(vio_point(3), 1.0, tolerance_);
}

// ============================================================================
// Edge Cases and Robustness Tests
// ============================================================================

TEST_F(TransformUtilsTest, SmallAngleRotation) {
    // Given - Very small rotation (testing numerical stability)
    Eigen::Vector3d small_rotation(1e-8, 2e-8, -1e-8);

    // When
    Eigen::Matrix3d rotation_matrix = utils::TransformUtils::axisAngleToRotationMatrix(small_rotation);
    Eigen::Vector3d recovered = utils::TransformUtils::rotationMatrixToAxisAngle(rotation_matrix);

    // Then
    EXPECT_TRUE(recovered.isApprox(small_rotation, 1e-10));
}

TEST_F(TransformUtilsTest, LargeAngleRotation) {
    // Given - Large rotation (> 180 degrees)
    Eigen::Vector3d large_rotation(0, 0, 7*M_PI/4); // 315 degrees

    // When
    Eigen::Matrix3d rotation_matrix = utils::TransformUtils::axisAngleToRotationMatrix(large_rotation);
    Eigen::Vector3d recovered = utils::TransformUtils::rotationMatrixToAxisAngle(rotation_matrix);

    // Then - Should be equivalent to -45 degrees (shortest rotation)
    double recovered_angle = recovered.norm();
    EXPECT_NEAR(recovered_angle, M_PI/4, 1e-6); // 45 degrees
}

TEST_F(TransformUtilsTest, ExtremelyLargeTranslation) {
    // Given - Very large translation values
    Eigen::Vector3d large_position(1e6, -5e5, 3e4);
    Eigen::Vector3d orientation(0.1, -0.05, 0.2);

    // When
    Eigen::Matrix4d transform = utils::TransformUtils::createTransform(large_position, orientation);
    Eigen::Vector3d extracted_position = utils::TransformUtils::extractPosition(transform);

    // Then
    EXPECT_TRUE(utils::TransformUtils::validateTransformMatrix(transform, "LargeTranslation"));
    EXPECT_TRUE(extracted_position.isApprox(large_position, 1e-6)); // Slightly relaxed tolerance for large values
}