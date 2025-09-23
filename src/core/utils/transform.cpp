//
// transform.cpp - Centralized transform matrix utilities
//

#include "lar/core/utils/transform.h"
#include <iostream>
#include <cmath>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace lar {
namespace utils {

// ============================================================================
// Rotation Conversions
// ============================================================================

Eigen::Vector3d TransformUtils::rotationMatrixToAxisAngle(const Eigen::Matrix3d& R) {
    Eigen::AngleAxisd axis_angle(R);
    return axis_angle.axis() * axis_angle.angle();
}

Eigen::Matrix3d TransformUtils::axisAngleToRotationMatrix(const Eigen::Vector3d& axis_angle) {
    double angle = axis_angle.norm();
    if (angle < 1e-10) {
        return Eigen::Matrix3d::Identity();
    }
    Eigen::Vector3d axis = axis_angle / angle;
    return Eigen::AngleAxisd(angle, axis).toRotationMatrix();
}

// ============================================================================
// Transform Validation
// ============================================================================

bool TransformUtils::validateTransformMatrix(const Eigen::Matrix4d& T, const std::string& name) {
    // Check if bottom row is [0 0 0 1]
    if (std::abs(T(3,0)) > DEFAULT_TOLERANCE ||
        std::abs(T(3,1)) > DEFAULT_TOLERANCE ||
        std::abs(T(3,2)) > DEFAULT_TOLERANCE ||
        std::abs(T(3,3) - 1.0) > DEFAULT_TOLERANCE) {
        std::cout << "WARNING: " << name << " has invalid bottom row: ["
                  << T(3,0) << ", " << T(3,1) << ", " << T(3,2) << ", " << T(3,3) << "]" << std::endl;
        return false;
    }

    // Check if rotation part is orthogonal
    Eigen::Matrix3d R = T.block<3,3>(0,0);
    Eigen::Matrix3d should_be_identity = R.transpose() * R;
    double orthogonality_error = (should_be_identity - Eigen::Matrix3d::Identity()).norm();

    if (orthogonality_error > ORTHOGONALITY_TOLERANCE) {
        std::cout << "WARNING: " << name << " rotation matrix is not orthogonal (error: "
                  << orthogonality_error << ")" << std::endl;
        return false;
    }

    // Check determinant (should be 1 for proper rotation, -1 for improper)
    double det = R.determinant();
    if (std::abs(det - 1.0) > DEFAULT_TOLERANCE && std::abs(det + 1.0) > DEFAULT_TOLERANCE) {
        std::cout << "WARNING: " << name << " rotation matrix has invalid determinant: "
                  << det << std::endl;
        return false;
    }

    return true;
}

// ============================================================================
// Debug Utilities
// ============================================================================

void TransformUtils::printTransform(const Eigen::Matrix4d& T, const std::string& name) {
    std::cout << name << ":" << std::endl;
    std::cout << "  Translation: [" << T(0,3) << ", " << T(1,3) << ", " << T(2,3) << "]" << std::endl;

    // Convert to axis-angle for rotation representation
    Eigen::Matrix3d R = T.block<3,3>(0,0);
    Eigen::Vector3d axis_angle_vec = rotationMatrixToAxisAngle(R);
    std::cout << "  Rotation (axis-angle): [" << axis_angle_vec(0) << ", "
              << axis_angle_vec(1) << ", " << axis_angle_vec(2) << "]" << std::endl;

    // Also show Euler angles for reference
    Eigen::Vector3d euler = R.eulerAngles(0, 1, 2) * 180.0 / M_PI;
    std::cout << "  Rotation (XYZ Euler degrees): [" << euler(0) << ", "
              << euler(1) << ", " << euler(2) << "]" << std::endl;
}

// ============================================================================
// Comparison Utilities
// ============================================================================

bool TransformUtils::isApproximatelyEqual(const Eigen::Matrix4d& T1,
                                         const Eigen::Matrix4d& T2,
                                         double tolerance) {
    return (T1 - T2).norm() < tolerance;
}

// ============================================================================
// Extract/Create Transform Components
// ============================================================================

Eigen::Vector3d TransformUtils::extractPosition(const Eigen::Matrix4d& T) {
    return T.block<3,1>(0,3);
}

Eigen::Matrix3d TransformUtils::extractRotation(const Eigen::Matrix4d& T) {
    return T.block<3,3>(0,0);
}

Eigen::Matrix4d TransformUtils::createTransform(const Eigen::Vector3d& position,
                                               const Eigen::Matrix3d& rotation) {
    Eigen::Matrix4d T = Eigen::Matrix4d::Identity();
    T.block<3,3>(0,0) = rotation;
    T.block<3,1>(0,3) = position;
    return T;
}

Eigen::Matrix4d TransformUtils::createTransform(const Eigen::Vector3d& position,
                                               const Eigen::Vector3d& orientation) {
    Eigen::Matrix3d rotation = axisAngleToRotationMatrix(orientation);
    return createTransform(position, rotation);
}

} // namespace utils
} // namespace lar