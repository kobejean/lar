//
// CoordinateTransformManager.cpp - VIO-LAR coordinate transform management
//
// COORDINATE CONVENTIONS:
// - LAR World: Right-handed, Y-up, landmarks and camera poses
// - VIO World: Right-handed, Y-up, VIO coordinate system (ARKit, ARCore, etc.)
// - T_lar_from_camera: Camera pose IN LAR world coordinates
// - T_vio_from_camera: Camera pose IN VIO world coordinates
// - T_lar_from_vio: Coordinate transform FROM VIO TO LAR world (output)

#include "lar/tracking/coordinate_transform_manager.h"
#include <iostream>
#include <cmath>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace lar {

// ============================================================================
// Constructor and Initialization
// ============================================================================

CoordinateTransformManager::CoordinateTransformManager()
    : has_vio_pose_(false) {

    current_vio_camera_pose_ = Eigen::Matrix4d::Identity();

    // Initialize empty transform result
    last_result_.success = false;
    last_result_.map_to_vio_transform = Eigen::Matrix4d::Identity();
    last_result_.confidence = 0.0;
    last_result_.timestamp = std::chrono::steady_clock::now();

    std::cout << "CoordinateTransformManager initialized" << std::endl;
}

// ============================================================================
// VIO Pose Management
// ============================================================================

void CoordinateTransformManager::updateVIOCameraPose(const Eigen::Matrix4d& T_vio_from_camera) {
    if (!validateTransformMatrix(T_vio_from_camera, "T_vio_from_camera")) {
        std::cout << "WARNING: Invalid VIO camera pose provided" << std::endl;
        return;
    }

    current_vio_camera_pose_ = T_vio_from_camera;
    has_vio_pose_ = true;
}

// ============================================================================
// Transform Computation
// ============================================================================

CoordinateTransformManager::TransformResult
CoordinateTransformManager::computeTransform(const Eigen::Matrix4d& T_lar_from_camera,
                                            double confidence) {
    TransformResult result;
    result.success = false;
    result.confidence = confidence;
    result.timestamp = std::chrono::steady_clock::now();

    if (!has_vio_pose_) {
        std::cout << "ERROR: No VIO camera pose available for transform computation" << std::endl;
        return result;
    }

    if (!validateTransformMatrix(T_lar_from_camera, "T_lar_from_camera")) {
        std::cout << "ERROR: Invalid LAR camera pose" << std::endl;
        return result;
    }

    // CRITICAL: Use synchronized poses to compute coordinate transform
    // We have:
    // - T_lar_from_camera: Camera pose in LAR world (filtered estimate)
    // - current_vio_camera_pose_: Camera pose from VIO (synchronized with measurement)
    //
    // We want T_vio_from_lar such that:
    // T_lar_from_camera = T_lar_from_vio * T_vio_from_camera
    // Therefore: T_lar_from_vio = T_lar_from_camera * T_vio_from_camera^-1
    // But we need the reverse transform: T_vio_from_lar = T_lar_from_vio^-1
    // So: map_to_vio_transform = (T_lar_from_camera * T_vio_from_camera^-1)^-1
    //                          = T_vio_from_camera * T_lar_from_camera^-1

    result.map_to_vio_transform = current_vio_camera_pose_ * T_lar_from_camera.inverse();

    if (!validateTransformMatrix(result.map_to_vio_transform, "map_to_vio_transform")) {
        std::cout << "ERROR: Computed transform is invalid" << std::endl;
        return result;
    }

    result.success = true;
    last_result_ = result;

    std::cout << "Transform computed successfully (confidence: " << confidence << ")" << std::endl;
    debugTransforms("computeTransform");

    return result;
}

// ============================================================================
// Transform Access
// ============================================================================

Eigen::Matrix4d CoordinateTransformManager::getCurrentTransform() const {
    return last_result_.map_to_vio_transform;
}

double CoordinateTransformManager::getTransformAge() const {
    if (!last_result_.success) {
        return std::numeric_limits<double>::infinity();
    }

    auto now = std::chrono::steady_clock::now();
    return std::chrono::duration<double>(now - last_result_.timestamp).count();
}

void CoordinateTransformManager::reset() {
    has_vio_pose_ = false;
    current_vio_camera_pose_ = Eigen::Matrix4d::Identity();

    last_result_.success = false;
    last_result_.map_to_vio_transform = Eigen::Matrix4d::Identity();
    last_result_.confidence = 0.0;
    last_result_.timestamp = std::chrono::steady_clock::now();

    std::cout << "CoordinateTransformManager reset" << std::endl;
}

// ============================================================================
// Debugging and Validation
// ============================================================================

void CoordinateTransformManager::debugTransforms(const std::string& context) const {
    std::cout << "=== COORDINATE TRANSFORMS (" << context << ") ===" << std::endl;

    if (has_vio_pose_) {
        printTransform(current_vio_camera_pose_, "T_vio_from_camera (current)");
    } else {
        std::cout << "No VIO camera pose available" << std::endl;
    }

    if (last_result_.success) {
        printTransform(last_result_.map_to_vio_transform, "T_vio_from_lar (last computed)");
        std::cout << "Transform age: " << getTransformAge() << " seconds" << std::endl;
        std::cout << "Transform confidence: " << last_result_.confidence << std::endl;
    } else {
        std::cout << "No valid transform available" << std::endl;
    }

    std::cout << "======================================" << std::endl;
}

bool CoordinateTransformManager::validateTransformMatrix(const Eigen::Matrix4d& T,
                                                        const std::string& name) const {
    // Check if bottom row is [0 0 0 1]
    if (std::abs(T(3,0)) > 1e-6 || std::abs(T(3,1)) > 1e-6 ||
        std::abs(T(3,2)) > 1e-6 || std::abs(T(3,3) - 1.0) > 1e-6) {
        std::cout << "WARNING: " << name << " has invalid bottom row: ["
                  << T(3,0) << ", " << T(3,1) << ", " << T(3,2) << ", " << T(3,3) << "]" << std::endl;
        return false;
    }

    // Check if rotation part is orthogonal
    Eigen::Matrix3d R = T.block<3,3>(0,0);
    Eigen::Matrix3d should_be_identity = R.transpose() * R;
    double orthogonality_error = (should_be_identity - Eigen::Matrix3d::Identity()).norm();

    if (orthogonality_error > 1e-3) {
        std::cout << "WARNING: " << name << " rotation is not orthogonal (error: "
                  << orthogonality_error << ")" << std::endl;
        return false;
    }

    // Check determinant (should be +1 for proper rotation)
    double det = R.determinant();
    if (std::abs(det - 1.0) > 1e-3) {
        std::cout << "WARNING: " << name << " rotation determinant is " << det
                  << " (should be 1.0)" << std::endl;
        return false;
    }

    return true;
}

void CoordinateTransformManager::printTransform(const Eigen::Matrix4d& T, const std::string& name) {
    std::cout << name << ":" << std::endl;
    std::cout << "  Translation: [" << T(0,3) << ", " << T(1,3) << ", " << T(2,3) << "]" << std::endl;

    // Convert to axis-angle for rotation representation
    Eigen::Matrix3d R = T.block<3,3>(0,0);
    Eigen::AngleAxisd axis_angle(R);
    Eigen::Vector3d axis_angle_vec = axis_angle.axis() * axis_angle.angle();
    std::cout << "  Rotation (axis-angle): [" << axis_angle_vec(0) << ", "
              << axis_angle_vec(1) << ", " << axis_angle_vec(2) << "]" << std::endl;

    // Convert to Euler angles for intuition
    Eigen::Vector3d euler = R.eulerAngles(0, 1, 2) * 180.0 / M_PI;
    std::cout << "  Rotation (XYZ Euler degrees): [" << euler(0) << ", "
              << euler(1) << ", " << euler(2) << "]" << std::endl;
}

} // namespace lar