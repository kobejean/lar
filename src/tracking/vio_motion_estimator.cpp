//
// VIOMotionEstimator.cpp - VIO motion estimation for FilteredTracker
//
// COORDINATE CONVENTIONS:
// - VIO World: Right-handed, Y-up, VIO coordinate system (ARKit, ARCore, etc.)
// - T_vio_from_camera: Camera pose IN VIO world coordinates
// - Motion transforms: Relative camera motion between frames

#include "lar/tracking/vio_motion_estimator.h"
#include <iostream>
#include <cmath>

namespace lar {

// ============================================================================
// Constructor and Initialization
// ============================================================================

VIOMotionEstimator::VIOMotionEstimator()
    : has_poses_(false) {

    last_vio_camera_pose_ = Eigen::Matrix4d::Identity();
    current_vio_camera_pose_ = Eigen::Matrix4d::Identity();
    last_update_time_ = std::chrono::steady_clock::now();

    std::cout << "VIOMotionEstimator initialized" << std::endl;
}

// ============================================================================
// VIO Pose Management
// ============================================================================

void VIOMotionEstimator::updateVIOCameraPose(const Eigen::Matrix4d& T_vio_from_camera) {
    if (!validateTransformMatrix(T_vio_from_camera, "T_vio_from_camera")) {
        std::cout << "WARNING: Invalid VIO camera pose provided to motion estimator" << std::endl;
        return;
    }

    // Update pose tracking
    if (!has_poses_) {
        // First pose - initialize both current and last
        last_vio_camera_pose_ = T_vio_from_camera;
        current_vio_camera_pose_ = T_vio_from_camera;
        has_poses_ = true;
        std::cout << "VIOMotionEstimator: First VIO pose received" << std::endl;
    } else {
        // Update poses: current becomes last, new pose becomes current
        last_vio_camera_pose_ = current_vio_camera_pose_;
        current_vio_camera_pose_ = T_vio_from_camera;
    }

    last_update_time_ = std::chrono::steady_clock::now();
}

// ============================================================================
// Motion Computation
// ============================================================================

Eigen::Matrix4d VIOMotionEstimator::getMotionDelta() {
    if (!has_poses_) {
        std::cout << "WARNING: No VIO poses available for motion computation" << std::endl;
        return Eigen::Matrix4d::Identity();
    }

    // Compute relative camera motion in camera's own frame
    // We need: T_camera_current_from_camera_last
    // This is: T_camera_from_vio_last * T_vio_from_camera_current
    Eigen::Matrix4d motion_delta = last_vio_camera_pose_.inverse() * current_vio_camera_pose_;

    if (!validateTransformMatrix(motion_delta, "motion_delta")) {
        std::cout << "WARNING: Computed motion delta is invalid, returning identity" << std::endl;
        return Eigen::Matrix4d::Identity();
    }

    // // Debug motion magnitude
    // Eigen::Vector3d translation = motion_delta.block<3,1>(0,3);
    // double translation_magnitude = translation.norm();

    // Eigen::Matrix3d R = motion_delta.block<3,3>(0,0);
    // Eigen::AngleAxisd axis_angle(R);
    // double rotation_magnitude = axis_angle.angle();

    // if (translation_magnitude > 0.1 || rotation_magnitude > 0.01) {
    //     std::cout << "VIO motion delta: translation=" << translation_magnitude
    //               << "m, rotation=" << (rotation_magnitude * 180.0 / M_PI) << "°" << std::endl;
    // }

    return motion_delta;
}

// ============================================================================
// State Management
// ============================================================================

void VIOMotionEstimator::reset() {
    has_poses_ = false;
    last_vio_camera_pose_ = Eigen::Matrix4d::Identity();
    current_vio_camera_pose_ = Eigen::Matrix4d::Identity();
    last_update_time_ = std::chrono::steady_clock::now();

    std::cout << "VIOMotionEstimator reset" << std::endl;
}

// ============================================================================
// Debugging and Validation
// ============================================================================

void VIOMotionEstimator::debugMotion(const std::string& context) const {
    std::cout << "=== VIO MOTION DEBUG (" << context << ") ===" << std::endl;

    if (has_poses_) {
        printTransform(last_vio_camera_pose_, "T_vio_from_camera (last)");
        printTransform(current_vio_camera_pose_, "T_vio_from_camera (current)");

        // Show time since last update
        auto now = std::chrono::steady_clock::now();
        double dt = std::chrono::duration<double>(now - last_update_time_).count();
        std::cout << "Time since last update: " << dt << " seconds" << std::endl;

        // Show motion delta
        Eigen::Matrix4d motion = last_vio_camera_pose_.inverse() * current_vio_camera_pose_;
        printTransform(motion, "Motion delta (T_current_from_last)");
    } else {
        std::cout << "No VIO poses available" << std::endl;
    }

    std::cout << "======================================" << std::endl;
}

bool VIOMotionEstimator::validateTransformMatrix(const Eigen::Matrix4d& T,
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

void VIOMotionEstimator::printTransform(const Eigen::Matrix4d& T, const std::string& name) {
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