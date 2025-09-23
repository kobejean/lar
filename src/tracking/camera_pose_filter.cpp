//
// CameraPoseFilter.cpp - Pure Kalman filter for camera poses
//
// COORDINATE CONVENTIONS:
// - LAR World: Right-handed, Y-up, landmarks and camera poses
// - T_lar_from_camera: Camera pose IN LAR world coordinates

#include "lar/tracking/camera_pose_filter.h"
#include <iostream>
#include <cmath>

namespace lar {

// ============================================================================
// CameraPoseState Implementation
// ============================================================================

Eigen::VectorXd CameraPoseFilter::CameraPoseState::toVector() const {
    Eigen::VectorXd vec(SIZE);
    vec.segment<3>(0) = position_lar;
    vec.segment<3>(3) = orientation_lar;
    return vec;
}

void CameraPoseFilter::CameraPoseState::fromVector(const Eigen::VectorXd& vec) {
    position_lar = vec.segment<3>(0);
    orientation_lar = vec.segment<3>(3);
}

Eigen::Matrix4d CameraPoseFilter::CameraPoseState::toTransform() const {
    Eigen::Matrix4d T_lar_from_camera = Eigen::Matrix4d::Identity();
    T_lar_from_camera.block<3,3>(0,0) = axisAngleToRotationMatrix(orientation_lar);
    T_lar_from_camera.block<3,1>(0,3) = position_lar;
    return T_lar_from_camera;
}

void CameraPoseFilter::CameraPoseState::fromTransform(const Eigen::Matrix4d& T_lar_from_camera) {
    position_lar = T_lar_from_camera.block<3,1>(0,3);
    orientation_lar = rotationMatrixToAxisAngle(T_lar_from_camera.block<3,3>(0,0));
}

// ============================================================================
// Mathematical Utilities
// ============================================================================

Eigen::Vector3d CameraPoseFilter::rotationMatrixToAxisAngle(const Eigen::Matrix3d& R) {
    Eigen::AngleAxisd axis_angle(R);
    return axis_angle.axis() * axis_angle.angle();
}

Eigen::Matrix3d CameraPoseFilter::axisAngleToRotationMatrix(const Eigen::Vector3d& axis_angle) {
    double angle = axis_angle.norm();
    if (angle < 1e-10) {
        return Eigen::Matrix3d::Identity();
    }
    Eigen::Vector3d axis = axis_angle / angle;
    return Eigen::AngleAxisd(angle, axis).toRotationMatrix();
}

// ============================================================================
// Constructor and Initialization
// ============================================================================

CameraPoseFilter::CameraPoseFilter(double initial_position_uncertainty,
                                   double initial_orientation_uncertainty)
    : is_initialized_(false)
    , initial_position_uncertainty_(initial_position_uncertainty)
    , initial_orientation_uncertainty_(initial_orientation_uncertainty) {

    // Initialize state
    state_.position_lar = Eigen::Vector3d::Zero();
    state_.orientation_lar = Eigen::Vector3d::Zero();

    // Initialize covariance
    covariance_ = Eigen::MatrixXd::Identity(6, 6);
    covariance_.block<3,3>(0,0) *= initial_position_uncertainty * initial_position_uncertainty;
    covariance_.block<3,3>(3,3) *= initial_orientation_uncertainty * initial_orientation_uncertainty;
}

void CameraPoseFilter::initialize(const Eigen::Matrix4d& T_lar_from_camera) {
    state_.fromTransform(T_lar_from_camera);

    // Reset covariance to initial values
    covariance_ = Eigen::MatrixXd::Identity(6, 6);
    covariance_.block<3,3>(0,0) *= initial_position_uncertainty_ * initial_position_uncertainty_;
    covariance_.block<3,3>(3,3) *= initial_orientation_uncertainty_ * initial_orientation_uncertainty_;

    is_initialized_ = true;

    std::cout << "CameraPoseFilter initialized with camera pose:" << std::endl;
    std::cout << "  Position: [" << state_.position_lar.transpose() << "]" << std::endl;
    std::cout << "  Orientation: [" << state_.orientation_lar.transpose() << "]" << std::endl;
}

// ============================================================================
// Prediction Step
// ============================================================================

void CameraPoseFilter::predict(const Eigen::Matrix4d& motion, double dt) {
    if (!is_initialized_) {
        return;
    }

    // Apply motion to current state
    Eigen::Matrix4d T_lar_from_camera_current = state_.toTransform();
    Eigen::Matrix4d T_lar_from_camera_predicted = T_lar_from_camera_current * motion;

    // Update state
    state_.fromTransform(T_lar_from_camera_predicted);

    // Compute process noise based on motion magnitude and time
    Eigen::Vector3d motion_translation = motion.block<3,1>(0,3);
    double motion_magnitude = motion_translation.norm();

    // Process noise scales with motion and time
    Eigen::MatrixXd Q = Eigen::MatrixXd::Identity(6, 6);
    Q.block<3,3>(0,0) *= (0.01 * motion_magnitude + 0.001 * dt); // Position uncertainty
    Q.block<3,3>(3,3) *= (0.005 * motion_magnitude + 0.0005 * dt); // Orientation uncertainty

    // Covariance prediction: P = P + Q
    covariance_ = covariance_ + Q;
}

// ============================================================================
// Measurement Update
// ============================================================================

void CameraPoseFilter::update(const Eigen::Matrix4d& measurement,
                              const Eigen::MatrixXd& measurement_noise) {
    if (!is_initialized_) {
        return;
    }

    // Convert measurement to state vector
    CameraPoseState measured_state;
    measured_state.fromTransform(measurement);
    Eigen::VectorXd z = measured_state.toVector();

    // Current state prediction
    Eigen::VectorXd x = state_.toVector();

    // Innovation (measurement residual)
    Eigen::VectorXd y = z - x;

    // Innovation covariance: S = H*P*H' + R
    // Since H = I (direct measurement), S = P + R
    Eigen::MatrixXd S = covariance_ + measurement_noise;

    // Kalman gain: K = P*H'*S^-1 = P*S^-1
    Eigen::MatrixXd K = covariance_ * S.inverse();

    // State update: x = x + K*y
    Eigen::VectorXd x_updated = x + K * y;
    state_.fromVector(x_updated);

    // Covariance update: P = (I - K*H)*P = (I - K)*P
    Eigen::MatrixXd I = Eigen::MatrixXd::Identity(6, 6);
    covariance_ = (I - K) * covariance_;

    std::cout << "CameraPoseFilter updated with measurement" << std::endl;
    std::cout << "  Innovation norm: " << y.norm() << std::endl;
    std::cout << "  Position uncertainty: " << getPositionUncertainty() << std::endl;
}

// ============================================================================
// State Access and Utilities
// ============================================================================

double CameraPoseFilter::getPositionUncertainty() const {
    return covariance_.block<3,3>(0,0).trace();
}

bool CameraPoseFilter::isOutlier(const Eigen::Matrix4d& measurement, double threshold) const {
    if (!is_initialized_) {
        return false; // Accept all measurements if not initialized
    }

    // Convert measurement to state vector
    CameraPoseState measured_state;
    measured_state.fromTransform(measurement);
    Eigen::VectorXd z = measured_state.toVector();

    // Current state prediction
    Eigen::VectorXd x = state_.toVector();

    // Innovation (measurement residual)
    Eigen::VectorXd y = z - x;

    // Mahalanobis distance using covariance matrix
    double mahalanobis_squared = y.transpose() * covariance_.inverse() * y;

    // Chi-squared test
    bool is_outlier = mahalanobis_squared > threshold;

    if (is_outlier) {
        std::cout << "Outlier detected: Mahalanobis distance = " << std::sqrt(mahalanobis_squared)
                  << " (threshold = " << std::sqrt(threshold) << ")" << std::endl;
    }

    return is_outlier;
}

void CameraPoseFilter::reset() {
    is_initialized_ = false;
    state_.position_lar = Eigen::Vector3d::Zero();
    state_.orientation_lar = Eigen::Vector3d::Zero();

    covariance_ = Eigen::MatrixXd::Identity(6, 6);
    covariance_.block<3,3>(0,0) *= initial_position_uncertainty_ * initial_position_uncertainty_;
    covariance_.block<3,3>(3,3) *= initial_orientation_uncertainty_ * initial_orientation_uncertainty_;

    std::cout << "CameraPoseFilter reset" << std::endl;
}

} // namespace lar