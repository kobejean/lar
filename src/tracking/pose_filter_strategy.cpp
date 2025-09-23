//
// PoseFilterStrategy.cpp - Pluggable filtering algorithms
//

#include "lar/tracking/pose_filter_strategy.h"
#include "lar/tracking/filtered_tracker_config.h"
#include <iostream>
#include <cmath>

namespace lar {

// ============================================================================
// PoseState Implementation
// ============================================================================

Eigen::VectorXd PoseState::toVector() const {
    Eigen::VectorXd vec(SIZE);
    vec.segment<3>(0) = position;
    vec.segment<3>(3) = orientation;
    return vec;
}

void PoseState::fromVector(const Eigen::VectorXd& vec) {
    position = vec.segment<3>(0);
    orientation = vec.segment<3>(3);
}

Eigen::Matrix4d PoseState::toTransform() const {
    return utils::TransformUtils::createTransform(position, orientation);
}

void PoseState::fromTransform(const Eigen::Matrix4d& T) {
    position = utils::TransformUtils::extractPosition(T);
    orientation = utils::TransformUtils::rotationMatrixToAxisAngle(utils::TransformUtils::extractRotation(T));
}


// ============================================================================
// ExtendedKalmanFilter Implementation
// ============================================================================

void ExtendedKalmanFilter::initialize(const Eigen::Matrix4d& initial_pose, const FilteredTrackerConfig& config) {
    state_.fromTransform(initial_pose);

    // Initialize covariance
    covariance_ = Eigen::MatrixXd::Identity(6, 6);
    covariance_.block<3,3>(0,0) *= config.initial_position_uncertainty * config.initial_position_uncertainty;
    covariance_.block<3,3>(3,3) *= config.initial_orientation_uncertainty * config.initial_orientation_uncertainty;

    is_initialized_ = true;

    if (config.enable_debug_output) {
        std::cout << "ExtendedKalmanFilter initialized with pose:" << std::endl;
        std::cout << "  Position: [" << state_.position.transpose() << "]" << std::endl;
        std::cout << "  Orientation: [" << state_.orientation.transpose() << "]" << std::endl;
    }
}

void ExtendedKalmanFilter::predict(const Eigen::Matrix4d& motion, double dt, const FilteredTrackerConfig& config) {
    if (!is_initialized_) {
        return;
    }

    // Apply motion to current state
    Eigen::Matrix4d T_current = state_.toTransform();
    Eigen::Matrix4d T_predicted = T_current * motion;
    state_.fromTransform(T_predicted);

    // Compute process noise based on motion magnitude and time
    Eigen::Vector3d motion_translation = motion.block<3,1>(0,3);
    double motion_magnitude = motion_translation.norm();

    // Process noise scales with motion and time
    Eigen::MatrixXd Q = Eigen::MatrixXd::Identity(6, 6);
    Q.block<3,3>(0,0) *= (config.motion_position_noise_scale * motion_magnitude + config.time_position_noise_scale * dt);
    Q.block<3,3>(3,3) *= (config.motion_orientation_noise_scale * motion_magnitude + config.time_orientation_noise_scale * dt);

    // Covariance prediction: P = P + Q
    covariance_ = covariance_ + Q;
}

void ExtendedKalmanFilter::update(const Eigen::Matrix4d& measurement,
                                 const Eigen::MatrixXd& measurement_noise,
                                 const FilteredTrackerConfig& config) {
    if (!is_initialized_) {
        return;
    }

    // Convert measurement to state vector
    PoseState measured_state;
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

    if (config.enable_debug_output) {
        std::cout << "ExtendedKalmanFilter updated with measurement" << std::endl;
        std::cout << "  Innovation norm: " << y.norm() << std::endl;
        std::cout << "  Position uncertainty: " << getPositionUncertainty() << std::endl;
    }
}

double ExtendedKalmanFilter::getPositionUncertainty() const {
    return covariance_.block<3,3>(0,0).trace();
}

void ExtendedKalmanFilter::reset() {
    is_initialized_ = false;
    state_.position = Eigen::Vector3d::Zero();
    state_.orientation = Eigen::Vector3d::Zero();
    covariance_ = Eigen::MatrixXd::Identity(6, 6);
}

// ============================================================================
// AveragingFilter Implementation
// ============================================================================

AveragingFilter::AveragingFilter(double alpha)
    : alpha_(alpha), position_uncertainty_(1.0), is_initialized_(false) {
}

void AveragingFilter::initialize(const Eigen::Matrix4d& initial_pose, const FilteredTrackerConfig& config) {
    state_.fromTransform(initial_pose);
    position_uncertainty_ = config.initial_position_uncertainty;
    is_initialized_ = true;

    if (config.enable_debug_output) {
        std::cout << "AveragingFilter initialized with alpha=" << alpha_ << std::endl;
    }
}

void AveragingFilter::predict(const Eigen::Matrix4d& motion, double dt, const FilteredTrackerConfig& config) {
    if (!is_initialized_) {
        return;
    }

    // Apply motion to current state
    Eigen::Matrix4d T_current = state_.toTransform();
    Eigen::Matrix4d T_predicted = T_current * motion;
    state_.fromTransform(T_predicted);

    // Increase uncertainty with motion
    Eigen::Vector3d motion_translation = motion.block<3,1>(0,3);
    double motion_magnitude = motion_translation.norm();
    position_uncertainty_ += config.motion_position_noise_scale * motion_magnitude + config.time_position_noise_scale * dt;
}

void AveragingFilter::update(const Eigen::Matrix4d& measurement,
                            const Eigen::MatrixXd& measurement_noise,
                            const FilteredTrackerConfig& config) {
    if (!is_initialized_) {
        return;
    }

    // Simple exponential moving average
    PoseState measured_state;
    measured_state.fromTransform(measurement);

    state_.position = alpha_ * state_.position + (1.0 - alpha_) * measured_state.position;
    state_.orientation = alpha_ * state_.orientation + (1.0 - alpha_) * measured_state.orientation;

    // Decrease uncertainty with measurement
    position_uncertainty_ = alpha_ * position_uncertainty_ + (1.0 - alpha_) * config.base_position_noise;

    if (config.enable_debug_output) {
        std::cout << "AveragingFilter updated with measurement (alpha=" << alpha_ << ")" << std::endl;
    }
}

Eigen::MatrixXd AveragingFilter::getCovariance() const {
    Eigen::MatrixXd cov = Eigen::MatrixXd::Identity(6, 6);
    cov.block<3,3>(0,0) *= position_uncertainty_ * position_uncertainty_;
    cov.block<3,3>(3,3) *= position_uncertainty_ * 0.1; // Approximate orientation uncertainty
    return cov;
}

double AveragingFilter::getPositionUncertainty() const {
    return position_uncertainty_;
}

void AveragingFilter::reset() {
    is_initialized_ = false;
    state_.position = Eigen::Vector3d::Zero();
    state_.orientation = Eigen::Vector3d::Zero();
    position_uncertainty_ = 1.0;
}

// ============================================================================
// PassThroughFilter Implementation
// ============================================================================

void PassThroughFilter::initialize(const Eigen::Matrix4d& initial_pose, const FilteredTrackerConfig& config) {
    state_.fromTransform(initial_pose);
    is_initialized_ = true;

    if (config.enable_debug_output) {
        std::cout << "PassThroughFilter initialized (no filtering)" << std::endl;
    }
}

void PassThroughFilter::predict(const Eigen::Matrix4d& motion, double dt, const FilteredTrackerConfig& config) {
    // No prediction - just keep last measurement
}

void PassThroughFilter::update(const Eigen::Matrix4d& measurement,
                              const Eigen::MatrixXd& measurement_noise,
                              const FilteredTrackerConfig& config) {
    if (!is_initialized_) {
        return;
    }

    // Simply use the measurement directly (no filtering)
    state_.fromTransform(measurement);

    if (config.enable_debug_output) {
        std::cout << "PassThroughFilter updated (direct measurement)" << std::endl;
    }
}

Eigen::MatrixXd PassThroughFilter::getCovariance() const {
    // Return a fixed covariance matrix
    Eigen::MatrixXd cov = Eigen::MatrixXd::Identity(6, 6);
    cov.block<3,3>(0,0) *= 0.1 * 0.1;  // 10cm position uncertainty
    cov.block<3,3>(3,3) *= 0.05 * 0.05; // ~3 degree orientation uncertainty
    return cov;
}

double PassThroughFilter::getPositionUncertainty() const {
    return 0.1; // Fixed 10cm uncertainty
}

void PassThroughFilter::reset() {
    is_initialized_ = false;
    state_.position = Eigen::Vector3d::Zero();
    state_.orientation = Eigen::Vector3d::Zero();
}

} // namespace lar