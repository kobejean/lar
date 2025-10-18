//
// SmoothFilter.cpp - Gradual transition to measurements over time
//

#include "lar/tracking/pose_filtering/smooth_filter.h"
#include "lar/tracking/filtered_tracker_config.h"
#include <iostream>
#include <cmath>

namespace lar {

SmoothFilter::SmoothFilter()
    : is_initialized_(false)
    , position_uncertainty_(10.0)
    , last_confidence_(0.0) {
}

void SmoothFilter::initialize(const MeasurementContext& context,
                              const FilteredTrackerConfig& config) {
    state_.fromTransform(context.measured_pose);
    target_state_ = state_;  // Start with target matching current state

    is_initialized_ = true;
    position_uncertainty_ = 1.0;  // Start with low uncertainty
    last_confidence_ = context.confidence;

    if (config.enable_debug_output) {
        std::cout << "SmoothFilter initialized with pose:" << std::endl;
        std::cout << "  Position: [" << state_.position.transpose() << "]" << std::endl;
        std::cout << "  Orientation: [" << state_.orientation.transpose() << "]" << std::endl;
    }
}

void SmoothFilter::predict(const Eigen::Matrix4d& motion,
                           double dt,
                           const FilteredTrackerConfig& config) {
    if (!is_initialized_) {
        return;
    }

    // Apply motion delta to current state (VIO-driven prediction)
    Eigen::Matrix4d T_current = state_.toTransform();
    Eigen::Matrix4d T_predicted = T_current * motion;
    state_.fromTransform(T_predicted);

    // Also apply motion to target state to keep them in sync
    Eigen::Matrix4d T_target = target_state_.toTransform();
    Eigen::Matrix4d T_target_predicted = T_target * motion;
    target_state_.fromTransform(T_target_predicted);

    // Calculate motion magnitude (both translation and rotation)
    Eigen::Vector3d motion_translation = motion.block<3,1>(0,3);
    double translation_magnitude = motion_translation.norm();

    // Extract rotation magnitude from motion matrix
    Eigen::Matrix3d rotation_matrix = motion.block<3,3>(0,0);
    Eigen::Vector3d axis_angle = utils::TransformUtils::rotationMatrixToAxisAngle(rotation_matrix);
    double rotation_magnitude = axis_angle.norm();

    // Combined motion metric: translation (meters) + rotation (radians * scale factor)
    // Scale rotation to be comparable to translation (e.g., 1 radian ≈ 1 meter of "motion feel")
    double combined_motion = translation_magnitude + rotation_magnitude * config.smooth_filter_rotation_scale;

    // Increase uncertainty slightly with motion and time
    position_uncertainty_ += config.motion_position_noise_scale * translation_magnitude +
                            config.time_position_noise_scale * dt;

    // Motion-proportional correction: correction distance scales with camera motion
    // When camera is stationary, no correction happens
    // When camera moves X meters, we can correct up to (alpha * X) meters without being noticed

    // Only apply correction if there's motion
    // Calculate how far we are from target
    Eigen::Vector3d position_delta = target_state_.position - state_.position;
    Eigen::Vector3d orientation_delta = target_state_.orientation - state_.orientation;

    double position_distance = position_delta.norm();
    double orientation_distance = orientation_delta.norm();

    // Maximum correction budget based on motion magnitude
    double max_correction_budget = combined_motion * config.smooth_filter_alpha + dt * 0.001;

    // Synchronized convergence: scale corrections proportionally so both complete at same time
    if (position_distance > 0.0001 && orientation_distance > 0.0001) {
        // Both position and orientation have errors
        // Distribute correction budget proportionally to error magnitudes
        double total_error = position_distance + orientation_distance;
        double position_fraction = position_distance / total_error;
        double orientation_fraction = orientation_distance / total_error;

        // Allocate correction budget proportionally
        double position_correction = std::min(max_correction_budget * position_fraction, position_distance);
        double orientation_correction = std::min(max_correction_budget * orientation_fraction, orientation_distance);

        // Apply corrections
        state_.position += (position_correction / position_distance) * position_delta;
        state_.orientation += (orientation_correction / orientation_distance) * orientation_delta;
    }
    else if (position_distance > 0.0001) {
        // Only position has error - use full budget
        double position_correction = std::min(max_correction_budget, position_distance);
        state_.position += (position_correction / position_distance) * position_delta;
    }
    else if (orientation_distance > 0.0001) {
        // Only orientation has error - use full budget
        double orientation_correction = std::min(max_correction_budget, orientation_distance);
        state_.orientation += (orientation_correction / orientation_distance) * orientation_delta;
    }
}

void SmoothFilter::update(const MeasurementContext& context,
                         const FilteredTrackerConfig& config) {
    if (!is_initialized_) {
        return;
    }

    // Update target state to new measurement
    target_state_.fromTransform(context.measured_pose);
    last_confidence_ = context.confidence;

    // Update uncertainty based on measurement quality
    // Higher confidence = lower uncertainty
    double measurement_uncertainty = config.base_position_noise / std::max(context.confidence, 0.1);

    // Blend current uncertainty with measurement uncertainty
    position_uncertainty_ = measurement_uncertainty;

    if (config.enable_debug_output) {
        std::cout << "SmoothFilter updated with measurement" << std::endl;
        std::cout << "  Confidence: " << context.confidence << std::endl;
        std::cout << "  Position uncertainty: " << position_uncertainty_ << std::endl;

        // Show distance to target (how much smoothing is left to do)
        Eigen::Vector3d position_delta = target_state_.position - state_.position;
        Eigen::Vector3d orientation_delta = target_state_.orientation - state_.orientation;
        std::cout << "  Distance to target: pos=" << position_delta.norm()
                  << "m, orient=" << orientation_delta.norm() << "rad" << std::endl;
    }
}

Eigen::MatrixXd SmoothFilter::getCovariance() const {
    // Return simple diagonal covariance based on uncertainty
    Eigen::MatrixXd cov = Eigen::MatrixXd::Identity(6, 6);
    cov.block<3,3>(0,0) *= position_uncertainty_ * position_uncertainty_;
    cov.block<3,3>(3,3) *= 0.1 * 0.1;  // Fixed orientation uncertainty
    return cov;
}

double SmoothFilter::getPositionUncertainty() const {
    return position_uncertainty_;
}

void SmoothFilter::reset() {
    is_initialized_ = false;
    state_.position = Eigen::Vector3d::Zero();
    state_.orientation = Eigen::Vector3d::Zero();
    target_state_.position = Eigen::Vector3d::Zero();
    target_state_.orientation = Eigen::Vector3d::Zero();
    position_uncertainty_ = 10.0;
    last_confidence_ = 0.0;
}

// ============================================================================
// Helper Methods
// ============================================================================


void SmoothFilter::smoothTowardsTarget(double alpha) {
    // Exponential smoothing: state = (1-alpha)*state + alpha*target
    // This is equivalent to: state += alpha * (target - state)

    // Smooth position
    Eigen::Vector3d position_delta = target_state_.position - state_.position;
    state_.position += alpha * position_delta;

    // Smooth orientation (axis-angle representation)
    // For small angles, linear interpolation is acceptable
    Eigen::Vector3d orientation_delta = target_state_.orientation - state_.orientation;
    state_.orientation += alpha * orientation_delta;

    // Note: For large orientation changes, SLERP would be more accurate,
    // but in practice with frequent updates and smoothing, linear interpolation
    // works well and is computationally cheaper
}

} // namespace lar