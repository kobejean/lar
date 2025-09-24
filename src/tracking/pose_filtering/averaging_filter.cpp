//
// AveragingFilter.cpp - Simple averaging filter implementation
//

#include "lar/tracking/pose_filtering/averaging_filter.h"
#include "lar/tracking/filtered_tracker_config.h"
#include <iostream>

namespace lar {

AveragingFilter::AveragingFilter(double alpha)
    : alpha_(alpha), position_uncertainty_(1.0), is_initialized_(false) {
}

void AveragingFilter::initialize(const MeasurementContext& context, const FilteredTrackerConfig& config) {
    state_.fromTransform(context.measured_pose);
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

void AveragingFilter::update(const MeasurementContext& context,
                            const FilteredTrackerConfig& config) {
    const Eigen::Matrix4d& measurement = context.measured_pose;
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

} // namespace lar