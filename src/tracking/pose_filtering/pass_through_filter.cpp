//
// PassThroughFilter.cpp - Pass-through filter (no filtering)
//

#include "lar/tracking/pose_filtering/pass_through_filter.h"
#include "lar/tracking/filtered_tracker_config.h"
#include <iostream>

namespace lar {

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