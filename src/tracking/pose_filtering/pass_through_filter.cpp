//
// PassThroughFilter.cpp - Pass-through filter (no filtering)
//

#include "lar/tracking/pose_filtering/pass_through_filter.h"
#include "lar/tracking/filtered_tracker_config.h"
#include <iostream>

namespace lar {

void PassThroughFilter::initialize(const MeasurementContext& context, const FilteredTrackerConfig& config) {
    state_.fromTransform(context.measured_pose);
    is_initialized_ = true;

    if (config.enable_debug_output) {
        std::cout << "PassThroughFilter initialized (no filtering)" << std::endl;
    }
}

void PassThroughFilter::predict(const Eigen::Matrix4d& motion, double dt, const FilteredTrackerConfig& config) {
    // No prediction - just keep last measurement
}

void PassThroughFilter::update(const MeasurementContext& context,
                              const FilteredTrackerConfig& config) {
    const Eigen::Matrix4d& measurement = context.measured_pose;

    // Simply use the measurement directly (no filtering)
    state_.fromTransform(measurement);

    if (config.enable_debug_output) {
        std::cout << "PassThroughFilter updated (direct measurement)" << std::endl;
    }
}

Eigen::MatrixXd PassThroughFilter::getCovariance() const {
    // Return a fixed covariance matrix
    Eigen::MatrixXd cov = Eigen::MatrixXd::Identity(6, 6);
    double posUncertainty = getPositionUncertainty();
    cov.block<3,3>(0,0) *= posUncertainty * posUncertainty;
    cov.block<3,3>(3,3) *= 0.2 * 0.2;
    return cov;
}

double PassThroughFilter::getPositionUncertainty() const {
    return 5;
}

void PassThroughFilter::reset() {
    is_initialized_ = false;
    state_.position = Eigen::Vector3d::Zero();
    state_.orientation = Eigen::Vector3d::Zero();
}

} // namespace lar