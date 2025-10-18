//
// DistanceOutlierDetector.cpp - Distance-based outlier detection
//

#include "lar/tracking/outlier_detection/distance_outlier_detector.h"
#include "lar/tracking/filtered_tracker_config.h"
#include <iostream>
#include <cmath>

namespace lar {

bool DistanceOutlierDetector::isOutlier(
    const Eigen::Matrix4d& measurement,
    const Eigen::Matrix4d& predicted_state,
    const Eigen::MatrixXd& covariance,
    double confidence,
    const FilteredTrackerConfig& config) const {


    // Position distance check
    Eigen::Vector3d pos_measured = measurement.block<3,1>(0,3);
    Eigen::Vector3d pos_predicted = predicted_state.block<3,1>(0,3);
    double position_distance = (pos_measured - pos_predicted).norm();

    if (position_distance > MAX_POSITION_DISTANCE) {
        if (config.enable_debug_output) {
            std::cout << "Distance outlier detected: position distance = " << position_distance
                      << " (threshold = " << MAX_POSITION_DISTANCE << ")" << std::endl;
        }
        return true;
    }

    // Orientation distance check
    Eigen::Matrix3d R_measured = measurement.block<3,3>(0,0);
    Eigen::Matrix3d R_predicted = predicted_state.block<3,3>(0,0);
    Eigen::Matrix3d R_diff = R_measured * R_predicted.transpose();

    Eigen::AngleAxisd axis_angle(R_diff);
    double orientation_distance = std::abs(axis_angle.angle());

    if (orientation_distance > MAX_ORIENTATION_DISTANCE) {
        if (config.enable_debug_output) {
            std::cout << "Distance outlier detected: orientation distance = " << orientation_distance
                      << " (threshold = " << MAX_ORIENTATION_DISTANCE << ")" << std::endl;
        }
        return true;
    }

    return false;
}

} // namespace lar