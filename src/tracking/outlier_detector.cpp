//
// OutlierDetector.cpp - Pluggable outlier detection strategies
//

#include "lar/tracking/outlier_detector.h"
#include "lar/tracking/filtered_tracker_config.h"
#include <iostream>
#include <cmath>

namespace lar {

// ============================================================================
// ChiSquaredOutlierDetector Implementation
// ============================================================================

bool ChiSquaredOutlierDetector::isOutlier(
    const Eigen::Matrix4d& measurement,
    const Eigen::Matrix4d& predicted_state,
    const Eigen::MatrixXd& covariance,
    double confidence,
    const FilteredTrackerConfig& config) const {

    if (!config.enable_outlier_detection) {
        return false;
    }

    // Convert to state vectors
    Eigen::VectorXd measured_vector = transformToVector(measurement);
    Eigen::VectorXd predicted_vector = transformToVector(predicted_state);

    // Innovation (measurement residual)
    Eigen::VectorXd innovation = measured_vector - predicted_vector;

    // Mahalanobis distance using covariance matrix
    double mahalanobis_squared = innovation.transpose() * covariance.inverse() * innovation;

    // Adjust threshold based on confidence
    double threshold = config.outlier_threshold / std::max(config.min_confidence_factor, confidence);

    bool is_outlier = mahalanobis_squared > threshold;

    if (config.enable_debug_output && is_outlier) {
        std::cout << "Chi-squared outlier detected: Mahalanobis distance = " << std::sqrt(mahalanobis_squared)
                  << " (threshold = " << std::sqrt(threshold) << ")" << std::endl;
    }

    return is_outlier;
}

Eigen::VectorXd ChiSquaredOutlierDetector::transformToVector(const Eigen::Matrix4d& T) const {
    Eigen::VectorXd vec(6);
    vec.segment<3>(0) = T.block<3,1>(0,3);  // position
    vec.segment<3>(3) = utils::TransformUtils::rotationMatrixToAxisAngle(T.block<3,3>(0,0));  // orientation
    return vec;
}


// ============================================================================
// DistanceOutlierDetector Implementation
// ============================================================================

bool DistanceOutlierDetector::isOutlier(
    const Eigen::Matrix4d& measurement,
    const Eigen::Matrix4d& predicted_state,
    const Eigen::MatrixXd& covariance,
    double confidence,
    const FilteredTrackerConfig& config) const {

    if (!config.enable_outlier_detection) {
        return false;
    }

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

// ============================================================================
// ConfidenceOutlierDetector Implementation
// ============================================================================

bool ConfidenceOutlierDetector::isOutlier(
    const Eigen::Matrix4d& measurement,
    const Eigen::Matrix4d& predicted_state,
    const Eigen::MatrixXd& covariance,
    double confidence,
    const FilteredTrackerConfig& config) const {

    if (!config.enable_outlier_detection) {
        return false;
    }

    bool is_outlier = confidence < MIN_CONFIDENCE_THRESHOLD;

    if (config.enable_debug_output && is_outlier) {
        std::cout << "Confidence outlier detected: confidence = " << confidence
                  << " (threshold = " << MIN_CONFIDENCE_THRESHOLD << ")" << std::endl;
    }

    return is_outlier;
}

} // namespace lar