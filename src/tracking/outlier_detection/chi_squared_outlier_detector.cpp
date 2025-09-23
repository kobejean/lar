//
// ChiSquaredOutlierDetector.cpp - Chi-squared test outlier detection
//

#include "lar/tracking/outlier_detection/chi_squared_outlier_detector.h"
#include "lar/tracking/filtered_tracker_config.h"
#include <iostream>
#include <cmath>

namespace lar {

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

} // namespace lar