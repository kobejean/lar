//
// SimpleConfidenceEstimator.cpp - Simple confidence estimation strategy
//

#include "lar/tracking/confidence_estimation/simple_confidence_estimator.h"
#include "lar/tracking/filtered_tracker_config.h"
#include "lar/core/landmark.h"
#include <algorithm>

namespace lar {

double SimpleConfidenceEstimator::calculateConfidence(
    const std::vector<std::pair<Landmark*, cv::KeyPoint>>& inliers,
    const Eigen::Matrix4d& T_lar_from_camera,
    const Frame& frame,
    const FilteredTrackerConfig& config) const {

    if (inliers.size() < config.min_inliers_for_tracking) {
        return 0.0;
    }

    // Simple confidence based only on inlier count
    return std::min(1.0, static_cast<double>(inliers.size()) / config.max_inliers_for_confidence);
}

Eigen::MatrixXd SimpleConfidenceEstimator::calculateMeasurementNoise(
    const std::vector<std::pair<Landmark*, cv::KeyPoint>>& inliers,
    const Eigen::Matrix4d& T_lar_from_camera,
    const Frame& frame,
    double confidence,
    const FilteredTrackerConfig& config) const {

    // Fixed noise based only on confidence
    double confidence_factor = 1.0 / std::max(config.min_confidence_factor, confidence);

    Eigen::MatrixXd R = Eigen::MatrixXd::Identity(6, 6);
    R.block<3,3>(0,0) *= (config.base_position_noise * confidence_factor) * (config.base_position_noise * confidence_factor);
    R.block<3,3>(3,3) *= (config.base_orientation_noise * confidence_factor) * (config.base_orientation_noise * confidence_factor);

    return R;
}

} // namespace lar