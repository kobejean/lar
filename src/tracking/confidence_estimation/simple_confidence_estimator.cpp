//
// SimpleConfidenceEstimator.cpp - Simple confidence estimation strategy
//

#include "lar/tracking/confidence_estimation/simple_confidence_estimator.h"
#include "lar/tracking/filtered_tracker_config.h"
#include "lar/core/landmark.h"
#include <algorithm>

namespace lar {

double SimpleConfidenceEstimator::calculateConfidence(
    const MeasurementContext& context,
    const FilteredTrackerConfig& config) const {

    if (!context.inliers) {
        throw std::runtime_error("MeasurementContext.inliers is null - this should never happen");
    }

    if (context.inliers->size() < config.min_inliers_for_tracking) {
        return 0.0;
    }

    // Simple confidence based only on inlier count
    return std::min(1.0, static_cast<double>(context.inliers->size()) / config.max_inliers_for_confidence);
}

Eigen::MatrixXd SimpleConfidenceEstimator::calculateMeasurementNoise(
    const MeasurementContext& context,
    const FilteredTrackerConfig& config) const {

    double confidence = context.confidence;

    // Fixed noise based only on confidence
    double confidence_factor = 1.0 / std::max(config.min_confidence_factor, confidence);

    Eigen::MatrixXd R = Eigen::MatrixXd::Identity(6, 6);
    R.block<3,3>(0,0) *= (config.base_position_noise * confidence_factor) * (config.base_position_noise * confidence_factor);
    R.block<3,3>(3,3) *= (config.base_orientation_noise * confidence_factor) * (config.base_orientation_noise * confidence_factor);

    return R;
}

} // namespace lar