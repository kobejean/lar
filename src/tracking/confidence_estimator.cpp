//
// ConfidenceEstimator.cpp - Pluggable confidence estimation strategies
//

#include "lar/tracking/confidence_estimator.h"
#include "lar/tracking/filtered_tracker_config.h"
#include "lar/core/landmark.h"
#include <opencv2/imgproc.hpp>
#include <algorithm>
#include <iostream>

namespace lar {

// ============================================================================
// GeometricConfidenceEstimator Implementation
// ============================================================================

double GeometricConfidenceEstimator::calculateConfidence(
    const std::vector<std::pair<Landmark*, cv::KeyPoint>>& inliers,
    const Eigen::Matrix4d& T_lar_from_camera,
    const FilteredTrackerConfig& config) const {

    if (inliers.size() < config.min_inliers_for_tracking) {
        return 0.0;
    }

    // Base confidence from number of inliers
    double inlier_confidence = std::min(1.0, static_cast<double>(inliers.size()) / config.max_inliers_for_confidence);

    // Geometric distribution factor
    Eigen::Vector3d camera_position = T_lar_from_camera.block<3,1>(0,3);
    double spatial_distribution = calculateSpatialDistribution(inliers, camera_position, config);

    // Combined confidence
    return inlier_confidence * spatial_distribution;
}

Eigen::MatrixXd GeometricConfidenceEstimator::calculateMeasurementNoise(
    const std::vector<std::pair<Landmark*, cv::KeyPoint>>& inliers,
    const Eigen::Matrix4d& T_lar_from_camera,
    double confidence,
    const FilteredTrackerConfig& config) const {

    // Scene-dependent measurement noise based on confidence and geometry
    double confidence_factor = 1.0 / std::max(config.min_confidence_factor, confidence);
    double sparsity_factor = std::max(1.0, config.min_inliers_for_noise / static_cast<double>(inliers.size()));
    double total_factor = confidence_factor * sparsity_factor;

    Eigen::MatrixXd R = Eigen::MatrixXd::Identity(6, 6);
    R.block<3,3>(0,0) *= (config.base_position_noise * total_factor) * (config.base_position_noise * total_factor);
    R.block<3,3>(3,3) *= (config.base_orientation_noise * total_factor) * (config.base_orientation_noise * total_factor);

    return R;
}

double GeometricConfidenceEstimator::calculateSpatialDistribution(
    const std::vector<std::pair<Landmark*, cv::KeyPoint>>& inliers,
    const Eigen::Vector3d& camera_position,
    const FilteredTrackerConfig& config) const {

    if (inliers.size() < config.min_inliers_for_tracking) {
        return 0.1; // Low confidence for sparse matches
    }

    // Calculate spread of landmarks in image coordinates
    std::vector<cv::Point2f> points;
    for (const auto& pair : inliers) {
        points.push_back(pair.second.pt);
    }

    // Compute bounding rectangle area as proxy for spatial distribution
    cv::Rect bounding_rect = cv::boundingRect(points);
    double area_normalized = (bounding_rect.width * bounding_rect.height) / (640.0 * 480.0); // Assume VGA resolution

    return std::min(1.0, area_normalized * config.spatial_distribution_scale);
}

// ============================================================================
// SimpleConfidenceEstimator Implementation
// ============================================================================

double SimpleConfidenceEstimator::calculateConfidence(
    const std::vector<std::pair<Landmark*, cv::KeyPoint>>& inliers,
    const Eigen::Matrix4d& T_lar_from_camera,
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