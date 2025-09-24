//
// DistanceStratifiedConfidenceEstimator.cpp - Distance-stratified confidence estimation
//

#include "lar/tracking/confidence_estimation/distance_stratified_confidence_estimator.h"
#include "lar/tracking/filtered_tracker_config.h"
#include "lar/core/landmark.h"
#include <algorithm>
#include <cmath>
#include <iostream>

namespace lar {

// ============================================================================
// FeatureDistribution Implementation
// ============================================================================

double FeatureDistribution::position_quality() const {
    if (total_count == 0) return 0.0;

    // Simple: more near features = better position quality
    double quality = std::min(0.9, static_cast<double>(near_count) / 12.0);

    // Floor: never below 0.2 if we have decent total features
    if (total_count >= 15) {
        quality = std::max(0.2, quality);
    }

    return quality;
}

double FeatureDistribution::rotation_quality() const {
    if (total_count == 0) return 0.0;

    // Simple: far + half of mid features help rotation
    int effective_far = far_count + mid_count / 2;
    double quality = std::min(0.8, static_cast<double>(effective_far) / 8.0);

    // Floor: never below 0.3 if we have decent total features
    if (total_count >= 15) {
        quality = std::max(0.3, quality);
    }

    return quality;
}

double FeatureDistribution::overall_quality() const {
    // Simple weighted average - position more important than rotation
    return 0.7 * position_quality() + 0.3 * rotation_quality();
}

// ============================================================================
// DistanceStratifiedConfidenceEstimator Implementation
// ============================================================================

double DistanceStratifiedConfidenceEstimator::calculateConfidence(
    const MeasurementContext& context,
    const FilteredTrackerConfig& config) const {

    const auto& inliers = context.inliers;
    const Eigen::Matrix4d& T_lar_from_camera = context.measured_pose;

    if (inliers.size() < config.min_inliers_for_tracking) {
        return 0.0;
    }

    // Analyze feature distribution
    FeatureDistribution distribution = analyzeFeatureDistribution(inliers, T_lar_from_camera, config);

    // Simple: combine feature count and distribution quality
    double count_factor = std::min(1.0, static_cast<double>(inliers.size()) / 50.0);  // 50 features = full count confidence
    double quality_factor = distribution.overall_quality();

    // Average them - no fancy math
    double final_confidence = (count_factor + quality_factor) / 2.0;

    if (config.enable_debug_output) {
        std::cout << "DistanceStratifiedConfidenceEstimator:" << std::endl;
        std::cout << "  Total features: " << inliers.size() << std::endl;
        std::cout << "  Near/Mid/Far: " << distribution.near_count << "/"
                  << distribution.mid_count << "/" << distribution.far_count << std::endl;
        std::cout << "  Mean distance: " << distribution.mean_distance << "m" << std::endl;
        std::cout << "  Position quality: " << distribution.position_quality() << std::endl;
        std::cout << "  Rotation quality: " << distribution.rotation_quality() << std::endl;
        std::cout << "  Count factor: " << count_factor << std::endl;
        std::cout << "  Quality factor: " << quality_factor << std::endl;
        std::cout << "  Final confidence: " << final_confidence << std::endl;
    }

    return final_confidence;
}

Eigen::MatrixXd DistanceStratifiedConfidenceEstimator::calculateMeasurementNoise(
    const MeasurementContext& context,
    const FilteredTrackerConfig& config) const {

    const auto& inliers = context.inliers;
    const Eigen::Matrix4d& T_lar_from_camera = context.measured_pose;
    double confidence = context.confidence;

    // Analyze feature distribution by distance
    FeatureDistribution distribution = analyzeFeatureDistribution(inliers, T_lar_from_camera, config);

    // Calculate independent uncertainties for position and rotation
    double position_uncertainty = calculatePositionUncertainty(distribution, config);
    double rotation_uncertainty = calculateRotationUncertainty(distribution, config);

    // Apply overall confidence scaling
    double confidence_factor = 1.0 / std::max(config.min_confidence_factor, confidence);
    position_uncertainty *= confidence_factor;
    rotation_uncertainty *= confidence_factor;

    // Build measurement noise covariance matrix
    Eigen::MatrixXd R = Eigen::MatrixXd::Identity(6, 6);
    R.block<3,3>(0,0) *= position_uncertainty * position_uncertainty;
    R.block<3,3>(3,3) *= rotation_uncertainty * rotation_uncertainty;

    // Add cross-correlation if features are primarily distant
    // (lever arm effect: rotation uncertainty affects position)
    if (distribution.mean_distance > config.far_threshold &&
        distribution.far_count > distribution.near_count) {

        double correlation_strength = distribution.mean_distance * 0.005; // tunable parameter
        correlation_strength = std::min(correlation_strength, 0.1); // cap correlation

        // Cross-correlation matrix (simplified - assumes viewing along Z axis)
        Eigen::Matrix3d cross_corr = Eigen::Matrix3d::Zero();
        cross_corr(0,1) = correlation_strength; // pitch affects X position
        cross_corr(1,0) = correlation_strength; // roll affects Y position

        R.block<3,3>(0,3) = cross_corr;
        R.block<3,3>(3,0) = cross_corr.transpose();
    }

    return R;
}

FeatureDistribution DistanceStratifiedConfidenceEstimator::analyzeFeatureDistribution(
    const std::vector<std::pair<Landmark*, cv::KeyPoint>>& inliers,
    const Eigen::Matrix4d& T_lar_from_camera,
    const FilteredTrackerConfig& config) const {

    FeatureDistribution dist;
    dist.total_count = inliers.size();

    if (inliers.empty()) {
        return dist;
    }

    // Transform landmarks to camera frame and calculate distances
    std::vector<double> distances;
    distances.reserve(inliers.size());

    Eigen::Matrix4d T_camera_from_lar = T_lar_from_camera.inverse();

    for (const auto& [landmark, keypoint] : inliers) {
        // Transform landmark to camera frame
        Eigen::Vector4d p_lar(landmark->position[0], landmark->position[1],
                              landmark->position[2], 1.0);
        Eigen::Vector4d p_camera = T_camera_from_lar * p_lar;

        // Calculate distance from camera
        double distance = p_camera.head<3>().norm();
        distances.push_back(distance);

        // Categorize by distance
        if (distance < config.near_threshold) {
            dist.near_count++;
        } else if (distance > config.far_threshold) {
            dist.far_count++;
        } else {
            dist.mid_count++;
        }
    }

    // Calculate distance statistics
    dist.min_distance = *std::min_element(distances.begin(), distances.end());
    dist.max_distance = *std::max_element(distances.begin(), distances.end());

    // Calculate mean distance
    double sum = 0.0;
    for (double d : distances) {
        sum += d;
    }
    dist.mean_distance = sum / distances.size();

    // Calculate standard deviation
    double variance = 0.0;
    for (double d : distances) {
        variance += (d - dist.mean_distance) * (d - dist.mean_distance);
    }
    dist.std_distance = std::sqrt(variance / distances.size());

    return dist;
}

double DistanceStratifiedConfidenceEstimator::calculatePositionUncertainty(
    const FeatureDistribution& distribution,
    const FilteredTrackerConfig& config) const {

    // Base position uncertainty
    double base_uncertainty = config.base_position_noise;

    // Position uncertainty improves with near features (better triangulation)
    double position_quality = distribution.position_quality();
    double triangulation_factor = 1.0 / std::max(0.1, position_quality);

    // Additional uncertainty from depth variance (mixed near/far is harder)
    double depth_variance_factor = 1.0;
    if (distribution.total_count > 1) {
        double relative_std = distribution.std_distance / std::max(0.1, distribution.mean_distance);
        depth_variance_factor = 1.0 + relative_std; // more spread = more uncertainty
    }

    // Feature count scaling (more features = lower uncertainty)
    double count_factor = std::sqrt(std::max(1.0, config.min_inliers_for_noise /
                                           static_cast<double>(distribution.total_count)));

    return base_uncertainty * triangulation_factor * depth_variance_factor * count_factor;
}

double DistanceStratifiedConfidenceEstimator::calculateRotationUncertainty(
    const FeatureDistribution& distribution,
    const FilteredTrackerConfig& config) const {

    // Base rotation uncertainty
    double base_uncertainty = config.base_orientation_noise;

    // Rotation uncertainty improves with distant features (better angular precision)
    double rotation_quality = distribution.rotation_quality();
    double angular_factor = 1.0 / std::max(0.1, rotation_quality);

    // Distance-based scaling (further features = better rotation estimates)
    double distance_factor = 1.0;
    if (distribution.mean_distance > 0) {
        // Uncertainty inversely proportional to distance (up to a limit)
        double distance_benefit = std::min(distribution.mean_distance / config.near_threshold, 3.0);
        distance_factor = 1.0 / distance_benefit;
    }

    // Feature count scaling
    double count_factor = std::sqrt(std::max(1.0, config.min_inliers_for_noise /
                                           static_cast<double>(distribution.total_count)));

    return base_uncertainty * angular_factor * distance_factor * count_factor;
}

} // namespace lar