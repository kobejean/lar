#ifndef LAR_TRACKING_CONFIDENCE_ESTIMATION_DISTANCE_STRATIFIED_CONFIDENCE_ESTIMATOR_H
#define LAR_TRACKING_CONFIDENCE_ESTIMATION_DISTANCE_STRATIFIED_CONFIDENCE_ESTIMATOR_H

#include "confidence_estimator_base.h"

namespace lar {

/**
 * Analyzes feature distribution by distance to provide separate confidence
 * estimates for position and rotation based on triangulation geometry.
 *
 * Key insight: Near features provide better position estimates,
 * distant features provide better rotation estimates.
 */
struct FeatureDistribution {
    int near_count = 0;     // Features < near_threshold
    int mid_count = 0;      // Features between thresholds
    int far_count = 0;      // Features > far_threshold
    int total_count = 0;

    double mean_distance = 0.0;
    double std_distance = 0.0;
    double min_distance = 0.0;
    double max_distance = 0.0;

    // Quality metrics (0-1, higher is better)
    double position_quality() const;
    double rotation_quality() const;
    double overall_quality() const;
};

/**
 * Distance-stratified confidence estimator that analyzes feature distribution
 * by distance to calculate independent confidence metrics for position vs rotation.
 */
class DistanceStratifiedConfidenceEstimator : public ConfidenceEstimator {
public:
    DistanceStratifiedConfidenceEstimator() = default;
    virtual ~DistanceStratifiedConfidenceEstimator() = default;

    /**
     * Calculate overall confidence based on feature distribution quality
     */
    double calculateConfidence(
        const std::vector<std::pair<Landmark*, cv::KeyPoint>>& inliers,
        const Eigen::Matrix4d& T_lar_from_camera,
        const Frame& frame,
        const FilteredTrackerConfig& config) const override;

    /**
     * Calculate measurement noise with separate position/rotation uncertainties
     * based on feature distance distribution
     */
    Eigen::MatrixXd calculateMeasurementNoise(
        const std::vector<std::pair<Landmark*, cv::KeyPoint>>& inliers,
        const Eigen::Matrix4d& T_lar_from_camera,
        const Frame& frame,
        double confidence,
        const FilteredTrackerConfig& config) const override;

private:
    /**
     * Analyze the distribution of features by distance from camera
     */
    FeatureDistribution analyzeFeatureDistribution(
        const std::vector<std::pair<Landmark*, cv::KeyPoint>>& inliers,
        const Eigen::Matrix4d& T_lar_from_camera,
        const FilteredTrackerConfig& config) const;

    /**
     * Calculate position uncertainty based on triangulation geometry
     */
    double calculatePositionUncertainty(
        const FeatureDistribution& distribution,
        const FilteredTrackerConfig& config) const;

    /**
     * Calculate rotation uncertainty based on angular measurement precision
     */
    double calculateRotationUncertainty(
        const FeatureDistribution& distribution,
        const FilteredTrackerConfig& config) const;
};

} // namespace lar

#endif /* LAR_TRACKING_CONFIDENCE_ESTIMATION_DISTANCE_STRATIFIED_CONFIDENCE_ESTIMATOR_H */