#ifndef LAR_TRACKING_CONFIDENCE_ESTIMATION_GEOMETRIC_CONFIDENCE_ESTIMATOR_H
#define LAR_TRACKING_CONFIDENCE_ESTIMATION_GEOMETRIC_CONFIDENCE_ESTIMATOR_H

#include "confidence_estimator_base.h"

namespace lar {

/**
 * Geometric confidence estimator based on inlier count and spatial distribution.
 * This is the current implementation extracted from FilteredTracker.
 */
class GeometricConfidenceEstimator : public ConfidenceEstimator {
public:
    double calculateConfidence(
        const MeasurementContext& context,
        const FilteredTrackerConfig& config) const override;

    Eigen::MatrixXd calculateMeasurementNoise(
        const MeasurementContext& context,
        const FilteredTrackerConfig& config) const override;

private:
    /**
     * Calculate spatial distribution factor based on landmark spread
     * @param inliers Vector of matched landmark-keypoint pairs
     * @param camera_position Camera position in LAR world
     * @param config Configuration parameters
     * @return Spatial distribution factor [0,1]
     */
    double calculateSpatialDistribution(
        const std::vector<std::pair<Landmark*, cv::KeyPoint>>& inliers,
        const Eigen::Vector3d& camera_position,
        const FilteredTrackerConfig& config) const;
};

} // namespace lar

#endif /* LAR_TRACKING_CONFIDENCE_ESTIMATION_GEOMETRIC_CONFIDENCE_ESTIMATOR_H */