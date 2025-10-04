#ifndef LAR_TRACKING_CONFIDENCE_ESTIMATION_CONFIDENCE_ESTIMATOR_BASE_H
#define LAR_TRACKING_CONFIDENCE_ESTIMATION_CONFIDENCE_ESTIMATOR_BASE_H

#include <vector>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <opencv2/core.hpp>
#include "lar/tracking/measurement_context.h"

namespace lar {

// Forward declarations
class Landmark;
class Frame;
struct FilteredTrackerConfig;

/**
 * Abstract interface for confidence estimation strategies.
 * Allows pluggable confidence calculation algorithms.
 */
class ConfidenceEstimator {
public:
    virtual ~ConfidenceEstimator() = default;

    /**
     * Calculate overall confidence for a measurement
     * @param context Measurement context containing inliers, poses, and metadata
     * @param config Configuration parameters
     * @return Confidence value in range [0,1]
     */
    virtual double calculateConfidence(
        const MeasurementContext& context,
        const FilteredTrackerConfig& config) const = 0;

    /**
     * Calculate measurement noise covariance matrix
     * @param context Measurement context containing inliers, poses, and metadata
     * @param config Configuration parameters
     * @return 6x6 measurement noise covariance matrix
     */
    virtual Eigen::MatrixXd calculateMeasurementNoise(
        const MeasurementContext& context,
        const FilteredTrackerConfig& config) const = 0;
};

} // namespace lar

#endif /* LAR_TRACKING_CONFIDENCE_ESTIMATION_CONFIDENCE_ESTIMATOR_BASE_H */