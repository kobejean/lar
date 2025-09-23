#ifndef LAR_TRACKING_OUTLIER_DETECTOR_H
#define LAR_TRACKING_OUTLIER_DETECTOR_H

#include <Eigen/Core>
#include <Eigen/Geometry>
#include "lar/core/utils/transform.h"

namespace lar {

// Forward declarations
struct FilteredTrackerConfig;

/**
 * Abstract interface for outlier detection strategies.
 * Allows pluggable outlier detection algorithms.
 */
class OutlierDetector {
public:
    virtual ~OutlierDetector() = default;

    /**
     * Check if a measurement is an outlier
     * @param measurement Camera pose measurement
     * @param predicted_state Predicted camera pose
     * @param covariance Current pose covariance matrix
     * @param confidence Measurement confidence
     * @param config Configuration parameters
     * @return True if measurement should be rejected as outlier
     */
    virtual bool isOutlier(
        const Eigen::Matrix4d& measurement,
        const Eigen::Matrix4d& predicted_state,
        const Eigen::MatrixXd& covariance,
        double confidence,
        const FilteredTrackerConfig& config) const = 0;
};

/**
 * Chi-squared test outlier detector using Mahalanobis distance.
 * This implements the standard statistical outlier detection.
 */
class ChiSquaredOutlierDetector : public OutlierDetector {
public:
    bool isOutlier(
        const Eigen::Matrix4d& measurement,
        const Eigen::Matrix4d& predicted_state,
        const Eigen::MatrixXd& covariance,
        double confidence,
        const FilteredTrackerConfig& config) const override;

private:
    /**
     * Convert transform matrix to 6D state vector (position + axis-angle orientation)
     */
    Eigen::VectorXd transformToVector(const Eigen::Matrix4d& T) const;
};

/**
 * Simple distance-based outlier detector.
 * Rejects measurements based on position/orientation distance thresholds.
 */
class DistanceOutlierDetector : public OutlierDetector {
public:
    bool isOutlier(
        const Eigen::Matrix4d& measurement,
        const Eigen::Matrix4d& predicted_state,
        const Eigen::MatrixXd& covariance,
        double confidence,
        const FilteredTrackerConfig& config) const override;

private:
    // Default thresholds (can be made configurable)
    static constexpr double MAX_POSITION_DISTANCE = 2.0;  // meters
    static constexpr double MAX_ORIENTATION_DISTANCE = 0.5; // radians (~30 degrees)
};

/**
 * Confidence-based outlier detector.
 * Rejects measurements below a confidence threshold.
 */
class ConfidenceOutlierDetector : public OutlierDetector {
public:
    bool isOutlier(
        const Eigen::Matrix4d& measurement,
        const Eigen::Matrix4d& predicted_state,
        const Eigen::MatrixXd& covariance,
        double confidence,
        const FilteredTrackerConfig& config) const override;

private:
    static constexpr double MIN_CONFIDENCE_THRESHOLD = 0.3;
};

} // namespace lar

#endif /* LAR_TRACKING_OUTLIER_DETECTOR_H */