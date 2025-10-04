#ifndef LAR_TRACKING_OUTLIER_DETECTION_DISTANCE_OUTLIER_DETECTOR_H
#define LAR_TRACKING_OUTLIER_DETECTION_DISTANCE_OUTLIER_DETECTOR_H

#include "outlier_detector_base.h"

namespace lar {

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

} // namespace lar

#endif /* LAR_TRACKING_OUTLIER_DETECTION_DISTANCE_OUTLIER_DETECTOR_H */