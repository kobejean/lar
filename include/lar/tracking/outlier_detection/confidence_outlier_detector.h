#ifndef LAR_TRACKING_OUTLIER_DETECTION_CONFIDENCE_OUTLIER_DETECTOR_H
#define LAR_TRACKING_OUTLIER_DETECTION_CONFIDENCE_OUTLIER_DETECTOR_H

#include "outlier_detector_base.h"

namespace lar {

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

#endif /* LAR_TRACKING_OUTLIER_DETECTION_CONFIDENCE_OUTLIER_DETECTOR_H */