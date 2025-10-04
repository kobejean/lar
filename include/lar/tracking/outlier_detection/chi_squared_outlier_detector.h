#ifndef LAR_TRACKING_OUTLIER_DETECTION_CHI_SQUARED_OUTLIER_DETECTOR_H
#define LAR_TRACKING_OUTLIER_DETECTION_CHI_SQUARED_OUTLIER_DETECTOR_H

#include "outlier_detector_base.h"
#include "lar/core/utils/transform.h"

namespace lar {

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

} // namespace lar

#endif /* LAR_TRACKING_OUTLIER_DETECTION_CHI_SQUARED_OUTLIER_DETECTOR_H */