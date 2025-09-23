#ifndef LAR_TRACKING_CONFIDENCE_ESTIMATION_SIMPLE_CONFIDENCE_ESTIMATOR_H
#define LAR_TRACKING_CONFIDENCE_ESTIMATION_SIMPLE_CONFIDENCE_ESTIMATOR_H

#include "confidence_estimator_base.h"

namespace lar {

/**
 * Simple confidence estimator that only considers inlier count.
 * Useful for comparison and debugging.
 */
class SimpleConfidenceEstimator : public ConfidenceEstimator {
public:
    double calculateConfidence(
        const std::vector<std::pair<Landmark*, cv::KeyPoint>>& inliers,
        const Eigen::Matrix4d& T_lar_from_camera,
        const Frame& frame,
        const FilteredTrackerConfig& config) const override;

    Eigen::MatrixXd calculateMeasurementNoise(
        const std::vector<std::pair<Landmark*, cv::KeyPoint>>& inliers,
        const Eigen::Matrix4d& T_lar_from_camera,
        const Frame& frame,
        double confidence,
        const FilteredTrackerConfig& config) const override;
};

} // namespace lar

#endif /* LAR_TRACKING_CONFIDENCE_ESTIMATION_SIMPLE_CONFIDENCE_ESTIMATOR_H */