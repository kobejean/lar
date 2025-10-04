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
        const MeasurementContext& context,
        const FilteredTrackerConfig& config) const override;

    Eigen::MatrixXd calculateMeasurementNoise(
        const MeasurementContext& context,
        const FilteredTrackerConfig& config) const override;
};

} // namespace lar

#endif /* LAR_TRACKING_CONFIDENCE_ESTIMATION_SIMPLE_CONFIDENCE_ESTIMATOR_H */