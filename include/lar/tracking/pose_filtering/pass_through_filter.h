#ifndef LAR_TRACKING_POSE_FILTERING_PASS_THROUGH_FILTER_H
#define LAR_TRACKING_POSE_FILTERING_PASS_THROUGH_FILTER_H

#include "pose_filter_strategy_base.h"

namespace lar {

/**
 * Pass-through filter (no filtering).
 * Useful for testing the impact of filtering vs raw measurements.
 */
class PassThroughFilter : public PoseFilterStrategy {
public:
    void initialize(const MeasurementContext& context, const FilteredTrackerConfig& config) override;
    void predict(const Eigen::Matrix4d& motion, double dt, const FilteredTrackerConfig& config) override;
    void update(const MeasurementContext& context,
               const FilteredTrackerConfig& config) override;

    PoseState getState() const override { return state_; }
    Eigen::MatrixXd getCovariance() const override;
    double getPositionUncertainty() const override;
    bool isInitialized() const override { return is_initialized_; }
    void reset() override;

private:
    PoseState state_;
    bool is_initialized_;
};

} // namespace lar

#endif /* LAR_TRACKING_POSE_FILTERING_PASS_THROUGH_FILTER_H */