#ifndef LAR_TRACKING_POSE_FILTERING_AVERAGING_FILTER_H
#define LAR_TRACKING_POSE_FILTERING_AVERAGING_FILTER_H

#include "pose_filter_strategy_base.h"

namespace lar {

/**
 * Simple averaging filter (for comparison/debugging).
 * Uses exponential moving average instead of full Kalman filtering.
 */
class AveragingFilter : public PoseFilterStrategy {
public:
    explicit AveragingFilter(double alpha = 0.9);

    void initialize(const Eigen::Matrix4d& initial_pose, const FilteredTrackerConfig& config) override;
    void predict(const Eigen::Matrix4d& motion, double dt, const FilteredTrackerConfig& config) override;
    void update(const MeasurementContext& context,
               const Eigen::MatrixXd& measurement_noise,
               const FilteredTrackerConfig& config) override;

    PoseState getState() const override { return state_; }
    Eigen::MatrixXd getCovariance() const override;
    double getPositionUncertainty() const override;
    bool isInitialized() const override { return is_initialized_; }
    void reset() override;

private:
    PoseState state_;
    double alpha_;                  // Averaging factor
    double position_uncertainty_;   // Simple uncertainty estimate
    bool is_initialized_;
};

} // namespace lar

#endif /* LAR_TRACKING_POSE_FILTERING_AVERAGING_FILTER_H */