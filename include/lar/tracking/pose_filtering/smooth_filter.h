#ifndef LAR_TRACKING_POSE_FILTERING_SMOOTH_FILTER_H
#define LAR_TRACKING_POSE_FILTERING_SMOOTH_FILTER_H

#include "pose_filter_strategy_base.h"
#include "pose_state.h"
#include <chrono>

namespace lar {

/**
 * SmoothFilter - Gradual transition to measurements over time
 *
 * Unlike PassThroughFilter which instantly adopts measurements, SmoothFilter
 * smoothly interpolates from the current state towards new measurements using
 * exponential smoothing. This provides visual continuity while still tracking
 * the measured pose.
 *
 * Algorithm:
 * - prediction: Apply VIO motion delta directly to current state
 * - update: Set target state to measurement, smoothly transition over time
 * - smoothing: Exponential interpolation controlled by smoothing_alpha
 *
 * Configuration parameters:
 * - smooth_filter_alpha: Base smoothing factor (0-1, default 0.3)
 */
class SmoothFilter : public PoseFilterStrategy {
public:
    SmoothFilter();
    ~SmoothFilter() override = default;

    // === Core Interface ===
    void initialize(const MeasurementContext& context,
                   const FilteredTrackerConfig& config) override;

    void predict(const Eigen::Matrix4d& motion,
                double dt,
                const FilteredTrackerConfig& config) override;

    void update(const MeasurementContext& context,
               const FilteredTrackerConfig& config) override;

    // === State Access ===
    PoseState getState() const override { return state_; }
    Eigen::MatrixXd getCovariance() const override;
    double getPositionUncertainty() const override;
    bool isInitialized() const override { return is_initialized_; }
    void reset() override;

private:
    // === Helper Methods ===

    /**
     * Smoothly interpolate from current state towards target state
     * @param alpha Interpolation factor (0 = keep current, 1 = adopt target)
     */
    void smoothTowardsTarget(double alpha);

    // === State Variables ===
    PoseState state_;              // Current smoothed state
    PoseState target_state_;       // Target state from latest measurement
    bool is_initialized_;          // Initialization flag

    // === Uncertainty Tracking ===
    double position_uncertainty_;  // Position uncertainty (meters)
    double last_confidence_;       // Last measurement confidence
};

} // namespace lar

#endif /* LAR_TRACKING_POSE_FILTERING_SMOOTH_FILTER_H */