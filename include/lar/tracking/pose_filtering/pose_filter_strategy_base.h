#ifndef LAR_TRACKING_POSE_FILTERING_POSE_FILTER_STRATEGY_BASE_H
#define LAR_TRACKING_POSE_FILTERING_POSE_FILTER_STRATEGY_BASE_H

#include <Eigen/Core>
#include <Eigen/Geometry>
#include "pose_state.h"
#include "lar/tracking/measurement_context.h"

namespace lar {

// Forward declarations
struct FilteredTrackerConfig;

/**
 * Abstract interface for pose filtering strategies.
 * Allows pluggable filtering algorithms (EKF, UKF, Particle Filter, etc.)
 */
class PoseFilterStrategy {
public:
    virtual ~PoseFilterStrategy() = default;

    /**
     * Initialize filter with first measurement context
     * @param context Complete measurement context with pose, inliers, confidence, and measurement_noise
     * @param config Configuration parameters
     */
    virtual void initialize(const MeasurementContext& context, const FilteredTrackerConfig& config) = 0;

    /**
     * Prediction step using motion model
     * @param motion Relative motion transform (T_current_from_last)
     * @param dt Time step in seconds
     * @param config Configuration parameters
     */
    virtual void predict(const Eigen::Matrix4d& motion, double dt, const FilteredTrackerConfig& config) = 0;

    /**
     * Measurement update step
     * @param context Measurement context containing pose, inliers, and metadata
     * @param config Configuration parameters
     */
    virtual void update(const MeasurementContext& context,
                       const FilteredTrackerConfig& config) = 0;

    /**
     * Get current state estimate
     */
    virtual PoseState getState() const = 0;

    /**
     * Get current covariance matrix
     */
    virtual Eigen::MatrixXd getCovariance() const = 0;

    /**
     * Get position uncertainty (trace of position covariance)
     */
    virtual double getPositionUncertainty() const = 0;

    /**
     * Check if filter is initialized
     */
    virtual bool isInitialized() const = 0;

    /**
     * Reset filter state
     */
    virtual void reset() = 0;
};

} // namespace lar

#endif /* LAR_TRACKING_POSE_FILTERING_POSE_FILTER_STRATEGY_BASE_H */