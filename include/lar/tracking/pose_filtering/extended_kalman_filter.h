#ifndef LAR_TRACKING_POSE_FILTERING_EXTENDED_KALMAN_FILTER_H
#define LAR_TRACKING_POSE_FILTERING_EXTENDED_KALMAN_FILTER_H

#include "pose_filter_strategy_base.h"
#include <deque>
#include <chrono>

namespace lar {

/**
 * Historical anchor pose for rotational drift correction
 */
struct AnchorPose {
    Eigen::Matrix4d transform;              // Pose transform in LAR coordinates
    double confidence;                      // Confidence when this pose was created
    double timestamp;                       // When this anchor was created (seconds)
    Eigen::Vector3d position;              // Position for quick distance checks
    size_t observation_count;              // Number of features used for this pose

    AnchorPose(const Eigen::Matrix4d& T, double conf, double time, size_t obs_count)
        : transform(T), confidence(conf), timestamp(time), observation_count(obs_count) {
        position = T.block<3,1>(0,3);
    }
};

/**
 * Extended Kalman Filter implementation with optional historical anchor constraints.
 * Full-featured EKF for camera pose estimation with motion-based process noise
 * and rotational drift correction using historical high-quality poses.
 */
class ExtendedKalmanFilter : public PoseFilterStrategy {
public:
    void initialize(const Eigen::Matrix4d& initial_pose, const FilteredTrackerConfig& config) override;
    void predict(const Eigen::Matrix4d& motion, double dt, const FilteredTrackerConfig& config) override;
    void update(const MeasurementContext& context,
               const Eigen::MatrixXd& measurement_noise,
               const FilteredTrackerConfig& config) override;

    /**
     * Enhanced update with anchor constraints (optional)
     * @param context Measurement context with pose, confidence, and observation count
     * @param measurement_noise Measurement covariance
     * @param config Configuration parameters
     */
    void updateWithAnchors(const MeasurementContext& context,
                          const Eigen::MatrixXd& measurement_noise,
                          const FilteredTrackerConfig& config);

    PoseState getState() const override { return state_; }
    Eigen::MatrixXd getCovariance() const override { return covariance_; }
    double getPositionUncertainty() const override;
    bool isInitialized() const override { return is_initialized_; }
    void reset() override;

private:
    PoseState state_;
    Eigen::MatrixXd covariance_;            // 6x6 covariance matrix
    bool is_initialized_;

    // Historical anchor constraints
    std::deque<AnchorPose> anchor_history_;
    std::chrono::steady_clock::time_point start_time_;

    // Helper methods for anchor constraints
    bool shouldAddAnchor(const Eigen::Matrix4d& measurement, double confidence, size_t observation_count, const FilteredTrackerConfig& config) const;
    void addAnchor(const Eigen::Matrix4d& measurement, double confidence, size_t observation_count, const FilteredTrackerConfig& config);
    void applyAnchorConstraints(const FilteredTrackerConfig& config);
    double getCurrentTimestamp() const;
};

} // namespace lar

#endif /* LAR_TRACKING_POSE_FILTERING_EXTENDED_KALMAN_FILTER_H */