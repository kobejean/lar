#ifndef LAR_TRACKING_POSE_FILTER_STRATEGY_H
#define LAR_TRACKING_POSE_FILTER_STRATEGY_H

#include <Eigen/Core>
#include <Eigen/Geometry>

namespace lar {

// Forward declarations
struct FilteredTrackerConfig;

/**
 * Pose state representation for filtering strategies.
 * Common interface for different filtering algorithms.
 */
struct PoseState {
    Eigen::Vector3d position;      // Camera position
    Eigen::Vector3d orientation;   // Camera orientation (axis-angle)

    static constexpr int SIZE = 6;

    // Convert to/from state vector for filter operations
    Eigen::VectorXd toVector() const;
    void fromVector(const Eigen::VectorXd& vec);

    // Convert to transform matrix
    Eigen::Matrix4d toTransform() const;
    void fromTransform(const Eigen::Matrix4d& T);

private:
    static Eigen::Vector3d rotationMatrixToAxisAngle(const Eigen::Matrix3d& R);
    static Eigen::Matrix3d axisAngleToRotationMatrix(const Eigen::Vector3d& axis_angle);
};

/**
 * Abstract interface for pose filtering strategies.
 * Allows pluggable filtering algorithms (EKF, UKF, Particle Filter, etc.)
 */
class PoseFilterStrategy {
public:
    virtual ~PoseFilterStrategy() = default;

    /**
     * Initialize filter with first measurement
     * @param initial_pose Initial camera pose
     * @param config Configuration parameters
     */
    virtual void initialize(const Eigen::Matrix4d& initial_pose, const FilteredTrackerConfig& config) = 0;

    /**
     * Prediction step using motion model
     * @param motion Relative motion transform (T_current_from_last)
     * @param dt Time step in seconds
     * @param config Configuration parameters
     */
    virtual void predict(const Eigen::Matrix4d& motion, double dt, const FilteredTrackerConfig& config) = 0;

    /**
     * Measurement update step
     * @param measurement Camera pose measurement
     * @param measurement_noise 6x6 measurement noise covariance matrix
     * @param config Configuration parameters
     */
    virtual void update(const Eigen::Matrix4d& measurement,
                       const Eigen::MatrixXd& measurement_noise,
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

/**
 * Extended Kalman Filter implementation.
 * Full-featured EKF for camera pose estimation with motion-based process noise.
 */
class ExtendedKalmanFilter : public PoseFilterStrategy {
public:
    void initialize(const Eigen::Matrix4d& initial_pose, const FilteredTrackerConfig& config) override;
    void predict(const Eigen::Matrix4d& motion, double dt, const FilteredTrackerConfig& config) override;
    void update(const Eigen::Matrix4d& measurement,
               const Eigen::MatrixXd& measurement_noise,
               const FilteredTrackerConfig& config) override;

    PoseState getState() const override { return state_; }
    Eigen::MatrixXd getCovariance() const override { return covariance_; }
    double getPositionUncertainty() const override;
    bool isInitialized() const override { return is_initialized_; }
    void reset() override;

private:
    PoseState state_;
    Eigen::MatrixXd covariance_;            // 6x6 covariance matrix
    bool is_initialized_;
};

/**
 * Simple averaging filter (for comparison/debugging).
 * Uses exponential moving average instead of full Kalman filtering.
 */
class AveragingFilter : public PoseFilterStrategy {
public:
    explicit AveragingFilter(double alpha = 0.9);

    void initialize(const Eigen::Matrix4d& initial_pose, const FilteredTrackerConfig& config) override;
    void predict(const Eigen::Matrix4d& motion, double dt, const FilteredTrackerConfig& config) override;
    void update(const Eigen::Matrix4d& measurement,
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

/**
 * Pass-through filter (no filtering).
 * Useful for testing the impact of filtering vs raw measurements.
 */
class PassThroughFilter : public PoseFilterStrategy {
public:
    void initialize(const Eigen::Matrix4d& initial_pose, const FilteredTrackerConfig& config) override;
    void predict(const Eigen::Matrix4d& motion, double dt, const FilteredTrackerConfig& config) override;
    void update(const Eigen::Matrix4d& measurement,
               const Eigen::MatrixXd& measurement_noise,
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

#endif /* LAR_TRACKING_POSE_FILTER_STRATEGY_H */