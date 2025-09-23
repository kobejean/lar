#ifndef LAR_TRACKING_CAMERA_POSE_FILTER_H
#define LAR_TRACKING_CAMERA_POSE_FILTER_H

#include <Eigen/Core>
#include <Eigen/Geometry>

namespace lar {

/**
 * Pure Kalman filter for camera poses in LAR world coordinates.
 * Handles prediction and measurement updates without any external dependencies.
 */
class CameraPoseFilter {
public:
    struct CameraPoseState {
        Eigen::Vector3d position_lar;      // Camera position IN LAR world coordinates
        Eigen::Vector3d orientation_lar;   // Camera orientation IN LAR world (axis-angle)

        static constexpr int SIZE = 6;

        // Convert to/from state vector for EKF operations
        Eigen::VectorXd toVector() const;
        void fromVector(const Eigen::VectorXd& vec);

        // Convert to transform matrix
        Eigen::Matrix4d toTransform() const; // Returns T_lar_from_camera
        void fromTransform(const Eigen::Matrix4d& T_lar_from_camera);
    };

    /**
     * Constructor
     * @param initial_position_uncertainty Initial position uncertainty (meters)
     * @param initial_orientation_uncertainty Initial orientation uncertainty (radians)
     */
    CameraPoseFilter(double initial_position_uncertainty = 1.0,
                     double initial_orientation_uncertainty = 0.1);

    /**
     * Initialize filter with first measurement
     */
    void initialize(const Eigen::Matrix4d& T_lar_from_camera);

    /**
     * Prediction step using motion model
     * @param motion Relative motion transform (T_current_from_last)
     * @param dt Time step in seconds
     */
    void predict(const Eigen::Matrix4d& motion, double dt);

    /**
     * Measurement update step
     * @param measurement Camera pose measurement (T_lar_from_camera)
     * @param measurement_noise 6x6 measurement noise covariance matrix
     */
    void update(const Eigen::Matrix4d& measurement, const Eigen::MatrixXd& measurement_noise);

    /**
     * Get current state estimate
     */
    CameraPoseState getState() const { return state_; }

    /**
     * Get current covariance matrix
     */
    Eigen::MatrixXd getCovariance() const { return covariance_; }

    /**
     * Get position uncertainty (trace of position covariance)
     */
    double getPositionUncertainty() const;

    /**
     * Check if filter is initialized
     */
    bool isInitialized() const { return is_initialized_; }

    /**
     * Reset filter state
     */
    void reset();

    /**
     * Check if measurement is outlier using chi-squared test
     * @param measurement Camera pose measurement
     * @param threshold Chi-squared threshold (default: 12.592 for 6 DOF, 95% confidence)
     */
    bool isOutlier(const Eigen::Matrix4d& measurement, double threshold = 12.592) const;

private:
    CameraPoseState state_;
    Eigen::MatrixXd covariance_;            // 6x6 covariance matrix
    bool is_initialized_;

    double initial_position_uncertainty_;
    double initial_orientation_uncertainty_;

    // Mathematical utilities
    static Eigen::Vector3d rotationMatrixToAxisAngle(const Eigen::Matrix3d& R);
    static Eigen::Matrix3d axisAngleToRotationMatrix(const Eigen::Vector3d& axis_angle);
};

} // namespace lar

#endif /* LAR_TRACKING_CAMERA_POSE_FILTER_H */