#ifndef LAR_TRACKING_VIO_MOTION_ESTIMATOR_H
#define LAR_TRACKING_VIO_MOTION_ESTIMATOR_H

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <chrono>

namespace lar {

/**
 * Estimates camera motion from VIO poses for FilteredTracker prediction steps.
 * Handles VIO pose tracking and relative motion computation.
 *
 * COORDINATE CONVENTIONS:
 * - VIO World: Right-handed, Y-up, VIO coordinate system (ARKit, ARCore, etc.)
 * - T_vio_from_camera: Camera pose IN VIO world coordinates
 * - Motion transforms: Relative camera motion between frames
 */
class VIOMotionEstimator {
public:
    /**
     * Constructor
     */
    VIOMotionEstimator();

    /**
     * Update current VIO camera pose
     * @param T_vio_from_camera Current VIO camera pose transform
     */
    void updateVIOCameraPose(const Eigen::Matrix4d& T_vio_from_camera);

    /**
     * Get relative motion since last update
     * @return T_current_from_last: Motion transform in camera frame
     */
    Eigen::Matrix4d getMotionDelta();

    /**
     * Get current VIO camera pose
     */
    Eigen::Matrix4d getCurrentVIOPose() const { return current_vio_camera_pose_; }

    /**
     * Get last VIO camera pose
     */
    Eigen::Matrix4d getLastVIOPose() const { return last_vio_camera_pose_; }

    /**
     * Check if VIO poses are available
     */
    bool hasValidPoses() const { return has_poses_; }

    /**
     * Reset motion estimator state
     */
    void reset();

    /**
     * Debugging utilities
     */
    void debugMotion(const std::string& context) const;
    bool validateTransformMatrix(const Eigen::Matrix4d& T, const std::string& name) const;

private:
    // VIO pose tracking
    Eigen::Matrix4d last_vio_camera_pose_;
    Eigen::Matrix4d current_vio_camera_pose_;
    bool has_poses_;

    // Motion tracking
    std::chrono::steady_clock::time_point last_update_time_;

    // Utilities
    static void printTransform(const Eigen::Matrix4d& T, const std::string& name);
};

} // namespace lar

#endif /* LAR_TRACKING_VIO_MOTION_ESTIMATOR_H */