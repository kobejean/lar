#ifndef LAR_TRACKING_COORDINATE_TRANSFORM_MANAGER_H
#define LAR_TRACKING_COORDINATE_TRANSFORM_MANAGER_H

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <chrono>

namespace lar {

/**
 * Manages coordinate transformations between VIO and LAR coordinate systems.
 * Handles synchronization of poses and computation of the VIO→LAR transform.
 *
 * COORDINATE CONVENTIONS:
 * - LAR World: Right-handed, Y-up, landmarks and camera poses
 * - VIO World: Right-handed, Y-up, VIO coordinate system (ARKit, ARCore, etc.)
 * - T_lar_from_camera: Camera pose IN LAR world coordinates
 * - T_vio_from_camera: Camera pose IN VIO world coordinates
 * - T_lar_from_vio: Coordinate transform FROM VIO TO LAR world (output)
 */
class CoordinateTransformManager {
public:
    struct TransformResult {
        bool success;
        Eigen::Matrix4d map_to_vio_transform;  // LAR world → VIO world coordinate transform
        double confidence;
        std::chrono::steady_clock::time_point timestamp;
    };

    /**
     * Constructor
     */
    CoordinateTransformManager();

    /**
     * Update current VIO camera pose
     * @param T_vio_from_camera Current VIO camera pose transform
     */
    void updateVIOCameraPose(const Eigen::Matrix4d& T_vio_from_camera);

    /**
     * Compute coordinate transform using synchronized poses
     * @param T_lar_from_camera Camera pose in LAR world (filtered or measured)
     * @param confidence Measurement confidence [0,1]
     * @return Transform result with success status and coordinate transform
     */
    TransformResult computeTransform(const Eigen::Matrix4d& T_lar_from_camera,
                                   double confidence);

    /**
     * Get current coordinate transform (last computed)
     */
    Eigen::Matrix4d getCurrentTransform() const;

    /**
     * Check if transform is available
     */
    bool hasValidTransform() const { return last_result_.success; }

    /**
     * Get age of current transform in seconds
     */
    double getTransformAge() const;

    /**
     * Reset transform state
     */
    void reset();

    /**
     * Debugging utilities
     */
    void debugTransforms(const std::string& context) const;
    bool validateTransformMatrix(const Eigen::Matrix4d& T, const std::string& name) const;

private:
    // VIO pose tracking
    Eigen::Matrix4d current_vio_camera_pose_;
    bool has_vio_pose_;

    // Last computed transform result
    TransformResult last_result_;

    // Utilities
    static void printTransform(const Eigen::Matrix4d& T, const std::string& name);
};

} // namespace lar

#endif /* LAR_TRACKING_COORDINATE_TRANSFORM_MANAGER_H */