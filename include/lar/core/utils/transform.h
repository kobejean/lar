#ifndef LAR_CORE_UTILS_TRANSFORM_H
#define LAR_CORE_UTILS_TRANSFORM_H

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <string>

// Forward declaration for g2o
namespace g2o {
    class SE3Quat;
}

namespace lar {
namespace utils {

/**
 * Transform matrix utilities for 4x4 homogeneous transforms
 */
class TransformUtils {
public:
    /**
     * Convert rotation matrix to axis-angle representation
     * @param R 3x3 rotation matrix
     * @return Axis-angle vector (axis * angle)
     */
    static Eigen::Vector3d rotationMatrixToAxisAngle(const Eigen::Matrix3d& R);

    /**
     * Convert axis-angle representation to rotation matrix
     * @param axis_angle Axis-angle vector (axis * angle)
     * @return 3x3 rotation matrix
     */
    static Eigen::Matrix3d axisAngleToRotationMatrix(const Eigen::Vector3d& axis_angle);

    /**
     * Validate that a 4x4 matrix is a valid homogeneous transform
     * @param T Transform matrix to validate
     * @param name Name for debugging/error messages
     * @return true if valid, false otherwise
     */
    static bool validateTransformMatrix(const Eigen::Matrix4d& T, const std::string& name = "Transform");

    /**
     * Print a transform matrix in readable format for debugging
     * @param T Transform matrix to print
     * @param name Name/description for the transform
     */
    static void printTransform(const Eigen::Matrix4d& T, const std::string& name = "Transform");

    /**
     * Check if two transforms are approximately equal within tolerance
     * @param T1 First transform
     * @param T2 Second transform
     * @param tolerance Maximum allowed difference (default: 1e-6)
     * @return true if transforms are approximately equal
     */
    static bool isApproximatelyEqual(const Eigen::Matrix4d& T1,
                                   const Eigen::Matrix4d& T2,
                                   double tolerance = 1e-6);

    /**
     * Extract position from transform matrix
     * @param T Transform matrix
     * @return 3D position vector
     */
    static Eigen::Vector3d extractPosition(const Eigen::Matrix4d& T);

    /**
     * Extract rotation matrix from transform matrix
     * @param T Transform matrix
     * @return 3x3 rotation matrix
     */
    static Eigen::Matrix3d extractRotation(const Eigen::Matrix4d& T);

    /**
     * Create transform from position and rotation matrix
     * @param position 3D position vector
     * @param rotation 3x3 rotation matrix
     * @return 4x4 transform matrix
     */
    static Eigen::Matrix4d createTransform(const Eigen::Vector3d& position,
                                         const Eigen::Matrix3d& rotation);

    /**
     * Create transform from position and axis-angle rotation
     * @param position 3D position vector
     * @param orientation Axis-angle rotation vector
     * @return 4x4 transform matrix
     */
    static Eigen::Matrix4d createTransform(const Eigen::Vector3d& position,
                                         const Eigen::Vector3d& orientation);

    /**
     * Interpolate between two SE(3) poses using proper Lie group interpolation
     * @param T1 Starting pose
     * @param T2 Ending pose
     * @param alpha Interpolation factor (0.0 = T1, 1.0 = T2)
     * @return Interpolated pose
     */
    static Eigen::Matrix4d interpolatePose(const Eigen::Matrix4d& T1,
                                         const Eigen::Matrix4d& T2,
                                         double alpha);

    // ========================================================================
    // g2o Coordinate System Conversion
    // ========================================================================

    /**
     * Convert ARKit extrinsics to g2o SE3Quat pose
     * Handles coordinate system conversion (Y/Z axis flips) and inversion
     * @param extrinsics ARKit camera extrinsics matrix
     * @return g2o SE3Quat pose for optimization
     */
    static g2o::SE3Quat arkitToG2oPose(const Eigen::Matrix4d& extrinsics);

    /**
     * Convert g2o SE3Quat pose back to ARKit extrinsics
     * Reverses the coordinate system conversion from arkitToG2oPose
     * @param pose g2o SE3Quat pose from optimization
     * @return ARKit camera extrinsics matrix
     */
    static Eigen::Matrix4d g2oToArkitPose(const g2o::SE3Quat& pose);

private:
    static constexpr double DEFAULT_TOLERANCE = 1e-6;
    static constexpr double ORTHOGONALITY_TOLERANCE = 1e-4;
};

} // namespace utils
} // namespace lar

#endif /* LAR_CORE_UTILS_TRANSFORM_H */