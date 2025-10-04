#ifndef LAR_TRACKING_POSE_FILTERING_POSE_STATE_H
#define LAR_TRACKING_POSE_FILTERING_POSE_STATE_H

#include <Eigen/Core>
#include <Eigen/Geometry>
#include "lar/core/utils/transform.h"

namespace lar {

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
};

} // namespace lar

#endif /* LAR_TRACKING_POSE_FILTERING_POSE_STATE_H */