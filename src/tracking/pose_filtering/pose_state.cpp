//
// PoseState.cpp - Pose state representation for filtering
//

#include "lar/tracking/pose_filtering/pose_state.h"

namespace lar {

Eigen::VectorXd PoseState::toVector() const {
    Eigen::VectorXd vec(SIZE);
    vec.segment<3>(0) = position;
    vec.segment<3>(3) = orientation;
    return vec;
}

void PoseState::fromVector(const Eigen::VectorXd& vec) {
    position = vec.segment<3>(0);
    orientation = vec.segment<3>(3);
}

Eigen::Matrix4d PoseState::toTransform() const {
    return utils::TransformUtils::createTransform(position, orientation);
}

void PoseState::fromTransform(const Eigen::Matrix4d& T) {
    position = utils::TransformUtils::extractPosition(T);
    orientation = utils::TransformUtils::rotationMatrixToAxisAngle(utils::TransformUtils::extractRotation(T));
}

} // namespace lar