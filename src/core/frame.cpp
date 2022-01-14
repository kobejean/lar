#include <Eigen/Core>

#include "geoar/core/frame.h"

namespace geoar {

  Frame::Frame(nlohmann::json& frame_data, size_t id) {
    this->id = id;
    this->frame_data = frame_data;
    this->transform = frame_data["transform"];
    this->intrinsics = frame_data["intrinsics"];
    createPose(transform);
  }

  void Frame::createPose(nlohmann::json& t) {
    Eigen::Matrix3d rot;
    rot << t[0][0], t[1][0], t[2][0],
            t[0][1], t[1][1], t[2][1],
            t[0][2], t[1][2], t[2][2];
    // Flipping y and z axis to align with image coordinates and depth direction
    rot(Eigen::indexing::all, 1) = -rot(Eigen::indexing::all, 1);
    rot(Eigen::indexing::all, 2) = -rot(Eigen::indexing::all, 2);
    
    Eigen::Vector3d position(t[3][0], t[3][1], t[3][2]);
    Eigen::Quaterniond orientation(rot);

    // TODO: see if we can do more numerically stable inverse
    pose = g2o::SE3Quat(orientation, position).inverse();
  }

}