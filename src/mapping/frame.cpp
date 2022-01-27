#include <Eigen/Core>

#include "geoar/mapping/frame.h"

namespace geoar {

  Frame::Frame() {
  }
  

  // Static Methods

  // g2o::SE3Quat Frame::poseFromTransform(nlohmann::json& t) {
  //   Eigen::Matrix3d rot;
  //   rot << t[0][0], t[1][0], t[2][0],
  //           t[0][1], t[1][1], t[2][1],
  //           t[0][2], t[1][2], t[2][2];
  //   // Flipping y and z axis to align with image coordinates and depth direction
  //   rot(Eigen::indexing::all, 1) = -rot(Eigen::indexing::all, 1);
  //   rot(Eigen::indexing::all, 2) = -rot(Eigen::indexing::all, 2);
    
  //   Eigen::Vector3d position(t[3][0], t[3][1], t[3][2]);
  //   Eigen::Quaterniond orientation(rot);

  //   // TODO: see if we can do more numerically stable inverse
  //   return g2o::SE3Quat(orientation, position).inverse();
  // }

  // nlohmann::json Frame::transformFromPose(g2o::SE3Quat& pose) {
  //   g2o::SE3Quat pose_inv = pose.inverse();
  //   Eigen::Vector3d t = pose_inv.translation();
  //   Eigen::Matrix3d R = pose_inv.rotation().toRotationMatrix();
  //   // Flipping y and z axis to align with image coordinates and depth direction
  //   R(Eigen::indexing::all, 1) = -R(Eigen::indexing::all, 1);
  //   R(Eigen::indexing::all, 2) = -R(Eigen::indexing::all, 2);

  //   return nlohmann::json::array({
  //     nlohmann::json::array({ R(0,0), R(1,0), R(2,0), 0. }),
  //     nlohmann::json::array({ R(0,1), R(1,1), R(2,1), 0. }),
  //     nlohmann::json::array({ R(0,2), R(1,2), R(2,2), 0. }),
  //     nlohmann::json::array({  t.x(),  t.y(),  t.z(), 1. }),
  //   });
  // }

}