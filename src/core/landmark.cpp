#include "lar/core/landmark.h"

namespace lar {

  Landmark::Landmark() {
    
  }

  Landmark::Landmark(Eigen::Vector3d &position, cv::Mat desc, size_t id) {
    this->id = id;
    this->position = position;
    this->desc = desc;
  }

  void Landmark::recordSighting(Eigen::Vector3d &cam_position, long long timestamp) {
    // TODO: find a good way to estimate the region where the landmark can be seen for good indexing
    if (sightings == 0) {
      Eigen::Vector2d position2(position.x(), position.z());
      Eigen::Vector2d cam_position2(cam_position.x(), cam_position.z());
      double distance = (position2 - cam_position2).norm();
      index_radius = distance;
      index_center = cam_position2;
    }
    sightings++;
    last_seen = timestamp;
  }

  bool Landmark::isUseable() const {
    return sightings >= 2;
  }

  // Static Methods

  void Landmark::concatDescriptions(std::vector<Landmark> landmarks, cv::Mat &desc) {
    for (size_t i = 0; i < landmarks.size(); i++) {
      desc.push_back(landmarks[i].desc);
    }
  }

}