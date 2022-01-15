#include "geoar/core/landmark.h"

namespace geoar {

  Landmark::Landmark() {
    
  }

  Landmark::Landmark(Eigen::Vector3d &position, cv::Mat desc, size_t id) {
    this->id = id;
    this->position = position;
    this->desc = desc;
  }

  void Landmark::recordSighting(nlohmann::json &cam_transform) {
    // TODO: find a good way to estimate the region where the landmark can be seen for good indexing
    if (sightings == 0) {
      Eigen::Vector2d position2(position.x(), position.z());
      Eigen::Vector2d cam_position2(cam_transform[3][0], cam_transform[3][2]);
      double distance = (position2 - cam_position2).norm();
      index_radius = distance;
      index_center = cam_position2;
    }
    sightings++;
  }

  bool Landmark::isUseable() const {
    return sightings >= 3;
  }

  // Static Methods

  void Landmark::concatDescriptions(std::vector<Landmark> landmarks, cv::Mat &desc) {
    for (size_t i = 0; i < landmarks.size(); i++) {
      desc.push_back(landmarks[i].desc);
    }
  }

}