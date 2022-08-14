#include "lar/core/landmark.h"

namespace lar {

  Landmark::Landmark() {
  }

  Landmark::Landmark(const Eigen::Vector3d& position, const cv::Mat& desc, size_t id) :
    id(id), position(position), desc(desc) {
  }

  // Static Methods

  cv::Mat Landmark::concatDescriptions(const std::vector<Landmark>& landmarks) {
    cv::Mat desc;
    for (size_t i = 0; i < landmarks.size(); i++) {
      desc.push_back(landmarks[i].desc);
    }
    return desc;
  }

#ifndef LAR_COMPACT_BUILD

  void Landmark::recordObservation(Observation observation) {
    // TODO: find a good way to estimate the region where the landmark can be seen for good indexing
    if (sightings == 0) {
      Eigen::Vector2d position2(position.x(), position.z());
      Eigen::Vector2d cam_position2(observation.cam_position.x(), observation.cam_position.z());
      distance = (position2 - cam_position2).norm();
      cam_position = observation.cam_position;
      orientation = observation.surface_normal;
    }
    sightings++;
    last_seen = observation.timestamp;
  }

  bool Landmark::isUseable() const {
    return sightings >= 3;
  }

#endif

}