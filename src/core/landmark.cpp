#include <algorithm>
#include "lar/core/landmark.h"

namespace lar {

  Landmark::Landmark() {
  }

  Landmark::Landmark(const Eigen::Vector3d& position, const cv::Mat& desc, size_t id) :
    id(id), position(position), desc(desc) {
  }

  // Static Methods

  cv::Mat Landmark::concatDescriptions(const std::vector<Landmark*>& landmarks) {
    cv::Mat desc;
    for (size_t i = 0; i < landmarks.size(); i++) {
      desc.push_back(landmarks[i]->desc);
    }
    return desc;
  }

  // #ifndef LAR_COMPACT_BUILD

  void Landmark::recordObservation(Observation observation) {
      obs.push_back(observation);
      
      // Update orientation (surface normal)
      orientation = observation.surface_normal;
      
      // Extract camera position from pose matrix (translation part)
      Eigen::Vector3d cam_position = observation.cam_pose.block<3,1>(0,3);
      
      // Calculate bounds for indexing (2D projection)
      Eigen::Vector2d position2(position.x(), position.z());
      Eigen::Vector2d cam_position2(cam_position.x(), cam_position.z());
      double bounds_diameter = (position2 - cam_position2).norm() * 2;
      
      // Create Point using 2D camera position for bounds calculation
      Point center = Point(cam_position.x(), cam_position.z());
      
      if (sightings == 0) {
          bounds = Rect(center, bounds_diameter, bounds_diameter);
      } else {
          Rect new_bounds(center, bounds_diameter, bounds_diameter);
          bounds = bounds.minBoundingBox(new_bounds);
      }
      
      sightings++;
      last_seen = observation.timestamp;
  }
  
  bool Landmark::isUseable() const {
    return sightings >= 3;
  }
  // #endif

}