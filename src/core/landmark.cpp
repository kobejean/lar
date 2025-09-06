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

  Rect Landmark::calculateSpatialBounds(const Eigen::Vector3d& landmark_position, 
                                       const std::vector<Eigen::Vector3d>& camera_positions) {
    if (camera_positions.empty()) {
      // Default bounds if no camera positions (1m radius)
      double default_size = 1.0;
      Point lower(landmark_position.x() - default_size, landmark_position.z() - default_size);
      Point upper(landmark_position.x() + default_size, landmark_position.z() + default_size);
      return Rect(lower, upper);
    }

    // Calculate distances from landmark to all cameras
    std::vector<double> distances;
    for (const auto& cam_pos : camera_positions) {
      double dist = (cam_pos - landmark_position).norm();
      distances.push_back(dist);
    }

    double margin = std::max(3.0, *std::max_element(distances.begin(), distances.end()) * 0.05);

    // Get X and Z coordinates of cameras (Y is up in ARKit convention)
    std::vector<double> cam_x_coords, cam_z_coords;
    for (const auto& cam_pos : camera_positions) {
      cam_x_coords.push_back(cam_pos.x());
      cam_z_coords.push_back(cam_pos.z());
    }

    double landmark_x = landmark_position.x();
    double landmark_z = landmark_position.z();

    // Include landmark position in bounds calculation (Colmap version is correct)
    double min_x = *std::min_element(cam_x_coords.begin(), cam_x_coords.end()) - margin;
    double max_x = *std::max_element(cam_x_coords.begin(), cam_x_coords.end()) + margin;
    double min_z = *std::min_element(cam_z_coords.begin(), cam_z_coords.end()) - margin;
    double max_z = *std::max_element(cam_z_coords.begin(), cam_z_coords.end()) + margin;

    Point lower(min_x, min_z);
    Point upper(max_x, max_z);
    return Rect(lower, upper);
  }

  void Landmark::updateBounds(const std::vector<Eigen::Vector3d>& camera_positions) {
    bounds = calculateSpatialBounds(position, camera_positions);
  }

#ifndef LAR_COMPACT_BUILD

  void Landmark::updateBoundsFromObservations() {
    std::vector<Eigen::Vector3d> camera_positions;
    camera_positions.reserve(obs.size());
    
    for (const auto& observation : obs) {
      // Extract camera position from the pose matrix (translation component)
      Eigen::Vector3d cam_pos = observation.cam_pose.block<3, 1>(0, 3);
      camera_positions.push_back(cam_pos);
    }
    
    updateBounds(camera_positions);
  }

#endif

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