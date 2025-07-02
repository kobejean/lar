#include <algorithm>
#include <tuple>
#include <deque>
#include <iostream>

#include <Eigen/Core>


#include "lar/mapping/location_matcher.h"
#include "lar/mapping/frame.h"


namespace lar {

  void LocationMatcher::recordPosition(long long timestamp, Eigen::Vector3d position) {
    positions.push_back(std::tuple<long long, Eigen::Vector3d>{ timestamp, position });
    updateMatches();
  }

  void LocationMatcher::recordLocation(long long timestamp, Eigen::Vector3d location, Eigen::Vector3d accuracy) {
    locations.push_back(std::tuple<long long, Eigen::Vector3d, Eigen::Vector3d>{ timestamp, location, accuracy });
    updateMatches();
  }

  void LocationMatcher::reinterpolateMatches(const std::vector<Frame>& frames) {
    // Re-interpolate GPS observations using updated frame positions
    for (auto& observation : matches) {
      long long timestamp = observation.timestamp;
      
      // Find frames that bracket this timestamp
      const Frame* prev_frame = nullptr;
      const Frame* next_frame = nullptr;
      
      for (size_t i = 0; i < frames.size(); i++) {
        const Frame& frame = frames[i];
        
        if (frame.timestamp <= timestamp) {
          prev_frame = &frame;
        }
        if (frame.timestamp >= timestamp && next_frame == nullptr) {
          next_frame = &frame;
          break;
        }
      }
      
      if (prev_frame && next_frame && prev_frame != next_frame) {
        // Interpolate position between the two frames
        long long prev_t = prev_frame->timestamp;
        long long next_t = next_frame->timestamp;
        Eigen::Vector3d prev_position = prev_frame->extrinsics.block<3,1>(0,3);
        Eigen::Vector3d next_position = next_frame->extrinsics.block<3,1>(0,3);
        
        long long denom = next_t - prev_t;
        double a = denom != 0 ? (timestamp - prev_t) / (double)denom : 0.5;
        
        // Update the relative position with interpolated position
        observation.relative = a * (next_position - prev_position) + prev_position;
      } else if (prev_frame) {
        // Use the closest frame if we can't interpolate
        observation.relative = prev_frame->extrinsics.block<3,1>(0,3);
      } else if (next_frame) {
        observation.relative = next_frame->extrinsics.block<3,1>(0,3);
      }
      // If no frames found, keep the original relative position
    }
  }

  // Private Methods
      
  void LocationMatcher::updateMatches() {
    // TODO: match location to frame
    if (locations.empty() || positions.size() < 2) return;
    
    auto location = locations.front();
    auto is_early = [&location](std::tuple<long long, Eigen::Vector3d> position){ return std::get<0>(position) < std::get<0>(location);};
    std::deque<std::tuple<long long, Eigen::Vector3d>>::iterator pp;

    while (!locations.empty() && (pp = std::partition_point(positions.begin(), positions.end(), is_early)) != positions.end()) {
      auto prev = pp;
      auto next = pp;
      long long timestamp = std::get<0>(locations.front());

      if (pp != positions.begin()) {
        prev = std::prev(pp);
      } else if (timestamp == std::get<0>(*pp) && std::next(pp) != positions.end()) {
        next = std::next(pp);
      }

      if (next != prev) {
        long long next_t = std::get<0>(*next);
        long long prev_t = std::get<0>(*prev);
        Eigen::Vector3d next_position = std::get<1>(*next);
        Eigen::Vector3d prev_position = std::get<1>(*prev);
        long long denom = next_t - prev_t;
        double a = denom != 0 ? (timestamp - prev_t) / (double)denom : 0.5;

        Eigen::Vector3d position = a * (next_position - prev_position) + prev_position;

        GPSObservation observation{
          .timestamp=timestamp,
          .relative=position,
          .global=std::get<1>(locations.front()),
          .accuracy=std::get<2>(locations.front()),
        };
        matches.push_back(observation);

        positions.erase(positions.begin(), prev);
      }
      
      locations.pop_front();
	    if (!locations.empty()) {
        location = locations.front();
      }
    }

    if (!locations.empty()) {
      // If there are still locations, we need to wait for more positions
      // but we can discard all positions except the last one. 
      positions.erase(positions.begin(), std::prev(positions.end()));
    }
  }
}
