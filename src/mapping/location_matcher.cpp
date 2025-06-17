#include <algorithm>
#include <tuple>
#include <deque>
#include <iostream>

#include <Eigen/Core>


#include "lar/mapping/location_matcher.h"


namespace lar {

  void LocationMatcher::recordPosition(long long timestamp, Eigen::Vector3d position) {
    positions.push_back(std::tuple<long long, Eigen::Vector3d>{ timestamp, position });
    updateMatches();
  }

  void LocationMatcher::recordLocation(long long timestamp, Eigen::Vector3d location, Eigen::Vector3d accuracy) {
    locations.push_back(std::tuple<long long, Eigen::Vector3d, Eigen::Vector3d>{ timestamp, location, accuracy });
    updateMatches();
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
