#ifndef LAR_MAPPING_LOCATION_MATCHER_H
#define LAR_MAPPING_LOCATION_MATCHER_H

#include <algorithm>
#include <tuple>
#include <deque>

#include <Eigen/Core>

#include "lar/core/utils/json.h"

namespace lar {

  struct GPSObservation {
    long long timestamp;
    Eigen::Vector3d relative;
    Eigen::Vector3d global;
    Eigen::Vector3d accuracy;
  };
  NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(GPSObservation, timestamp, relative, global, accuracy)

  class LocationMatcher {
    public:
      void recordPosition(long long timestamp, Eigen::Vector3d position);
      void recordLocation(long long timestamp, Eigen::Vector3d location, Eigen::Vector3d accuracy);
      std::vector<GPSObservation> matches;
      
    private:
      std::deque<std::tuple<long long, Eigen::Vector3d>> positions;
      std::deque<std::tuple<long long, Eigen::Vector3d, Eigen::Vector3d>> locations;

      void updateMatches();
  };

}

#endif /* LAR_MAPPING_LOCATION_MATCHER_H */