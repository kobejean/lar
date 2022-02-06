#ifndef LAR_MAPPING_MAPPER_H
#define LAR_MAPPING_MAPPER_H

#include <filesystem>

#include <Eigen/Core>
#include <nlohmann/json.hpp>
#include <opencv2/core.hpp>

#include "lar/core/utils/json.h"
#include "lar/core/map.h"
#include "lar/mapping/frame.h"
#include "lar/mapping/location_matcher.h"

namespace fs = std::filesystem;

namespace lar {

  class Mapper {
    public:
      class Data {
        public:
          Map map;
          std::vector<Frame> frames;
          std::vector<GPSObservation> gps_obs;
          fs::path directory;

          Data() {};
          fs::path getPathPrefix(int id) {
            std::string id_string = std::to_string(id);
            int zero_count = 8 - static_cast<int>(id_string.length());
            std::string prefix = std::string(zero_count, '0') + id_string + '_';
            return directory / prefix;
          };
      };

      Data data;
      LocationMatcher location_matcher;

      Mapper(fs::path directory);

      void addFrame(Frame frame, cv::InputArray image, cv::InputArray depth, cv::InputArray confidence);
      void addPosition(Eigen::Vector3d position, long long timestamp);
      void addLocation(Eigen::Vector3d location, Eigen::Vector3d accuracy, long long timestamp);
      void writeMetadata();
      void readMetadata();
  };

}

#endif /* LAR_MAPPING_MAPPER_H */