#ifndef GEOAR_MAPPING_MAPPER_H
#define GEOAR_MAPPING_MAPPER_H

#include <filesystem>

#include <Eigen/Core>
#include <nlohmann/json.hpp>
#include <opencv2/core.hpp>

#include "geoar/core/utils/json.h"
#include "geoar/mapping/frame.h"
#include "geoar/mapping/location_matcher.h"

namespace fs = std::filesystem;

namespace geoar {

  class Mapper {
    public:
      class Data {
        public:
          Map map;
          std::vector<Frame> frames;
          std::vector<GPSObservation> gps_observations;
          cv::Mat desc;
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

#endif /* GEOAR_MAPPING_MAPPER_H */