#ifndef GEOAR_MAPPING_MAPPER_H
#define GEOAR_MAPPING_MAPPER_H

#include <filesystem>

#include <Eigen/Core>
#include <nlohmann/json.hpp>
#include <opencv2/core.hpp>

#include "geoar/core/utils/json.h"
#include "geoar/mapping/location_matcher.h"

namespace fs = std::filesystem;

namespace geoar {

  class Mapper {
    public:
      struct FrameMetadata {
        int id;
        long long timestamp;
        Eigen::Matrix3d intrinsics;
        Eigen::Matrix4d extrinsics;
      };

      fs::path directory;
      LocationMatcher location_matcher;
      std::vector<FrameMetadata> frames;
      std::vector<GPSObservation> gps_observations;

      Mapper(fs::path directory);

      void addFrame(cv::InputArray image, cv::InputArray depth, cv::InputArray confidence, FrameMetadata metadata);
      void addPosition(Eigen::Vector3d position, long long timestamp);
      void addLocation(Eigen::Vector3d location, Eigen::Vector3d accuracy, long long timestamp);
      void writeMetadata();

    private:
      fs::path getPathPrefix(int id);
  };

  NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(Mapper::FrameMetadata, id, timestamp, intrinsics, extrinsics)

}

#endif /* GEOAR_MAPPING_MAPPER_H */