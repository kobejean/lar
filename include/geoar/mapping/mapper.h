#ifndef GEOAR_MAPPING_MAPPER_H
#define GEOAR_MAPPING_MAPPER_H

#include <filesystem>

#include <Eigen/Core>
#include <nlohmann/json.hpp>
#include <opencv2/core.hpp>

#include "geoar/core/utils/json.h"

namespace fs = std::filesystem;

namespace geoar {

  class Mapper {
    public:
      struct FrameMetadata {
        int id;
        long int timestamp;
        Eigen::Matrix3d intrinsics;
        Eigen::Matrix4d extrinsics;
      };

      struct GPSObservation {
        long int timestamp;
        Eigen::Vector3d relative;
        Eigen::Vector3d global;
        Eigen::Vector3d accuracy;
      };

      fs::path directory;
      std::vector<FrameMetadata> frames;
      std::vector<GPSObservation> gps_observations;

      Mapper(fs::path directory);

      void addFrame(cv::InputArray image, cv::InputArray depth, cv::InputArray confidence, FrameMetadata metadata);
      void addGPSObservation(GPSObservation observation);
      void writeMetadata();

    private:
      fs::path getPathPrefix(int id);
  };

  NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(Mapper::FrameMetadata, id, timestamp, intrinsics, extrinsics)
  NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(Mapper::GPSObservation, timestamp, relative, global, accuracy)

}

#endif /* GEOAR_MAPPING_MAPPER_H */