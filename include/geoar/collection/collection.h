#ifndef GEOAR_COLLECTION_COLLECTION_H
#define GEOAR_COLLECTION_COLLECTION_H

#include <filesystem>

#include <Eigen/Core>
#include <nlohmann/json.hpp>
#include <opencv2/core.hpp>

namespace fs = std::filesystem;

namespace geoar {

  class Collection {
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

      Collection(fs::path directory);

      void addFrame(cv::InputArray image, cv::InputArray depth, cv::InputArray confidence, FrameMetadata metadata);
      void addGPSObservation(GPSObservation observation);

    private:
      fs::path getPathPrefix(int id);
  };

  NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(Collection::FrameMetadata, id, timestamp, intrinsics, extrinsics)
  NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(Collection::GPSObservation, timestamp, relative, global, accuracy)

}

#endif /* GEOAR_COLLECTION_COLLECTION_H */