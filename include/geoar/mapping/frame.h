#ifndef GEOAR_MAPPING_FRAME_H
#define GEOAR_MAPPING_FRAME_H

#include <nlohmann/json.hpp>
#include <opencv2/features2d.hpp>

#include "geoar/core/utils/json.h"

namespace geoar {

  class Frame {
    public:
      size_t id;
      long long timestamp;
      Eigen::Matrix3d intrinsics;
      Eigen::Matrix4d extrinsics;
      // Processing
      std::vector<cv::KeyPoint> kpts;
      std::vector<float> depth;
      std::vector<float> confidence;
      std::vector<size_t> landmark_ids;

      Frame();
  };
  NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(Frame, id, timestamp, intrinsics, extrinsics)
}

#endif /* GEOAR_MAPPING_FRAME_H */