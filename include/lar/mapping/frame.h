#ifndef LAR_MAPPING_FRAME_H
#define LAR_MAPPING_FRAME_H

#include <nlohmann/json.hpp>
#include <opencv2/features2d.hpp>

#include "lar/core/utils/json.h"
#include "lar/core/landmark.h"

namespace lar {

  class Frame {
    public:
      // TODO: Make these attributes const
      size_t id;
      long long timestamp;
      Eigen::Matrix3d intrinsics;
      Eigen::Matrix4d extrinsics;
      // Auxilary data
      bool processed{false};
      std::vector<Landmark::Observation> obs;

      Frame();
  };
  NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(Frame, id, timestamp, intrinsics, extrinsics)
}

#endif /* LAR_MAPPING_FRAME_H */