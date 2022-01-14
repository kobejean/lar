#ifndef GEOAR_FRAME_H
#define GEOAR_FRAME_H

#include <nlohmann/json.hpp>

#include <opencv2/features2d.hpp>

#include "g2o/types/slam3d/se3quat.h"

#include "geoar/core/landmark.h"

namespace geoar {

  class Frame {
    public:
      size_t id;
      nlohmann::json frame_data;
      nlohmann::json transform;
      nlohmann::json intrinsics;
      g2o::SE3Quat pose;
      std::vector<cv::KeyPoint> kpts;
      std::vector<float> depth;
      std::vector<float> confidence;
      std::vector<size_t> landmarks;

      Frame(nlohmann::json& frame_data);

      static g2o::SE3Quat poseFromTransform(nlohmann::json& t);
      static nlohmann::json transformFromPose(g2o::SE3Quat& pose);
  };
}

#endif /* GEOAR_FRAME_H */