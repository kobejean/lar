#ifndef GEOAR_FRAME_H
#define GEOAR_FRAME_H

#include <nlohmann/json.hpp>

#include <opencv2/features2d.hpp>

#include "g2o/types/slam3d/se3quat.h"

#include "geoar/core/landmark.h"

namespace geoar {

  class Frame {
    public:
      nlohmann::json frame_data;
      nlohmann::json transform;
      nlohmann::json intrinsics;
      g2o::SE3Quat pose;
      vector<cv::KeyPoint> kpts;
      vector<double> depth;
      vector<size_t> landmarks;

      Frame(nlohmann::json& frame_data);

    private:

      void createPose(nlohmann::json& t);
  };
}

#endif /* GEOAR_FRAME_H */