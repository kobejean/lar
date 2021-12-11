#ifndef GEOAR_FRAME_H
#define GEOAR_FRAME_H

#include <nlohmann/json.hpp>
#include <Eigen/Core>
#include <opencv2/features2d.hpp>

#include "g2o/types/sba/types_six_dof_expmap.h"
#include "g2o/types/slam3d/types_slam3d.h"

#include "geoar/core/landmark.h"

using namespace Eigen;
using json = nlohmann::json;

namespace geoar {

  class Frame {
    public:
      json transform;
      json intrinsics;
      g2o::SE3Quat pose;
      vector<cv::KeyPoint> kpts;
      vector<float> depth;
      vector<size_t> landmarks;

      Frame(json& frame_data);

    private:

      void createPose(json& t);
  };
}

#endif /* GEOAR_FRAME_H */