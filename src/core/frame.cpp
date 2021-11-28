
#include <stdint.h>

#include <iostream>
#include <unordered_set>

#include <nlohmann/json.hpp>
#include <Eigen/Core>

#include <opencv2/core.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/imgcodecs.hpp>

#include "geoar/core/frame.h"


using namespace Eigen;
using namespace std;
using namespace cv;
using json = nlohmann::json;

namespace geoar {

  Frame::Frame(json& frame_data, std::string directory) {
      transform = frame_data["transform"];
      createPose(transform);
  }

  void Frame::createPose(json& t) {
    Matrix3d rot;
    rot << t[0][0], t[1][0], t[2][0],
            t[0][1], t[1][1], t[2][1],
            t[0][2], t[1][2], t[2][2]; 
    Vector3d position(t[3][0], t[3][1], t[3][2]);
    Quaterniond orientation(rot);

    pose = g2o::SE3Quat(orientation, position).inverse();
  }

}