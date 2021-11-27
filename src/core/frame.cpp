
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
      createFeatures(frame_data["id"], directory);
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

  void Frame::createFeatures(int id, string directory) {
    // Create filename prefix
    string id_string = to_string(id);
    int zero_count = 8 - id_string.length();
    string prefix = string(zero_count, '0') + id_string + '_';

    string img_filepath = directory + '/' + prefix + "image.jpeg";
    cout << "loading: " << img_filepath << endl;
    Mat image = imread(img_filepath, IMREAD_GRAYSCALE);

    string confidence_filepath = directory + '/' + prefix + "confidence.pfm";
    cout << "loading: " << confidence_filepath << endl;
    Mat confidence = imread(confidence_filepath, IMREAD_UNCHANGED);

    string depth_filepath = directory + '/' + prefix + "depth.pfm";
    cout << "loading: " << depth_filepath << endl;
    Mat depth = imread(depth_filepath, IMREAD_UNCHANGED);
  }

}