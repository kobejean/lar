#include "geoar/core/landmark.h"

namespace geoar {

  Landmark::Landmark(Eigen::Vector3d &position, cv::KeyPoint &kpt, cv::Mat desc) {
    this->position = position;
    this->kpt = kpt;
    this->desc = desc;
  }

  void Landmark::concatDescriptions(std::vector<Landmark> landmarks, cv::Mat &desc) {
    for (size_t i = 0; i < landmarks.size(); i++) {
      desc.push_back(landmarks[i].desc);
    }
  }

}