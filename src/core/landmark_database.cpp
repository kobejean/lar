#include "geoar/core/landmark_database.h"

namespace geoar {

  LandmarkDatabase::LandmarkDatabase() {
  }

  Landmark& LandmarkDatabase::operator[](size_t id) {
    return _landmarks[id];
  }
  const Landmark& LandmarkDatabase::operator[](size_t id) const {
    return _landmarks[id];
  }

  void LandmarkDatabase::insert(std::vector<Landmark> &landmarks) {
    _landmarks.reserve(_landmarks.size() + std::distance(landmarks.begin(), landmarks.end()));
    _landmarks.insert(_landmarks.end(), landmarks.begin(), landmarks.end());
  }

  size_t LandmarkDatabase::size() const {
    return _landmarks.size();
  }

  void LandmarkDatabase::cull() {
    std::vector<Landmark> landmarks;
    for (size_t i = 0; i < _landmarks.size(); i++) {
      Landmark& landmark = _landmarks[i];
      if (landmark.isUseable()) {
        landmarks.push_back(landmark);
      }
    }
    _landmarks = landmarks;
  }

  cv::Mat LandmarkDatabase::getDescriptions() {
    cv::Mat desc;
    Landmark::concatDescriptions(_landmarks, desc);
    return desc;
  }

}