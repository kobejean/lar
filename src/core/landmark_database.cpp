#include "geoar/core/landmark_database.h"

namespace geoar {

  LandmarkDatabase::LandmarkDatabase() {
  }

  Landmark& LandmarkDatabase::operator[](size_t id) {
    return _landmarks[id];
  }

  void LandmarkDatabase::insert(std::vector<Landmark> &landmarks) {
    _landmarks.reserve(_landmarks.size() + std::distance(landmarks.begin(), landmarks.end()));
    _landmarks.insert(_landmarks.end(), landmarks.begin(), landmarks.end());
  }

  size_t LandmarkDatabase::size(){
    return _landmarks.size();
  }
}