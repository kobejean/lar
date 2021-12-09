#include "geoar/core/landmark_database.h"

using namespace Eigen;
using namespace std;

namespace geoar {

  LandmarkDatabase::LandmarkDatabase() {
  }

  void LandmarkDatabase::addLandmarks(vector<Landmark> &landmarks) {
    this->landmarks.reserve(this->landmarks.size() + distance(landmarks.begin(), landmarks.end()));
    this->landmarks.insert(this->landmarks.end(), landmarks.begin(), landmarks.end());
  }

}