#include "lar/core/landmark_database.h"

namespace lar {

  LandmarkDatabase::LandmarkDatabase() {
  }

  Landmark& LandmarkDatabase::operator[](size_t id) {
    return all[id];
  }
  const Landmark& LandmarkDatabase::operator[](size_t id) const {
    return all[id];
  }

  void LandmarkDatabase::insert(const std::vector<Landmark>& landmarks) {
    all.reserve(all.size() + std::distance(landmarks.begin(), landmarks.end()));
    all.insert(all.end(), landmarks.begin(), landmarks.end());
  }

  size_t LandmarkDatabase::size() const {
    return all.size();
  }

  void LandmarkDatabase::cull() {
    std::vector<Landmark> landmarks;
    for (size_t i = 0; i < all.size(); i++) {
      Landmark& landmark = all[i];
      if (landmark.isUseable()) {
        landmarks.push_back(landmark);
      }
    }
    all = landmarks;
  }

  cv::Mat LandmarkDatabase::getDescriptions() const {
    cv::Mat desc;
    Landmark::concatDescriptions(all, desc);
    return desc;
  }

}