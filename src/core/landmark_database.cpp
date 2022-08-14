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
    for (const Landmark& landmark : landmarks) {
      Point center = Point(landmark.cam_position.x(), landmark.cam_position.y());
      // std::cout << "landmark.distance: " << landmark.distance << std::endl;
      Rect bounds(center, landmark.distance, landmark.distance);
      _rtree.insert(landmark.id, bounds, landmark.id);
    }
  }

  std::vector<Landmark> LandmarkDatabase::find(const Rect &query) const {
    std::vector<size_t> ids = _rtree.find(query);
    std::vector<Landmark> result;
    std::transform(ids.cbegin(), ids.cend(), std::back_inserter(result),
                   [this](size_t id) { return all[id]; });
    return result;
  }

  size_t LandmarkDatabase::size() const {
    return all.size();
  }

#ifndef LAR_COMPACT_BUILD

  void LandmarkDatabase::cull() {
    // TODO: cull rtree nodes as well
    std::vector<Landmark> landmarks;
    for (Landmark& landmark : all) {
      if (landmark.isUseable()) {
        landmark.id = landmarks.size();
        landmarks.push_back(landmark);
      }
    }
    all = landmarks;
  }
  
#endif

}