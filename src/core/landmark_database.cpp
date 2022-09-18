#include "lar/core/landmark_database.h"

namespace lar {

  LandmarkDatabase::LandmarkDatabase() : _rtree() {
  }

  Landmark& LandmarkDatabase::operator[](size_t id) {
    return _rtree[id];
  }

  void LandmarkDatabase::insert(const std::vector<Landmark>& landmarks) {
    for (const Landmark& landmark : landmarks) {
      _rtree.insert(landmark, landmark.bounds, landmark.id);
    }
  }

  std::vector<Landmark> LandmarkDatabase::find(const Rect &query) const {
    return _rtree.find(query);
  }

  size_t LandmarkDatabase::size() const {
    return _rtree.size();
  }

  size_t LandmarkDatabase::createID() {
    return next_id++;
  }

  std::vector<Landmark> LandmarkDatabase::all() const {
    return _rtree.all();
  }

#ifndef LAR_COMPACT_BUILD

  void LandmarkDatabase::cull() {
    // TODO: cull rtree nodes as well
    RegionTree<Landmark> rtree;
    for (Landmark& landmark : all()) {
      if (landmark.isUseable()) {
        rtree.insert(landmark, landmark.bounds, landmark.id);
      }
    }
    _rtree = rtree;
  }
  
#endif

}