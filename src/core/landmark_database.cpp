#include "lar/core/landmark_database.h"

namespace lar {

  LandmarkDatabase::LandmarkDatabase() : _rtree() {
  }

  LandmarkDatabase::LandmarkDatabase(LandmarkDatabase&& other)
    : _rtree(std::move(other._rtree)), next_id(other.next_id.load()) {
  }

  LandmarkDatabase& LandmarkDatabase::operator=(LandmarkDatabase&& other) {
    if (this != &other) {
      _rtree = std::move(other._rtree);
      next_id.store(other.next_id.load());
    }
    return *this;
  }

  Landmark& LandmarkDatabase::operator[](size_t id) {
    return _rtree[id];
  }

  std::vector<size_t> LandmarkDatabase::insert(std::vector<Landmark>& landmarks) {
    std::vector<size_t> ids;
    ids.reserve(landmarks.size());
    
    for (Landmark& landmark : landmarks) {
      if (landmark.id == 0) {  // Assuming 0 means unassigned
        landmark.id = next_id.fetch_add(1);
      }
      ids.push_back(landmark.id);
      _rtree.insert(landmark, landmark.bounds, landmark.id);
    }
    return ids;
  }

  std::vector<Landmark*> LandmarkDatabase::find(const Rect &query) const {
    return _rtree.find(query);
  }

  size_t LandmarkDatabase::size() const {
    return _rtree.size();
  }


  std::vector<Landmark*> LandmarkDatabase::all() const {
    return _rtree.all();
  }

// #ifndef LAR_COMPACT_BUILD

  void LandmarkDatabase::cull() {
    // TODO: find better way to do this
    std::vector<Landmark> landmarks;
    for (Landmark* landmark : all()) {
      landmarks.push_back(*landmark);
    }
    for (Landmark landmark : landmarks) {
      if (!landmark.isUseable()) {
        _rtree.erase(landmark.id);
      }
    }
  }

  void LandmarkDatabase::addObservation(size_t id, Landmark::Observation observation) {
    Landmark &landmark = _rtree[id];
    landmark.recordObservation(observation);
    _rtree.reinsert(id, landmark.bounds);
  }
  
// #endif

}