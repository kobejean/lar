#include "lar/core/landmark_database.h"

namespace lar {

  LandmarkDatabase::LandmarkDatabase() : rtree_() {
  }

  LandmarkDatabase::LandmarkDatabase(LandmarkDatabase&& other)
    : rtree_(std::move(other.rtree_)), next_id_(other.next_id_.load()) {
  }

  LandmarkDatabase& LandmarkDatabase::operator=(LandmarkDatabase&& other) {
    if (this != &other) {
      rtree_ = std::move(other.rtree_);
      next_id_.store(other.next_id_.load());
    }
    return *this;
  }

  Landmark& LandmarkDatabase::operator[](size_t id) {
    std::shared_lock lock(mutex_);
    return rtree_[id];
  }

  std::vector<size_t> LandmarkDatabase::insert(std::vector<Landmark>& landmarks) {
    std::vector<size_t> ids;
    ids.reserve(landmarks.size());
    
    std::unique_lock lock(mutex_);
    for (Landmark& landmark : landmarks) {
      if (landmark.id == 0) {  // Assuming 0 means unassigned
        landmark.id = next_id_.fetch_add(1);
      }
      ids.push_back(landmark.id);
      rtree_.insert(landmark, landmark.bounds, landmark.id);
    }
    return ids;
  }

  std::vector<Landmark*> LandmarkDatabase::find(const Rect &query) const {
    std::shared_lock lock(mutex_);
    return rtree_.find(query);
  }

  size_t LandmarkDatabase::size() const {
    std::shared_lock lock(mutex_);
    return rtree_.size();
  }


  std::vector<Landmark*> LandmarkDatabase::all() const {
    std::shared_lock lock(mutex_);
    return rtree_.all();
  }

// #ifndef LAR_COMPACT_BUILD

  void LandmarkDatabase::cull() {
    std::unique_lock lock(mutex_);
    // TODO: find better way to do this
    std::vector<Landmark> landmarks;
    for (Landmark* landmark : all()) {
      landmarks.push_back(*landmark);
    }
    for (Landmark landmark : landmarks) {
      if (!landmark.isUseable()) {
        rtree_.erase(landmark.id);
      }
    }
  }

  void LandmarkDatabase::addObservation(size_t id, Landmark::Observation observation) {
    std::unique_lock lock(mutex_);
    Landmark &landmark = rtree_[id];
    landmark.recordObservation(observation);
    rtree_.reinsert(id, landmark.bounds);
  }
  
// #endif

}