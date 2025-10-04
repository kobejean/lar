#include "lar/core/landmark_database.h"

namespace lar {

  LandmarkDatabase::LandmarkDatabase() : rtree_() {
  }

  LandmarkDatabase::LandmarkDatabase(LandmarkDatabase&& other) noexcept {
    std::unique_lock<std::shared_mutex> this_lock(mutex_, std::defer_lock);
    std::unique_lock<std::shared_mutex> other_lock(other.mutex_, std::defer_lock);
    std::lock(this_lock, other_lock);

    rtree_ = std::move(other.rtree_);
    next_id_ = other.next_id_;
  }

  LandmarkDatabase& LandmarkDatabase::operator=(LandmarkDatabase&& other) noexcept {
    if (this != &other) {
      std::unique_lock<std::shared_mutex> this_lock(mutex_, std::defer_lock);
      std::unique_lock<std::shared_mutex> other_lock(other.mutex_, std::defer_lock);
      std::lock(this_lock, other_lock);

      rtree_ = std::move(other.rtree_);
      next_id_ = other.next_id_;
    }
    return *this;
  }

  Landmark& LandmarkDatabase::operator[](size_t id) {
    std::shared_lock lock(mutex_);
    return rtree_[id];
  }

  void LandmarkDatabase::insert(std::vector<Landmark>& landmarks, std::vector<Landmark*>* out_pointers) {
    std::unique_lock lock(mutex_);
    for (Landmark& landmark : landmarks) {
      if (landmark.id == 0) {  // Assuming 0 means unassigned
        landmark.id = ++next_id_;
      }
      Landmark* ptr = rtree_.insert(landmark, landmark.bounds, landmark.id);

      if (out_pointers) {
        out_pointers->push_back(ptr);
      }
    }
  }

  void LandmarkDatabase::find(const Rect &query, std::vector<Landmark*> &results, int limit) const {
    std::shared_lock lock(mutex_);
    rtree_.find(query, results);
    if (limit >= 0 && results.size() >= limit) {
      std::partial_sort(results.begin(), results.begin() + limit, results.end(), [](const Landmark* a, const Landmark* b) {
        return a->sightings > b->sightings;
      });
      results.resize(limit);
    }
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
    for (Landmark* landmark : rtree_.all()) {
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
    Rect bounds = landmark.bounds; // Copy because landmark may move
    rtree_.updateBounds(id, bounds);
  }
  
// #endif

}