#ifndef LAR_CORE_LANDMARK_DATABASE_H
#define LAR_CORE_LANDMARK_DATABASE_H

#include <shared_mutex>
#include <nlohmann/json.hpp>
#include "lar/core/utils/json.h"
#include "lar/core/landmark.h"
#include "lar/core/spacial/region_tree.h"

namespace lar {

  class LandmarkDatabase {
    public:

      LandmarkDatabase();
      LandmarkDatabase(const LandmarkDatabase& other) = delete;
      LandmarkDatabase& operator=(const LandmarkDatabase& other) = delete;
      LandmarkDatabase(LandmarkDatabase&& other);
      LandmarkDatabase& operator=(LandmarkDatabase&& other);
      
      Landmark& operator[](size_t id);

      std::vector<size_t> insert(std::vector<Landmark> &landmarks);
      std::vector<Landmark*> find(const Rect &query) const;
      size_t size() const;
      std::vector<Landmark*> all() const;

// #ifndef LAR_COMPACT_BUILD
      void cull();
      void addObservation(size_t id, Landmark::Observation observation);
// #endif
    private:
      RegionTree<Landmark> rtree_;
      std::atomic<size_t> next_id_{0};
      mutable std::shared_mutex mutex_;
  };
  

  static void to_json(nlohmann::json& j, const LandmarkDatabase& l) {
    std::vector<Landmark> landmarks;
    for (Landmark* landmark : l.all()) {
      landmarks.push_back(*landmark);
    }

    // TODO: use hilbert curve ordering to improve bulk insert performance
    std::sort(landmarks.begin(), landmarks.end(), [](const Landmark& a, const Landmark& b) {
      return a.id < b.id;
    });
    j = landmarks;
  }

  static void from_json(const nlohmann::json& j, LandmarkDatabase& l) {
    std::vector<Landmark> landmarks = j;
    l.insert(landmarks);
  }
  
}

#endif /* LAR_CORE_LANDMARK_DATABASE_H */