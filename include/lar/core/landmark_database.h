#ifndef LAR_CORE_LANDMARK_DATABASE_H
#define LAR_CORE_LANDMARK_DATABASE_H

#include <shared_mutex>
#include <nlohmann/json.hpp>
#include "lar/core/utils/json.h"
#include "lar/core/landmark.h"
#include "lar/core/spatial/region_tree.h"

namespace lar {

  class LandmarkDatabase {
    public:

      LandmarkDatabase();
      LandmarkDatabase(const LandmarkDatabase& other) = delete;
      LandmarkDatabase& operator=(const LandmarkDatabase& other) = delete;
      LandmarkDatabase(LandmarkDatabase&& other) noexcept;
      LandmarkDatabase& operator=(LandmarkDatabase&& other) noexcept;
      
      Landmark& operator[](size_t id);

      void insert(std::vector<Landmark>& landmarks, std::vector<Landmark*>* pointers = nullptr);
      void find(const Rect &query, std::vector<Landmark*> &results, int limit = -1) const;
      size_t size() const;
      std::vector<Landmark*> all() const;

// #ifndef LAR_COMPACT_BUILD
      void cull();
      void addObservation(size_t id, Landmark::Observation observation);
// #endif

      friend void to_json(nlohmann::json& j, const LandmarkDatabase& l) {
        // TODO: use hilbert curve ordering to improve bulk insert performance
        j = nlohmann::json::array();
        for (Landmark* landmark : l.all()) {
          j.push_back(*landmark);
        }
      }

      friend void from_json(const nlohmann::json& j, LandmarkDatabase& l) {
        std::vector<Landmark> landmarks;
        landmarks.reserve(j.size());
        landmarks = std::move(j);
        l.insert(landmarks);
      }

    private:
      RegionTree<Landmark> rtree_;
      size_t next_id_ = 0;
      mutable std::shared_mutex mutex_;
  };

}

#endif /* LAR_CORE_LANDMARK_DATABASE_H */