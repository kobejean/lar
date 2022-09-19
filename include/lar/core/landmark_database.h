#ifndef LAR_CORE_LANDMARK_DATABASE_H
#define LAR_CORE_LANDMARK_DATABASE_H

#include <nlohmann/json.hpp>
#include "lar/core/utils/json.h"
#include "lar/core/landmark.h"
#include "lar/core/spacial/region_tree.h"

namespace lar {

  class LandmarkDatabase {
    public:

      LandmarkDatabase();
      
      Landmark& operator[](size_t id);

      void insert(const std::vector<Landmark>& landmarks);
      std::vector<Landmark> find(const Rect &query) const;
      size_t size() const;
      size_t createID();
      std::vector<Landmark> all() const;

#ifndef LAR_COMPACT_BUILD
      void cull();
      void addObservation(size_t id, Landmark::Observation observation);
#endif
    private:
      RegionTree<Landmark> _rtree;
      size_t next_id = 0;
  };
  

  static void to_json(nlohmann::json& j, const LandmarkDatabase& l) {
    std::vector<Landmark> landmarks = l.all();

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