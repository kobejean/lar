#ifndef LAR_CORE_LANDMARK_DATABASE_H
#define LAR_CORE_LANDMARK_DATABASE_H

#include "lar/core/landmark.h"
#include "lar/core/spacial/region_tree.h"

namespace lar {

  class LandmarkDatabase {
      size_t next_id = 0;
    public:
      std::vector<Landmark> all;

      LandmarkDatabase();
      
      Landmark& operator[](size_t id);
      const Landmark& operator[](size_t id) const;

      void insert(const std::vector<Landmark>& landmarks);
      std::vector<Landmark> find(const Rect &query) const;
      size_t size() const;
      size_t createID();

#ifndef LAR_COMPACT_BUILD
      void cull();
#endif
    // private: // TODO: make private
      RegionTree<size_t> _rtree;
  };
  
}

#endif /* LAR_CORE_LANDMARK_DATABASE_H */