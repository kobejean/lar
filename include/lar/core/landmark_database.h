#ifndef LAR_CORE_LANDMARK_DATABASE_H
#define LAR_CORE_LANDMARK_DATABASE_H

#include "lar/core/landmark.h"

namespace lar {

  class LandmarkDatabase {
    public:
      std::vector<Landmark> all;

      LandmarkDatabase();
      
      Landmark& operator[](size_t id);
      const Landmark& operator[](size_t id) const;

      void insert(const std::vector<Landmark>& landmarks);
      size_t size() const;
      cv::Mat getDescriptions() const;

#ifndef LAR_COMPACT_BUILD
      void cull();
#endif
  };
  
}

#endif /* LAR_CORE_LANDMARK_DATABASE_H */