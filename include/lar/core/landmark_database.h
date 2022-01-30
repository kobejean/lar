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

      void insert(std::vector<Landmark> &landmarks);
      size_t size() const;
      void cull();
      cv::Mat getDescriptions();
  };
  
}

#endif /* LAR_CORE_LANDMARK_DATABASE_H */