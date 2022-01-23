#ifndef GEOAR_CORE_LANDMARK_DATABASE_H
#define GEOAR_CORE_LANDMARK_DATABASE_H

#include "geoar/core/landmark.h"

namespace geoar {

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

#endif /* GEOAR_CORE_LANDMARK_DATABASE_H */