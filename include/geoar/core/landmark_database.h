#ifndef GEOAR_LANDMARK_DATABASE_H
#define GEOAR_LANDMARK_DATABASE_H

#include "geoar/core/landmark.h"

namespace geoar {

  class LandmarkDatabase {
    public:
      LandmarkDatabase();
      
      Landmark& operator[](size_t id);

      void insert(std::vector<Landmark> &landmarks);
      size_t size();
    private:
      std::vector<Landmark> _landmarks;
  };
}

#endif /* GEOAR_LANDMARK_DATABASE_H */