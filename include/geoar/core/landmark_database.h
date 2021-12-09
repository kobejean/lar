#ifndef GEOAR_LANDMARK_DATABASE_H
#define GEOAR_LANDMARK_DATABASE_H

#include "geoar/core/landmark.h"

namespace geoar {

  class LandmarkDatabase {
    public:
      vector<Landmark> landmarks;

      LandmarkDatabase();

      void addLandmarks(vector<Landmark> &landmarks);
  };
}

#endif /* GEOAR_LANDMARK_DATABASE_H */