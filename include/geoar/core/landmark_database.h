#ifndef LANDMARK_DATABASE_H
#define LANDMARK_DATABASE_H

#include "geoar/core/landmark.h"

namespace geoar {

  class LandmarkDatabase {
    public:
      vector<Landmark> landmarks;

      LandmarkDatabase();

      void addLandmarks(vector<Landmark> &landmarks);
  };
}

#endif /* LANDMARK_DATABASE_H */