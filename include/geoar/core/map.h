#ifndef MAP_H
#define MAP_H

#include "geoar/core/landmark_database.h"

namespace geoar {

  class Map {
    public: 
      LandmarkDatabase landmarkDatabase;

      Map();
  };
}

#endif /* MAP_H */