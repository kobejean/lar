#ifndef GEOAR_MAP_H
#define GEOAR_MAP_H

#include "geoar/core/landmark_database.h"

namespace geoar {

  class Map {
    public: 
      LandmarkDatabase landmarkDatabase;

      Map();
  };
}

#endif /* GEOAR_MAP_H */