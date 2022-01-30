#ifndef LAR_CORE_MAP_H
#define LAR_CORE_MAP_H

#include "lar/core/landmark_database.h"

namespace lar {

  class Map {
    public: 
      LandmarkDatabase landmarks;

      Map();
  };
}

#endif /* LAR_CORE_MAP_H */