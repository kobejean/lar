#ifndef LAR_CORE_MAP_H
#define LAR_CORE_MAP_H

#include <Eigen/Dense>
#include "lar/core/landmark_database.h"

namespace lar {

  class Map {
    public: 
      LandmarkDatabase landmarks;
      Eigen::Transform<double,3,Eigen::Affine> origin{Eigen::Transform<double,3,Eigen::Affine>::Identity()};

      Map();
      bool globalPointFrom(const Eigen::Vector3d& relative, Eigen::Vector3d& global);
      bool relativePointFrom(const Eigen::Vector3d& global, Eigen::Vector3d& relative);
  };
  
}

#endif /* LAR_CORE_MAP_H */