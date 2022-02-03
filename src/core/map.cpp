#include "lar/core/map.h"

namespace lar {

  Map::Map() {
  }

  bool Map::globalPointFrom(const Eigen::Vector3d& relative, Eigen::Vector3d& global) {
    if (origin.isApprox(Eigen::Transform<double,3,Eigen::Affine>::Identity())) return false;
    global = origin * relative;
    return true;
  }

  bool Map::relativePointFrom(const Eigen::Vector3d& global, Eigen::Vector3d& relative) {
    if (origin.isApprox(Eigen::Transform<double,3,Eigen::Affine>::Identity())) return false;
    relative = origin.inverse() * global;
    return true;
  }

}