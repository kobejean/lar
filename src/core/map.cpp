#include "lar/core/map.h"

namespace lar {

  Map::Map() {
  }

  Eigen::Vector3d Map::globalPointFrom(const Eigen::Vector3d& relative) {
    return origin * relative;
  }

  Eigen::Vector3d Map::relativePointFrom(const Eigen::Vector3d& global) {
    return origin.inverse() * global;
  }

}