#ifndef LAR_CORE_ANCHOR_H
#define LAR_CORE_ANCHOR_H

#include "lar/core/utils/json.h"
#include <Eigen/Dense>

namespace lar {

  struct Anchor {
    public: 
      int id;
      Eigen::Transform<double,3,Eigen::Affine> transform{Eigen::Transform<double,3,Eigen::Affine>::Identity()};
  };
  
  NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(Anchor, id, transform)
}

#endif /* LAR_CORE_ANCHOR_H */