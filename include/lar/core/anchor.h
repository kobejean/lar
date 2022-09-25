#ifndef LAR_CORE_ANCHOR_H
#define LAR_CORE_ANCHOR_H

#include "lar/core/utils/json.h"
#include <Eigen/Dense>

namespace lar {

  struct Anchor {
    public: 
      int id;
      Eigen::Transform<double,4,Eigen::Affine> transform{Eigen::Transform<double,4,Eigen::Affine>::Identity()};
  
#ifndef LAR_COMPACT_BUILD
      std::size_t frame_id;
      Eigen::Transform<double,4,Eigen::Affine> relative_transform{Eigen::Transform<double,4,Eigen::Affine>::Identity()};
#endif
  };
  
  NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(Anchor, id, transform)
}

#endif /* LAR_CORE_ANCHOR_H */