#ifndef LAR_CORE_ANCHOR_H
#define LAR_CORE_ANCHOR_H

#include "lar/core/utils/json.h"
#include <Eigen/Dense>

namespace lar {

  struct Anchor {
    public: 
      using Transform = Eigen::Transform<double,3,Eigen::Affine>;
      std::size_t id;
      Transform transform;

      Anchor();
      Anchor(std::size_t id, Transform transform);
  
#ifndef LAR_COMPACT_BUILD
      std::size_t frame_id;
      Transform relative_transform;
#endif
  };
  
  NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(Anchor, id, transform)
}

#endif /* LAR_CORE_ANCHOR_H */