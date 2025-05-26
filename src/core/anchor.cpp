#include "lar/core/anchor.h"

namespace lar {

  Anchor::Anchor() : 
    id(-1),
    transform(Transform::Identity()),
    frame_id(0),
    relative_transform(Transform::Identity()) {
  }

  Anchor::Anchor(std::size_t id, Transform transform) :
    id(id),
    transform(transform),
    frame_id(0),
    relative_transform(Transform::Identity()) {
  }

}