#ifndef LAR_CORE_MAP_H
#define LAR_CORE_MAP_H

#include <Eigen/Dense>
#include "lar/core/utils/json.h"
#include "lar/core/anchor.h"
#include "lar/core/landmark_database.h"

namespace lar {

  class Map {
    public: 
      LandmarkDatabase landmarks;
      Eigen::Transform<double,3,Eigen::Affine> origin{Eigen::Transform<double,3,Eigen::Affine>::Identity()};
      std::vector<Anchor> anchors;

      Map();
      bool globalPointFrom(const Eigen::Vector3d& relative, Eigen::Vector3d& global);
      bool relativePointFrom(const Eigen::Vector3d& global, Eigen::Vector3d& relative);
  };
  

  static void to_json(nlohmann::json& j, const Map& m) {
    j = nlohmann::json{
      {"landmarks", m.landmarks.all},
      {"origin", m.origin},
      {"anchors", m.anchors},
    };
  }

  static void from_json(const nlohmann::json& j, Map& m) {
    std::vector<Landmark> landmarks = j.at("landmarks").get<std::vector<Landmark>>();
    m.landmarks.insert(landmarks);
    j.at("origin").get_to(m.origin);
    j.at("anchors").get_to(m.anchors);
  }
  // NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(Map, landmarks, origin, anchors)
}

#endif /* LAR_CORE_MAP_H */