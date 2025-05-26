#ifndef LAR_CORE_MAP_H
#define LAR_CORE_MAP_H

#include <unordered_map>
#include <nlohmann/json.hpp>
#include <Eigen/Dense>
#include "lar/core/utils/json.h"
#include "lar/core/anchor.h"
#include "lar/core/landmark_database.h"

namespace lar {

  class Map {
    public: 
      using Transform = Eigen::Transform<double,3,Eigen::Affine>;
      bool origin_ready = false;
      LandmarkDatabase landmarks;
      Transform origin{Transform::Identity()};
      std::unordered_map<std::size_t, Anchor> anchors;
      std::unordered_map<std::size_t, std::vector<std::size_t>> edges;
      
      Map();
      void updateOrigin(const Transform &origin);
      Anchor& createAnchor(const Transform& transform);
      void updateAnchor(Anchor& anchor, const Transform& transform);
      void removeAnchor(const Anchor& anchor);
      void addEdge(std::size_t anchor_id_u, std::size_t anchor_id_v);
      std::vector<Anchor*> getPath(std::size_t start_id, std::size_t goal_id);
      void globalPointFrom(const Eigen::Vector3d& relative, Eigen::Vector3d& global);
      void globalPointFrom(const Anchor& anchor, Eigen::Vector3d& global);
      void relativePointFrom(const Eigen::Vector3d& global, Eigen::Vector3d& relative);
      
      using DidAddAnchorCallback = std::function<void(Anchor&)>;
      using DidUpdateAnchorCallback = std::function<void(Anchor&)>;
      using WillRemoveAnchorCallback = std::function<void(Anchor&)>;
      void setDidAddAnchorCallback(DidAddAnchorCallback callback);
      void setDidUpdateAnchorCallback(DidUpdateAnchorCallback callback);
      void setWillRemoveAnchorCallback(WillRemoveAnchorCallback callback);
    private:
      DidAddAnchorCallback on_did_add_anchor;
      DidUpdateAnchorCallback on_did_update_anchor;
      WillRemoveAnchorCallback on_will_remove_anchor;
      
      void notifyDidAddAnchor(Anchor& anchor);
      void notifyDidUpdateAnchor(Anchor& anchor);
      void notifyWillRemoveAnchor(Anchor& anchor);
  };
  
  NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(Map, landmarks, origin, anchors, edges, origin_ready)
}

#endif /* LAR_CORE_MAP_H */