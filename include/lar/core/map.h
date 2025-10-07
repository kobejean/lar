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
      void rescale(double scale_factor);
      Anchor& createAnchor(const Transform& transform);
      void updateAnchor(Anchor& anchor, const Transform& transform);
      void removeAnchor(const Anchor& anchor);
      void addEdge(std::size_t anchor_id_u, std::size_t anchor_id_v);
      std::vector<Anchor*> getPath(std::size_t start_id, std::size_t goal_id);
      void globalPointFrom(const Eigen::Vector3d& relative, Eigen::Vector3d& global);
      void globalPointFrom(const Anchor& anchor, Eigen::Vector3d& global);
      void relativePointFrom(const Eigen::Vector3d& global, Eigen::Vector3d& relative);
      
      // Bulk notification callbacks
      using DidAddAnchorsCallback = std::function<void(const std::vector<std::reference_wrapper<Anchor>>&)>;
      using DidUpdateAnchorsCallback = std::function<void(const std::vector<std::reference_wrapper<Anchor>>&)>;
      using WillRemoveAnchorsCallback = std::function<void(const std::vector<std::reference_wrapper<const Anchor>>&)>;
      using DidUpdateOriginCallback = std::function<void(const Transform&)>;
      using DidAddEdgeCallback = std::function<void(std::size_t from_id, std::size_t to_id)>;

      void setDidAddAnchorsCallback(DidAddAnchorsCallback callback);
      void setDidUpdateAnchorsCallback(DidUpdateAnchorsCallback callback);
      void setWillRemoveAnchorsCallback(WillRemoveAnchorsCallback callback);
      void setDidUpdateOriginCallback(DidUpdateOriginCallback callback);
      void setDidAddEdgeCallback(DidAddEdgeCallback callback);

      // Bulk operations
      std::vector<std::reference_wrapper<Anchor>> createAnchors(const std::vector<Transform>& transforms);
      void updateAnchors(const std::vector<std::pair<std::reference_wrapper<Anchor>, Transform>>& updates);
      void removeAnchors(const std::vector<std::reference_wrapper<const Anchor>>& anchors);

      // Public notification methods (for BundleAdjustment and other processing classes)
      void notifyDidUpdateAnchors(const std::vector<std::reference_wrapper<Anchor>>& anchors);

    private:
      DidAddAnchorsCallback on_did_add_anchors;
      DidUpdateAnchorsCallback on_did_update_anchors;
      WillRemoveAnchorsCallback on_will_remove_anchors;
      DidUpdateOriginCallback on_did_update_origin;
      DidAddEdgeCallback on_did_add_edge;

      void notifyDidAddAnchors(const std::vector<std::reference_wrapper<Anchor>>& anchors);
      void notifyWillRemoveAnchors(const std::vector<std::reference_wrapper<const Anchor>>& anchors);
      void notifyDidUpdateOrigin(const Transform& origin);
      void notifyDidAddEdge(std::size_t from_id, std::size_t to_id);
  };
  
  NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(Map, landmarks, origin, anchors, edges, origin_ready)
}

#endif /* LAR_CORE_MAP_H */