#include <queue>
#include <unordered_set>
#include <algorithm>
#include <iostream>
#include "lar/core/map.h"

namespace lar {

  Map::Map() {
  }

  // Bulk operations
  std::vector<std::reference_wrapper<Anchor>> Map::createAnchors(const std::vector<Transform>& transforms) {
    std::vector<std::reference_wrapper<Anchor>> created;
    created.reserve(transforms.size());

    for (const auto& transform : transforms) {
      // Generate ID as max existing ID + 1 to avoid collisions after deletions
      std::size_t id = 0;
      if (!anchors.empty()) {
        auto max_it = std::max_element(anchors.begin(), anchors.end(),
          [](const auto& a, const auto& b) { return a.first < b.first; });
        id = max_it->first + 1;
      }
      auto [it, inserted] = anchors.emplace(id, Anchor{id, transform});
      created.push_back(it->second);
    }

    notifyDidAddAnchors(created);
    return created;
  }

  void Map::updateAnchors(const std::vector<std::pair<std::reference_wrapper<Anchor>, Transform>>& updates) {
    std::vector<std::reference_wrapper<Anchor>> updated;
    updated.reserve(updates.size());

    for (auto& [anchor_ref, transform] : updates) {
      anchor_ref.get().transform = transform;
      updated.push_back(anchor_ref);
    }

    notifyDidUpdateAnchors(updated);
  }

  void Map::removeAnchors(const std::vector<std::reference_wrapper<const Anchor>>& anchors_to_remove) {
    // Notify BEFORE deletion so delegates have access to anchor data
    notifyWillRemoveAnchors(anchors_to_remove);

    // Now delete
    for (const auto& anchor_ref : anchors_to_remove) {
      std::size_t anchor_id = anchor_ref.get().id;
      anchors.erase(anchor_id);
      edges.erase(anchor_id);
      for (auto& [id, edge_list] : edges) {
        edge_list.erase(
          std::remove(edge_list.begin(), edge_list.end(), anchor_id),
          edge_list.end()
        );
      }
    }
  }

  // Convenience methods - single operations call bulk internally
  Anchor& Map::createAnchor(const Transform &transform) {
    auto result = createAnchors({transform});
    return result[0].get();
  }

  void Map::updateAnchor(Anchor& anchor, const Transform& transform) {
    updateAnchors({{std::ref(anchor), transform}});
  }

  void Map::removeAnchor(const Anchor& anchor) {
    removeAnchors({std::cref(anchor)});
  }

  void Map::addEdge(std::size_t anchor_id_u, std::size_t anchor_id_v) {
    edges[anchor_id_u].push_back(anchor_id_v);
    edges[anchor_id_v].push_back(anchor_id_u);
    notifyDidAddEdge(anchor_id_u, anchor_id_v);
  }

  void Map::removeEdge(std::size_t anchor_id_u, std::size_t anchor_id_v) {
    // Remove anchor_id_v from anchor_id_u's edge list
    auto& edges_u = edges[anchor_id_u];
    edges_u.erase(
      std::remove(edges_u.begin(), edges_u.end(), anchor_id_v),
      edges_u.end()
    );

    // Remove anchor_id_u from anchor_id_v's edge list
    auto& edges_v = edges[anchor_id_v];
    edges_v.erase(
      std::remove(edges_v.begin(), edges_v.end(), anchor_id_u),
      edges_v.end()
    );

    notifyDidRemoveEdge(anchor_id_u, anchor_id_v);
  }

  std::vector<Anchor*> Map::getPath(std::size_t start_id, std::size_t goal_id) {
    std::vector<Anchor*> path;
    std::unordered_map<std::size_t, double> g_score;
    std::unordered_map<std::size_t, std::size_t> came_from;
    std::unordered_set<std::size_t> closed_set;
    std::priority_queue<std::pair<double, std::size_t>, 
                       std::vector<std::pair<double, std::size_t>>, 
                       std::greater<>> open_queue;
    
    // Heuristic: Euclidean distance between anchor positions
    auto heuristic = [this](std::size_t from, std::size_t to) {
      return (anchors[to].transform.translation() - 
              anchors[from].transform.translation()).norm();
    };
    
    g_score[start_id] = 0.0;
    open_queue.push({heuristic(start_id, goal_id), start_id});
    
    while (!open_queue.empty()) {
      auto [f_score, current] = open_queue.top();
      open_queue.pop();
      
      if (closed_set.count(current)) continue;
      closed_set.insert(current);
      
      if (current == goal_id) {
        // Reconstruct path
        std::vector<std::size_t> path_ids;
        for (std::size_t node = goal_id; ; node = came_from[node]) {
          path_ids.push_back(node);
          if (node == start_id) break;
        }
        
        std::reverse(path_ids.begin(), path_ids.end());
        for (std::size_t id : path_ids) {
          path.push_back(&anchors[id]);
        }
        break;
      }
      
      for (std::size_t neighbor : edges[current]) {
        if (closed_set.count(neighbor)) continue;
        
        double tentative_g = g_score[current] + 
          (anchors[neighbor].transform.translation() - 
            anchors[current].transform.translation()).norm();
        
        if (!g_score.count(neighbor) || tentative_g < g_score[neighbor]) {
          came_from[neighbor] = current;
          g_score[neighbor] = tentative_g;
          open_queue.push({tentative_g + heuristic(neighbor, goal_id), neighbor});
        }
      }
    }
    return path;
  }

  void Map::updateOrigin(const Transform &origin) {
    this->origin = origin;
    origin_ready = true;
    notifyDidUpdateOrigin(origin);
  }

  void Map::rescale(double scale_factor) {
    if (scale_factor <= 0.0) {
      std::cout << "Invalid scale factor: " << scale_factor << std::endl;
      return;
    }

    // Find anchor point (use first anchor's position, or origin if no anchors)
    Eigen::Vector3d anchor_position = Eigen::Vector3d::Zero();
    if (!anchors.empty()) {
      anchor_position = anchors.begin()->second.transform.translation();
    }

    std::cout << "Rescaling map by factor " << scale_factor
              << " relative to anchor position: " << anchor_position.transpose() << std::endl;

    // Rescale all anchors
    for (auto& [id, anchor] : anchors) {
      Eigen::Vector3d position = anchor.transform.translation();
      Eigen::Vector3d scaled_position = anchor_position + (position - anchor_position) * scale_factor;

      // Update transform with new position (preserving rotation)
      Transform new_transform = anchor.transform;
      new_transform.translation() = scaled_position;
      anchor.transform = new_transform;
    }

    // Rescale all landmarks
    for (Landmark* landmark : landmarks.all()) {
      // Scale position
      landmark->position = anchor_position + (landmark->position - anchor_position) * scale_factor;

      // Scale bounds
      double min_x = landmark->bounds.lower.x;
      double min_z = landmark->bounds.lower.y;
      double max_x = landmark->bounds.upper.x;
      double max_z = landmark->bounds.upper.y;

      double scaled_min_x = anchor_position.x() + (min_x - anchor_position.x()) * scale_factor;
      double scaled_min_z = anchor_position.z() + (min_z - anchor_position.z()) * scale_factor;
      double scaled_max_x = anchor_position.x() + (max_x - anchor_position.x()) * scale_factor;
      double scaled_max_z = anchor_position.z() + (max_z - anchor_position.z()) * scale_factor;

      landmark->bounds = Rect(
        Point(scaled_min_x, scaled_min_z),
        Point(scaled_max_x, scaled_max_z)
      );
    }

    // Notify that all anchors have been updated
    std::vector<std::reference_wrapper<Anchor>> all_anchors;
    for (auto& [id, anchor] : anchors) {
      all_anchors.push_back(anchor);
    }
    notifyDidUpdateAnchors(all_anchors);

    std::cout << "Rescaling complete" << std::endl;
  }

  void Map::globalPointFrom(const Eigen::Vector3d& relative, Eigen::Vector3d& global) {
    global = origin * relative;
  }

  void Map::globalPointFrom(const Anchor& anchor, Eigen::Vector3d& global) {
    global = origin * anchor.transform.translation();
  }

  void Map::relativePointFrom(const Eigen::Vector3d& global, Eigen::Vector3d& relative) {
    // TODO: store inverse for efficiency 
    relative = origin.inverse() * global;
  }

  // Callback setters
  void Map::setDidAddAnchorsCallback(DidAddAnchorsCallback callback) {
    on_did_add_anchors = callback;
  }

  void Map::setDidUpdateAnchorsCallback(DidUpdateAnchorsCallback callback) {
    on_did_update_anchors = callback;
  }

  void Map::setWillRemoveAnchorsCallback(WillRemoveAnchorsCallback callback) {
    on_will_remove_anchors = callback;
  }

  void Map::setDidUpdateOriginCallback(DidUpdateOriginCallback callback) {
    on_did_update_origin = callback;
  }

  void Map::setDidAddEdgeCallback(DidAddEdgeCallback callback) {
    on_did_add_edge = callback;
  }

  void Map::setDidRemoveEdgeCallback(DidRemoveEdgeCallback callback) {
    on_did_remove_edge = callback;
  }

  // Notification methods
  void Map::notifyDidAddAnchors(const std::vector<std::reference_wrapper<Anchor>>& anchors) {
    if (on_did_add_anchors) {
      on_did_add_anchors(anchors);
    }
  }

  void Map::notifyDidUpdateAnchors(const std::vector<std::reference_wrapper<Anchor>>& anchors) {
    if (on_did_update_anchors) {
      on_did_update_anchors(anchors);
    }
  }

  void Map::notifyWillRemoveAnchors(const std::vector<std::reference_wrapper<const Anchor>>& anchors) {
    if (on_will_remove_anchors) {
      on_will_remove_anchors(anchors);
    }
  }

  void Map::notifyDidUpdateOrigin(const Transform& origin) {
    if (on_did_update_origin) {
      on_did_update_origin(origin);
    }
  }

  void Map::notifyDidAddEdge(std::size_t from_id, std::size_t to_id) {
    if (on_did_add_edge) {
      on_did_add_edge(from_id, to_id);
    }
  }

  void Map::notifyDidRemoveEdge(std::size_t from_id, std::size_t to_id) {
    if (on_did_remove_edge) {
      on_did_remove_edge(from_id, to_id);
    }
  }
}
