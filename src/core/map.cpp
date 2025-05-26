#include <queue>
#include <unordered_set>
#include <algorithm>
#include "lar/core/map.h"

namespace lar {

  Map::Map() {
  }

  Anchor& Map::createAnchor(const Transform &transform) {
    // TODO: reconsider id generation
    std::size_t id = anchors.size();
    auto [it, inserted] = anchors.emplace(id,Anchor{id,transform});
    notifyDidAddAnchor(it->second);
    return it->second;
  }

  void Map::updateAnchor(Anchor& anchor, const Transform& transform) {
    anchor.transform = transform;
    notifyDidUpdateAnchor(anchor);
  }

  void Map::removeAnchor(const Anchor& anchor) {
    std::size_t anchor_id = anchor.id;
    auto it = anchors.find(anchor_id);
    if (it != anchors.end()) {
      notifyWillRemoveAnchor(it->second);
      anchors.erase(it);
      edges.erase(anchor_id);
      for (auto& [id, edge_list] : edges) {
        edge_list.erase(
          std::remove(edge_list.begin(), edge_list.end(), anchor_id),
          edge_list.end()
        );
      }
    }
  }

  void Map::addEdge(std::size_t anchor_id_u, std::size_t anchor_id_v) {
    edges[anchor_id_u].push_back(anchor_id_v);
    edges[anchor_id_v].push_back(anchor_id_u);
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

  void Map::setDidAddAnchorCallback(DidAddAnchorCallback callback) {
    on_did_add_anchor = callback;
  }

  void Map::setDidUpdateAnchorCallback(DidUpdateAnchorCallback callback) {
    on_did_update_anchor = callback;
  }

  void Map::setWillRemoveAnchorCallback(WillRemoveAnchorCallback callback) {
    on_will_remove_anchor = callback;
  }

  void Map::notifyDidAddAnchor(Anchor& anchor) {
    if (on_did_add_anchor) {
        on_did_add_anchor(anchor);
    }
  }

  void Map::notifyDidUpdateAnchor(Anchor& anchor) {
    if (on_did_update_anchor) {
      on_did_update_anchor(anchor);
    }
  }

  void Map::notifyWillRemoveAnchor(Anchor& anchor) {
    if (on_will_remove_anchor) {
      on_will_remove_anchor(anchor);
    }
  }

}