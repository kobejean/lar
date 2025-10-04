#ifndef LAR_CORE_SPATIAL_REGION_TREE_H
#define LAR_CORE_SPATIAL_REGION_TREE_H

#include <vector>
#include <unordered_map>
#include <memory>
#include <iostream>
#include <functional>
#include "lar/core/spatial/rect.h"

namespace lar {

  /**
   * RegionTree - Spatial index using R-tree structure for fast 2D queries.
   *
   * Thread Safety:
   *   This class is NOT thread-safe and has no internal synchronization.
   *
   *   For concurrent access, use LandmarkDatabase which provides thread-safe
   *   wrappers with std::shared_mutex protection. Direct usage of RegionTree
   *   requires external synchronization.
   *
   *   Design Decision: RegionTree must remain lock-free to avoid potential
   *   deadlocks when called from synchronized wrappers like LandmarkDatabase.
   *
   * Pointer Stability:
   *   Pointers returned by insert() remain valid until erase() is called on
   *   that specific element. Pointers are stable across updateBounds() calls
   *   and tree rebalancing operations.
   */
  template <typename T>
  class RegionTree {
    public:
      // lifecycle
      RegionTree();
      ~RegionTree();
      RegionTree(const RegionTree& other) = delete;
      RegionTree& operator=(const RegionTree& other) = delete;
      RegionTree(RegionTree&& other);
      RegionTree& operator=(RegionTree&& other);

      // operations
      T& operator[](size_t id);

      // Returns pointer to inserted value. Pointer remains valid until erase().
      T* insert(T&& value, Rect bounds, size_t id);

      void erase(size_t id);

      // Updates spatial bounds while preserving pointer stability.
      // Pointers returned by insert() remain valid after this call.
      void updateBounds(size_t id, const Rect &bounds);

      void find(const Rect &query, std::vector<T*> &results) const;
      void print(std::ostream &os);

      // collection
      size_t size() const;
      std::vector<T*> all() const;

    private:
      class Node;
      class LeafNode;
      std::unique_ptr<Node> root_;
      using leaf_container = std::unordered_map<size_t, LeafNode*>;
      leaf_container leaf_map_;

      // Helper to insert node into tree
      void rootInsert(std::unique_ptr<Node> node);
  };

}

#endif /* LAR_CORE_SPATIAL_REGION_TREE_H */