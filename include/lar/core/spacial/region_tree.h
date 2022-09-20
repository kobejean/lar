#ifndef LAR_CORE_SPACIAL_REGION_TREE_H
#define LAR_CORE_SPACIAL_REGION_TREE_H

#include <vector>
#include <unordered_map>
#include <memory>
#include <iostream>
#include "lar/core/spacial/rect.h"
#include "lar/core/landmark.h"

namespace lar {

  template <typename T>
  class RegionTree {
    public: 
      static const std::size_t MAX_CHILDREN;
      std::unordered_map<size_t, T> entities;

      // lifecycle
      RegionTree();

      // operations
      T& operator[](size_t id);
      void insert(T value, Rect bounds, size_t id);
      void erase(size_t id);
      std::vector<T> find(const Rect &query) const;
      void print(std::ostream &os);

      // collection
      size_t size() const;
      std::vector<T> all() const;

    private:
      class _Node;
      std::shared_ptr<_Node> root;
      std::unordered_map<size_t, _Node*> leaf_map;
  };

}

#endif /* LAR_CORE_SPACIAL_REGION_TREE_H */