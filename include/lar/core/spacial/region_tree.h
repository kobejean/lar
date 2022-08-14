#ifndef LAR_CORE_SPACIAL_REGION_TREE_H
#define LAR_CORE_SPACIAL_REGION_TREE_H

#include <array>
#include <vector>
#include <iostream>
#include "lar/core/spacial/rect.h"

namespace lar {

  template <typename T>
  class RegionTree {
    public: 
      static constexpr std::size_t MAX_CHILDREN = 50;
      Rect bounds;
      T value;
      size_t id;
      std::vector<RegionTree*> children;

      RegionTree();
      RegionTree(T value, Rect bounds, size_t id);
      ~RegionTree();
      void insert(T value, Rect bounds, size_t id);
      std::vector<T> find(const Rect &query) const;
      void print(std::ostream &os);
  };

}

#endif /* LAR_CORE_SPACIAL_REGION_TREE_H */