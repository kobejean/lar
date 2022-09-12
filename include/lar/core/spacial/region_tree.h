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
      static constexpr std::size_t MAX_CHILDREN = 7;
      
      RegionTree();
      void insert(T value, Rect bounds, size_t id);
      std::vector<T> find(const Rect &query) const;
      void print(std::ostream &os);

    private:
      std::shared_ptr<void> root;
  };

}

#endif /* LAR_CORE_SPACIAL_REGION_TREE_H */