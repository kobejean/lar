#ifndef LAR_CORE_SPACIAL_REGION_TREE_H
#define LAR_CORE_SPACIAL_REGION_TREE_H

#include <array>
#include <vector>
#include <iostream>
#include "lar/core/spacial/rectangle.h"

namespace lar {

  class RegionTree {
    public: 
      static const std::size_t MAX_CHILDREN = 10;
      Rectangle bounds;
      int id;
      std::vector<RegionTree> children;

      RegionTree();
      ~RegionTree();
      bool insert(RegionTree &node);
      void find(const Rectangle &query, std::vector<RegionTree> &result);
      void print(std::ostream &os);

    private:
  };

}

#endif /* LAR_CORE_SPACIAL_REGION_TREE_H */