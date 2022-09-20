#include <iostream>
#include <algorithm>
#include <array>
#include <vector>
#include <cassert>
#include "lar/core/spacial/region_tree.h"
#include "lar/core/landmark.h"

#include "_node.h"

namespace lar {

template <typename T>
class RegionTree<T>::_Node::_Partition {
  static void linearPickSeeds(overflow_container &nodes, _Node **seed1, _Node **seed2);
  static void distribute(overflow_container &nodes, _Node *lower_split, _Node *upper_split);

  template <typename Score, typename Compare>
  static void extractSeed(overflow_container &nodes, _Node **seed, Score score, Compare comp);

  template <typename Comparator>
  static void populateSplit(overflow_container &nodes, size_t m, _Node *split, Comparator comp);

  friend class RegionTree<T>::_Node;
};

} // namespace lar