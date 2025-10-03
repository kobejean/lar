#include <iostream>
#include <algorithm>
#include <array>
#include <vector>
#include <cassert>
#include "lar/core/spacial/region_tree.h"
#include "lar/core/landmark.h"

#include "node.h"

namespace lar {

template <typename T>
class RegionTree<T>::Node::Partition {
  friend class RegionTree<T>;
public:
  static void partition(overflow_container &children, Node *lower_split, Node *upper_split);
private:
  static void linearPickSeeds(overflow_container &nodes, std::unique_ptr<Node> *seed1, std::unique_ptr<Node> *seed2);
  static void distribute(overflow_container &nodes, Node *lower_split, Node *upper_split);

  template <typename Score, typename Compare>
  static void extractSeed(overflow_container &nodes, std::unique_ptr<Node> *seed, Score score, Compare comp);

  template <typename Comparator>
  static void populateSplit(overflow_container &nodes, size_t m, Node *split, Comparator comp);

};

} // namespace lar