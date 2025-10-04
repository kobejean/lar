#include <iostream>
#include <algorithm>
#include <array>
#include <vector>
#include <cassert>
#include "lar/core/spatial/region_tree.h"
#include "lar/core/landmark.h"

#include "node_partition.h"

namespace lar {

// partition strategy based on: https://www.just.edu.jo/~qmyaseen/rtree1.pdf
template <typename T>
void RegionTree<T>::Node::Partition::partition(overflow_container &nodes, Node *lower_split, Node *upper_split) {
  std::unique_ptr<Node> seed1, seed2;
  Partition::linearPickSeeds(nodes, &seed1, &seed2);

  lower_split->bounds = seed1->bounds;
  lower_split->linkChild(std::move(seed1));
  upper_split->bounds = seed2->bounds;
  upper_split->linkChild(std::move(seed2));

  Partition::distribute(nodes, lower_split, upper_split);
}


// linearPickSeeds()

template <typename T>
void RegionTree<T>::Node::Partition::linearPickSeeds(overflow_container &nodes, std::unique_ptr<Node> *seed1, std::unique_ptr<Node> *seed2) {
  extractSeed(nodes, seed1,
    [](const std::unique_ptr<Node> &node) { return node->bounds.lower.l1(); },
    [](double value, double best) { return value < best; }
  );
  extractSeed(nodes, seed2,
    [](const std::unique_ptr<Node> &node) { return node->bounds.upper.l1(); },
    [](double value, double best) { return value > best; }
  );
}

// extractSeed()
template <typename T>
template <typename Score, typename Compare>
void RegionTree<T>::Node::Partition::extractSeed(overflow_container &nodes, std::unique_ptr<Node> *seed, Score score, Compare comp) {
  double best = score(nodes[0]);
  size_t index = 0;
  for (size_t i = 1; i < nodes.size(); i++) {
    double value = score(nodes[i]);
    if (comp(value, best)) {
      best = value;
      index = i;
    }
  }
  *seed = std::move(nodes[index]);
  nodes.erase(index);
}


// distribute()

template <typename T>
void RegionTree<T>::Node::Partition::distribute(overflow_container &nodes, Node *lower_split, Node *upper_split) {
  Point lower_vert = lower_split->children[0]->bounds.lower; // lower vertex of seed seed1
  Point upper_vert = upper_split->children[0]->bounds.upper; // upper vertex of seed seed2
  size_t m = nodes.size() / 2;


  // score nodes by upper vertex distance to lower_vert
  populateSplit(nodes, m, lower_split, [&lower_vert] (const std::unique_ptr<Node> &a, const std::unique_ptr<Node> &b) {
    return a->bounds.upper.dist2(lower_vert) < b->bounds.upper.dist2(lower_vert);
  });

  // score nodes by lower vertex distance to upper_vert
  populateSplit(nodes, m, upper_split, [&upper_vert] (const std::unique_ptr<Node> &a, const std::unique_ptr<Node> &b) {
    return a->bounds.lower.dist2(upper_vert) < b->bounds.lower.dist2(upper_vert);
  });

  // distribute remaining nodes to lower and upper splits by whichever is closest
  for (auto & node : nodes) {
    if (node->bounds.upper.dist2(lower_vert) < node->bounds.lower.dist2(upper_vert)) {
      lower_split->bounds = lower_split->bounds.minBoundingBox(node->bounds);
      lower_split->linkChild(std::move(node));
    } else {
      upper_split->bounds = upper_split->bounds.minBoundingBox(node->bounds);
      upper_split->linkChild(std::move(node));
    }
  }
}


// populateSplit()

template <typename T>
template <typename Comparator>
void RegionTree<T>::Node::Partition::populateSplit(overflow_container &nodes, size_t m, Node *split, Comparator comp) {
  std::partial_sort(nodes.begin(), nodes.begin() + m, nodes.end(), comp);
  // add the closest m nodes to split
  for (size_t i = 0; i < m; i++) {
    split->bounds = split->bounds.minBoundingBox(nodes[i]->bounds);
    split->linkChild(std::move(nodes[i]));
  }

  nodes.pop_front(m);
}

} // namespace lar