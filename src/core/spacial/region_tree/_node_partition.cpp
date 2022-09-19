#include <iostream>
#include <algorithm>
#include <array>
#include <vector>
#include <cassert>
#include "lar/core/spacial/region_tree.h"
#include "lar/core/landmark.h"

#include "_node.h"

namespace lar {

namespace {

template <typename T, std::size_t N>
using overflow_collection = typename _Node<T,N>::overflow_collection;

template <typename T, std::size_t N>
void _linearPickSeeds(overflow_collection<T,N> &nodes, _Node<T,N> **seed1, _Node<T,N> **seed2);

template <typename T, std::size_t N>
void _distribute(overflow_collection<T,N> &nodes, _Node<T,N> *lower_split, _Node<T,N> *upper_split);

// partition strategy based on: https://www.just.edu.jo/~qmyaseen/rtree1.pdf
template <typename T, std::size_t N>
void _Node<T,N>::partition(overflow_collection &nodes, _Node *lower_split, _Node *upper_split) {
  _Node *seed1, *seed2;
  _linearPickSeeds<T,N>(nodes, &seed1, &seed2);

  lower_split->bounds = seed1->bounds;
  lower_split->linkChild(seed1);
  upper_split->bounds = seed2->bounds;
  upper_split->linkChild(seed2);

  _distribute<T,N>(nodes, lower_split, upper_split);
}


// linearPickSeeds()

template <typename T, std::size_t N, typename Score, typename Compare>
void _extractSeed(overflow_collection<T,N> &nodes, _Node<T,N> **seed, Score score, Compare comp);

template <typename T, std::size_t N>
void _linearPickSeeds(overflow_collection<T,N> &nodes, _Node<T,N> **seed1, _Node<T,N> **seed2) {
  _extractSeed(nodes, seed1,
    [](const _Node<T,N> *node) { return node->bounds.lower.l1(); },
    [](double value, double best) { return value < best; }
  );
  _extractSeed(nodes, seed2,
    [](const _Node<T,N> *node) { return node->bounds.upper.l1(); },
    [](double value, double best) { return value > best; }
  );
}

// extractSeed()

template <typename T, std::size_t N, typename Score, typename Compare>
void _extractSeed(overflow_collection<T,N> &nodes, _Node<T,N> **seed, Score score, Compare comp) {
  double best = score(nodes[0]);
  size_t index = 0;
  for (size_t i = 1; i < nodes.size(); i++) {
    double value = score(nodes[i]);
    if (comp(value, best)) {
      best = value;
      index = i;
    }
  }
  *seed = nodes[index];
  nodes.erase(index);
}


// distribute()

template <typename T, std::size_t N, typename Comparator>
void _populateSplit(overflow_collection<T,N> &nodes, size_t m, _Node<T,N> *split, Comparator comp);

template <typename T, std::size_t N>
void _distribute(overflow_collection<T,N> &nodes, _Node<T,N> *lower_split, _Node<T,N> *upper_split) {
  Point lower_vert = lower_split->children[0]->bounds.lower; // lower vertex of seed seed1
  Point upper_vert = upper_split->children[0]->bounds.upper; // upper vertex of seed seed2
  size_t m = nodes.size() / 2;


  // score nodes by upper vertex distance to lower_vert
  _populateSplit<T,N>(nodes, m, lower_split, [&lower_vert] (const _Node<T,N> *a, const _Node<T,N> *b) {
    return a->bounds.upper.dist2(lower_vert) < b->bounds.upper.dist2(lower_vert);
  });

  // score nodes by lower vertex distance to upper_vert
  _populateSplit<T,N>(nodes, m, upper_split, [&upper_vert] (const _Node<T,N> *a, const _Node<T,N> *b) {
    return a->bounds.lower.dist2(upper_vert) < b->bounds.lower.dist2(upper_vert);
  });

  // distribute remaining nodes to lower and upper splits by whichever is closest
  for (auto & node : nodes) {
    if (node->bounds.upper.dist2(lower_vert) < node->bounds.lower.dist2(upper_vert)) {
      lower_split->bounds = lower_split->bounds.minBoundingBox(node->bounds);
      lower_split->linkChild(node);
    } else {
      upper_split->bounds = upper_split->bounds.minBoundingBox(node->bounds);
      upper_split->linkChild(node);
    }
  }
}


// populateSplit()

template <typename T, std::size_t N, typename Comparator>
void _populateSplit(overflow_collection<T,N> &nodes, size_t m, _Node<T,N> *split, Comparator comp) {
  std::partial_sort(nodes.begin(), nodes.begin() + m, nodes.end(), comp);
  // add the closest m nodes to split
  for (size_t i = 0; i < m; i++) {
    _Node<T,N> *node = nodes[i];
    split->bounds = split->bounds.minBoundingBox(node->bounds);
    split->linkChild(node);
  }

  nodes.pop_front(m);
}

} // namespace

} // namespace lar