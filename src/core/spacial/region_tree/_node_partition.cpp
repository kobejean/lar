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

template <typename T>
using overflow_collection = typename _Node<T>::overflow_collection;

template <typename T>
void _linearPickSeeds(overflow_collection<T> &nodes, _Node<T> **seed1, _Node<T> **seed2);

template <typename T>
void _distribute(overflow_collection<T> &nodes, _Node<T> *lower_split, _Node<T> *upper_split);

// partition strategy based on: https://www.just.edu.jo/~qmyaseen/rtree1.pdf
template <typename T>
void _Node<T>::partition(overflow_collection &nodes, _Node *lower_split, _Node *upper_split) {
  _Node *seed1, *seed2;
  _linearPickSeeds<T>(nodes, &seed1, &seed2);

  lower_split->bounds = seed1->bounds;
  lower_split->linkChild(seed1);
  upper_split->bounds = seed2->bounds;
  upper_split->linkChild(seed2);

  _distribute<T>(nodes, lower_split, upper_split);
}


// linearPickSeeds()

template <typename T>
void _linearPickSeeds(overflow_collection<T> &nodes, _Node<T> **seed1, _Node<T> **seed2) {
  double lowest = nodes[0]->bounds.lower.l1();
  double highest = nodes[0]->bounds.upper.l1();
  size_t lx = 0;
  size_t hx = 0;

  for (size_t i = 1; i < nodes.size(); i++) {
    double lowxy = nodes[i]->bounds.lower.l1();
    
    if (lowxy < lowest) {
      lowest = lowxy;
      lx = i;
    }
  }
  *seed1 = nodes[lx];
  nodes.erase(lx);

  for (size_t i = 1; i < nodes.size(); i++) {\
    double highxy = nodes[i]->bounds.upper.l1();
    
    if (highxy > highest) {
      highest = highxy;
      hx = i;
    }
  }
  *seed2 = nodes[hx];
  nodes.erase(hx);

  // // break tie
  // if (lx == hx) {
  //   double lowest2 = nodes[0]->bounds.lower.l1();
  //   size_t original_lx = lx;
  //   lx = 0;
  //   if (original_lx == 0) {
  //     lowest2 = nodes[1]->bounds.lower.l1();
  //     lx = 1;
  //   }
  //   for (size_t i = 1; i < nodes.size(); i++) {
  //     double lowxy = nodes[i]->bounds.lower.l1();
  //     if (i != original_lx && lowxy < lowest2) {
  //       lowest2 = lowxy;
  //       lx = i;
  //     }
  //   }
  // }

  // *seed1 = nodes[lx];
  // *seed2 = nodes[hx];
  // // remove seeds from node by swapping in the last child
  // if (hx == nodes.size() - 1) {
  //   nodes.pop_back();
  //   nodes[lx] = nodes.back();
  //   nodes.pop_back();
  // } else {
  //   nodes[lx] = nodes.back();
  //   nodes.pop_back();
  //   nodes[hx] = nodes.back();
  //   nodes.pop_back();
  // }
}


// distribute()

template <typename T, typename Comparator>
void _populateSplit(overflow_collection<T> &nodes, size_t m, _Node<T> *split, Comparator comp);

template <typename T>
void _distribute(overflow_collection<T> &nodes, _Node<T> *lower_split, _Node<T> *upper_split) {
  Point lower_vert = lower_split->children[0]->bounds.lower; // lower vertex of seed seed1
  Point upper_vert = upper_split->children[0]->bounds.upper; // upper vertex of seed seed2
  size_t m = nodes.size() / 2;


  // score nodes by upper vertex distance to lower_vert
  _populateSplit<T>(nodes, m, lower_split, [&lower_vert] (const _Node<T> *a, const _Node<T> *b) {
    return a->bounds.upper.dist2(lower_vert) < b->bounds.upper.dist2(lower_vert);
  });

  // score nodes by lower vertex distance to upper_vert
  _populateSplit<T>(nodes, m, upper_split, [&upper_vert] (const _Node<T> *a, const _Node<T> *b) {
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

template <typename T, typename Comparator>
void _populateSplit(overflow_collection<T> &nodes, size_t m, _Node<T> *split, Comparator comp) {
  std::partial_sort(nodes.begin(), nodes.begin() + m, nodes.end(), comp);
  // add the closest m nodes to split
  for (size_t i = 0; i < m; i++) {
    _Node<T> *node = nodes[i];
    split->bounds = split->bounds.minBoundingBox(node->bounds);
    split->linkChild(node);
  }
  nodes.pop_front(m);
}


} // namespace

} // namespace lar