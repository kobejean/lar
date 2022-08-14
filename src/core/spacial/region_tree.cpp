#include <iostream>
#include <algorithm>
#include <array>
#include <vector>
#include "lar/core/spacial/region_tree.h"
#include "lar/core/landmark.h"

namespace lar {

namespace {

template <typename T>
RegionTree<T>* _insert(RegionTree<T> *root, RegionTree<T> *node);

template <typename T>
void _print(std::ostream &os, const RegionTree<T> *node, int depth);

template <typename T>
inline bool _isLeaf(const RegionTree<T> *node);

} // namespace







template <typename T>
RegionTree<T>::RegionTree() {
  
}

template <typename T>
RegionTree<T>::RegionTree(T value, Rect bounds, size_t id) : bounds(bounds), value(value), id(id) {
  
}

template <typename T>
RegionTree<T>::~RegionTree() {
  
}

template <typename T>
void RegionTree<T>::insert(T value, Rect bounds, size_t id) {
  RegionTree<T> *node = new RegionTree<T>(value, bounds, id);
  if (this->children.size() == 0) {
    this->bounds = node->bounds;
    this->children.push_back(node);
    return;
  }

  RegionTree<T> *split = _insert(this, node);
  if (split != nullptr) {
    // if we have a spit at root, create a new root
    // with a copy of the old root and the split as a children
    RegionTree<T> *copy = new RegionTree<T>(*this);
    this->children.clear();
    this->children.push_back(copy);
    this->children.push_back(split);
  }
}

template <typename T>
void RegionTree<T>::find(const Rect &query, std::vector<T> &result) {
  if (_isLeaf(this)) {
    result.push_back(this->value);
    return;
  }

  for (auto & child : children) {
    if (child->bounds.intersectsWith(query)) {
      child->find(query, result);
    }
  }
}

template <typename T>
void RegionTree<T>::print(std::ostream &os) {
  _print(os, this, 0);
}






namespace {


template <typename T>
void _partition(std::vector<RegionTree<T>*> &children, RegionTree<T> *lower_split, RegionTree<T> *upper_split);

template <typename T>
void _linearPickSeeds(std::vector<RegionTree<T>*> &children, RegionTree<T> **seed1, RegionTree<T> **seed2);

template <typename T>
void _distribute(std::vector<RegionTree<T>*> &nodes, RegionTree<T> *lower_split, RegionTree<T> *upper_split);

template <typename T>
RegionTree<T>* _addChild(RegionTree<T> *parent, RegionTree<T> *child);

template <typename T>
RegionTree<T> *_findBestInsertChild(RegionTree<T> *root, RegionTree<T> *node);
template <typename T, typename Comparator>
void _populateSplit(std::vector<RegionTree<T>*> &nodes, size_t m, RegionTree<T> *split, Comparator comp);
  
template <typename T>
void _linearPickSeeds(std::vector<RegionTree<T>*> &children, RegionTree<T> **seed1, RegionTree<T> **seed2) {
  double lowest = children[0]->bounds.lower.l1();
  double highest = children[0]->bounds.upper.l1();
  size_t lx = 0;
  size_t hx = 0;

  for (size_t i = 1; i < children.size(); i++) {
    double lowxy = children[i]->bounds.lower.l1();
    double highxy = children[i]->bounds.upper.l1();
    
    if (lowxy < lowest) {
      lowest = lowxy;
      lx = i;
    }
    if (highxy > highest) {
      highest = highxy;
      hx = i;
    }
  }

  // break tie
  if (lx == hx) {
    double lowest2 = children[0]->bounds.lower.l1();
    if (lx == 0) {
      lowest2 = children[1]->bounds.lower.l1();
      lx = 1;
    }
    for (size_t i = 1; i < children.size(); i++) {
      double lowxy = children[i]->bounds.lower.l1();
      if (lowxy != lowxy < lowest2) {
        lowest2 = lowxy;
        lx = i;
      }
    }
  }
  *seed1 = children[lx];
  *seed2 = children[hx];

  // remove seeds from node by swapping in the last child
  children[lx] = children.back();
  children.pop_back();
  children[hx] = children.back();
  children.pop_back();
}


template <typename T, typename Comparator>
void _populateSplit(std::vector<RegionTree<T>*> &nodes, size_t m, RegionTree<T> *split, Comparator comp) {
  std::partial_sort(nodes.begin(), nodes.begin() + m, nodes.end(), comp);
  // add the closest m nodes to split
  for (size_t i = 0; i < m; i++) {
    RegionTree<T> *node = nodes[i];
    split->bounds = split->bounds.minBoundingBox(node->bounds);
    split->children.push_back(node);
  }
  nodes.erase(nodes.begin(), nodes.begin() + m);
}

template <typename T>
void _distribute(std::vector<RegionTree<T>*> &nodes, RegionTree<T> *lower_split, RegionTree<T> *upper_split) {
  Point lower_vert = lower_split->children[0]->bounds.lower; // lower vertex of seed seed1
  Point upper_vert = upper_split->children[0]->bounds.upper; // upper vertex of seed seed2
  size_t m = nodes.size() / 2;


  // score nodes by upper vertex distance to lower_vert
  _populateSplit(nodes, m, lower_split, [&lower_vert] (const RegionTree<T> *a, const RegionTree<T> *b) {
    return a->bounds.upper.dist2(lower_vert) < b->bounds.upper.dist2(lower_vert);
  });

  // score nodes by lower vertex distance to upper_vert
  _populateSplit(nodes, m, upper_split, [&upper_vert] (const RegionTree<T> *a, const RegionTree<T> *b) {
    return a->bounds.lower.dist2(upper_vert) < b->bounds.lower.dist2(upper_vert);
  });

  // distribute remaining nodes to lower and upper splits by whichever is closest
  for (auto & node : nodes) {
    if (node->bounds.upper.dist2(lower_vert) < node->bounds.lower.dist2(upper_vert)) {
      lower_split->bounds = lower_split->bounds.minBoundingBox(node->bounds);
      lower_split->children.push_back(node);
    } else {
      upper_split->bounds = upper_split->bounds.minBoundingBox(node->bounds);
      upper_split->children.push_back(node);
    }
  }
}


// partition strategy based on: https://www.just.edu.jo/~qmyaseen/rtree1.pdf
template <typename T>
void _partition(std::vector<RegionTree<T>*> &nodes, RegionTree<T> *lower_split, RegionTree<T> *upper_split) {
  RegionTree<T> *seed1, *seed2;
  _linearPickSeeds(nodes, &seed1, &seed2);

  lower_split->bounds = seed1->bounds;
  lower_split->children.push_back(seed1);
  upper_split->bounds = seed2->bounds;
  upper_split->children.push_back(seed2);

  _distribute(nodes, lower_split, upper_split);
}


template <typename T>
RegionTree<T>* _insert(RegionTree<T> *root, RegionTree<T> *node) {
  root->bounds = root->bounds.minBoundingBox(node->bounds);

  RegionTree<T> *best_child = _findBestInsertChild(root, node);
  if (best_child != nullptr) {
    RegionTree<T> *child_split = _insert(best_child, node);
    return child_split == nullptr ? nullptr : _addChild(root, child_split);
  } else {
    return _addChild(root, node);
  }
}


/* insert int into array, adjusting capacity when needed */
template <typename T>
RegionTree<T>* _addChild(RegionTree<T> *parent, RegionTree<T> *child) {
  if (parent->children.size() < RegionTree<T>::MAX_CHILDREN) {
    parent->children.push_back(child);
    return nullptr;
  } else {
    std::vector<RegionTree<T>*> nodes(parent->children);
    nodes.push_back(child);
    RegionTree<T> *split = new RegionTree<T>();
    // reset parent
    parent->children.clear();
    _partition(nodes, parent, split);
    return split;
  }
}


template <typename T>
inline bool _isLeaf(const RegionTree<T> *node) {
  return node->children.size() == 0;
}


template <typename T>
void _print(std::ostream &os, const RegionTree<T> *node, int depth) {
  // print leading tabs
  for (int i = 0; i < depth; i++) os << "\t";

  // print node info
  node->bounds.print(os);
  if (_isLeaf(node)) os << " - #" << node->id;
  os  << "\n";
  
  // print children
  for (auto & child : node->children) _print(os, child, depth + 1);
}


template <typename T>
struct _InsertScore {
  double overlap, expansion, area;

  _InsertScore() : overlap(0), expansion(0), area(0) {}
  _InsertScore(const RegionTree<T> *parent, const RegionTree<T> *child) {
    overlap = parent->bounds.overlap(child->bounds);
    area = parent->bounds.area();
    expansion = parent->bounds.minBoundingBox(child->bounds).area() - area;
  }

  bool operator<(const _InsertScore &other) const {
    if (this->overlap < other.overlap) return true;
    if (this->expansion > other.expansion) return true;
    if (this->area < other.area) return true;
    return false;
  }
};

template <typename T>
RegionTree<T> *_findBestInsertChild(RegionTree<T> *root, RegionTree<T> *node) {
  if (_isLeaf(root->children[0])) return nullptr;
  struct _InsertScore<T> best_score;
  RegionTree<T> *best_child = root->children[0];
  // find the best child to insert into
  for (auto & child : root->children) {
    struct _InsertScore<T> score(child, node);
    if (best_score < score ) {
      best_score = score;
      best_child = child;
    }
  }
  return best_child;
}

} // namespace

// explicit instantiations
template class RegionTree<int>;
template class RegionTree<Landmark>;

} // namespace lar