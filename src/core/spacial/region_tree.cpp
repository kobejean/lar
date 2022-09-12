#include <iostream>
#include <algorithm>
#include <array>
#include <vector>
#include <cassert>
#include "lar/core/spacial/region_tree.h"
#include "lar/core/landmark.h"


namespace lar {

namespace {
  
// Internal RegionTree node declaration
template <typename T>
class _Node {
  public:
    Rect bounds;
    T value;
    size_t id;
    // TODO: use better choice of container
    std::vector<_Node*> children;

    _Node() {};
    _Node(T value, Rect bounds, size_t id) : bounds(bounds), value(value), id(id) {};
    ~_Node() {
      for (auto &child : children) {
        delete child;
      }
    };
    
    _Node* insert(_Node *node);
    void find(const Rect &query, std::vector<T> &result) const;
    void print(std::ostream &os, int depth) const;

    inline bool isLeaf() const;
    _Node *findBestInsertChild(const Rect &bounds) const;
    _Node *addChild(_Node *child);
};

} // namespace




template <typename T>
RegionTree<T>::RegionTree() : root(new _Node<T>()) {
  
}

template <typename T>
void RegionTree<T>::insert(T value, Rect bounds, size_t id) {
  _Node<T> *root = static_cast<_Node<T>*>(this->root.get());

  _Node<T> *node = new _Node(value, bounds, id);
  if (root->children.size() == 0) {
    root->bounds = node->bounds;
    root->children.push_back(node);
    return;
  }

  _Node<T> *split = root->insert(node);
  if (split != nullptr) {
    // if we have a spit at root, create a new root
    // with a copy of the old root and the split as a children
    _Node<T> *copy = new _Node(*root);
    root->children.clear();
    root->children.push_back(copy);
    root->children.push_back(split);
  }
}

template <typename T>
std::vector<T> RegionTree<T>::find(const Rect &query) const {
  _Node<T> *root = static_cast<_Node<T>*>(this->root.get());

  std::vector<T> result;
  if (root->children.size() > 0) root->find(query, result);
  return result;
}

template <typename T>
void RegionTree<T>::print(std::ostream &os) {
  _Node<T> *root = static_cast<_Node<T>*>(this->root.get());

  root->print(os, 0);
}






namespace {


// insert()

template <typename T>
_Node<T>* _Node<T>::insert(_Node<T> *node) {
  this->bounds = this->bounds.minBoundingBox(node->bounds);

  if (!this->children[0]->isLeaf()) {
    _Node<T> *best_child = this->findBestInsertChild(node->bounds);
    _Node<T> *child_split = best_child->insert(node);
    return child_split == nullptr ? nullptr : this->addChild(child_split);
  } else {
    return this->addChild(node);
  }
}

// find()

template <typename T>
void _Node<T>::find(const Rect &query, std::vector<T> &result) const {
  if (this->isLeaf()) {
    result.push_back(this->value);
    return;
  }

  for (auto & child : this->children) {
    if (child->bounds.intersectsWith(query)) {
      child->find(query, result);
    }
  }
}

// print()

template <typename T>
void _Node<T>::print(std::ostream &os, int depth) const {
  // print leading tabs
  for (int i = 0; i < depth; i++) os << "\t";

  // print node info
  this->bounds.print(os);
  if (this->isLeaf()) os << " - #" << this->id;
  os << "\n";
  
  // print children
  for (auto & child : this->children) child->print(os, depth + 1);
}


// isLeaf()

template <typename T>
inline bool _Node<T>::isLeaf() const {
  return this->children.size() == 0;
}


// findBestInsertChild()

template <typename T>
struct _InsertScore {
  double overlap, expansion, area;

  _InsertScore() : overlap(0), expansion(0), area(0) {}
  _InsertScore(const _Node<T> *parent, const Rect &bounds) :
    overlap(parent->bounds.overlap(bounds)),
    area(parent->bounds.area()) {
    expansion = parent->bounds.minBoundingBox(bounds).area() - area;
  }

  bool operator<(const _InsertScore &other) const {
    if (this->overlap < other.overlap) return true;
    if (this->expansion > other.expansion) return true;
    if (this->area < other.area) return true;
    return false;
  }
};


template <typename T>
_Node<T> *_Node<T>::findBestInsertChild(const Rect &bounds) const {
  struct _InsertScore<T> best_score(this->children[0], bounds);
  _Node<T> *best_child = this->children[0];
  // find the best child to insert into
  for (auto &child : this->children) {
    struct _InsertScore<T> score(child, bounds);
    if (best_score < score ) {
      best_score = score;
      best_child = child;
    }
  }
  return best_child;
}


// addChild()

template <typename T>
void _partition(std::vector<_Node<T>*> &children, _Node<T> *lower_split, _Node<T> *upper_split);

/* insert int into array, adjusting capacity when needed */
template <typename T>
_Node<T>* _Node<T>::addChild(_Node<T> *child) {
  if (this->children.size() < RegionTree<T>::MAX_CHILDREN) {
    this->children.push_back(child);
    return nullptr;
  } else {
    std::vector<_Node<T>*> nodes(this->children);
    nodes.push_back(child);
    _Node<T> *split = new _Node<T>();
    // reset parent
    this->children.clear();
    _partition<T>(nodes, this, split);
    return split;
  }
}


// _partition()

template <typename T>
void _linearPickSeeds(std::vector<_Node<T>*> &nodes, _Node<T> **seed1, _Node<T> **seed2);

template <typename T>
void _distribute(std::vector<_Node<T>*> &nodes, _Node<T> *lower_split, _Node<T> *upper_split);

// partition strategy based on: https://www.just.edu.jo/~qmyaseen/rtree1.pdf
template <typename T>
void _partition(std::vector<_Node<T>*> &nodes, _Node<T> *lower_split, _Node<T> *upper_split) {
  _Node<T> *seed1, *seed2;
  _linearPickSeeds<T>(nodes, &seed1, &seed2);

  lower_split->bounds = seed1->bounds;
  lower_split->children.push_back(seed1);
  upper_split->bounds = seed2->bounds;
  upper_split->children.push_back(seed2);

  _distribute<T>(nodes, lower_split, upper_split);
}


// linearPickSeeds()

template <typename T>
void _linearPickSeeds(std::vector<_Node<T>*> &nodes, _Node<T> **seed1, _Node<T> **seed2) {
  double lowest = nodes[0]->bounds.lower.l1();
  double highest = nodes[0]->bounds.upper.l1();
  size_t lx = 0;
  size_t hx = 0;

  for (size_t i = 1; i < nodes.size(); i++) {
    double lowxy = nodes[i]->bounds.lower.l1();
    double highxy = nodes[i]->bounds.upper.l1();
    
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
    double lowest2 = nodes[0]->bounds.lower.l1();
    size_t original_lx = lx;
    lx = 0;
    if (original_lx == 0) {
      lowest2 = nodes[1]->bounds.lower.l1();
      lx = 1;
    }
    for (size_t i = 1; i < nodes.size(); i++) {
      double lowxy = nodes[i]->bounds.lower.l1();
      if (i != original_lx && lowxy < lowest2) {
        lowest2 = lowxy;
        lx = i;
      }
    }
  }
  assert(lx != hx);

  *seed1 = nodes[lx];
  *seed2 = nodes[hx];
  // remove seeds from node by swapping in the last child
  if (hx == nodes.size() - 1) {
    nodes.pop_back();
    nodes[lx] = nodes.back();
    nodes.pop_back();
  } else {
    nodes[lx] = nodes.back();
    nodes.pop_back();
    nodes[hx] = nodes.back();
    nodes.pop_back();
  }
}


// distribute()

template <typename T, typename Comparator>
void _populateSplit(std::vector<_Node<T>*> &nodes, size_t m, _Node<T> *split, Comparator comp);

template <typename T>
void _distribute(std::vector<_Node<T>*> &nodes, _Node<T> *lower_split, _Node<T> *upper_split) {
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
      lower_split->children.push_back(node);
    } else {
      upper_split->bounds = upper_split->bounds.minBoundingBox(node->bounds);
      upper_split->children.push_back(node);
    }
  }
}


// populateSplit()

template <typename T, typename Comparator>
void _populateSplit(std::vector<_Node<T>*> &nodes, size_t m, _Node<T> *split, Comparator comp) {
  std::partial_sort(nodes.begin(), nodes.begin() + m, nodes.end(), comp);
  // add the closest m nodes to split
  for (size_t i = 0; i < m; i++) {
    _Node<T> *node = nodes[i];
    split->bounds = split->bounds.minBoundingBox(node->bounds);
    split->children.push_back(node);
  }
  nodes.erase(nodes.begin(), nodes.begin() + m);
}


} // namespace

// explicit instantiations
template class RegionTree<size_t>;
// template class RegionTree<Landmark>;

} // namespace lar