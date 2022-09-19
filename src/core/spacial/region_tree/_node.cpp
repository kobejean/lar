#include <iostream>
#include <algorithm>
#include <vector>
#include <cassert>
#include "lar/core/spacial/region_tree.h"
#include "lar/core/landmark.h"

#include "_node.h"
#include "_node_partition.cpp"

namespace lar {

namespace {


template <typename T, std::size_t N>
_Node<T,N>::_Node(std::size_t height) : height(height), parent(nullptr) {
  
}

template <typename T, std::size_t N>
_Node<T,N>::_Node(T value, Rect bounds, size_t id) : bounds(bounds), value(value), id(id), height(0) {

};

template <typename T, std::size_t N>
_Node<T,N>::~_Node() {
  for (auto &child : children) {
    delete child;
  }
};

template <typename T, std::size_t N>
_Node<T,N>* _Node<T,N>::insert(_Node *node) {
  this->bounds = this->bounds.minBoundingBox(node->bounds);

  if (this->height != node->height+1) {
    _Node<T,N> *best_child = this->findBestInsertChild(node->bounds);
    _Node<T,N> *child_split = best_child->insert(node);
    return child_split == nullptr ? nullptr : this->addChild(child_split);
  } else {
    return this->addChild(node);
  }
}

template <typename T, std::size_t N>
_Node<T,N>* _Node<T,N>::erase() {
  if (this->parent != nullptr) {
    _Node<T,N>* underflow = nullptr;
    size_t index = this->parent->findChildIndex(this);
    this->parent->children.erase(index);
    this->parent->subtractBounds(this->bounds);

    if (this->parent->children.size() < (N / 2) && this->parent->parent != nullptr) {
      child_collection children = this->parent->children;
      this->parent->children.clear();
      underflow = this->parent->erase();
      this->parent = nullptr;
      for (auto &child : children) {
        underflow->insert(child);
      }
    } else {
      underflow = this->parent;
    }
    delete this;
    return underflow;
  }
  return nullptr;
}

template <typename T, std::size_t N>
void _Node<T,N>::find(const Rect &query, std::vector<T> &result) const { 
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

template <typename T, std::size_t N>
void _Node<T,N>::print(std::ostream &os, int depth) const {
  // print leading tabs
  for (int i = 0; i < depth; i++) os << "\t";

  // print node info
  this->bounds.print(os);
  if (this->isLeaf()) os << " - #" << this->id;
  os << "\n";
  
  // print children
  for (auto & child : this->children) child->print(os, depth + 1);
}

template <typename T, std::size_t N>
inline bool _Node<T,N>::isLeaf() const {
  return this->height == 0;
}

template <typename T, std::size_t N>
struct _InsertScore {
  double overlap, expansion, area;

  _InsertScore() : overlap(0), expansion(0), area(0) {}
  _InsertScore(const _Node<T,N> *parent, const Rect &bounds) :
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

template <typename T, std::size_t N>
_Node<T,N> *_Node<T,N>::findBestInsertChild(const Rect &bounds) const {
  _InsertScore<T,N> best_score(this->children[0], bounds);
  _Node<T,N> *best_child = this->children[0];
  // find the best child to insert into
  for (auto &child : this->children) {
    _InsertScore<T,N> score(child, bounds);
    if (best_score < score ) {
      best_score = score;
      best_child = child;
    }
  }
  return best_child;
}


template <typename T, std::size_t N>
_Node<T,N> *_Node<T,N>::addChild(_Node *child) {
  if (this->children.size() < MAX_CHILDREN) {
    linkChild(child);
    return nullptr;
  } else {
    overflow_collection nodes;
    for (auto &child : this->children) nodes.push_back(child);
    nodes.push_back(child);
    _Node<T,N> *split = new _Node<T,N>(this->height);
    // reset parent
    this->children.clear();
    partition(nodes, this, split);
    return split;
  }
}


template <typename T, std::size_t N>
void _Node<T,N>::linkChild(_Node *child) {
  this->children.push_back(child);
  child->parent = this;
}


template <typename T, std::size_t N>
std::size_t _Node<T,N>::findChildIndex(_Node *child) const {
  for (size_t i = 0; i < this->children.size(); i++) {
    if (this->children[i] == child) return i;
  }
  return -1;
}


template <typename T, std::size_t N>
void _Node<T,N>::subtractBounds(const Rect &bounds) {
  if (!bounds.isInsideOf(this->bounds)) {
    this->bounds = children[0]->bounds;
    for (size_t i = 1; i < children.size(); i++) {
      this->bounds = this->bounds.minBoundingBox(children[i]->bounds);
    }
    if (this->parent != nullptr) {
      this->parent->subtractBounds(bounds);
    }
  }
}

} // namespace

} // namespace lar