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


template <typename T>
_Node<T>::_Node() {
  
}

template <typename T>
_Node<T>::_Node(T value, Rect bounds, size_t id) : bounds(bounds), value(value), id(id) {

};

template <typename T>
_Node<T>::~_Node() {
  for (size_t i = 0; i < children.size(); i++) {
    delete children[i];
  }
  // for (auto &child : children) {
  //   delete child;
  // }
};

template <typename T>
_Node<T>* _Node<T>::insert(_Node *node) {
  this->bounds = this->bounds.minBoundingBox(node->bounds);

  if (!this->children[0]->isLeaf()) {
    _Node<T> *best_child = this->findBestInsertChild(node->bounds);
    _Node<T> *child_split = best_child->insert(node);
    return child_split == nullptr ? nullptr : this->addChild(child_split);
  } else {
    return this->addChild(node);
  }
}

template <typename T>
void _Node<T>::erase() {
  // if (this->parent != nullptr) {

  // }
  delete this;
}

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

template <typename T>
inline bool _Node<T>::isLeaf() const {
  return this->children.size() == 0;
}

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
  _InsertScore<T> best_score(this->children[0], bounds);
  _Node<T> *best_child = this->children[0];
  // find the best child to insert into
  for (auto &child : this->children) {
    _InsertScore<T> score(child, bounds);
    if (best_score < score ) {
      best_score = score;
      best_child = child;
    }
  }
  return best_child;
}


template <typename T>
_Node<T> *_Node<T>::addChild(_Node *child) {
  if (this->children.size() < MAX_CHILDREN) {
    linkChild(child);
    return nullptr;
  } else {
    overflow_collection nodes;
    for (auto &child : this->children) nodes.push_back(child);
    nodes.push_back(child);
    _Node<T> *split = new _Node<T>();
    // reset parent
    this->children.clear();
    partition(nodes, this, split);
    return split;
  }
}


template <typename T>
void _Node<T>::linkChild(_Node *child) {
  this->children.push_back(child);
  child->parent = this;
}

} // namespace

} // namespace lar