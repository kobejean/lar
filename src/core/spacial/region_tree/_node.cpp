#include <iostream>
#include <algorithm>
#include <vector>
#include <cassert>
#include "lar/core/spacial/region_tree.h"
#include "lar/core/landmark.h"

#include "_node.h"
#include "_node_partition.cpp"

namespace lar {

template <typename T>
RegionTree<T>::_Node::_Node(std::size_t height) : height(height), parent(nullptr) {
  
}

template <typename T>
RegionTree<T>::_Node::~_Node() {
  for (auto &child : children) {
    delete child;
  }
};

template <typename T>
typename RegionTree<T>::_Node *RegionTree<T>::_Node::insert(_Node *node) {
  bounds = bounds.minBoundingBox(node->bounds);

  if (height != node->height+1) {
    _Node *best_child = findBestInsertChild(node->bounds);
    _Node *child_split = best_child->insert(node);
    return child_split == nullptr ? nullptr : addChild(child_split);
  } else {
    return addChild(node);
  }
}

template <typename T>
typename RegionTree<T>::_Node *RegionTree<T>::_Node::erase() {
  if (parent != nullptr) {
    _Node* insert_root = nullptr;
    size_t index = parent->findChildIndex(this);
    parent->children.erase(index);
    parent->subtractBounds(bounds);

    if (parent->children.size() < (MAX_CHILDREN / 2) && parent->parent != nullptr) {
      children_container siblings = parent->children;
      parent->children.clear();
      insert_root = parent->erase();
      parent = nullptr;
      for (auto &sibling : siblings) {
        insert_root->insert(sibling);
      }
    } else {
      insert_root = parent;
    }
    delete this;
    return insert_root;
  }
  return nullptr;
}

template <typename T>
void RegionTree<T>::_Node::find(const Rect &query, std::vector<T> &result) const { 
  if (this->isLeaf()) {
    const _LeafNode *leaf = static_cast<const _LeafNode*>(this);
    result.push_back(leaf->value);
    return;
  }

  for (auto & child : this->children) {
    if (child->bounds.intersectsWith(query)) {
      child->find(query, result);
    }
  }
}

template <typename T>
void RegionTree<T>::_Node::print(std::ostream &os, int depth) const {
  // print leading tabs
  for (int i = 0; i < depth; i++) os << "\t";

  // print node info
  this->bounds.print(os);
  if (this->isLeaf()) {
    const _LeafNode *leaf = static_cast<const _LeafNode*>(this);
    os << " - #" << leaf->id;
  }
  os << "\n";
  
  // print children
  for (auto & child : this->children) child->print(os, depth + 1);
}

template <typename T>
inline bool RegionTree<T>::_Node::isLeaf() const {
  return this->height == 0;
}

template <typename T>
struct RegionTree<T>::_Node::_InsertScore {
  double overlap, expansion, area;

  _InsertScore() : overlap(0), expansion(0), area(0) {}
  _InsertScore(const _Node *parent, const Rect &bounds) :
    overlap(parent->bounds.overlap(bounds)),
    area(parent->bounds.area()) {
    expansion = parent->bounds.minBoundingBox(bounds).area() - area;
  }

  bool operator<(const _InsertScore &other) const {
    if (overlap < other.overlap) return true;
    if (expansion > other.expansion) return true;
    if (area < other.area) return true;
    return false;
  }
};

template <typename T>
typename RegionTree<T>::_Node *RegionTree<T>::_Node::findBestInsertChild(const Rect &bounds) const {
  _InsertScore best_score(children[0], bounds);
  _Node *best_child = children[0];
  // find the best child to insert into
  for (auto &child : children) {
    _InsertScore score(child, bounds);
    if (best_score < score ) {
      best_score = score;
      best_child = child;
    }
  }
  return best_child;
}


template <typename T>
typename RegionTree<T>::_Node *RegionTree<T>::_Node::addChild(_Node *child) {
  if (children.size() < MAX_CHILDREN) {
    linkChild(child);
    return nullptr;
  } else {
    overflow_container nodes;
    for (auto &child : children) nodes.push_back(child);
    nodes.push_back(child);
    RegionTree<T>::_Node *split = new RegionTree<T>::_Node(height);
    // reset parent
    children.clear();
    partition(nodes, this, split);
    return split;
  }
}


template <typename T>
void RegionTree<T>::_Node::linkChild(_Node *child) {
  children.push_back(child);
  child->parent = this;
}


template <typename T>
std::size_t RegionTree<T>::_Node::findChildIndex(_Node *child) const {
  for (size_t i = 0; i < children.size(); i++) {
    if (children[i] == child) return i;
  }
  return -1;
}


template <typename T>
void RegionTree<T>::_Node::subtractBounds(const Rect &bounds) {
  if (!bounds.isInsideOf(this->bounds)) {
    this->bounds = children[0]->bounds;
    for (size_t i = 1; i < children.size(); i++) {
      this->bounds = this->bounds.minBoundingBox(children[i]->bounds);
    }
    if (parent != nullptr) {
      parent->subtractBounds(bounds);
    }
  }
}

template <typename T>
RegionTree<T>::_LeafNode::_LeafNode(T value, Rect bounds, size_t id) : _Node(0), id(id), value(value) {
  this->bounds = bounds;
  this->height = 0;
};

} // namespace lar