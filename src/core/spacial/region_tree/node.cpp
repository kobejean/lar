#include <iostream>
#include <algorithm>
#include <vector>
#include <cassert>
#include "lar/core/spacial/region_tree.h"
#include "lar/core/landmark.h"

#include "node.h"
#include "node_partition.cpp"

namespace lar {

template <typename T>
RegionTree<T>::Node::Node(std::size_t height) : height(height), parent(nullptr) {
  
}

template <typename T>
RegionTree<T>::Node::~Node() {
  for (auto &child : children) {
    delete child;
  }
};

template <typename T>
typename RegionTree<T>::Node *RegionTree<T>::Node::insert(Node *node) {
  bounds = bounds.minBoundingBox(node->bounds);

  if (height != node->height+1) {
    Node *best_child = findBestInsertChild(node->bounds);
    Node *child_split = best_child->insert(node);
    return child_split == nullptr ? nullptr : addChild(child_split);
  } else {
    return addChild(node);
  }
}

template <typename T>
typename RegionTree<T>::Node *RegionTree<T>::Node::erase() {
  if (parent != nullptr) {
    Node* insert_root = nullptr;
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
void RegionTree<T>::Node::find(const Rect &query, std::vector<T> &result) const { 
  if (this->isLeaf()) {
    const LeafNode *leaf = static_cast<const LeafNode*>(this);
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
void RegionTree<T>::Node::print(std::ostream &os, int depth) const {
  // print leading tabs
  for (int i = 0; i < depth; i++) os << "\t";

  // print node info
  this->bounds.print(os);
  if (this->isLeaf()) {
    const LeafNode *leaf = static_cast<const LeafNode*>(this);
    os << " - #" << leaf->id;
  }
  os << "\n";
  
  // print children
  for (auto & child : this->children) child->print(os, depth + 1);
}

template <typename T>
inline bool RegionTree<T>::Node::isLeaf() const {
  return this->height == 0;
}

template <typename T>
struct RegionTree<T>::Node::_InsertScore {
  double overlap, expansion, area;

  _InsertScore() : overlap(0), expansion(0), area(0) {}
  _InsertScore(const Node *parent, const Rect &bounds) :
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
typename RegionTree<T>::Node *RegionTree<T>::Node::findBestInsertChild(const Rect &bounds) const {
  _InsertScore best_score(children[0], bounds);
  Node *best_child = children[0];
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
typename RegionTree<T>::Node *RegionTree<T>::Node::addChild(Node *child) {
  if (children.size() < MAX_CHILDREN) {
    linkChild(child);
    return nullptr;
  } else {
    overflow_container nodes;
    for (auto &child : children) nodes.push_back(child);
    nodes.push_back(child);
    RegionTree<T>::Node *split = new RegionTree<T>::Node(height);
    // reset parent
    children.clear();
    Partition::partition(nodes, this, split);
    return split;
  }
}


template <typename T>
void RegionTree<T>::Node::linkChild(Node *child) {
  children.push_back(child);
  child->parent = this;
}


template <typename T>
std::size_t RegionTree<T>::Node::findChildIndex(Node *child) const {
  for (size_t i = 0; i < children.size(); i++) {
    if (children[i] == child) return i;
  }
  return -1;
}


template <typename T>
void RegionTree<T>::Node::subtractBounds(const Rect &bounds) {
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
RegionTree<T>::LeafNode::LeafNode(T value, Rect bounds, size_t id) : Node(0), id(id), value(value) {
  this->bounds = bounds;
  this->height = 0;
};

} // namespace lar