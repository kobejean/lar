#include <iostream>
#include <algorithm>
#include <vector>
#include <cassert>
#include "lar/core/spacial/region_tree.h"
#include "lar/core/landmark.h"

#include "node.h"

// import node partition implementation in same translation unit
// for template instantiation
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
void RegionTree<T>::Node::insert(Node *node) {
  bounds = bounds.minBoundingBox(node->bounds);

  if (height > node->height+1) {
    Node *best_child = findBestInsertChild(node->bounds);
    best_child->insert(node);
  } else {
    assert(height == node->height+1);
    addChild(node);
  }
}

template <typename T>
typename RegionTree<T>::Node *RegionTree<T>::Node::erase() {
  if (parent != nullptr) {
    Node* insert_root = nullptr;
    size_t index = parent->findChildIndex(this);
    parent->children.erase(index);
    assert(parent != nullptr);
    parent->subtractBounds(bounds);

    if (parent->children.size() < (MAX_CHILDREN / 2)) {
      children_container siblings = parent->children;
      // unlink from parent
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
  } else {
    // we will not delete the root
    // we will just return the root as the insert root
    return this;
  }
}

template <typename T>
void RegionTree<T>::Node::find(const Rect &query, std::vector<T*> &result) { 
  if (this->isLeaf()) {
    LeafNode *leaf = static_cast<LeafNode*>(this);
    result.push_back(&leaf->value);
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
struct RegionTree<T>::Node::InsertScore {
  double overlap, expansion, area;

  InsertScore() : overlap(0), expansion(0), area(0) {}
  InsertScore(const Node *parent, const Rect &bounds) :
    overlap(parent->bounds.overlap(bounds)),
    area(parent->bounds.area()) {
    expansion = parent->bounds.minBoundingBox(bounds).area() - area;
  }

  bool operator<(const InsertScore &other) const {
    if (overlap < other.overlap) return true;
    if (expansion > other.expansion) return true;
    if (area < other.area) return true;
    return false;
  }
};

template <typename T>
typename RegionTree<T>::Node *RegionTree<T>::Node::findBestInsertChild(const Rect &bounds) const {
  InsertScore best_score(children[0], bounds);
  Node *best_child = children[0];
  // find the best child to insert into
  for (auto &child : children) {
    InsertScore score(child, bounds);
    if (best_score < score ) {
      best_score = score;
      best_child = child;
    }
  }
  return best_child;
}


template <typename T>
void RegionTree<T>::Node::addChild(Node *child) {
  if (children.size() < MAX_CHILDREN) {
    linkChild(child);
  } else {
    overflow_container nodes;
    for (auto &child : children) nodes.push_back(child);
    nodes.push_back(child);
    RegionTree<T>::Node *split = new RegionTree<T>::Node(height);
    // reset parent
    children.clear();
    Partition::partition(nodes, this, split);
    if (this->parent != nullptr) {
      this->parent->addChild(split);
    } else {
      // if we have a spit at root, create a new root
      // with a copy of the old root and the split as a children
      Node *copy = new Node(*this);
      for (auto &child : children) {
        child->parent = copy;
      }
      children.clear();
      linkChild(copy);
      linkChild(split);
      this->height++;
    }
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