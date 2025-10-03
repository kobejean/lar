#include <iostream>
#include <algorithm>
#include <memory>
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
void RegionTree<T>::Node::insert(std::unique_ptr<Node> node) {
  bounds = bounds.minBoundingBox(node->bounds);

  if (height > node->height+1) {
    Node *best_child = findBestInsertChild(node->bounds);
    best_child->insert(std::move(node));
  } else {
    assert(height == node->height+1);
    addChild(std::move(node));
  }
}

template <typename T>
typename RegionTree<T>::Node *RegionTree<T>::Node::erase() {
  if (parent != nullptr) {
    Node* insert_root = nullptr;
    Node* parent_node = parent;
    size_t index = parent_node->findChildIndex(this);
    parent_node->children.erase(index);
    parent_node->subtractBounds(bounds);

    if (parent_node->children.size() < (MAX_CHILDREN / 2)) {
      children_container siblings;
      for (auto &child : parent_node->children) {
        siblings.push_back(std::move(child));
      }
      // unlink from parent
      parent_node->children.clear();
      insert_root = parent_node->erase();
      for (auto &sibling : siblings) {
        insert_root->insert(std::move(sibling));
      }
    } else {
      insert_root = parent_node;
    }
    return insert_root;
  } else {
    // we will not delete the root
    // we will just return the root as the insert root
    return this;
  }
}

template <typename T>
void RegionTree<T>::Node::find(const Rect &query, std::vector<T*> &results) { 
  if (this->isLeaf()) {
    LeafNode *leaf = static_cast<LeafNode*>(this);
    results.push_back(&leaf->value);
    return;
  }

  for (auto & child : this->children) {
    if (child->bounds.intersectsWith(query)) {
      child->find(query, results);
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
  double coverage, expansion, area;

  InsertScore() : coverage(0), expansion(0), area(0) {}
  InsertScore(const Node &child, const Rect &bounds) :
    coverage(child.bounds.overlap(bounds)),
    area(child.bounds.area()) {
    expansion = child.bounds.minBoundingBox(bounds).area() - area;
  }

  bool operator<(const InsertScore &other) const {
    // Returns true if 'this' is worse than 'other' for insertion
    // Prefer: high coverage, low expansion, small area
    if (std::abs(coverage - other.coverage) > 1e-5) return coverage < other.coverage;
    if (std::abs(expansion - other.expansion) > 1e-5) return expansion > other.expansion;
    return area > other.area;
  }
};

template <typename T>
typename RegionTree<T>::Node *RegionTree<T>::Node::findBestInsertChild(const Rect &bounds) const {
  InsertScore best_score(*children[0], bounds);
  Node *best_child = children[0].get();
  // find the best child to insert into
  for (auto &child : children) {
    InsertScore score(*child, bounds);
    if (best_score < score ) {
      best_score = score;
      best_child = child.get();
    }
  }
  return best_child;
}


template <typename T>
void RegionTree<T>::Node::addChild(std::unique_ptr<Node> child) {
  if (children.size() < MAX_CHILDREN) {
    linkChild(std::move(child));
  } else {
    overflow_container nodes;
    for (auto &child : children) nodes.push_back(std::move(child));
    nodes.push_back(std::move(child));
    // reset parent
    children.clear();
    if (this->parent != nullptr) {
      auto split = std::make_unique<Node>(height);
      Partition::partition(nodes, this, split.get());
      this->parent->addChild(std::move(split));
    } else {
      auto split1 = std::make_unique<Node>(height);
      auto split2 = std::make_unique<Node>(height);
      Partition::partition(nodes, split1.get(), split2.get());
      linkChild(std::move(split1));
      linkChild(std::move(split2));
      this->height++;
    }
  }
}


template <typename T>
void RegionTree<T>::Node::linkChild(std::unique_ptr<Node> child) {
  child->parent = this;
  children.push_back(std::move(child));
}


template <typename T>
std::size_t RegionTree<T>::Node::findChildIndex(Node *child) const {
  for (size_t i = 0; i < children.size(); i++) {
    if (children[i].get() == child) return i;
  }
  assert(false && "Child not found in parent - tree structure corrupted");
  return static_cast<std::size_t>(-1);  // Unreachable, but silences warnings
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
