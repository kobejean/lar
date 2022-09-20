#include <iostream>
#include <algorithm>
#include <array>
#include <vector>
#include <cassert>
#include "lar/core/spacial/region_tree.h"
#include "lar/core/landmark.h"

// import node implementation in same translation unit
// for template instantiation
#include "node.cpp"

namespace lar {

// lifecycle

template <typename T>
RegionTree<T>::RegionTree() : root(new Node(1)) {
  
}


// operations

template <typename T>
T& RegionTree<T>::operator[](size_t id) {
  return leaf_map[id]->value;
}

template <typename T>
void RegionTree<T>::insert(T value, Rect bounds, size_t id) {
  // create new leaf node
  LeafNode *node = new LeafNode(value, bounds, id);
  leaf_map.emplace(id, node);

  if (root->children.size() == 0) {
    // if tree is empty
    root->bounds = node->bounds;
    root->linkChild(node);
    return;
  }

  Node *split = root->insert(node);
  if (split != nullptr) {
    // if we have a spit at root, create a new root
    // with a copy of the old root and the split as a children
    Node *copy = new Node(*root);
    for (auto &child : root->children) {
      child->parent = copy;
    }
    root->children.clear();
    root->linkChild(copy);
    root->linkChild(split);
    root->height++;
  }
}

template <typename T>
void RegionTree<T>::erase(size_t id) {
  Node *node = leaf_map.extract(id).mapped();
  node->erase();
  if (root->children.size() == 1 && root->height > 1) {
    // if root has only one child, replace root with child
    Node *child = root->children[0];
    child->parent = nullptr;
    // clear children of root so that it doesn't delete them
    // when the root is deleted
    root->children.clear();
    this->root.reset(child);
  }
}

template <typename T>
std::vector<T> RegionTree<T>::find(const Rect &query) const {
  std::vector<T> result;
  if (root->children.size() > 0) root->find(query, result);
  return result;
}

template <typename T>
void RegionTree<T>::print(std::ostream &os) {
  root->print(os, 0);
}


// collection

template <typename T>
size_t RegionTree<T>::size() const {
  return leaf_map.size();
}

template <typename T>
std::vector<T> RegionTree<T>::all() const {
  std::vector<T> all;
  all.reserve(leaf_map.size());
  for(auto kv : leaf_map) {
    all.push_back(kv.second->value);  
  }
  return all;
}

// explicit instantiations
template class RegionTree<size_t>;
template class RegionTree<Landmark>;

} // namespace lar