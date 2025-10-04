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
RegionTree<T>::RegionTree() : root(std::make_unique<Node>(1)) {

}

template <typename T>
RegionTree<T>::~RegionTree() = default;

template <typename T>
RegionTree<T>::RegionTree(RegionTree&& other) = default;

template <typename T>
RegionTree<T>& RegionTree<T>::operator=(RegionTree&& other) = default;


// operations

template <typename T>
T& RegionTree<T>::operator[](size_t id) {
  return leaf_map[id]->value;
}

template <typename T>
T* RegionTree<T>::insert(T value, Rect bounds, size_t id) {
  // create new leaf node
  auto node = std::make_unique<LeafNode>(value, bounds, id);
  LeafNode* leaf_ptr = node.get();
  leaf_map.emplace(id, leaf_ptr);

  if (root->children.size() == 0) {
    // if tree is empty
    root->bounds = node->bounds;
    root->linkChild(std::move(node));
    return &leaf_ptr->value;
  }

  root->insert(std::move(node));
  return &leaf_ptr->value;
}

template <typename T>
void RegionTree<T>::erase(size_t id) {
  Node *node = leaf_map.extract(id).mapped();
  node->erase();
  if (root->children.size() == 1 && root->height > 1) {
    // if root has only one child, replace root with child
    auto child = std::move(root->children[0]);
    child->parent = nullptr;
    // clear children of root so that it doesn't delete them
    // when the root is deleted
    root->children.clear();
    this->root = std::move(child);
  }
}

template <typename T>
void RegionTree<T>::updateBounds(size_t id, Rect &bounds) {
  T item = std::move((*this)[id]);
  this->erase(id);
  this->insert(std::move(item), bounds, id);
}

template <typename T>
void RegionTree<T>::find(const Rect &query, std::vector<T*> &results) const {
  if (root->children.size() > 0) root->find(query, results);
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
std::vector<T*> RegionTree<T>::all() const {
  std::vector<T*> all;
  all.reserve(leaf_map.size());
  for(auto kv : leaf_map) {
    all.push_back(&kv.second->value);  
  }
  return all;
}

// explicit instantiations
template class RegionTree<size_t>;
template class RegionTree<Landmark>;

} // namespace lar
