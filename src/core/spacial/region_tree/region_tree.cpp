#include <iostream>
#include <algorithm>
#include <array>
#include <vector>
#include <cassert>
#include "lar/core/spacial/region_tree.h"
#include "lar/core/landmark.h"

// import rtree node implementation in same translation unit
#include "_node.cpp"

namespace lar {

// lifecycle

template <typename T>
RegionTree<T>::RegionTree() : root(new _Node(1)) {
  
}


// operations

template <typename T>
T& RegionTree<T>::operator[](size_t id) {
  return entities[id];
}

template <typename T>
void RegionTree<T>::insert(T value, Rect bounds, size_t id) {
  // create new leaf node
  _Node *node = new _Node(value, bounds, id);
  entities.emplace(id, value);
  leaf_map.emplace(id, node);

  if (root->children.size() == 0) {
    // if tree is empty
    root->bounds = node->bounds;
    root->linkChild(node);
    return;
  }

  _Node *split = root->insert(node);
  if (split != nullptr) {
    // if we have a spit at root, create a new root
    // with a copy of the old root and the split as a children
    _Node *copy = new _Node(*root);
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
  entities.erase(id);
  _Node *node = leaf_map.extract(id).mapped();
  node->erase();
  if (root->children.size() == 1 && root->height > 1) {
    // if root has only one child, replace root with child
    _Node *child = root->children[0];
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
  return entities.size();
}

template <typename T>
std::vector<T> RegionTree<T>::all() const {
  std::vector<T> all;
  all.reserve(entities.size());
  for(auto kv : entities) {
    all.push_back(kv.second);  
  }
  return all;
}

// RegionTree<size_t> is used for testing purposes
// MAX_CHILDREN is set to 4 to make testing easier
template<>
const std::size_t RegionTree<size_t>::MAX_CHILDREN = 4;

// RegionTree<Landmark> is used in the landmark database
template<>
const std::size_t RegionTree<Landmark>::MAX_CHILDREN = 25;

// explicit instantiations
template class RegionTree<size_t>;
template class RegionTree<Landmark>;

} // namespace lar