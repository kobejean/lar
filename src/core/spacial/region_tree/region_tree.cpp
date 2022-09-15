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


template <typename T>
RegionTree<T>::RegionTree() : root(new _Node<T>()) {
  
}

template <typename T>
T& RegionTree<T>::operator[](size_t id) {
  return entities[id];
}

template <typename T>
void RegionTree<T>::insert(T value, Rect bounds, size_t id) {
  _Node<T> *root = static_cast<_Node<T>*>(this->root.get());

  // create new leaf node
  _Node<T> *node = new _Node<T>(value, bounds, id);
  entities[id] = value;
  leaf_map[id] = node;

  if (root->children.size() == 0) {
    // if tree is empty
    root->bounds = node->bounds;
    root->linkChild(node);
    return;
  }

  _Node<T> *split = root->insert(node);
  if (split != nullptr) {
    // if we have a spit at root, create a new root
    // with a copy of the old root and the split as a children
    _Node<T> *copy = new _Node<T>(*root);
    root->children.clear();
    root->linkChild(copy);
    root->linkChild(split);
  }
}

template <typename T>
void RegionTree<T>::erase(size_t id) {
  entities.erase(id);
  _Node<T> *node = static_cast<_Node<T>*>(leaf_map.extract(id).mapped());
  delete node;
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

// explicit instantiations
template class RegionTree<size_t>;
template class RegionTree<Landmark>;

} // namespace lar