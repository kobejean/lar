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

template <typename T, std::size_t N>
RegionTree<T,N>::RegionTree() : root(new _Node<T,N>(1)) {
  
}


// operations

template <typename T, std::size_t N>
T& RegionTree<T,N>::operator[](size_t id) {
  return entities[id];
}

template <typename T, std::size_t N>
void RegionTree<T,N>::insert(T value, Rect bounds, size_t id) {
  _Node<T,N> *root = static_cast<_Node<T,N>*>(this->root.get());

  // create new leaf node
  _Node<T,N> *node = new _Node<T,N>(value, bounds, id);
  entities.emplace(id, value);
  leaf_map.emplace(id, node);

  if (root->children.size() == 0) {
    // if tree is empty
    root->bounds = node->bounds;
    root->linkChild(node);
    return;
  }

  _Node<T,N> *split = root->insert(node);
  if (split != nullptr) {
    // if we have a spit at root, create a new root
    // with a copy of the old root and the split as a children
    _Node<T,N> *copy = new _Node<T,N>(*root);
    for (auto &child : root->children) {
      child->parent = copy;
    }
    root->children.clear();
    root->linkChild(copy);
    root->linkChild(split);
    root->height++;
  }
}

template <typename T, std::size_t N>
void RegionTree<T,N>::erase(size_t id) {
  _Node<T,N> *root = static_cast<_Node<T,N>*>(this->root.get());

  entities.erase(id);
  _Node<T,N> *node = static_cast<_Node<T,N>*>(leaf_map.extract(id).mapped());
  node->erase();
  if (root->children.size() == 1 && root->height > 1) {
    root->children.clear();
    // if root has only one child, replace root with child
    _Node<T,N> *child = root->children[0];
    child->parent = nullptr;
    this->root.reset(child);
  }
}

template <typename T, std::size_t N>
std::vector<T> RegionTree<T,N>::find(const Rect &query) const {
  _Node<T,N> *root = static_cast<_Node<T,N>*>(this->root.get());

  std::vector<T> result;
  if (root->children.size() > 0) root->find(query, result);
  return result;
}

template <typename T, std::size_t N>
void RegionTree<T,N>::print(std::ostream &os) {
  _Node<T,N> *root = static_cast<_Node<T,N>*>(this->root.get());

  root->print(os, 0);
}


// collection

template <typename T, std::size_t N>
size_t RegionTree<T,N>::size() const {
  return entities.size();
}

template <typename T, std::size_t N>
std::vector<T> RegionTree<T,N>::all() const {
  std::vector<T> all;
  all.reserve(entities.size());
  for(auto kv : entities) {
    all.push_back(kv.second);  
  }
  return all;
}


// explicit instantiations
template class RegionTree<size_t, 4>;
template class RegionTree<Landmark>;

} // namespace lar