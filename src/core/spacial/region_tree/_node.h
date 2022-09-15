#ifndef LAR_CORE_SPACIAL_REGION_TREE__NODE_H
#define LAR_CORE_SPACIAL_REGION_TREE__NODE_H

#include <iostream>
#include <algorithm>
#include <vector>
#include "lar/core/spacial/region_tree.h"
#include "lar/core/landmark.h"

namespace lar {

namespace {
  
// internal RegionTree node class
template <typename T>
class _Node {
  public:
    Rect bounds;
    T value;
    size_t id;
    _Node<T> *parent;
    // TODO: use better choice of container
    std::vector<_Node*> children;

    // lifecycle
    _Node();
    _Node(T value, Rect bounds, size_t id);
    ~_Node();

    // operations
    _Node* insert(_Node *node);
    void find(const Rect &query, std::vector<T> &result) const;
    void print(std::ostream &os, int depth) const;

    // helpers
    inline bool isLeaf() const;
    _Node *findBestInsertChild(const Rect &bounds) const;
    _Node *addChild(_Node *child);
    void linkChild(_Node *child);

    static void partition(std::vector<_Node*> &children, _Node *lower_split, _Node *upper_split);
};

} // namespace

} // namespace lar


#endif /* LAR_CORE_SPACIAL_REGION_TREE__NODE_H */