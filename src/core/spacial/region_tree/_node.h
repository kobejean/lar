#ifndef LAR_CORE_SPACIAL_REGION_TREE__NODE_H
#define LAR_CORE_SPACIAL_REGION_TREE__NODE_H

#include <iostream>
#include <algorithm>
#include <vector>
#include "lar/core/data_structures/unordered_array.h"
#include "lar/core/spacial/region_tree.h"
#include "lar/core/landmark.h"

#include "_node_traits.h"

namespace lar {

// internal RegionTree node class
template <typename T>
class RegionTree<T>::_Node {
  public:
    static constexpr std::size_t MAX_CHILDREN = _NodeTraits<T>::MAX_CHILDREN;
    using children_container = unordered_array<_Node*, MAX_CHILDREN>;
    using overflow_container = unordered_array<_Node*, MAX_CHILDREN+1>;

    Rect bounds;
    uint8_t height;
    _Node *parent;
    children_container children;

    // lifecycle
    _Node(std::size_t height);
    ~_Node();

    // operations
    _Node *insert(_Node *node);
    _Node *erase();
    void find(const Rect &query, std::vector<T> &result) const;
    void print(std::ostream &os, int depth) const;

    // helpers
    inline bool isLeaf() const;
    _Node *findBestInsertChild(const Rect &bounds) const;
    _Node *addChild(_Node *child);
    void linkChild(_Node *child);
    std::size_t findChildIndex(_Node *child) const;
    void subtractBounds(const Rect &bounds);

    static void partition(overflow_container &children, _Node *lower_split, _Node *upper_split);
    friend class RegionTree<T>;
    friend class RegionTree<T>::_LeafNode;
  private:
    struct _InsertScore;
    class _Partition;
};


template <typename T>
class RegionTree<T>::_LeafNode : RegionTree<T>::_Node {
  public:
    size_t id;
    T value;
    _LeafNode(T value, Rect bounds, size_t id);
    friend class RegionTree<T>;
    friend class RegionTree<T>::_Node;
};

} // namespace lar


#endif /* LAR_CORE_SPACIAL_REGION_TREE__NODE_H */