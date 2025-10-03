#ifndef LAR_CORE_SPACIAL_REGION_TREE__NODE_H
#define LAR_CORE_SPACIAL_REGION_TREE__NODE_H

#include <iostream>
#include <algorithm>
#include <memory>
#include <vector>
#include "lar/core/data_structures/unordered_array.h"
#include "lar/core/spacial/region_tree.h"
#include "lar/core/landmark.h"

#include "node_traits.h"

namespace lar {

// internal RegionTree node class
template <typename T>
class RegionTree<T>::Node {
    friend class RegionTree<T>;
  public:
    static constexpr std::size_t MAX_CHILDREN = NodeTraits<T>::MAX_CHILDREN;
    using children_container = unordered_array<std::unique_ptr<Node>, MAX_CHILDREN>;
    using overflow_container = unordered_array<std::unique_ptr<Node>, MAX_CHILDREN+1>;

    Rect bounds;
    uint8_t height;
    Node *parent;
    children_container children;

    // lifecycle
    Node(std::size_t height);
    ~Node();

    // operations
    void insert(std::unique_ptr<Node> node);
    Node *erase();
    void find(const Rect &query, std::vector<T*> &result);
    void print(std::ostream &os, int depth) const;

    // helpers
    inline bool isLeaf() const;
    Node *findBestInsertChild(const Rect &bounds) const;
    void addChild(std::unique_ptr<Node> child);
    void linkChild(std::unique_ptr<Node> child);
    std::size_t findChildIndex(Node *child) const;
    void subtractBounds(const Rect &bounds);
  private:
    struct InsertScore;
    class Partition;
};


template <typename T>
class RegionTree<T>::LeafNode : public Node {
    friend class RegionTree<T>;
  public:
    size_t id;
    T value;
    LeafNode(T value, Rect bounds, size_t id);
};

} // namespace lar


#endif /* LAR_CORE_SPACIAL_REGION_TREE__NODE_H */