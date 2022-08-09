#include <iostream>
#include <array>
#include <vector>
#include "lar/core/spacial/region_tree.h"

namespace lar {
  struct _RTOverflowStatus {
    bool succeeded;
    RegionTree *split1, *split2;
  };

  struct _RTInsertScore {
    double overlap, expansion, area;
  };

  bool _partition(RegionTree &root, RegionTree *n1, RegionTree *n2);
  static _RTOverflowStatus _insert(RegionTree &root, RegionTree &node);
  static _RTOverflowStatus _addChild(RegionTree &parent, RegionTree &child);
  static void _extendBounds(RegionTree &node, const Rectangle &other_rect);
  static void _print(std::ostream &os, const RegionTree &node, int depth);
  static inline bool _isLeaf(const RegionTree &node);

  RegionTree::RegionTree() {
    
  }

  RegionTree::~RegionTree() {
    
  }

  bool RegionTree::insert(RegionTree &node) {
    std::cout << "inserting node " << node.id << " into tree" << std::endl;
    _RTOverflowStatus stat = _insert(*this, node);
    if (stat.split1 != NULL && stat.split2 != NULL) {
        _addChild(*this, *stat.split1);
        _addChild(*this, *stat.split2);
    }
    std::cout << "insertion status: " << stat.succeeded << std::endl;
    return stat.succeeded;
  }

  void RegionTree::find(const Rectangle &query, std::vector<RegionTree> &result) {
    if (_isLeaf(*this)) {
      result.push_back(*this);
      return;
    }

    for (size_t i = 0; i < children.size(); i++) {
      if (children[i].bounds.intersectsWith(query)) {
        children[i].find(query, result);
      }
    }
  }

  void RegionTree::print(std::ostream &os) {
    _print(os, *this, 0);
  }


// partition strategy based on: https://www.just.edu.jo/~qmyaseen/rtree1.pdf
bool _partition(RegionTree &root, RegionTree *n1, RegionTree *n2) {
    RegionTree *s1;
    RegionTree *s2;
    // _rtree_linear_pick_seeds(root, &s1, &s2);

    // rtree_init(n1);
    // _rtree_extend_bounds(n1, &s1->bounds);
    // if (!_rtree_add_child(n1, s1).succeeded) return false;
    
    // rtree_init(n2);
    // _rtree_extend_bounds(n2, &s2->bounds);
    // if (!_rtree_add_child(n2, s2).succeeded) return false;

    // _rtree_ndistribute(root, n1, n2);

    return true;
}

  static void _extendBounds(RegionTree &node, const Rectangle &other_rect) {
      node.bounds = _isLeaf(node) ? other_rect : node.bounds.min_bounding(other_rect);
  }

  static _RTOverflowStatus _insert(RegionTree &root, RegionTree &node) {
    if (!node.bounds.isInsideOf(root.bounds)) _extendBounds(root, node.bounds);

    // int best = _rtree_find_insert_child(root, node);
    // if (best >= 0) {
    //     struct _RTOverflowStatus stat = _rtree_insert(root->children[best], node);
    //     if (stat.split1 != NULL && stat.split2 != NULL) {
    //         rtree_deinit(root->children[best]);
    //         root->children[best] = stat.split1;
    //         return _rtree_addChild(root, stat.split2);
    //     }
    //     return stat;
    // } else {
      return _addChild(root, node);
    // }
  }

  /* insert int into array, adjusting capacity when needed */
  static _RTOverflowStatus _addChild(RegionTree &parent, RegionTree &child) {
      _RTOverflowStatus stat;
      if (parent.children.size() < RegionTree::MAX_CHILDREN) {
        parent.children.push_back(child);
        stat.succeeded = true;
        stat.split1 = NULL;
        stat.split2 = NULL;
      } else {
        stat.split1 = NULL;
        stat.split2 = NULL;
        stat.succeeded = _partition(parent, stat.split1, stat.split2);
      }
      return stat;
  }
  
  static void _print(std::ostream &os, const RegionTree &node, int depth) {
    // print leading tabs
    for (int i = 0; i < depth; i++) os << "\t";

    // print node info
    node.bounds.print(os);
    if (_isLeaf(node)) os << " - #" << node.id;
    os << "\n";
    
    // print children
    for (size_t i = 0; i < node.children.size(); i++) _print(os, node.children[i], depth+1);
  }

  static inline bool _isLeaf(const RegionTree &node) {
    return node.children.size() == 0;
  }
}
