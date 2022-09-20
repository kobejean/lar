#include <stdint.h>

#include <iostream>
#include "lar/core/spacial/region_tree.h"

using namespace std;


int main(int argc, const char* argv[]){
  lar::RegionTree<size_t> tree;

  tree.insert(1, lar::Rect(12, 4, 24, 15), 1);
  tree.print(std::cout);
  std::cout << "node1 inserted #1\n";
  tree.insert(2, lar::Rect(23, 24, 26, 26), 2);
  tree.print(std::cout);
  std::cout << "node2 inserted #2\n";
  tree.insert(3, lar::Rect(12, 22, 16, 26), 3);
  tree.print(std::cout);
  std::cout << "node3 inserted #3\n";
  tree.insert(4, lar::Rect(2, 16, 23, 17), 4);
  tree.print(std::cout);
  std::cout << "node4 inserted #4\n";
  tree.insert(5, lar::Rect(1, 1, 2, 4), 5);
  tree.print(std::cout);
  std::cout << "node5 inserted #5\n";
  tree.insert(6, lar::Rect(9, 14, 13, 18), 6);
  tree.print(std::cout);
  std::cout << "node6 inserted #6\n";
  tree.insert(7, lar::Rect(5, 19, 8, 23), 7);
  tree.print(std::cout);
  std::cout << "node7 inserted #7\n";
  tree.insert(8, lar::Rect(20, 14, 26, 22), 8);
  tree.print(std::cout);
  std::cout << "node8 inserted #8\n";
  tree.insert(9, lar::Rect(3, 9, 6, 12), 9);
  tree.print(std::cout);
  std::cout << "node9 inserted #9\n";
  tree.insert(10, lar::Rect(6, 3, 8, 7), 10);
  tree.print(std::cout);
  std::cout << "node10 inserted #10\n";
  tree.insert(11, lar::Rect(8, 21, 17, 28), 11);
  tree.print(std::cout);
  std::cout << "node11 inserted #11\n";
  tree.insert(12, lar::Rect(16, 5, 18, 13), 12);
  tree.print(std::cout);
  std::cout << "node12 inserted #12\n";
  tree.insert(13, lar::Rect(2, 22, 3, 26), 13);
  tree.print(std::cout);
  std::cout << "node13 inserted #13\n";
  tree.insert(14, lar::Rect(12, 20, 14, 22), 14);
  tree.print(std::cout);
  std::cout << "node14 inserted #14\n";
  tree.insert(15, lar::Rect(11, 10, 14, 12), 15);
  tree.print(std::cout);
  std::cout << "node15 inserted #15\n";
  tree.insert(16, lar::Rect(2, 5, 5, 7), 16);
  tree.print(std::cout);
  std::cout << "node16 inserted #16\n";
  tree.insert(17, lar::Rect(20, 6, 14, 9), 17);
  tree.print(std::cout);
  std::cout << "node17 inserted #17\n";
  

  size_t n = 25;
  for (size_t i = 18; i < n; i++) {
    tree.insert(i, lar::Rect(i, i, i+1, i+1), i);
    tree.print(std::cout);
    std::cout << "node" << i << " inserted #" << i << '\n';
  }

  for (size_t i = 1; i < n; i++) {
    tree.erase(i);
    tree.print(std::cout);
    std::cout << "node" << i << " removed #" << i << '\n';
  }
  return 0;
}
