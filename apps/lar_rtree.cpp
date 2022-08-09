#include <stdint.h>

#include <iostream>
#include "lar/core/spacial/region_tree.h"

using namespace std;

int main(int argc, const char* argv[]){
  lar::Rectangle rect(1, 2, 3, 4);
  std::cout << "rectangle: " << std::endl;
  rect.print(std::cout);
  std::cout << std::endl;

  lar::RegionTree tree;

  lar::RegionTree node1;
  node1.bounds = rect;
  node1.id = 1;
  std::cout << "node: " << std::endl;
  node1.print(std::cout);

  tree.insert(node1);
  // tree.print(std::cout);
  return 0;
}
