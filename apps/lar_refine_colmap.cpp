#include <stdint.h>

#include <iostream>
#include <sys/stat.h>
#include <unordered_set>

#include "lar/processing/colmap_refiner.h"

using namespace std;

int main(int argc, const char* argv[]){
  // string input = "./input/dwell-crane-202";
  string input = "./input/aizu-park-4-etx-2";
  // string input = "./input/my-room-3";
  if (argc > 1) {
    input = argv[1];
  }
  string output = "./output/map";

  struct stat st;
  int status = stat(input.c_str(), &st);

  if (status != 0) {
    cout << "Could not read directory at '" << input << endl;
    return 1;
  }

  lar::Mapper mapper(input);
  mapper.readMetadata();

  lar::ColmapRefiner refiner(mapper.data);
  
  // Comment/uncomment to switch methods:
  // refiner.process();  // Original tracker-based method
  refiner.processWithColmapData(input + "/colmap");  // New COLMAP-based method
  
  refiner.optimize();
  refiner.saveMap(output);
  return 0;
}
