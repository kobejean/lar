#include <stdint.h>

#include <iostream>
#include <sys/stat.h>
#include <unordered_set>

#include "geoar/process/map_processing.h"

using namespace std;
using namespace geoar;

int main(int argc, const char* argv[]){
  string input = "./input/snapshot";
  // string output = "./output/map.g2o";

  struct stat st;
  int status = stat(input.c_str(), &st);

  if (status != 0) {
    cout << "Could not read directory at '" << input << endl;
    return 0;
  }

  geoar::MapProcessing processor;
  processor.createMap(input);
}
