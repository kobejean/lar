#include <stdint.h>

#include <iostream>
#include <sys/stat.h>
#include <unordered_set>

#include "geoar/processing/map_processor.h"

using namespace std;

int main(int argc, const char* argv[]){
  string input = "./input/snapshot";
  string output = "./output/map";

  struct stat st;
  int status = stat(input.c_str(), &st);

  if (status != 0) {
    cout << "Could not read directory at '" << input << endl;
    return 0;
  }

  geoar::MapProcessor processor;
  processor.createMap(input, output);
}
