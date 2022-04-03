#include <stdint.h>

#include <iostream>
#include <sys/stat.h>
#include <unordered_set>

#include "lar/processing/map_processor.h"

using namespace std;

int main(int argc, const char* argv[]){
  string input = "./input/snapshot";
  string output = "./output/map";

  struct stat st;
  int status = stat(input.c_str(), &st);

  if (status != 0) {
    cout << "Could not read directory at '" << input << endl;
    return 1;
  }

  lar::Mapper mapper(input);
  mapper.readMetadata();
  lar::MapProcessor processor(mapper.data);
  processor.createMap(output);
  return 0;
}
