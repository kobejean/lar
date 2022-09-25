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

  Eigen::Matrix4d transform;
  transform << 1., 0., 0., 4.,
               0., 1., 0., 3.,
               0., 0., 1., 1.,
               0., 0., 0., 1.;
  lar::Anchor anchor;
  anchor.id = 0;
  anchor.transform = transform;
  mapper.addAnchor(anchor);


  lar::MapProcessor processor(mapper.data);
  processor.process();
  processor.saveMap(output);
  return 0;
}
