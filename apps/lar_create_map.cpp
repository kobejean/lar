#include <stdint.h>

#include <iostream>
#include <sys/stat.h>
#include <unordered_set>

#include "lar/processing/map_processor.h"

using namespace std;

int main(int argc, const char* argv[]){
  // string input = "./input/snapshot";
  string input = "./input/iimori1";
  // string input = "./input/u-aizu-out";
  string output = "./output/map";

  struct stat st;
  int status = stat(input.c_str(), &st);

  if (status != 0) {
    cout << "Could not read directory at '" << input << endl;
    return 1;
  }

  lar::Mapper mapper(input);
  mapper.readMetadata();

  Eigen::Matrix4d transformMat;
  transformMat <<  0.982631504535675,  0.0, -0.18556788563728333,  0.041521620005369186,
                0.0,                1.0,  0.0,                 -0.18674816191196442,
                0.1855679154396057, 0.0,  0.9826314449310303,  -0.2180960327386856,
                0.0,                0.0,  0.0,                  1.0;
  lar::Anchor::Transform transform(transformMat);
  lar::Anchor anchor(0, transform);
  mapper.addAnchor(anchor);


  lar::MapProcessor processor(mapper.data);
  processor.process();
  processor.optimize();
  processor.saveMap(output);
  return 0;
}
