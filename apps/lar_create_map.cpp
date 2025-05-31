#include <stdint.h>

#include <iostream>
#include <sys/stat.h>
#include <unordered_set>

#include "lar/processing/map_processor.h"

using namespace std;

int main(int argc, const char* argv[]){
  // string input = "./input/snapshot";
  // string input = "./input/iimori1";
  // string input = "./input/student_hall_2023-03-27-0900";
  // string input = "./input/u-aizu-out";
  string input = "./input/aizu-park";
  // string input = "./input/aizu-park-2";
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
  processor.process();
  processor.optimize();

  Eigen::Matrix4d transformMat;
  transformMat << 1.0, 0.0, 0.0, -951.465,
                  0.0, 1.0, 0.0, -42.2072,
                  0.0, 0.0, 1.0, -2588.11,
                  0.0, 0.0, 0.0, 1.0;
  lar::Anchor::Transform transform(transformMat);
  mapper.createAnchor(transform);
  
  processor.saveMap(output);
  return 0;
}
