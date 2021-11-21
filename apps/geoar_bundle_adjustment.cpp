#include <stdint.h>

#include <iostream>
#include <unordered_set>

#include "geoar/process/map_processor.h"

using namespace std;
using namespace geoar;

int main(int argc, const char* argv[]){
  std::string input = "../input/map.json";
  std::string output = "../output/map.g2o";

  ifstream ifs(input);
  if (ifs.fail()) {
    cout << "Could not read file at '" << input << endl;
    return 0;
  }
  geoar::MapProcessor processor;
  processor.parseMap(ifs);

  cout << endl;
  processor.optimizer.save(output.c_str());
  cout << "Saved g2o file to: " << output << endl;

  processor.optimizer.initializeOptimization();
  processor.optimizer.setVerbose(true);

  cout << "Performing full Bundle Adjustment:" << endl;
  processor.optimizer.optimize(10);
  cout << endl;
}
