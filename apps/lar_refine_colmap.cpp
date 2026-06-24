#include <stdint.h>

#include <filesystem>
#include <iostream>
#include <sys/stat.h>
#include <unordered_set>

#include "lar/processing/colmap_refiner.h"

namespace fs = std::filesystem;
using namespace std;

int main(int argc, const char* argv[]){
  if (argc < 2) {
    cout << "Usage: lar_refine_colmap <input_dir> [output_dir]" << endl;
    cout << "  input_dir   session directory (with metadata + colmap/ subdir)" << endl;
    cout << "  output_dir  where to save the refined map" << endl;
    cout << "              (default: ./output/<input-name>-refined)" << endl;
    return 1;
  }

  string input = argv[1];
  // Default output: ./output/<input-basename>-refined  (override with argv[2])
  string output = "./output/" + fs::path(input).filename().string() + "-refined";
  if (argc > 2) {
    output = argv[2];
  }

  struct stat st;
  int status = stat(input.c_str(), &st);

  if (status != 0) {
    cout << "Could not read directory at '" << input << endl;
    return 1;
  }

  fs::create_directories(output);
  cout << "Input:  " << input << endl;
  cout << "Output: " << output << endl;

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
