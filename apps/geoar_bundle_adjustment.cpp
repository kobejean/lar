#include <stdint.h>

#include <iostream>
#include <unordered_set>

#include "g2o/core/optimization_algorithm_factory.h"
#include "g2o/core/robust_kernel_impl.h"
#include "g2o/core/sparse_optimizer.h"
#include "g2o/solvers/structure_only/structure_only_solver.h"
#include "g2o/stuff/sampler.h"
#include "g2o/types/sba/types_six_dof_expmap.h"
#include "g2o/types/slam3d/types_slam3d.h"
#include "g2o/core/factory.h"

#include "geoar/process/map_processor.h"

#if defined G2O_HAVE_CHOLMOD
G2O_USE_OPTIMIZATION_LIBRARY(cholmod);
#else
G2O_USE_OPTIMIZATION_LIBRARY(eigen);
#endif

G2O_USE_OPTIMIZATION_LIBRARY(dense);

using namespace Eigen;
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
