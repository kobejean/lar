
#include <stdint.h>

#include <iostream>
#include <unordered_set>

#include <nlohmann/json.hpp>

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>

#include "g2o/core/optimization_algorithm_factory.h"
#include "g2o/core/robust_kernel_impl.h"
#include "g2o/core/sparse_optimizer.h"
#include "g2o/solvers/structure_only/structure_only_solver.h"
#include "g2o/stuff/sampler.h"
#include "g2o/types/sba/types_six_dof_expmap.h"
#include "g2o/types/slam3d/types_slam3d.h"

#include "geoar/process/map_processor.h"

G2O_USE_OPTIMIZATION_LIBRARY(eigen);
G2O_USE_OPTIMIZATION_LIBRARY(dense);

using namespace Eigen;
using namespace std;
using json = nlohmann::json;

namespace geoar {

  MapProcessor::MapProcessor() : graph_construction(data) {
    data.optimizer.setVerbose(true);
    string solver_name = "lm_fix6_3";
    g2o::OptimizationAlgorithmProperty solver_property;
    auto algorithm = g2o::OptimizationAlgorithmFactory::instance()->construct(solver_name, solver_property);
    data.optimizer.setAlgorithm(algorithm);
  }


  void MapProcessor::createMap(string directory) {
    graph_construction.processRawData(directory);
  }

}