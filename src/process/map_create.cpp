
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

#if defined G2O_HAVE_CHOLMOD
G2O_USE_OPTIMIZATION_LIBRARY(cholmod);
#else
G2O_USE_OPTIMIZATION_LIBRARY(eigen);
#endif

G2O_USE_OPTIMIZATION_LIBRARY(dense);

using namespace Eigen;
using namespace std;
using json = nlohmann::json;

namespace geoar {


  void MapProcessor::createMap(string directory) {
    createFrames(directory);
  }


  // Private methods

  void MapProcessor::createFrames(string directory) {
    ifstream metadata_ifs(directory + "/metadata.json");
    json metadata = json::parse(metadata_ifs);

    for (json frame_data : metadata) {
      Frame frame(frame_data, directory);
      frames.push_back(frame);
    }
  }
}