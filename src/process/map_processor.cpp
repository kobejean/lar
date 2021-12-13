
#include <stdint.h>

#include <iostream>
#include <unordered_set>

#include <nlohmann/json.hpp>

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>

#include "g2o/core/factory.h"
#include "g2o/core/optimization_algorithm_factory.h"
#include "g2o/core/robust_kernel_impl.h"
#include "g2o/core/sparse_optimizer.h"
#include "g2o/solvers/structure_only/structure_only_solver.h"
#include "g2o/stuff/sampler.h"
#include "g2o/types/sba/types_six_dof_expmap.h"
#include "g2o/types/slam3d/types_slam3d.h"

#include "geoar/process/map_processor.h"

using namespace Eigen;
using namespace std;
using json = nlohmann::json;

namespace geoar {

  MapProcessor::MapProcessor() : bundle_adjustment(data) {
  }

  void MapProcessor::createMap(std::string directory) {
    data.loadRawData(directory);
    bundle_adjustment.construct(directory);
    std::string output = directory + "/map.g2o";

    cout << endl;
    bundle_adjustment.optimizer.save(output.c_str());
    cout << "Saved g2o file to: " << output << endl;

    bundle_adjustment.optimizer.initializeOptimization();
    bundle_adjustment.optimizer.setVerbose(true);

    cout << "Performing full Bundle Adjustment:" << endl;
    bundle_adjustment.optimizer.optimize(2);
  }

}