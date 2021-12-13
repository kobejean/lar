
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

G2O_USE_OPTIMIZATION_LIBRARY(eigen);
G2O_USE_OPTIMIZATION_LIBRARY(dense);

namespace g2o {
G2O_REGISTER_TYPE_GROUP(expmap);
G2O_REGISTER_TYPE(PARAMS_CAMERAPARAMETERS, CameraParameters);
G2O_REGISTER_TYPE(VERTEX_SE3:EXPMAP, VertexSE3Expmap);
G2O_REGISTER_TYPE(EDGE_SE3:EXPMAP, EdgeSE3Expmap);
G2O_REGISTER_TYPE(EDGE_PROJECT_XYZ2UV:EXPMAP, EdgeProjectXYZ2UV);

G2O_REGISTER_TYPE_GROUP(slam3d);
G2O_REGISTER_TYPE(VERTEX_TRACKXYZ, VertexPointXYZ);
}

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


  void MapProcessor::createMap(std::string directory) {
    graph_construction.processRawData(directory);
    graph_construction.construct(directory);
    std::string output = directory + "/map.g2o";

    cout << endl;
    data.optimizer.save(output.c_str());
    cout << "Saved g2o file to: " << output << endl;

    data.optimizer.initializeOptimization();
    data.optimizer.setVerbose(true);

    cout << "Performing full Bundle Adjustment:" << endl;
    data.optimizer.optimize(2);
  }

}