
#include <stdint.h>

#include <iostream>
#include <unordered_set>

#include <nlohmann/json.hpp>

#include "g2o/core/optimization_algorithm_factory.h"
#include "g2o/core/robust_kernel_impl.h"
#include "g2o/core/sparse_optimizer.h"
#include "g2o/solvers/structure_only/structure_only_solver.h"
#include "g2o/stuff/sampler.h"
#include "g2o/types/sba/types_six_dof_expmap.h"
#include "g2o/types/sba/types_six_dof_expmap.h"

#if defined G2O_HAVE_CHOLMOD
G2O_USE_OPTIMIZATION_LIBRARY(cholmod);
#else
G2O_USE_OPTIMIZATION_LIBRARY(eigen);
#endif

G2O_USE_OPTIMIZATION_LIBRARY(dense);

using namespace Eigen;
using namespace std;
using json = nlohmann::json;

namespace process {
  g2o::SparseOptimizer parse_map(std::ifstream map_ifs) {
    json map = json::parse(map_ifs);
    json pointCloud = map["pointCloud"];
    json cameraPoints = pointCloud["cameraPoints"];
    json featurePoints = pointCloud["featurePoints"];
    json locationPoints = pointCloud["locationPoints"];
    
    for (auto& el : cameraPoints[0].items()) {
      std::cout << "cameraPoints." << el.key() << " = " << el.value() << '\n';
    }
    for (auto& el : cameraPoints[2]["featurePoints"][0].items()) {
      std::cout << "cameraPoints[*].featurePoints[*]." << el.key() << '\n';
    }
    for (auto& el : cameraPoints[2]["intrinsics"].items()) {
      std::cout << "cameraPoints[*].intrinsics." << el.key() << '\n';
    }
    for (auto& el : cameraPoints[2]["intrinsics"]["principlePoint"].items()) {
      std::cout << "cameraPoints[*].intrinsics.principlePoint." << el.key() << '\n';
    }
    for (auto& el : featurePoints[0].items()) {
      std::cout << "featurePoints[*]." << el.key() << '\n';
    }
    for (auto& el : locationPoints[0].items()) {
      std::cout << "locationPoints[*]." << el.key() << '\n';
    }

    g2o::SparseOptimizer optimizer;
    optimizer.setVerbose(true);
    string solverName = "lm_fix6_3";

    g2o::OptimizationAlgorithmProperty solverProperty;
    auto algorithm = g2o::OptimizationAlgorithmFactory::instance()->construct(solverName, solverProperty);
    optimizer.setAlgorithm(algorithm);

    return optimizer;
  }
}