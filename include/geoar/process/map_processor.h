#ifndef MAP_PROCESSOR_H
#define MAP_PROCESSOR_H

#include "g2o/core/optimization_algorithm_factory.h"
#include "g2o/core/robust_kernel_impl.h"
#include "g2o/core/sparse_optimizer.h"
#include "g2o/solvers/structure_only/structure_only_solver.h"
#include "g2o/stuff/sampler.h"
#include "g2o/types/sba/types_six_dof_expmap.h"
#include "g2o/types/slam3d/types_slam3d.h"

#include "geoar/process/graph_construction.h"
#include "geoar/core/map.h"

using namespace Eigen;
using json = nlohmann::json;

namespace geoar {

  class MapProcessor {
    public: 
      g2o::SparseOptimizer optimizer;
      Map map;
      GraphConstruction graphConstruction;

      MapProcessor();
      void createMap(std::string directory);
  };
}

#endif /* MAP_PROCESSOR_H */