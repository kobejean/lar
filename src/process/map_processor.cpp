#include "geoar/process/map_processor.h"

namespace geoar {

  MapProcessor::MapProcessor() : bundle_adjustment(data) {
  }

  void MapProcessor::createMap(std::string directory) {
    data.loadRawData(directory);
    bundle_adjustment.construct();
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