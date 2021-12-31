#include "geoar/process/frame_processing.h"
#include "geoar/process/map_processing.h"

namespace geoar {

  MapProcessing::MapProcessing() : bundle_adjustment(data) {
  }

  void MapProcessing::createMap(std::string directory) {
    loadData(directory);
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

  void MapProcessing::loadData(std::string directory) {
    std::ifstream metadata_ifs(directory + "/metadata.json");
    nlohmann::json metadata = nlohmann::json::parse(metadata_ifs);

    FrameProcessing frame_processing(data);
    for (nlohmann::json frame_data : metadata["frames"]) {
      Frame frame = frame_processing.process(frame_data, directory);
      data.frames.push_back(frame);
    }
  }

}