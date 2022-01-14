#include "geoar/process/bundle_adjustment.h"
#include "geoar/process/frame_processing.h"
#include "geoar/process/map_processing_data.h"
#include "geoar/process/map_processing.h"

namespace geoar {

  // Helper function forward declarations
  MapProcessingData loadData(std::string directory);

  MapProcessing::MapProcessing() {
  }

  void MapProcessing::createMap(std::string directory) {
    MapProcessingData data = loadData(directory);
    BundleAdjustment bundle_adjustment(data);
    bundle_adjustment.construct();

    std::string output = directory + "/map.g2o";

    cout << endl;
    bundle_adjustment.optimizer.save(output.c_str());
    cout << "Saved g2o file to: " << output << endl;

    bundle_adjustment.optimize();
  }

  MapProcessingData loadData(std::string directory) {
    std::ifstream metadata_ifs(directory + "/metadata.json");
    nlohmann::json metadata = nlohmann::json::parse(metadata_ifs);
    MapProcessingData data;

    FrameProcessing frame_processing(data);
    for (nlohmann::json frame_data : metadata["frames"]) {
      Frame frame = frame_processing.process(frame_data, directory);
      data.frames.push_back(frame);
    }
    return data;
  }

}