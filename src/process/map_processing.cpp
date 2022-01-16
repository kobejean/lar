#include "geoar/process/bundle_adjustment.h"
#include "geoar/process/frame_processing.h"
#include "geoar/process/map_processing_data.h"
#include "geoar/process/map_processing.h"
#include "geoar/core/utils/json.h"

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

    std::cout << std::endl;
    bundle_adjustment.optimizer.save(output.c_str());
    std::cout << "Saved g2o file to: " << output << std::endl;

    bundle_adjustment.optimize();

    // Serialize
    nlohmann::json map_json = data.map;
    // Save
    std::ofstream file(directory + "/map.json");
    file << std::setw(2) << map_json << std::endl;
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