#include <filesystem>

#include "geoar/processing/bundle_adjustment.h"
#include "geoar/processing/frame_processor.h"
#include "geoar/processing/map_processor.h"
#include "geoar/core/utils/json.h"

namespace geoar {

  MapProcessor::MapProcessor(Mapper::Data& data) {
    this->data = &data;
  }

  void MapProcessor::createMap(std::string out_dir) {
    std::filesystem::create_directory(out_dir);
    
    // Process frames
    FrameProcessor frame_processor(*data);
    for (Frame& frame : data->frames) {
      frame_processor.process(frame);
    }

    // Optimize
    BundleAdjustment bundle_adjustment(*data);
    bundle_adjustment.construct();

    std::string output = out_dir + "/map.g2o";

    std::cout << std::endl;
    bundle_adjustment.optimizer.save(output.c_str());
    std::cout << "Saved g2o file to: " << output << std::endl;

    bundle_adjustment.optimize();

    // Serialize
    nlohmann::json map_json = data->map;
    // Save
    std::ofstream file(out_dir + "/map.json");
    file << std::setw(2) << map_json << std::endl;
  }

}