#include <filesystem>

#include "lar/processing/map_processor.h"
#include "lar/core/utils/json.h"

namespace lar {

  MapProcessor::MapProcessor(std::shared_ptr<Mapper::Data> data) :
    data(data), bundle_adjustment(data), global_alignment(data), frame_processor(data) {
  }

  void MapProcessor::process() {
    // Update GPS alignment
    global_alignment.updateAlignment();
    std::cout << data->map.origin.matrix() << std::endl;

    // Process frames
    for (Frame& frame : data->frames) {
      frame_processor.process(frame);
    }
  }

  void MapProcessor::optimize() {
    // Optimize
    bundle_adjustment.reset();
    bundle_adjustment.construct();
    bundle_adjustment.optimize();
  }

  void MapProcessor::saveMap(std::string dir) {
    std::filesystem::create_directory(dir);
    std::string output = dir + "/map.g2o";

    std::cout << std::endl;
    bundle_adjustment.optimizer.save(output.c_str());
    std::cout << "Saved g2o file to: " << output << std::endl;

    data->map.landmarks.cull();

    // Serialize
    nlohmann::json map_json = data->map;
    // Save
    std::ofstream file(dir + "/map.json");
    file << std::setw(2) << map_json << std::endl;
  }

}