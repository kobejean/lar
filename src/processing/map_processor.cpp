#include <filesystem>

#include "lar/processing/bundle_adjustment.h"
#include "lar/processing/frame_processor.h"
#include "lar/processing/global_alignment.h"
#include "lar/processing/map_processor.h"
#include "lar/core/utils/json.h"

namespace lar {

  MapProcessor::MapProcessor(std::shared_ptr<Mapper::Data> data) : data(data) {
  }

  void MapProcessor::process() {
    // Update GPS alignment
    GlobalAlignment global_alignment(data);
    global_alignment.updateAlignment();
    std::cout << data->map.origin.matrix() << std::endl;

    // Process frames
    FrameProcessor frame_processor(data);
    for (Frame& frame : data->frames) {
      frame_processor.process(frame);
    }

    // Optimize
    BundleAdjustment bundle_adjustment(data);
    bundle_adjustment.construct();
    bundle_adjustment.optimize();
  }

  void MapProcessor::createMap(std::string out_dir) {
    std::filesystem::create_directory(out_dir);

    // Update GPS alignment
    GlobalAlignment global_alignment(data);
    global_alignment.updateAlignment();
    
    // Process frames
    FrameProcessor frame_processor(data);
    for (Frame& frame : data->frames) {
      frame_processor.process(frame);
    }

    // Optimize
    BundleAdjustment bundle_adjustment(data);
    bundle_adjustment.construct();

    std::string output = out_dir + "/map.g2o";

    std::cout << std::endl;
    bundle_adjustment.optimizer.save(output.c_str());
    std::cout << "Saved g2o file to: " << output << std::endl;

    bundle_adjustment.optimize();
    data->map.landmarks.cull();

    // Serialize
    nlohmann::json map_json = data->map;
    // Save
    std::ofstream file(out_dir + "/map.json");
    file << std::setw(2) << map_json << std::endl;
  }

}