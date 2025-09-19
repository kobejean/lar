#include <filesystem>

#include "lar/processing/map_processor.h"
#include "lar/core/utils/json.h"
#include "lar/mapping/location_matcher.h"

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
    bundle_adjustment.update();

    // Re-interpolate GPS observations using updated frame positions
    LocationMatcher temp_matcher;
    temp_matcher.matches = data->gps_obs;
    temp_matcher.reinterpolateMatches(data->frames);
    data->gps_obs = temp_matcher.matches;
  }

  void MapProcessor::rescale(double scale_factor) {
    if (scale_factor <= 0.0) {
      std::cout << "Invalid scale factor: " << scale_factor << std::endl;
      return;
    }
    
    std::cout << "Manual rescaling by factor: " << scale_factor << std::endl;
    
    // Ensure bundle adjustment is constructed before rescaling
    bundle_adjustment.reset();
    bundle_adjustment.construct();
    
    // Perform rescaling using the core implementation
    bundle_adjustment.performRescaling(scale_factor);
    
    // Update the data structures with scaled values
    bundle_adjustment.update();
    
    // Re-interpolate GPS observations using updated frame positions
    LocationMatcher temp_matcher;
    temp_matcher.matches = data->gps_obs;
    temp_matcher.reinterpolateMatches(data->frames);
    data->gps_obs = temp_matcher.matches;
    
    std::cout << "Manual rescaling complete" << std::endl;
  }

  void MapProcessor::saveMap(std::string dir) {
    std::filesystem::create_directory(dir);
    std::string output = dir + "/map.g2o";

    std::cout << std::endl;
    bundle_adjustment.optimizer.save(output.c_str());
    std::cout << "Saved g2o file to: " << output << std::endl;

    // data->map.landmarks.cull();

    // Serialize
    nlohmann::json map_json = data->map;
    // Save
    std::ofstream file(dir + "/map.json");
    file << std::setw(2) << map_json << std::endl;
  }

}