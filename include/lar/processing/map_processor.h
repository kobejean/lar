#ifndef LAR_PROCESSING_MAP_PROCESSOR_H
#define LAR_PROCESSING_MAP_PROCESSOR_H

#include <string>

#include "lar/core/map.h"
#include "lar/mapping/mapper.h"
#include "lar/processing/bundle_adjustment.h"
#include "lar/processing/frame_processor.h"
#include "lar/processing/global_alignment.h"

namespace lar {

  class MapProcessor {
    public:
      std::shared_ptr<Mapper::Data> data;
      
      MapProcessor(std::shared_ptr<Mapper::Data> data);
      void process();
      void optimize();
      void saveMap(std::string dir);
    // private:
      BundleAdjustment bundle_adjustment;
      GlobalAlignment global_alignment;
      FrameProcessor frame_processor;
  };
}

#endif /* LAR_PROCESSING_MAP_PROCESSOR_H */