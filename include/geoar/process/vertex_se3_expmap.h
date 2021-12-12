
#ifndef GEOAR_VERTEXSE3EXPMAP_H
#define GEOAR_VERTEXSE3EXPMAP_H

#include "g2o/config.h"
#include "g2o/core/base_vertex.h"
#include "g2o/core/hyper_graph_action.h"
#include "g2o/types/sba/g2o_types_sba_api.h"

namespace g2o {
#ifdef G2O_HAVE_OPENGL
  /**
   * \brief visualize the 3D pose vertex
   */
  class G2O_TYPES_SBA_API VertexSE3ExpmapDrawAction: public DrawAction{
    public:
      VertexSE3ExpmapDrawAction();
      virtual HyperGraphElementAction* operator()(HyperGraph::HyperGraphElement* element, HyperGraphElementAction::Parameters* params_);
    protected:
      virtual bool refreshPropertyPtrs(HyperGraphElementAction::Parameters* params_);
      FloatProperty* _triangleX, *_triangleY;
  };

#endif
}

#endif /* GEOAR_VERTEXSE3EXPMAP_H */