
#include "geoar/process/vertex_se3_expmap.h"
#include "g2o/core/factory.h"

#include "g2o/types/slam3d/se3_ops.h"
#include "g2o/stuff/misc.h"

#ifdef G2O_HAVE_OPENGL
#include "g2o/stuff/opengl_wrapper.h"
#include "g2o/stuff/opengl_primitives.h"
#endif

#include <iostream>
#include "g2o/core/cache.h"
#include "g2o/types/sba/vertex_se3_expmap.h"
#include "g2o/core/hyper_graph_action.h"

using namespace Eigen;

namespace g2o {

#ifdef G2O_HAVE_OPENGL
  void drawTriangle(float xSize, float ySize){
    Vector3F p[3];
    glBegin(GL_TRIANGLES);
    p[0] << 0., 0., 0.;
    p[1] << -xSize, ySize, 0.;
    p[2] << -xSize, -ySize, 0.;
    for (int i = 1; i < 2; ++i) {
      Vector3F normal = (p[i] - p[0]).cross(p[i+1] - p[0]);
      glNormal3f(normal.x(), normal.y(), normal.z());
      glVertex3f(p[0].x(), p[0].y(), p[0].z());
      glVertex3f(p[i].x(), p[i].y(), p[i].z());
      glVertex3f(p[i+1].x(), p[i+1].y(), p[i+1].z());
    }
    glEnd();
  }

  VertexSE3ExpmapDrawAction::VertexSE3ExpmapDrawAction()
      : DrawAction(typeid(VertexSE3Expmap).name()), _triangleX(nullptr), _triangleY(nullptr) {
    _cacheDrawActions = 0;
  }

  bool VertexSE3ExpmapDrawAction::refreshPropertyPtrs(HyperGraphElementAction::Parameters* params_){
    if (!DrawAction::refreshPropertyPtrs(params_))
      return false;
    if (_previousParams){
      _triangleX = _previousParams->makeProperty<FloatProperty>(_typeName + "::TRIANGLE_X", .2f);
      _triangleY = _previousParams->makeProperty<FloatProperty>(_typeName + "::TRIANGLE_Y", .05f);
    } else {
      _triangleX = 0;
      _triangleY = 0;
    }
    return true;
  }

  HyperGraphElementAction* VertexSE3ExpmapDrawAction::operator()(HyperGraph::HyperGraphElement* element,
                 HyperGraphElementAction::Parameters* params_){
    if (typeid(*element).name()!=_typeName)
      return nullptr;
    initializeDrawActionsCache();
    refreshPropertyPtrs(params_);

    if (! _previousParams)
      return this;

    if (_show && !_show->value())
      return this;

    VertexSE3Expmap* that = static_cast<VertexSE3Expmap*>(element);

    glColor3f(POSE_VERTEX_COLOR);
    glPushMatrix();
    g2o::Isometry3 isometry = that->estimate();
    glMultMatrixd(isometry.matrix().cast<double>().eval().data());
    opengl::drawArrow2D(_triangleX->value(), _triangleY->value(), _triangleX->value()*.3f);
    drawCache(that->cacheContainer(), params_);
    drawUserData(that->userData(), params_);
    glPopMatrix();
    return this;
  }

  G2O_REGISTER_ACTION(VertexSE3ExpmapDrawAction);
#endif
}