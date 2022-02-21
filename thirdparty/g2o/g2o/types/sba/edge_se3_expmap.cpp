// g2o - General Graph Optimization
// Copyright (C) 2011 H. Strasdat
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are
// met:
//
// * Redistributions of source code must retain the above copyright notice,
//   this list of conditions and the following disclaimer.
// * Redistributions in binary form must reproduce the above copyright
//   notice, this list of conditions and the following disclaimer in the
//   documentation and/or other materials provided with the distribution.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS
// IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED
// TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
// PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
// HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
// SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED
// TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
// LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
// NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
// SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

#include "edge_se3_expmap.h"

#ifdef G2O_HAVE_OPENGL
#include "g2o/stuff/opengl_wrapper.h"
#include "g2o/stuff/opengl_primitives.h"
#endif

namespace g2o {

EdgeSE3Expmap::EdgeSE3Expmap()
    : BaseBinaryEdge<6, SE3Quat, VertexSE3Expmap, VertexSE3Expmap>() {}

bool EdgeSE3Expmap::read(std::istream& is) {
  Vector7 meas;
  internal::readVector(is, meas);
  setMeasurement(SE3Quat(meas).inverse());
  return readInformationMatrix(is);
}

bool EdgeSE3Expmap::write(std::ostream& os) const {
  internal::writeVector(os, measurement().inverse().toVector());
  return writeInformationMatrix(os);
}

void EdgeSE3Expmap::computeError() {
  const VertexSE3Expmap* v1 = static_cast<const VertexSE3Expmap*>(_vertices[0]);
  const VertexSE3Expmap* v2 = static_cast<const VertexSE3Expmap*>(_vertices[1]);

  SE3Quat C(_measurement);
  SE3Quat error_ = v2->estimate().inverse() * C * v1->estimate();
  _error = error_.log();
}

void EdgeSE3Expmap::linearizeOplus() {
  VertexSE3Expmap* vi = static_cast<VertexSE3Expmap*>(_vertices[0]);
  SE3Quat Ti(vi->estimate());

  VertexSE3Expmap* vj = static_cast<VertexSE3Expmap*>(_vertices[1]);
  SE3Quat Tj(vj->estimate());

  const SE3Quat& Tij = _measurement;
  SE3Quat invTij = Tij.inverse();

  SE3Quat invTj_Tij = Tj.inverse() * Tij;
  SE3Quat infTi_invTij = Ti.inverse() * invTij;

  _jacobianOplusXi = invTj_Tij.adj();
  _jacobianOplusXj = -infTi_invTij.adj();
}

#ifdef G2O_HAVE_OPENGL
  EdgeSE3ExpmapDrawAction::EdgeSE3ExpmapDrawAction(): DrawAction(typeid(EdgeSE3Expmap).name()){}

  HyperGraphElementAction* EdgeSE3ExpmapDrawAction::operator()(HyperGraph::HyperGraphElement* element,
               HyperGraphElementAction::Parameters* params_){
    if (typeid(*element).name()!=_typeName)
      return nullptr;
    refreshPropertyPtrs(params_);
    if (! _previousParams)
      return this;

    if (_show && !_show->value())
      return this;

    EdgeSE3Expmap* e =  static_cast<EdgeSE3Expmap*>(element);
    VertexSE3Expmap* fromEdge = static_cast<VertexSE3Expmap*>(e->vertices()[0]);
    VertexSE3Expmap* toEdge   = static_cast<VertexSE3Expmap*>(e->vertices()[1]);
    if (! fromEdge || ! toEdge)
      return this;
    glColor3f(POSE_EDGE_COLOR);
    glPushAttrib(GL_ENABLE_BIT);
    glDisable(GL_LIGHTING);
    glBegin(GL_LINES);
    Eigen::Vector3d fromTranslation = fromEdge->estimate().inverse().translation();
    Eigen::Vector3d toTranslation = toEdge->estimate().inverse().translation();
    glVertex3f(fromTranslation.x(),fromTranslation.y(),fromTranslation.z());
    glVertex3f(toTranslation.x(),toTranslation.y(),toTranslation.z());
    glEnd();
    glPopAttrib();
    return this;
  }
#endif

}  // namespace g2o
