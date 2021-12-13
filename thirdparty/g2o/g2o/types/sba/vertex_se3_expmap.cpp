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

#include "vertex_se3_expmap.h"
#include "g2o/core/factory.h"
#ifdef G2O_HAVE_OPENGL
#include "g2o/stuff/opengl_wrapper.h"
#include "g2o/stuff/opengl_primitives.h"
#endif

#include "g2o/stuff/misc.h"

namespace g2o {

VertexSE3Expmap::VertexSE3Expmap() : BaseVertex<6, SE3Quat>() {}

bool VertexSE3Expmap::read(std::istream& is) {
  Vector7 est;
  internal::readVector(is, est);
  setEstimate(SE3Quat(est).inverse());
  return true;
}

bool VertexSE3Expmap::write(std::ostream& os) const {
  return internal::writeVector(os, estimate().inverse().toVector());
}

void VertexSE3Expmap::setToOriginImpl() { _estimate = SE3Quat(); }

void VertexSE3Expmap::oplusImpl(const number_t* update_) {
  Eigen::Map<const Vector6> update(update_);
  setEstimate(SE3Quat::exp(update) * estimate());
}

#ifdef G2O_HAVE_OPENGL

  VertexSE3ExpmapDrawAction::VertexSE3ExpmapDrawAction()
      : DrawAction(typeid(VertexSE3Expmap).name()), _length(nullptr), _depth(nullptr) {
    _cacheDrawActions = 0;
  }

  bool VertexSE3ExpmapDrawAction::refreshPropertyPtrs(HyperGraphElementAction::Parameters* params_){
    if (!DrawAction::refreshPropertyPtrs(params_))
      return false;
    if (_previousParams){
      _length = _previousParams->makeProperty<FloatProperty>(_typeName + "::LENGTH", .2f);
      _depth = _previousParams->makeProperty<FloatProperty>(_typeName + "::DEPTH", .2f);
    } else {
      _length = 0;
      _depth = 0;
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
    g2o::Isometry3 isometry = that->estimate().inverse();
    Eigen::Matrix4d transform = isometry.matrix();
    transform(Eigen::indexing::all, 1) = -transform(Eigen::indexing::all, 1);
    transform(Eigen::indexing::all, 2) = -transform(Eigen::indexing::all, 2);
    glMultMatrixd(transform.cast<double>().eval().data());
    opengl::drawPyramid(_length->value(), _depth->value());
    drawCache(that->cacheContainer(), params_);
    drawUserData(that->userData(), params_);
    glPopMatrix();
    return this;
  }

#endif
}  // namespace g2o
