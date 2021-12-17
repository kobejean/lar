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

#include "edge_project_xyz2uv.h"
#ifdef G2O_HAVE_OPENGL
#include "g2o/stuff/opengl_wrapper.h"
#include "g2o/stuff/opengl_primitives.h"
#endif

namespace g2o {

EdgeProjectXYZ2UV::EdgeProjectXYZ2UV()
    : BaseBinaryEdge<2, Vector2, VertexPointXYZ, VertexSE3Expmap>() {
  _cam = 0;
  resizeParameters(1);
  installParameter(_cam, 0);
}

bool EdgeProjectXYZ2UV::read(std::istream& is) {
  readParamIds(is);
  internal::readVector(is, _measurement);
  return readInformationMatrix(is);
}

bool EdgeProjectXYZ2UV::write(std::ostream& os) const {
  writeParamIds(os);
  internal::writeVector(os, measurement());
  return writeInformationMatrix(os);
}

void EdgeProjectXYZ2UV::computeError() {
  const VertexSE3Expmap* v1 = static_cast<const VertexSE3Expmap*>(_vertices[1]);
  const VertexPointXYZ* v2 = static_cast<const VertexPointXYZ*>(_vertices[0]);
  const CameraParameters* cam = static_cast<const CameraParameters*>(parameter(0));
  _error = measurement() - cam->cam_map(v1->estimate().map(v2->estimate()));
}

void EdgeProjectXYZ2UV::linearizeOplus() {
  VertexSE3Expmap* vj = static_cast<VertexSE3Expmap*>(_vertices[1]);
  SE3Quat T(vj->estimate());
  VertexPointXYZ* vi = static_cast<VertexPointXYZ*>(_vertices[0]);
  Vector3 xyz = vi->estimate();
  Vector3 xyz_trans = T.map(xyz);

  number_t x = xyz_trans[0];
  number_t y = xyz_trans[1];
  number_t z = xyz_trans[2];
  number_t z_2 = z * z;

  const CameraParameters* cam = static_cast<const CameraParameters*>(parameter(0));

  Eigen::Matrix<number_t, 2, 3, Eigen::ColMajor> intrinsics;
  intrinsics(0, 0) = -1. / z * cam->focal_length;
  intrinsics(0, 1) = 0;
  intrinsics(0, 2) = x / z_2 * cam->focal_length;

  intrinsics(1, 0) = 0;
  intrinsics(1, 1) = -1. / z * cam->focal_length;
  intrinsics(1, 2) = y / z_2 * cam->focal_length;

  _jacobianOplusXi = intrinsics * T.rotation().toRotationMatrix();

  _jacobianOplusXj(0, 0) = x * y / z_2 * cam->focal_length;
  _jacobianOplusXj(0, 1) = -(1 + (x * x / z_2)) * cam->focal_length;
  _jacobianOplusXj(0, 2) = y / z * cam->focal_length;

  _jacobianOplusXj(1, 0) = (1 + y * y / z_2) * cam->focal_length;
  _jacobianOplusXj(1, 1) = -x * y / z_2 * cam->focal_length;
  _jacobianOplusXj(1, 2) = -x / z * cam->focal_length;

  _jacobianOplusXj.block<2,3>(0, 3) = intrinsics;
}

#ifdef G2O_HAVE_OPENGL

  Eigen::Vector3d getDirectionVector(const CameraParameters* cam, Eigen::Vector2d kpt) {
    double scale = 1.0 / (double)cam->focal_length;
    double x = (kpt.x() - cam->principle_point[0]) * scale;
    double y = (kpt.y() - cam->principle_point[1]) * scale;
    double z = 1;
    return Eigen::Vector3d(x, y, z).normalized();
  }

  EdgeProjectXYZ2UVDrawAction::EdgeProjectXYZ2UVDrawAction(): DrawAction(typeid(EdgeProjectXYZ2UV).name()){}

  HyperGraphElementAction* EdgeProjectXYZ2UVDrawAction::operator()(HyperGraph::HyperGraphElement* element,
               HyperGraphElementAction::Parameters* params_){
    if (typeid(*element).name()!=_typeName)
      return nullptr;
    refreshPropertyPtrs(params_);
    if (! _previousParams)
      return this;

    if (_show && !_show->value())
      return this;

    EdgeProjectXYZ2UV* e =  static_cast<EdgeProjectXYZ2UV*>(element);
    const CameraParameters* cam = static_cast<const CameraParameters*>(e->parameter(0));
    Eigen::Vector2d kpt = e->measurement();
    Eigen::Vector3d direction = getDirectionVector(cam, kpt);

    VertexPointXYZ* fromEdge = static_cast<VertexPointXYZ*>(e->vertices()[0]);
    VertexSE3Expmap* toEdge  = static_cast<VertexSE3Expmap*>(e->vertices()[1]);
    if (! fromEdge || ! toEdge)
      return this;

    Eigen::Vector3d fromTranslation = fromEdge->estimate();
    Eigen::Vector3d toTranslation = toEdge->estimate().inverse().translation();

    Eigen::Vector3d compliment = direction * (toTranslation-fromTranslation).norm();
    Eigen::Vector3d target = toEdge->estimate().inverse() * compliment;

    glPushAttrib(GL_ENABLE_BIT);
    glDisable(GL_LIGHTING);

    // Draw line in direction of key point
    glColor3f(0.4f, 0.7f, 0.4f);
    glBegin(GL_LINES);
    glVertex3f(toTranslation.x(),toTranslation.y(),toTranslation.z());
    glVertex3f(target.x(),target.y(),target.z());

    
    // Draw line to vertex
    glColor3f(0.7f, 0.4f, 0.4f);
    glBegin(GL_LINES);
    glVertex3f(target.x(),target.y(),target.z());
    glVertex3f(fromTranslation.x(),fromTranslation.y(),fromTranslation.z());

    glEnd();
    glPopAttrib();
    return this;
  }
#endif

}  // namespace g2o
