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

#include "edge_project_xyz2uvd.h"
#ifdef G2O_HAVE_OPENGL
#include "g2o/stuff/opengl_wrapper.h"
#include "g2o/stuff/opengl_primitives.h"
#endif

namespace g2o {

EdgeProjectXYZ2UVD::EdgeProjectXYZ2UVD()
    : BaseBinaryEdge<3, Vector3, VertexPointXYZ, VertexSE3Expmap>() {
  _cam = 0;
  resizeParameters(1);
  installParameter(_cam, 0);
}

bool EdgeProjectXYZ2UVD::read(std::istream& is) {
  readParamIds(is);
  internal::readVector(is, _measurement);
  return readInformationMatrix(is);
}

bool EdgeProjectXYZ2UVD::write(std::ostream& os) const {
  writeParamIds(os);
  internal::writeVector(os, measurement());
  return writeInformationMatrix(os);
}

void EdgeProjectXYZ2UVD::computeError() {
  const VertexSE3Expmap* v1 = static_cast<const VertexSE3Expmap*>(_vertices[1]);
  const VertexPointXYZ* v2 = static_cast<const VertexPointXYZ*>(_vertices[0]);
  const CameraParameters* cam = static_cast<const CameraParameters*>(parameter(0));
  const Vector3 cam_space = v1->estimate().map(v2->estimate());
  
  // Check if point is behind camera (negative Z)
  if (cam_space[2] <= 0.0) {
    const number_t large_error = 1e9;
    _error = Vector3(large_error, large_error, large_error);
    return;
  }
  
  const Vector2 img_coord = cam->cam_map(cam_space);
  const Vector3 img_space(img_coord[0], img_coord[1], cam_space[2]);
  _error = measurement() - img_space;
}


void EdgeProjectXYZ2UVD::linearizeOplus() {
  VertexSE3Expmap* vj = static_cast<VertexSE3Expmap*>(_vertices[1]);
  SE3Quat T(vj->estimate());
  VertexPointXYZ* vi = static_cast<VertexPointXYZ*>(_vertices[0]);
  Vector3 xyz = vi->estimate();
  Vector3 xyz_trans = T.map(xyz);

  number_t x = xyz_trans[0];
  number_t y = xyz_trans[1];
  number_t z = xyz_trans[2];
  
  // Check if point is behind camera
  if (z <= 0.0) {
    // Set zero Jacobians when point is behind camera
    // This prevents the optimizer from using invalid gradients
    _jacobianOplusXi.setZero();
    _jacobianOplusXj.setZero();
    return;
  }
  
  number_t z_2 = z * z;
  const CameraParameters* cam = static_cast<const CameraParameters*>(parameter(0));

  Eigen::Matrix<number_t, 3, 3, Eigen::ColMajor> J_intr;
  J_intr(0, 0) = -1. / z * cam->focal_length;
  J_intr(0, 1) = 0;
  J_intr(0, 2) = x / z_2 * cam->focal_length;

  J_intr(1, 0) = 0;
  J_intr(1, 1) = -1. / z * cam->focal_length;
  J_intr(1, 2) = y / z_2 * cam->focal_length;

  J_intr(2, 0) = 0;
  J_intr(2, 1) = 0;
  J_intr(2, 2) = 1;

  _jacobianOplusXi = J_intr * T.rotation().toRotationMatrix();

  /*
  Translation Jacobian:
  J_intr * [[   0,  -z,   y],
            [   z,   0,  -x],
            [-y*f, x*f,   0]]
  */
  _jacobianOplusXj(0, 0) = x * y / z_2 * cam->focal_length;
  _jacobianOplusXj(0, 1) = -(1 + (x * x / z_2)) * cam->focal_length;
  _jacobianOplusXj(0, 2) = y / z * cam->focal_length;

  _jacobianOplusXj(1, 0) = (1 + y * y / z_2) * cam->focal_length;
  _jacobianOplusXj(1, 1) = -x * y / z_2 * cam->focal_length;
  _jacobianOplusXj(1, 2) = -x / z * cam->focal_length;

  _jacobianOplusXj(2, 0) = -y * cam->focal_length;
  _jacobianOplusXj(2, 1) = x * cam->focal_length;
  _jacobianOplusXj(2, 2) = 0;

  /*
  Rotation Jacobian:
  */
  _jacobianOplusXj.block<3,3>(0, 3) = J_intr;
}

#ifdef G2O_HAVE_OPENGL

  Eigen::Vector3d EdgeProjectXYZ2UVDDrawAction::getDirectionVector(const CameraParameters* cam, Eigen::Vector2d kpt) {
    double scale = 1.0 / (double)cam->focal_length;
    double x = (kpt.x() - cam->principle_point[0]) * scale;
    double y = (kpt.y() - cam->principle_point[1]) * scale;
    double z = 1;
    return Eigen::Vector3d(x, y, z).normalized();
  }

  EdgeProjectXYZ2UVDDrawAction::EdgeProjectXYZ2UVDDrawAction(): DrawAction(typeid(EdgeProjectXYZ2UVD).name()){}

  HyperGraphElementAction* EdgeProjectXYZ2UVDDrawAction::operator()(HyperGraph::HyperGraphElement* element,
               HyperGraphElementAction::Parameters* params_){
    if (typeid(*element).name()!=_typeName)
      return nullptr;
    refreshPropertyPtrs(params_);
    if (! _previousParams)
      return this;

    if (_show && !_show->value())
      return this;

    EdgeProjectXYZ2UVD* e =  static_cast<EdgeProjectXYZ2UVD*>(element);
    const CameraParameters* cam = static_cast<const CameraParameters*>(e->parameter(0));
    Eigen::Vector3d img_space = e->measurement();
    Eigen::Vector2d kpt(img_space[0], img_space[1]);
    Eigen::Vector3d direction = getDirectionVector(cam, kpt);

    VertexPointXYZ* fromEdge = static_cast<VertexPointXYZ*>(e->vertices()[0]);
    VertexSE3Expmap* toEdge  = static_cast<VertexSE3Expmap*>(e->vertices()[1]);
    if (! fromEdge || ! toEdge)
      return this;

    Eigen::Vector3d fromTranslation = fromEdge->estimate();
    Eigen::Vector3d toTranslation = toEdge->estimate().inverse().translation();

    Eigen::Vector3d compliment = direction * (toTranslation-fromTranslation).norm();
    Eigen::Vector3d measurement = direction * img_space[2];
    Eigen::Vector3d target = toEdge->estimate().inverse() * measurement;
    Eigen::Vector3d continuation = toEdge->estimate().inverse() * compliment;

    glPushAttrib(GL_ENABLE_BIT);
    glDisable(GL_LIGHTING);

    // Draw line in direction of key point
    glColor3f(0.4f, 0.7f, 0.4f);
    glBegin(GL_LINES);
    glVertex3f(toTranslation.x(),toTranslation.y(),toTranslation.z());
    glVertex3f(target.x(),target.y(),target.z());

    // 
    glColor3f(0.7f, 0.7f, 0.7f);
    glBegin(GL_LINES);
    glVertex3f(target.x(),target.y(),target.z());
    glVertex3f(continuation.x(),continuation.y(),continuation.z());

    
    // Draw line to vertex
    glColor3f(0.7f, 0.4f, 0.4f);
    glBegin(GL_LINES);
    glVertex3f(continuation.x(),continuation.y(),continuation.z());
    glVertex3f(fromTranslation.x(),fromTranslation.y(),fromTranslation.z());

    glEnd();
    glPopAttrib();
    return this;
  }
#endif

}  // namespace g2o
