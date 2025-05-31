#ifndef EDGE_SE3_EXPMAP_GRAVITY_H_
#define EDGE_SE3_EXPMAP_GRAVITY_H_

#include "g2o/core/base_unary_edge.h"
#include "g2o/types/sba/types_six_dof_expmap.h"

#ifdef G2O_HAVE_OPENGL
#include "g2o/stuff/opengl_wrapper.h"
#include "g2o/stuff/opengl_primitives.h"
#include "g2o/core/hyper_graph_action.h"
#endif

namespace g2o {

/**
 * \brief Gravity alignment constraint for SE3Expmap poses
 *
 * Constrains a camera pose so that R * g_camera ≈ (0,1,0) where:
 * - g_camera is the gravity direction in camera coordinates (measurement)
 * - (0,1,0) is the world gravity direction in g2o coordinates
 */
class EdgeSE3ExpmapGravity : public BaseUnaryEdge<3, Eigen::Vector3d, VertexSE3Expmap> {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
    
    EdgeSE3ExpmapGravity() {}
    
    /**
     * Compute the error: R * g_camera - g_world
     */
    void computeError() override {
        const VertexSE3Expmap* vertex = static_cast<const VertexSE3Expmap*>(_vertices[0]);
        const SE3Quat& pose = vertex->estimate();
        
        // Get rotation matrix from pose
        Eigen::Matrix3d R = pose.rotation().toRotationMatrix();
        
        // Transform camera gravity direction to world coordinates
        Eigen::Vector3d predicted_world_gravity = R * _measurement;
        
        // Expected world gravity direction in g2o coordinates
        Eigen::Vector3d expected_world_gravity(0, -1, 0);
        
        // Error = predicted - expected
        _error = predicted_world_gravity - expected_world_gravity;
    }
    
    /**
     * Compute Jacobian w.r.t. SE3Expmap parameterization
     */
    void linearizeOplus() override {
        const VertexSE3Expmap* vertex = static_cast<const VertexSE3Expmap*>(_vertices[0]);
        const SE3Quat& pose = vertex->estimate();
        
        Eigen::Matrix3d R = pose.rotation().toRotationMatrix();
        
        // Jacobian w.r.t. SE3 tangent space [translation, rotation]
        // The jacobian matrix should already be sized correctly by the base class
        _jacobianOplusXi.setZero();
        
        // Error doesn't depend on translation (first 3 columns stay zero)
        
        // For rotation part: de/dφ = d(R * g_camera)/dφ = [R * g_camera]×
        Eigen::Vector3d rotated_gravity = R * _measurement;
        _jacobianOplusXi.block<3, 3>(0, 3) = skewSymmetric(rotated_gravity);
    }
    
    bool read(std::istream& is) override {
        Eigen::Vector3d camera_gravity;
        for (int i = 0; i < 3; i++) {
            is >> camera_gravity[i];
        }
        setMeasurement(camera_gravity);
        return readInformationMatrix(is);
    }
    
    bool write(std::ostream& os) const override {
        for (int i = 0; i < 3; i++) {
            os << _measurement[i] << " ";
        }
        return writeInformationMatrix(os);
    }

private:
    /**
     * Create skew-symmetric matrix from 3D vector
     * [v]× = [0 -v3 v2; v3 0 -v1; -v2 v1 0]
     */
    Eigen::Matrix3d skewSymmetric(const Eigen::Vector3d& v) const {
        Eigen::Matrix3d skew;
        skew << 0, -v(2), v(1),
                v(2), 0, -v(0),
                -v(1), v(0), 0;
        return skew;
    }
};

#ifdef G2O_HAVE_OPENGL
/**
 * \brief Visualize gravity constraint as a purple vector
 */
class EdgeSE3ExpmapGravityDrawAction : public DrawAction {
public:
    EdgeSE3ExpmapGravityDrawAction() : DrawAction(typeid(EdgeSE3ExpmapGravity).name()) {}
    
    virtual HyperGraphElementAction* operator()(HyperGraph::HyperGraphElement* element,
                                              HyperGraphElementAction::Parameters* params_) {
        if (typeid(*element).name() != _typeName)
            return nullptr;
            
        refreshPropertyPtrs(params_);
        if (!_previousParams)
            return this;
            
        if (_show && !_show->value())
            return this;
            
        EdgeSE3ExpmapGravity* e = static_cast<EdgeSE3ExpmapGravity*>(element);
        VertexSE3Expmap* vertex = static_cast<VertexSE3Expmap*>(e->vertices()[0]);
        
        if (!vertex)
            return this;
            
        // Get camera position and gravity direction in world coordinates
        const SE3Quat& pose = vertex->estimate();
        Eigen::Vector3d translation = pose.inverse().translation();
        Eigen::Vector3d world_gravity = pose.rotation().toRotationMatrix() * e->measurement();
        Eigen::Vector3d gravity_end = translation + 0.5 * world_gravity;
        
        // Draw purple gravity vector
        glColor3f(0.8f, 0.2f, 0.8f);
        glPushAttrib(GL_ENABLE_BIT);
        glDisable(GL_LIGHTING);
        glBegin(GL_LINES);
        glVertex3f(translation.x(), translation.y(), translation.z());
        glVertex3f(gravity_end.x(), gravity_end.y(), gravity_end.z());
        glEnd();
        glPopAttrib();
        
        return this;
    }
};
#endif

} // namespace g2o

#endif // EDGE_SE3_EXPMAP_GRAVITY_H_