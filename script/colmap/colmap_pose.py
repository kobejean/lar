import numpy as np


def quaternion_to_rotation_matrix(qw, qx, qy, qz):
    """Convert quaternion to 3x3 rotation matrix"""
    norm = np.sqrt(qw*qw + qx*qx + qy*qy + qz*qz)
    qw, qx, qy, qz = qw/norm, qx/norm, qy/norm, qz/norm
    
    R = np.array([
        [1 - 2*(qy*qy + qz*qz), 2*(qx*qy - qw*qz), 2*(qx*qz + qw*qy)],
        [2*(qx*qy + qw*qz), 1 - 2*(qx*qx + qz*qz), 2*(qy*qz - qw*qx)],
        [2*(qx*qz - qw*qy), 2*(qy*qz + qw*qx), 1 - 2*(qx*qx + qy*qy)]
    ])
    return R

def rotation_matrix_to_quaternion(R):
    """Convert 3x3 rotation matrix to quaternion (w, x, y, z)"""
    # Shepperd's method for numerical stability
    trace = np.trace(R)
    
    if trace > 0:
        s = np.sqrt(trace + 1.0) * 2  # s = 4 * qw
        qw = 0.25 * s
        qx = (R[2, 1] - R[1, 2]) / s
        qy = (R[0, 2] - R[2, 0]) / s
        qz = (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2  # s = 4 * qx
        qw = (R[2, 1] - R[1, 2]) / s
        qx = 0.25 * s
        qy = (R[0, 1] + R[1, 0]) / s
        qz = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2  # s = 4 * qy
        qw = (R[0, 2] - R[2, 0]) / s
        qx = (R[0, 1] + R[1, 0]) / s
        qy = 0.25 * s
        qz = (R[1, 2] + R[2, 1]) / s
    else:
        s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2  # s = 4 * qz
        qw = (R[1, 0] - R[0, 1]) / s
        qx = (R[0, 2] + R[2, 0]) / s
        qy = (R[1, 2] + R[2, 1]) / s
        qz = 0.25 * s
    
    return np.array([qw, qx, qy, qz])

class ColmapPose:
    """Container class for COLMAP pose data with utility methods"""
    def __init__(self, parts):
        image_id = int(parts[0])
        qw, qx, qy, qz = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
        tx, ty, tz = float(parts[5]), float(parts[6]), float(parts[7])
        camera_id = int(parts[8])
        image_name = parts[9]
        
        quaternion = np.array([qw, qx, qy, qz])
        translation = np.array([tx, ty, tz])
        self.image_name = image_name
        self.image_id = image_id
        self.camera_id = camera_id
        self.quaternion = quaternion  # [qw, qx, qy, qz]
        self.translation = translation  # [tx, ty, tz] - world-to-camera translation
        # Preconpute rotation matrix
        qw, qx, qy, qz = self.quaternion
        self._rotation_matrix = quaternion_to_rotation_matrix(qw, qx, qy, qz)
        # Cached properties
        self._camera_to_world_matrix = None
        self._camera_position = None

    @property
    def camera_position(self):
        """Get camera position in world coordinates (ARKit convention)"""
        if self._camera_position is None:
            # Camera position: -R^T * t
            self._camera_position = -self._rotation_matrix.T @ self.translation
            self._camera_position[1] = -self._camera_position[1]
            self._camera_position[2] = -self._camera_position[2]
        return self._camera_position
    
    @property
    def camera_to_world_matrix(self):
        """Get ARKit convention 4x4 camera-to-world transformation matrix"""
        if self._camera_to_world_matrix is None:
            R = self._rotation_matrix
            t = self.camera_position
            self._camera_to_world_matrix = np.array(
                [
                    [ R[0,0], -R[1,0], -R[2,0],  t[0]],
                    [-R[0,1],  R[1,1],  R[2,1],  t[1]],
                    [-R[0,2],  R[1,2],  R[2,2],  t[2]],
                    [      0,       0,       0,     1],
                ],
                dtype=np.float64
            )
        return self._camera_to_world_matrix
