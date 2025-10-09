import numpy as np


# ============================================================================
# ARKit Extrinsics Parsing and Coordinate Conversion
# ============================================================================

def parse_arkit_extrinsics(extrinsics):
    """
    Parse ARKit extrinsics array into 4x4 transformation matrix.

    ARKit extrinsics is a 16-element array representing a 4x4 transformation
    matrix in column-major order (camera-from-world transform).

    Args:
        extrinsics: 16-element array or list from ARKit

    Returns:
        4x4 numpy array representing camera-from-world transformation
    """
    return np.array(extrinsics).reshape(4, 4, order='F')

def extract_rotation_translation_from_extrinsics(extrinsics, apply_colmap_conversion=False):
    """
    Extract rotation matrix and translation vector from ARKit extrinsics.

    Args:
        extrinsics: 16-element array [R00, R10, R20, 0, R01, R11, R21, 0,
                                       R02, R12, R22, 0, tx, ty, tz, 1]
        apply_colmap_conversion: If True, apply Y/Z axis flip for COLMAP coordinate system

    Returns:
        (R, t): 3x3 rotation matrix and 3-element translation vector
    """
    matrix = parse_arkit_extrinsics(extrinsics)
    R = matrix[:3, :3]
    t = matrix[:3, 3]

    if apply_colmap_conversion:
        # COLMAP has opposite Y and Z axis from ARKit
        # Apply coordinate conversion to translation
        t = np.array([t[0], -t[1], -t[2]])
        # Apply coordinate conversion to rotation
        R = R.copy()
        R[:, 1] = -R[:, 1]  # Flip Y column
        R[:, 2] = -R[:, 2]  # Flip Z column

    return R, t

def compute_relative_pose_from_arkit(extrinsics1, extrinsics2, for_colmap=False):
    """
    Compute relative pose from camera1 to camera2 from ARKit extrinsics.

    Given two camera-from-world transforms T1 and T2, compute the
    camera2-from-camera1 transform: T_rel = T2 * T1^-1

    Args:
        extrinsics1: ARKit extrinsics for camera 1
        extrinsics2: ARKit extrinsics for camera 2
        for_colmap: If True, work in COLMAP coordinate system (Y/Z flipped)

    Returns:
        (R_rel, t_rel): Relative rotation matrix and translation vector
    """
    # Extract poses (convert to COLMAP coordinates if needed)
    R1, t1 = extract_rotation_translation_from_extrinsics(extrinsics1, apply_colmap_conversion=for_colmap)
    R2, t2 = extract_rotation_translation_from_extrinsics(extrinsics2, apply_colmap_conversion=for_colmap)

    # Compute world-from-camera1 (invert T1)
    R1_inv = R1.T
    t1_inv = -R1.T @ t1

    # Compute camera2-from-camera1: T2 * T1^-1
    R_rel = R2 @ R1_inv
    t_rel = R2 @ t1_inv + t2

    return R_rel, t_rel


# ============================================================================
# Quaternion and Rotation Matrix Conversions
# ============================================================================

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
