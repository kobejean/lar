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

# ARKit camera axes (x-right, y-up, looks down -z) -> COLMAP/OpenCV camera axes
# (x-right, y-down, looks down +z). This is a CAMERA-side flip, applied on the
# left of the (inverted) rotation. The world frame is kept as ARKit's.
_ARKIT_TO_COLMAP_CAM = np.diag([1.0, -1.0, -1.0])

def extract_rotation_translation_from_extrinsics(extrinsics, apply_colmap_conversion=False):
    """
    Extract a world-to-camera pose from an ARKit extrinsics matrix.

    ARKit `extrinsics` is a column-major 4x4 **camera-to-world** transform whose
    translation column is the camera center C in ARKit world coordinates.

    Args:
        extrinsics: 16-element ARKit extrinsics (column-major camera-to-world)
        apply_colmap_conversion: If True, return the COLMAP world-to-camera pose
            (R_w2c, t_w2c) with R_w2c = F @ R_c2w^T, t_w2c = -R_w2c @ C, where
            F flips the camera y/z axes. If False, return the raw ARKit
            camera-to-world (R_c2w, C).

    Returns:
        (R, t): 3x3 rotation and 3-vector. world-to-camera when
        apply_colmap_conversion=True, else raw camera-to-world.
    """
    matrix = parse_arkit_extrinsics(extrinsics)
    R_c2w = matrix[:3, :3]
    C = matrix[:3, 3]  # camera center in ARKit world coordinates

    if apply_colmap_conversion:
        R = _ARKIT_TO_COLMAP_CAM @ R_c2w.T  # world-to-camera (COLMAP camera axes)
        t = -R @ C
        return R, t

    return R_c2w, C

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
        """Camera center C in world coordinates (= ARKit world frame)."""
        if self._camera_position is None:
            # world-to-camera (R,t) -> center C = -R^T t. World frame is ARKit's,
            # so no extra axis flip is needed.
            self._camera_position = -self._rotation_matrix.T @ self.translation
        return self._camera_position

    @property
    def camera_to_world_matrix(self):
        """Get ARKit convention 4x4 camera-to-world transformation matrix.

        Inverse of extract_rotation_translation_from_extrinsics: given the COLMAP
        world-to-camera R_w2c, the ARKit camera-to-world rotation is
        R_c2w = R_w2c^T @ F (F flips the camera y/z axes back).
        """
        if self._camera_to_world_matrix is None:
            R_c2w = self._rotation_matrix.T @ _ARKIT_TO_COLMAP_CAM
            C = self.camera_position
            M = np.eye(4, dtype=np.float64)
            M[:3, :3] = R_c2w
            M[:3, 3] = C
            self._camera_to_world_matrix = M
        return self._camera_to_world_matrix
