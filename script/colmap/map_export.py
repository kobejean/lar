import json
import numpy as np
from database_operations import read_colmap_poses
from feature_extraction import get_descriptor_for_3d_point

def calculate_spatial_bounds(landmark_position, camera_positions, max_distance_factor=1.5):
    """Calculate spatial bounds for a landmark based on camera positions that observe it"""
    if not camera_positions:
        return {
            "lower": {"x": landmark_position[0] - 1.0, "y": landmark_position[2] - 1.0},
            "upper": {"x": landmark_position[0] + 1.0, "y": landmark_position[2] + 1.0}
        }
    
    camera_positions = np.array(camera_positions)
    distances = np.linalg.norm(camera_positions - landmark_position, axis=1)
    max_distance = np.max(distances) if len(distances) > 0 else 1.0
    
    landmark_x, landmark_z = landmark_position[0], landmark_position[2]
    camera_x_coords = camera_positions[:, 0]
    camera_z_coords = camera_positions[:, 2]
    
    extent = max_distance * max_distance_factor
    
    min_x = min(np.min(camera_x_coords), landmark_x) - extent * 0.5
    max_x = max(np.max(camera_x_coords), landmark_x) + extent * 0.5
    min_z = min(np.min(camera_z_coords), landmark_z) - extent * 0.5
    max_z = max(np.max(camera_z_coords), landmark_z) + extent * 0.5
    
    return {
        "lower": {"x": min_x, "y": min_z},
        "upper": {"x": max_x, "y": max_z}
    }

def matrix_to_list_column_major(matrix):
    """Convert a 4x4 numpy matrix to a flat list in column-major order (ARKit format)"""
    return matrix.T.flatten().tolist()

def list_to_matrix_column_major(transform_list):
    """Convert a flat list to a 4x4 numpy matrix from column-major order (ARKit format)"""
    return np.array(transform_list).reshape(4, 4, order='F')

def find_arkit_frame_by_id(arkit_frames, frame_id):
    """Helper function to find ARKit frame by ID"""
    for frame in arkit_frames:
        if frame["id"] == frame_id:
            return frame
    return None

def update_anchor_transforms(anchors, arkit_frames, colmap_poses):
    """
    Alternative approach: Use the full COLMAP pose (rotation + translation)
    if the position-only update doesn't work well.
    """
    updated_anchors = []
    
    print(f"Updating {len(anchors)} anchors using full COLMAP poses...")
    
    for anchor_id, anchor_data in anchors:
        frame_id = anchor_data["frame_id"]
        arkit_frame = arkit_frames[frame_id]
        
        # Find the corresponding COLMAP pose
        colmap_pose = colmap_poses.get(frame_id)
        if colmap_pose is None:
            print(f"Warning: COLMAP pose not found for frame {frame_id}, keeping original anchor {anchor_id}")
            updated_anchors.append([anchor_id, anchor_data])
            continue
        try:
            # Get the original ARKit camera-to-world transform
            new_camera_to_world = colmap_pose.camera_to_world_matrix
            anchor_to_camera = list_to_matrix_column_major(anchor_data["relative_transform"])
            new_anchor_to_world = new_camera_to_world @ anchor_to_camera
            updated_anchor_data = anchor_data.copy()
            updated_anchor_data["transform"] = matrix_to_list_column_major(new_anchor_to_world)
            updated_anchors.append([anchor_id, updated_anchor_data])
            print(f"Updated anchor {anchor_id} for frame {frame_id}")
            print(list_to_matrix_column_major(arkit_frame['extrinsics']))
            print(new_camera_to_world)
            print(list_to_matrix_column_major(anchor_data['transform']))
            print(new_anchor_to_world)
            
        except Exception as e:
            print(f"Error updating anchor {anchor_id}: {e}")
            updated_anchors.append([anchor_id, anchor_data])
            continue
    
    return updated_anchors

def load_existing_map(map_file_path):
    """Load existing map.json file"""
    try:
        with open(map_file_path, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Warning: Existing map file {map_file_path} not found")
        return None
    except json.JSONDecodeError as e:
        print(f"Error parsing existing map file: {e}")
        return None

def export_aligned_map_json(poses_dir, database_path, output_file, arkit_frames=None, existing_map_path=None):
    """Export landmarks in map.json format from aligned model (no coordinate conversions)"""
    try:
        # Read reconstruction files from aligned model
        images_file = poses_dir / "images.txt"
        points3d_file = poses_dir / "points3D.txt"
        
        if not images_file.exists() or not points3d_file.exists():
            print("Could not find reconstruction files for map export")
            return False
        
        print("Exporting aligned map.json (assuming aligned coordinates)...")
        
        # Read poses from aligned model
        colmap_poses = read_colmap_poses(images_file)
        
        # Load existing map data if provided
        existing_map = None
        if existing_map_path:
            existing_map = load_existing_map(existing_map_path)
        
        # Parse 3D points and create landmarks
        landmarks = []
        processed = 0
        
        with open(points3d_file, 'r') as f:
            for line in f:
                if line.startswith('#') or not line.strip():
                    continue
                parts = line.strip().split()
                if len(parts) >= 8:
                    try:
                        point3d_id = int(parts[0])
                        # convert from COLMAP to ARKit convention
                        x, y, z = float(parts[1]), -float(parts[2]), -float(parts[3])
                        
                        # Parse the track
                        track = []
                        observing_cameras = []
                        
                        for i in range(8, len(parts), 2):
                            if i + 1 < len(parts):
                                img_id = int(parts[i])
                                point2d_idx = int(parts[i + 1])
                                track.append((img_id, point2d_idx))
                                
                                if img_id in colmap_poses:
                                    cam_pos = colmap_poses[img_id].camera_position
                                    observing_cameras.append(cam_pos)
                        
                        sightings = len(track)
                        
                        # Get descriptor
                        descriptor_b64 = get_descriptor_for_3d_point(
                            track, colmap_poses, database_path
                        )
                        if descriptor_b64 is None:
                            continue
                        
                        # Calculate bounds
                        bounds = calculate_spatial_bounds(
                            landmark_position=np.array([x, y, z]),
                            camera_positions=observing_cameras,
                            max_distance_factor=2.0
                        )
                        
                        landmark = {
                            "bounds": bounds,
                            "desc": descriptor_b64,
                            "id": point3d_id,
                            "orientation": [0.0, 0.0, 1.0],
                            "position": [x, y, z],
                            "sightings": sightings
                        }
                        
                        landmarks.append(landmark)
                        processed += 1
                        
                    except (ValueError, IndexError) as e:
                        continue
        
        # Update anchors if we have the necessary data
        updated_anchors = []
        updated_edges = []
        updated_origin = None
        origin_ready = True
        
        if existing_map and arkit_frames:
            updated_anchors = update_anchor_transforms(
                existing_map["anchors"], 
                arkit_frames, 
                colmap_poses
            )
            
            # Copy other existing data
            if "edges" in existing_map:
                updated_edges = existing_map["edges"]
                print(f"Preserved {len(updated_edges)} edges from existing map")
            if "origin" in existing_map:
                updated_origin = existing_map["origin"]
                print("Preserved origin from existing map")
            if "origin_ready" in existing_map:
                origin_ready = existing_map["origin_ready"]
        
        # Create the map structure
        map_data = {
            "anchors": updated_anchors,
            "edges": updated_edges,
            "landmarks": landmarks,
        }
        
        # Add origin data if available
        if updated_origin is not None:
            map_data["origin"] = updated_origin
            map_data["origin_ready"] = origin_ready
        
        # Save as JSON
        with open(output_file, 'w') as f:
            json.dump(map_data, f, indent=2)
        
        print(f"✅ Successfully exported aligned map.json:")
        print(f"  - {len(landmarks)} landmarks with SIFT descriptors")
        print(f"  - {len(updated_anchors)} updated anchors (no coordinate conversion)")
        print(f"  - {len(updated_edges)} edges")
        print(f"  - Origin data: {'included' if updated_origin else 'not available'}")
        print(f"  - Saved to: {output_file}")
        
        return True
        
    except Exception as e:
        print(f"Error exporting aligned map.json: {e}")
        import traceback
        traceback.print_exc()
        return False