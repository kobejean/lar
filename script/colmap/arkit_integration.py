import json
import numpy as np

def load_arkit_data(frames_json_path):
    """Load ARKit frames data from JSON file"""
    with open(frames_json_path, 'r') as f:
        frames = json.load(f)
    
    print(f"Loaded {len(frames)} ARKit frames")
    return frames

def create_reference_file_from_arkit(arkit_frames, output_file):
    """Create reference coordinates file from ARKit frames for model_aligner"""
    with open(output_file, 'w') as f:
        for frame in arkit_frames:
            image_name = f"{frame['id']:08d}_image.jpeg"
            
            # ARKit extrinsics are camera-to-world transformation
            matrix = np.array(frame['extrinsics']).reshape(4, 4, order='F')
            camera_position = matrix[:3, 3]
            
            # Write in format: image_name X Y Z (colmap has opposite Y and Z axis from ARKit)
            f.write(f"{image_name} {camera_position[0]:.10f} {-camera_position[1]:.10f} {-camera_position[2]:.10f}\n")
    
    print(f"Created reference coordinates file with {len(arkit_frames)} camera positions")
    return True