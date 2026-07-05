import json
from colmap_pose import extract_rotation_translation_from_extrinsics

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

            # world-to-camera (R,t) in COLMAP convention -> camera center C = -R^T t
            R, t = extract_rotation_translation_from_extrinsics(
                frame['extrinsics'],
                apply_colmap_conversion=True
            )
            camera_position = -R.T @ t

            # Write in format: image_name X Y Z
            f.write(f"{image_name} {camera_position[0]:.10f} {camera_position[1]:.10f} {camera_position[2]:.10f}\n")

    print(f"Created reference coordinates file with {len(arkit_frames)} camera positions")
    return True