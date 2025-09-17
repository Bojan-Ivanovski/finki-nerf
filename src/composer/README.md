# Composer Package

This package contains camera orchestration, scene setup, and video generation functionality for FINKI-NeRF. It bridges the gap between raw datasets and renderable camera configurations.

## Structure

```
composer/
├── __init__.py              # Camera generation and video path functions
├── synthentic_worker.py     # Synthetic dataset handling
└── README.md                # This file
```

## Key Components

### Camera Generation (`__init__.py`)

#### Basic Camera Setup
- **`generate_camera_options(num_options=5)`** - Generate random camera positions
- **`create_boilerplate(frame, width, height)`** - Create ray data for rendering

#### Video Path Generation
- **`generate_smooth_camera_path(path_type, num_frames, ...)`** - Create smooth camera trajectories
- **`interpolate_camera_poses(pose1, pose2, t)`** - Smooth camera interpolation
- **`generate_interpolated_path(keyframes, num_frames)`** - Keyframe-based paths

#### Supported Path Types
1. **Orbit**: Circular motion around object at fixed height
2. **Spiral**: Dynamic spiral with changing height/radius
3. **Linear**: Straight-line motion between two points
4. **Keyframe**: Interpolation between actual dataset camera positions

### Synthetic Dataset Handling (`synthentic_worker.py`)
- **SyntheticObject**: Individual object processing and frame extraction
- **SyntheticDataset**: Dataset-level operations and object enumeration
- **SyntheticFrame**: Individual camera frame representation with pose data

## Camera Path Generation

### Orbit Path
Creates circular camera movement around the scene center:
```python
frames = generate_smooth_camera_path(
    path_type="orbit",
    num_frames=60,
    radius=5.0,
    height=1.5,
    start_angle=0,
    end_angle=2*np.pi
)
```

**Parameters:**
- `radius`: Distance from scene center
- `height`: Camera height above ground plane
- `start_angle`/`end_angle`: Angular range for orbit

### Spiral Path
Creates spiral motion with changing height and radius:
```python
frames = generate_smooth_camera_path(
    path_type="spiral",
    num_frames=90,
    radius=5.0,
    height=1.5
)
```

**Characteristics:**
- Starts at specified radius/height
- Spirals inward while rising up
- Creates dramatic camera movements

### Linear Path
Creates straight-line camera motion:
```python
frames = generate_smooth_camera_path(
    path_type="linear",
    num_frames=30,
    radius=5.0,  # Used for start/end positions
    height=1.5
)
```

**Movement:** Side-to-side motion while maintaining view of center

### Keyframe-Based Path
Uses actual dataset camera positions:
```python
# Extract keyframes from dataset
obj = SyntheticObject(data_path, "lego")
keyframes = list(obj.list_frames())[:5]

# Generate interpolated path
frames = generate_interpolated_path(keyframes, 60)
```

**Benefits:**
- Uses real camera positions from training data
- Smooth interpolation between known good viewpoints
- Validates model on actual dataset cameras

## Camera Mathematics

### Coordinate System
- **Origin**: Scene center at (0, 0, 0)
- **Up Vector**: Positive Y-axis (0, 1, 0)
- **Forward**: Camera looks toward origin
- **Right**: Cross product of up and forward vectors

### Transformation Matrices
4×4 homogeneous transformation matrices:
```
[ R11 R12 R13 Tx ]
[ R21 R22 R23 Ty ]
[ R31 R32 R33 Tz ]
[  0   0   0   1 ]
```

Where:
- **R**: 3×3 rotation matrix (camera orientation)
- **T**: 3×1 translation vector (camera position)

### Camera Pose Interpolation
Spherical linear interpolation (SLERP) for smooth camera motion:
```python
def interpolate_camera_poses(pose1, pose2, t):
    # Linear interpolation for position
    pos = (1 - t) * pos1 + t * pos2

    # SLERP for rotation
    rot = slerp(rot1, rot2, t)

    return combine_pose(rot, pos)
```

## Integration Points

### With Tools Package
- Uses `save_pixels()` to create rendering boilerplates
- Integrates with ray generation and pixel processing
- Handles coordinate transformations

### With Data Package
- Accesses dataset paths via `SYNTHETIC_TRAIN_DATA_PATH`
- Processes synthetic dataset directory structures
- Handles multiple object types and camera configurations

### With CLI Package
- Video command uses camera path generation functions
- Predict command uses basic camera setup
- Provides camera position options for user selection

### With Core Package
- DataManager uses synthetic worker classes
- Camera frames integrate with rendering pipeline
- Supports both training data generation and inference

## Boilerplate Generation

### Process
1. **Camera Setup**: Define camera pose and intrinsics
2. **Ray Generation**: Create rays for each pixel in output image
3. **Data Packaging**: Save ray data to HDF5 format
4. **Rendering Ready**: File ready for model inference

### HDF5 Structure
Each boilerplate file contains:
```
rays: [H*W, 11] array
  - Ray origins (3D)
  - Ray directions (3D)
  - Pixel coordinates (2D)
  - Placeholder data (3D)
```

### Usage Example
```python
from composer import create_boilerplate, generate_camera_options

# Generate camera position
cameras = generate_camera_options(num_options=1)
selected_camera = cameras[0]

# Create rendering boilerplate
create_boilerplate(selected_camera, width=800, height=800)

# File created: {frame_path}.h5 ready for rendering
```

## Video Generation Workflow

### Complete Pipeline
1. **Path Planning**: Choose path type and parameters
2. **Frame Generation**: Create camera frames along path
3. **Boilerplate Creation**: Generate ray data for each frame
4. **Model Rendering**: Process each frame through NeRF
5. **Video Assembly**: Combine frames into MP4 video

### Performance Considerations
- **Memory Usage**: One boilerplate file per frame (temporary)
- **Disk Space**: 800×800 images ≈ 2MB per frame
- **Processing Time**: Depends on model complexity and GPU
- **Cleanup**: Temporary boilerplate files automatically removed

## Synthetic Dataset Format

### Expected Structure
```
train_dataset_synthetic/
├── lego/
│   ├── transforms_train.json    # Camera poses and intrinsics
│   └── train/
│       ├── r_0.png             # Training images
│       ├── r_1.png
│       └── ...
└── chair/
    ├── transforms_train.json
    └── train/
```

### Camera Pose Format
JSON format following NeRF convention:
```json
{
    "camera_angle_x": 0.6911503076553345,
    "frames": [
        {
            "file_path": "./train/r_0",
            "rotation": 0.012566370614359171,
            "transform_matrix": [
                [0.6911, 0.0, 0.7227, 2.7],
                [0.2274, 0.9495, -0.2176, -0.8],
                [-0.6861, 0.3138, 0.6563, 2.4],
                [0.0, 0.0, 0.0, 1.0]
            ]
        }
    ]
}
```

## Error Handling

### Camera Generation
- **Invalid Parameters**: Validate radius, height, angles
- **Insufficient Keyframes**: Ensure minimum 2 keyframes for interpolation
- **Matrix Validation**: Check transformation matrix validity

### Dataset Processing
- **Missing Files**: Handle missing transforms.json or image files
- **Invalid Format**: Validate JSON structure and camera parameters
- **Path Resolution**: Handle dataset path configuration issues

## Best Practices

### Camera Path Design
1. **Smooth Motion**: Use appropriate frame counts for smooth video
2. **Scene Coverage**: Ensure path shows interesting scene aspects
3. **Collision Avoidance**: Keep cameras outside scene bounds
4. **Consistent Speed**: Maintain uniform camera motion

### Performance Optimization
1. **Batch Processing**: Process multiple frames efficiently
2. **Memory Management**: Clean up temporary files promptly
3. **Path Caching**: Reuse computed camera paths when possible
4. **Progressive Generation**: Generate frames incrementally

## Usage Examples

### Basic Camera Setup
```python
from composer import generate_camera_options, create_boilerplate

# Generate random camera positions
cameras = generate_camera_options(num_options=5)

# User selects camera
selected = cameras[int(input("Pick camera (0-4): "))]

# Create boilerplate for rendering
create_boilerplate(selected, 800, 800)
```

### Video Path Generation
```python
from composer import generate_smooth_camera_path

# Create orbit video path
orbit_frames = generate_smooth_camera_path(
    path_type="orbit",
    num_frames=120,
    radius=4.0,
    height=2.0
)

# Create spiral path
spiral_frames = generate_smooth_camera_path(
    path_type="spiral",
    num_frames=90
)
```

### Keyframe Interpolation
```python
from composer import generate_interpolated_path
from composer.synthentic_worker import SyntheticObject

# Load object keyframes
obj = SyntheticObject("./data/train_dataset_synthetic", "lego")
keyframes = list(obj.list_frames())[:8]

# Generate smooth interpolation
smooth_path = generate_interpolated_path(keyframes, 200)
```
