# Tools Package

This package contains utility functions and data structures for ray operations, pixel processing, and rendering in FINKI-NeRF.

## Structure

```
tools/
├── __init__.py          # Core utilities (data loading, rendering, sampling)
├── ray.py               # Ray data structures and operations
├── pixel.py             # Pixel-level data structures
└── README.md            # This file
```

## Key Components

### Core Utilities (`__init__.py`)

#### Data Management
- **`get_dataset_length(filename)`** - Get dimensions of HDF5 dataset
- **`save_pixels(pixels, filename)`** - Save pixel data to HDF5 format
- **`load_pixels(filename, batch_size, shuffle)`** - Load training data in batches

#### PDF Sampling
- **`sample_pdf(bins, weights, N_importance)`** - Importance sampling for hierarchical NeRF
  - Converts weights to probability distribution
  - Performs inverse transform sampling
  - Returns fine sampling points for second network pass

#### Image Rendering
- **`generate_image_from_boilerplate(coarse_model, fine_model, filename)`** - Complete rendering pipeline
  - Loads ray data from HDF5 boilerplate
  - Performs coarse network forward pass
  - Executes importance sampling
  - Runs fine network for final colors
  - Assembles and saves rendered image

### Ray Operations (`ray.py`)
Data structures and operations for ray-based computations:
- Ray representation and manipulation
- Coordinate transformations
- Ray-scene intersection utilities

### Pixel Processing (`pixel.py`)
Pixel-level data structures and operations:
- Frame pixel representation
- Ground truth color association
- Coordinate management

## HDF5 Data Format

### Dataset Structure
```
rays: [N, 11] float32 array
    - Columns 0-2: Ray origin (ox, oy, oz)
    - Columns 3-5: Ray direction (dx, dy, dz)
    - Columns 6-8: Ground truth RGB (r, g, b)
    - Columns 9-10: Pixel coordinates (px, py)
```

### Data Flow
1. **Generation**: Synthetic objects → Ray-pixel pairs → HDF5 files
2. **Training**: HDF5 files → Batched loading → Network training
3. **Inference**: Camera poses → Ray generation → Rendering → Images

## Rendering Pipeline

### Complete Rendering Process
```
Camera Pose → Ray Generation → Coarse Sampling → Network Forward Pass
     ↓
Importance Sampling → Fine Sampling → Fine Network → Volume Rendering
     ↓
RGB Assembly → Image Reconstruction → File Output
```

### Batch Processing
- **Memory Efficiency**: Process rays in configurable batch sizes
- **GPU Utilization**: Optimize batch sizes for available memory
- **Progress Tracking**: Monitor rendering progress for long operations

## Sampling Algorithms

### Hierarchical Volume Sampling
1. **Coarse Sampling**: Uniform distribution along ray
   ```python
   t_vals = linspace(0.0, 1.0, N_coarse)  # N_coarse = 64
   z_vals = near * (1 - t_vals) + far * t_vals
   ```

2. **Importance Sampling**: Based on coarse network weights
   ```python
   z_vals_fine = sample_pdf(bins, weights, N_fine)  # N_fine = 128
   z_vals_all = sort(concatenate([z_vals_coarse, z_vals_fine]))
   ```

### PDF Sampling Implementation
- **Input**: Bins (ray segments) and weights from coarse network
- **Process**: Convert to PDF → Compute CDF → Inverse transform sampling
- **Output**: Fine sampling points concentrated in high-density regions

## Performance Optimization

### Batch Size Guidelines
| GPU Memory | Recommended Batch Size |
|------------|----------------------|
| 4GB | 2,000-5,000 |
| 8GB | 5,000-15,000 |
| 12GB | 10,000-25,000 |
| 16GB+ | 20,000+ |

### Memory Management
```python
# Efficient data loading
for rays, colors, coords in load_pixels(filename, batch_size=10000, shuffle=True):
    # Process batch
    pass  # Automatic memory cleanup between batches
```

### Rendering Speed Optimization
- **Vectorized Operations**: Use NumPy/TensorFlow vectorization
- **Memory Mapping**: Efficient HDF5 access for large datasets
- **Batch Processing**: Balance memory usage vs. computation efficiency

## Integration Points

### With Model Package
- Provides data loading functions for training
- Supplies sampling algorithms for hierarchical NeRF
- Handles volume rendering computations

### With Composer Package
- Receives ray data from camera/scene setup
- Processes synthetic dataset frames
- Handles coordinate transformations

### With Core Package
- Used by `TrainingManager` for data loading
- Integrated with model training pipelines
- Provides validation utilities

## Data Validation

### HDF5 File Validation
```python
def validate_h5_file(filename):
    with h5py.File(filename, 'r') as f:
        rays = f['rays']
        print(f"Shape: {rays.shape}")  # Expected: [N, 11]
        print(f"Dtype: {rays.dtype}")  # Expected: float32

        # Check for invalid values
        data = rays[:]
        nan_count = np.isnan(data).sum()
        inf_count = np.isinf(data).sum()

        return nan_count == 0 and inf_count == 0
```

### Data Quality Metrics
- **Ray Direction Normalization**: Ensure unit vectors
- **Coordinate Bounds**: Validate pixel coordinates within image bounds
- **Color Range**: Ensure RGB values in [0, 1] range
- **Numerical Stability**: Check for NaN/Inf values

## Usage Examples

### Basic Data Loading
```python
from tools import load_pixels

for rays, colors, coords in load_pixels("lego.h5", batch_size=1000):
    # rays: [1000, 2, 3] - origins and directions
    # colors: [1000, 3] - ground truth RGB
    # coords: [1000, 2] - pixel coordinates
    pass
```

### Importance Sampling
```python
from tools import sample_pdf

# After coarse network forward pass
bins = 0.5 * (z_vals[1:] + z_vals[:-1])  # Midpoints
fine_samples = sample_pdf(bins, coarse_weights, N_fine=128)
```

### Complete Rendering
```python
from tools import generate_image_from_boilerplate

img = generate_image_from_boilerplate(
    coarse_model, fine_model,
    filename="camera_view.h5",
    batch_size=10000
)
```

## Error Handling

### Common Issues
1. **Memory Errors**: Reduce batch size
2. **File Access**: Validate HDF5 file existence and format
3. **Invalid Sampling**: Check weight normalization
4. **Coordinate Errors**: Validate pixel bounds

### Debugging Tools
- Enable detailed logging for data loading operations
- Validate intermediate results in sampling pipeline
- Check ray generation and coordinate transformations
- Monitor memory usage during batch processing