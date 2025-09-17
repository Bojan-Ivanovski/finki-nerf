# Data Package

This package handles data configuration and path management for FINKI-NeRF datasets.

## Structure

```
data/
├── __init__.py               # Dataset paths and configuration
└── README.md                 # This file
└── train_dataset_synthetic/  # Directory with objects from dataset

```

## Key Components

### Dataset Path Configuration
Centralizes dataset path management with environment variable support:

**Default Paths:**
- `DEFAULT_SYNTHETIC_TRAIN_DATA_PATH` - Training dataset location
- `DEFAULT_SYNTHETIC_TEST_DATA_PATH` - Test dataset location
- `DEFAULT_SYNTHETIC_VAL_DATA_PATH` - Validation dataset location

**Environment Variables:**
- `SYNTHETIC_TRAIN_DATA_PATH` - Override training path
- `SYNTHETIC_TEST_DATA_PATH` - Override test path
- `SYNTHETIC_VAL_DATA_PATH` - Override validation path

### Configuration Management
Provides flexible dataset location configuration:
```python
# Default behavior (relative paths)
SYNTHETIC_TRAIN_DATA_PATH = "./data/train_dataset_synthetic"

# Environment override
export SYNTHETIC_TRAIN_DATA_PATH="/custom/path/to/training/data"
```

## Supported Dataset Formats

### NeRF Synthetic Dataset
Standard synthetic dataset format with:
- **Images**: PNG format, typically 800×800 resolution
- **Camera Poses**: `transforms_train.json` with 4×4 transformation matrices
- **Scene Information**: Camera intrinsics, near/far planes
- **Objects**: Multiple 3D objects (lego, chair, drums, etc.)

### Directory Structure
Expected dataset organization:
```
train_dataset_synthetic/
├── lego/
│   ├── metadata.json
│   └── images/
│       ├── r_0.png
│       ├── r_1.png
│       └── ...
├── chair/
│   ├── metadata.json
│   └── images/
└── ...
```

## Dataset Objects

### Available Objects (Synthetic)
Standard NeRF synthetic dataset includes:
- **lego** - LEGO bulldozer (most commonly used)
- **chair** - Wooden chair
- **drums** - Drum kit
- **ficus** - Bonsai ficus tree
- **hotdog** - Hot dog
- **materials** - Metallic spheres with different materials
- **mic** - Microphone
- **ship** - Pirate ship

### Object Characteristics
| Object | Complexity | Training Time | Quality |
|--------|------------|---------------|---------|
| hotdog | Simple | Fast | High |
| lego | Medium | Medium | High |
| chair | Medium | Medium | High |
| drums | Complex | Slow | Medium |
| materials | Complex | Slow | Medium |
| ship | Complex | Slow | Medium |

## Configuration Examples

### Default Configuration
```python
# Uses relative paths - good for development
from data import SYNTHETIC_TRAIN_DATA_PATH
# Results in: "./data/train_dataset_synthetic"
```

### Custom Configuration
```bash
# Set custom paths via environment variables
export SYNTHETIC_TRAIN_DATA_PATH="/mnt/datasets/nerf_synthetic/train"
export SYNTHETIC_TEST_DATA_PATH="/mnt/datasets/nerf_synthetic/test"
```

### Docker Configuration
```dockerfile
# Mount external dataset volume
VOLUME ["/app/data"]
ENV SYNTHETIC_TRAIN_DATA_PATH="/app/data/train_dataset_synthetic"
```

## Integration Points

### With Core Package
- `DataManager` uses these paths to locate datasets
- Provides consistent path resolution across the application
- Supports environment-based configuration

### With Composer Package
- `SyntheticObject` and `SyntheticDataset` use these paths
- Enables flexible dataset location management
- Supports multiple dataset environments

### With CLI Package
- Command handlers use these paths for data operations
- Consistent path handling across all commands
- Environment variable support for deployment

## Path Resolution

### Resolution Order
1. **Environment Variable**: Check for custom path in environment
2. **Default Value**: Fall back to relative path in project
3. **Validation**: Ensure path exists and is accessible

### Path Validation
```python
import os
from data import SYNTHETIC_TRAIN_DATA_PATH

# Validate path exists
if not os.path.exists(SYNTHETIC_TRAIN_DATA_PATH):
    print(f"Dataset path not found: {SYNTHETIC_TRAIN_DATA_PATH}")

# Check for specific object
lego_path = os.path.join(SYNTHETIC_TRAIN_DATA_PATH, "lego")
if os.path.exists(lego_path):
    print("Lego dataset available")
```

## Dataset Setup

### Download and Setup
```bash
# Create data directory
mkdir -p data

# Download NeRF synthetic dataset
cd data
wget https://drive.google.com/uc?id=18JxhpWD-4ZmuFKLzKlAw-w5PpzZxXOcG -O nerf_synthetic.zip
unzip nerf_synthetic.zip
mv nerf_synthetic train_dataset_synthetic

# Verify structure
ls train_dataset_synthetic/
# Should show: chair drums ficus hotdog lego materials mic ship
```

### Environment Configuration
```bash
# Create .env file for custom paths
echo "SYNTHETIC_TRAIN_DATA_PATH=/custom/path/to/data" > .env

# Or set in shell environment
export SYNTHETIC_TRAIN_DATA_PATH="/mnt/ssd/nerf_data/train"
```

## Storage Requirements

### Dataset Sizes
| Dataset | Size | Objects | Images per Object |
|---------|------|---------|------------------|
| Synthetic Train | ~1.7GB | 8 | 100 |
| Synthetic Test | ~170MB | 8 | 200 |
| Synthetic Val | ~170MB | 8 | 200 |

### Storage Optimization
- **SSD Recommended**: Faster data loading during training
- **Network Storage**: Supported via path configuration
- **Compression**: Original PNG format provides good compression

## Troubleshooting

### Common Issues

**Dataset Not Found:**
```
FileNotFoundError: Dataset path does not exist
```
- Verify dataset is downloaded and extracted
- Check environment variable configuration
- Ensure correct directory structure

**Permission Errors:**
```
PermissionError: Cannot access dataset directory
```
- Check file permissions: `chmod -R 755 data/`
- Verify user has read access to dataset location

**Path Resolution Issues:**
```bash
# Debug path resolution
python -c "from data import SYNTHETIC_TRAIN_DATA_PATH; print(SYNTHETIC_TRAIN_DATA_PATH)"

# Check environment
env | grep SYNTHETIC

# Verify directory contents
ls -la $SYNTHETIC_TRAIN_DATA_PATH
```

## Best Practices

1. **Use Environment Variables**: For deployment flexibility
2. **Absolute Paths**: In production environments
3. **Path Validation**: Always check dataset existence
4. **Consistent Structure**: Maintain expected directory layout
5. **Version Control**: Don't commit large dataset files to git

## Future Extensions

### Planned Support
- Real scene datasets (LLFF format)
- Custom dataset formats
- Automatic dataset download and setup
- Multiple dataset version management
- Dataset integrity validation