# Core Business Logic

This package contains the core business logic classes that handle the primary operations of FINKI-NeRF. These classes are separate from CLI concerns and can be used independently for programmatic access.

## Structure

```
core/
├── __init__.py           # Package exports
├── models.py            # Model management and initialization
├── data.py              # Data processing and validation
├── training.py          # Training coordination and checkpointing
└── README.md            # This file
```

## Key Components

### ModelManager (`models.py`)
Handles NeRF model lifecycle management:
- **Model Creation**: Creates standardized NeRF models with paper-compliant configurations
- **Weight Loading**: Loads existing model weights with user confirmation
- **Model Initialization**: Handles dummy forward passes and compilation

**Key Methods:**
- `create_model(eager=False)` - Create and initialize NeRF model
- `load_model_if_exists(path, model)` - Conditionally load existing weights

### DataManager (`data.py`)
Manages data operations and validation:
- **Dataset Exploration**: Lists available objects in synthetic datasets
- **Data Generation**: Converts synthetic objects to HDF5 training data
- **Validation**: Ensures data integrity and format compliance

**Key Methods:**
- `list_objects_in_dataset(data_type)` - Show available dataset objects
- `generate_object_data(object_name, batch_size)` - Generate training data

### TrainingManager (`training.py`)
Coordinates the complete training process:
- **Training Setup**: Model initialization and checkpoint configuration
- **Training Execution**: Manages the dual network training process
- **Progress Monitoring**: Comprehensive logging and validation

**Key Classes:**
- `TrainingManager` - Main training coordinator
- `DualModelCheckpoint` - Custom checkpoint callback for coarse/fine models

**Key Methods:**
- `train_model(model_name, data, batch_size, sample_size, epochs, eager)` - Complete training workflow

## Design Principles

### 1. Separation of Concerns
- Business logic is completely separate from CLI/UI concerns
- Each manager class handles a specific domain (models, data, training)
- Clean interfaces that can be used programmatically

### 2. Error Handling
- Comprehensive input validation
- Meaningful exception messages
- Graceful degradation when possible

### 3. Logging and Monitoring
- Detailed progress reporting
- Debug-level information for troubleshooting
- Performance metrics and timing

### 4. Configuration Management
- Standardized model configurations following NeRF paper specifications
- Configurable parameters with sensible defaults
- Environment-based configuration support

## Usage Examples

### Programmatic Model Training
```python
from core import ModelManager, DataManager, TrainingManager

# Generate training data
DataManager.generate_object_data("lego", batch_size=10000)

# Train model
model = TrainingManager.train_model(
    model_name="lego_model.weights.h5",
    data="lego.h5",
    batch_size=20000,
    sample_size=2000,
    epochs=100,
    eager=False
)
```

### Model Management
```python
from core import ModelManager

# Create new model
model = ModelManager.create_model(eager=False)

# Load existing weights if available
ModelManager.load_model_if_exists("my_model.weights.h5", model)
```

## Integration Points

### With CLI Package
- CLI command handlers import and use these manager classes
- CLI handles user interaction while core handles business logic
- Clean separation allows for easy testing and alternative interfaces

### With Model Package
- `ModelManager` integrates with `model/` package for NeRF architecture
- Handles model configuration and initialization
- Manages model lifecycle and checkpointing

### With Tools Package
- Uses utility functions from `tools/` for data loading and processing
- Integrates rendering and ray generation utilities
- Handles HDF5 data operations

## Error Handling Strategy

Each manager class implements consistent error handling:
- **Input Validation**: Check parameters before processing
- **Resource Validation**: Verify file existence and accessibility
- **Runtime Error Handling**: Catch and re-raise with meaningful messages
- **Cleanup**: Ensure resources are properly cleaned up on errors

## Testing Strategy

Core classes are designed for easy unit testing:
- No external dependencies on CLI or user interaction
- Clear input/output contracts
- Mockable external dependencies (file system, models)
- Isolated functionality for focused testing