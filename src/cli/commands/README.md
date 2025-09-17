# Command Handlers

This directory contains individual command handler implementations for the FINKI-NeRF CLI.

## Files

### `base.py`
Abstract base class (`CommandHandler`) that defines the interface all command handlers must implement:
- `execute(args)` - Main execution method that processes parsed arguments

### `train.py` - Training Operations
Handles all training-related operations:
- **List Objects**: `--list-objects` - Show available objects in dataset
- **Generate Data**: `--generate-data-for-object` - Create HDF5 training data
- **Train Model**: `--data` - Train NeRF models on preprocessed data

**Key Features:**
- Validates training parameters
- Integrates with `DataManager` and `TrainingManager`
- Comprehensive logging and progress reporting

### `predict.py` - Single Image Prediction
Handles single image generation from trained models:
- Loads trained coarse and fine models
- Interactive camera position selection
- Generates single novel view images

**Key Features:**
- Model validation and loading
- Camera position management
- Boilerplate generation and cleanup

### `video.py` - Video Generation
Handles video sequence generation:
- Multiple camera path types (orbit, spiral, linear, keyframe)
- Batch frame rendering
- Automatic video encoding with ffmpeg
- Frame organization and cleanup

**Key Features:**
- Advanced camera path generation
- Progress tracking for long operations
- Optional video post-processing
- Comprehensive error handling

### `data.py` - Data Inspection
Handles dataset inspection operations:
- Dataset size and structure reporting
- File validation and integrity checks

**Key Features:**
- Data file validation
- Comprehensive error reporting

## Command Handler Pattern

All command handlers follow a consistent pattern:

```python
class CommandHandler(CommandHandler):
    def execute(self, args):
        # 1. Validate arguments
        # 2. Execute business logic
        # 3. Handle errors gracefully
        # 4. Provide user feedback
```

## Error Handling

Each command handler is responsible for:
- Validating input arguments
- Catching and handling exceptions
- Providing meaningful error messages
- Logging detailed debugging information

## Integration with Core Logic

Command handlers serve as the interface between CLI arguments and core business logic:
- Import classes from `core/` package for business operations
- Handle CLI-specific concerns (user interaction, progress reporting)
- Delegate complex operations to appropriate manager classes

## Testing

Each command handler can be tested independently by:
- Creating mock argument objects
- Calling the `execute()` method directly
- Verifying expected behavior and outputs