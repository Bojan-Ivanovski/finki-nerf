# Logs Package

This package provides centralized logging functionality for FINKI-NeRF with configurable levels and consistent formatting across all components.

## Structure

```
logs/
├── __init__.py          # Logger configuration and setup
└── README.md            # This file
```

## Key Components

### Logger Configuration
Provides a centralized logger instance used throughout the application:
- **Consistent Formatting**: Unified log message format
- **Configurable Levels**: Support for all Python logging levels
- **Module Integration**: Easy import and use across packages

## Logging Levels

### Available Levels
- **DEBUG**: Detailed diagnostic information
- **INFO**: General operational messages
- **WARNING**: Warning messages for unusual conditions
- **ERROR**: Error messages for serious problems
- **CRITICAL**: Critical errors that may cause program termination

### Level Usage Guidelines

#### DEBUG Level
```python
logger.debug("Detailed information for debugging")
logger.debug(f"Processing batch {i}/{total_batches}")
logger.debug(f"Model weights shape: {weights.shape}")
```

#### INFO Level
```python
logger.info("Starting training process")
logger.info(f"Training completed successfully in {elapsed_time:.2f}s")
logger.info(f"Generated {num_frames} video frames")
```

#### WARNING Level
```python
logger.warning("ffmpeg not found, video creation skipped")
logger.warning(f"Using CPU training (no GPU detected)")
logger.warning("Model checkpoint not found, starting fresh")
```

#### ERROR Level
```python
logger.error(f"Failed to load dataset: {error_message}")
logger.error("Invalid camera position selection")
logger.error(f"Training failed: {exception}")
```

## Integration Across Packages

### Core Package Integration
```python
# core/training.py
from logs import logger

logger.info("Starting training with parameters: ...")
logger.debug(f"Batch {i}: loss = {loss:.6f}")
logger.error(f"Training failed: {e}")
```

### CLI Package Integration
```python
# cli/commands/train.py
from logs import logger

logger.debug("(CLI) Train Input: %s %s %s", arg1, arg2, arg3)
logger.info("Training command executed successfully")
```

### Model Package Integration
```python
# model/__init__.py
from logs import logger

def log(self, message, level=INFO):
    if self.eager_execution:
        logger.log(level, f"(EAGER) {message}")
```

## Usage Patterns

### Command Execution Logging
```python
def execute(self, args):
    logger.debug(f"(CLI) Executing {self.__class__.__name__}")
    logger.info("Starting operation...")

    try:
        # Operation logic
        logger.info("Operation completed successfully")
    except Exception as e:
        logger.error(f"Operation failed: {e}")
```

### Progress Reporting
```python
def process_data(items):
    logger.info(f"Processing {len(items)} items")

    for i, item in enumerate(items):
        if i % 100 == 0:
            logger.debug(f"Processed {i}/{len(items)} items")

        # Process item

    logger.info("Processing completed")
```

### Performance Monitoring
```python
import time

start_time = time.time()
logger.info("Starting expensive operation")

# Expensive operation
result = expensive_function()

elapsed = time.time() - start_time
logger.info(f"Operation completed in {elapsed:.2f}s")
```

## Log Level Configuration

### CLI Configuration
Set logging level via command line:
```bash
# Different logging levels
python main.py --log-level DEBUG train --data lego.h5
python main.py --log-level INFO predict  # Default level
python main.py --log-level WARNING video --num-frames 30
```

### Programmatic Configuration
```python
from logs import logger
import logging

# Set specific level
logger.setLevel(logging.DEBUG)

# Temporary level change
old_level = logger.level
logger.setLevel(logging.DEBUG)
# ... debug operations ...
logger.setLevel(old_level)
```

### Environment Configuration
```bash
# Set via environment variable (if implemented)
export FINKI_NERF_LOG_LEVEL=DEBUG
python main.py train --data lego.h5
```

## Output Examples

### Training Session Logs
```
INFO:root:Starting training with: model=lego.weights.h5, data=lego.h5, batch_size=20000
DEBUG:root:(CLI) Train Input: lego lego.h5 False 20000 2000 100
INFO:root:Training completed successfully
```

### Video Generation Logs
```
INFO:root:Starting video generation with path_type: orbit, frames: 60
INFO:root:Generated orbit camera path with 60 frames
INFO:root:Rendering frame 1/60
INFO:root:Rendering frame 2/60
...
INFO:root:Rendered 60 frames to outputs/video_finki_nerf_orbit/
INFO:root:Creating video: outputs/video_finki_nerf_orbit/video_finki_nerf_orbit.mp4
INFO:root:Video created successfully
```

### Error Handling Logs
```
ERROR:root:Coarse model not found: coarse_missing_model.weights.h5
ERROR:root:Failed to render frame 5: CUDA out of memory
WARNING:root:ffmpeg not found. Video creation skipped.
```

## Debugging Guidelines

### Debug Level Usage
Enable debug logging for troubleshooting:
```bash
python main.py --log-level DEBUG train --data lego.h5 --eager
```

### Component-Specific Debugging
```python
# In specific modules, add detailed debug info
logger.debug(f"Input tensor shape: {input.shape}")
logger.debug(f"Processing ray batch: {batch_start}-{batch_end}")
logger.debug(f"Memory usage: {get_memory_usage()}MB")
```

### Performance Debugging
```python
def debug_performance(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        logger.debug(f"Starting {func.__name__}")

        result = func(*args, **kwargs)

        elapsed = time.time() - start_time
        logger.debug(f"{func.__name__} completed in {elapsed:.3f}s")

        return result
    return wrapper
```

## Best Practices

### 1. Consistent Message Format
```python
# Good: Descriptive with context
logger.info(f"Training epoch {epoch}/{total_epochs}: loss={loss:.6f}")

# Avoid: Vague messages
logger.info("Training")
```

### 2. Appropriate Log Levels
```python
# DEBUG: Technical details for developers
logger.debug(f"Tensor operation: {operation_details}")

# INFO: User-relevant progress updates
logger.info("Training started")

# WARNING: Issues that don't stop execution
logger.warning("Using fallback method")

# ERROR: Failures that affect functionality
logger.error("Cannot load model weights")
```

### 3. Context Information
```python
# Include relevant context
logger.error(f"Failed to process frame {frame_id}: {error}")
logger.info(f"Processed {processed}/{total} items ({processed/total*100:.1f}%)")
```

### 4. Exception Logging
```python
try:
    risky_operation()
except Exception as e:
    logger.error(f"Operation failed: {e}")
    raise  # Re-raise if needed
```

## Integration with External Tools

### TensorBoard Integration (Future)
```python
# Potential extension for training visualization
def log_to_tensorboard(metrics, step):
    logger.debug(f"Logging metrics to TensorBoard: step={step}")
    # TensorBoard logging logic
```

### File Logging (Extension)
```python
# Add file handler for persistent logs
import logging

file_handler = logging.FileHandler('finki_nerf.log')
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)
```

## Troubleshooting

### Common Issues

**Logs Not Appearing:**
- Check log level setting
- Verify logger import: `from logs import logger`
- Ensure proper log level: `logger.setLevel(level)`

**Too Much Debug Output:**
- Use higher log level: `--log-level INFO`
- Filter specific components
- Use conditional debug logging

**Missing Context:**
- Include relevant variables in log messages
- Add operation identifiers
- Include timing information for long operations