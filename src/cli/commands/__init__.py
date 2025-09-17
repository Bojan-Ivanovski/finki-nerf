"""
CLI command handlers package.

This package contains individual command handlers for each FINKI-NeRF operation:

- base: Abstract base class for all command handlers
- train: Training command implementation
- predict: Single image prediction command
- video: Video generation command
- data: Data inspection commands

Each command handler is responsible for:
- Validating command-line arguments
- Executing the appropriate business logic
- Handling errors and providing user feedback
- Logging command execution details
"""

from .base import CommandHandler
from .train import TrainCommandHandler
from .predict import PredictCommandHandler
from .video import VideoCommandHandler
from .data import DataLengthCommandHandler

__all__ = [
    'CommandHandler',
    'TrainCommandHandler',
    'PredictCommandHandler',
    'VideoCommandHandler',
    'DataLengthCommandHandler'
]