"""
Core business logic package for FINKI-NeRF.

This package contains the core business logic classes that handle:
- Model management and initialization
- Data processing and validation
- Training coordination and checkpointing

These classes are separate from CLI concerns and can be used
independently for programmatic access to FINKI-NeRF functionality.
"""

from .models import ModelManager
from .data import DataManager
from .training import TrainingManager, DualModelCheckpoint

__all__ = [
    'ModelManager',
    'DataManager',
    'TrainingManager',
    'DualModelCheckpoint'
]