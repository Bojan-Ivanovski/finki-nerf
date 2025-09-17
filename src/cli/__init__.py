"""
Command Line Interface package for FINKI-NeRF.

This package contains all CLI-related functionality:
- Command-line argument parsing and management
- Individual command handlers for each operation
- CLI workflow coordination

The CLI is designed to be modular with separate handlers for each command type:
- train: Model training operations
- predict: Single image generation
- video: Video sequence generation
- data-length: Dataset inspection

Main entry points:
- CLIManager: Main CLI coordinator class
- Individual command handlers in the commands/ subpackage
"""

from .manager import CLIManager

__all__ = ['CLIManager']