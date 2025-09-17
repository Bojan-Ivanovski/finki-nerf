"""
Data inspection command handler.

This module implements data-related commands for:
- Inspecting dataset properties and statistics
- Validating data file integrity
- Reporting dataset information
"""

import os
from logs import logger
from tools import get_dataset_length
from .base import CommandHandler


class DataLengthCommandHandler(CommandHandler):
    """Handles data inspection and validation operations."""

    def execute(self, args):
        """Execute data length inspection command.

        Args:
            args: Parsed command-line arguments (currently unused)

        This command inspects the default lego.h5 dataset file and reports:
        - Total number of rays in the dataset
        - Data structure dimensions
        - File existence and accessibility
        """
        filename = "lego.h5"

        # Check if data file exists
        if not os.path.exists(filename):
            logger.error(f"Data file not found: {filename}")
            return

        try:
            # Get dataset length information
            length = get_dataset_length(filename)
            print("Full length:", length)

        except Exception as e:
            logger.error(f"Failed to get dataset length: {e}")