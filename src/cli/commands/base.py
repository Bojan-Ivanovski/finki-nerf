"""
Base command handler for CLI operations.

This module defines the abstract base class that all command handlers must inherit from.
It provides the common interface for command execution and ensures consistency
across all command implementations.
"""

from abc import ABC, abstractmethod


class CommandHandler(ABC):
    """Abstract base class for all CLI command handlers.

    Each command handler must implement the execute method to handle
    command-specific logic and argument processing.
    """

    @abstractmethod
    def execute(self, args):
        """Execute the command with the provided arguments.

        Args:
            args: Parsed command-line arguments from argparse

        This method should handle all command-specific logic including:
        - Argument validation
        - Business logic execution
        - Error handling and user feedback
        - Progress reporting and logging
        """
        pass