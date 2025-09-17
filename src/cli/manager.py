"""
CLI Manager and argument parsing.

This module contains the main CLI management class that handles:
- Command-line argument parsing and validation
- Command routing and execution
- Error handling and user feedback
- Help text generation
"""

import sys
import argparse
from logs import logger
from .commands import (
    TrainCommandHandler,
    PredictCommandHandler,
    VideoCommandHandler,
    DataLengthCommandHandler
)


class CLIManager:
    """Main CLI coordinator that manages argument parsing and command execution."""

    def __init__(self):
        """Initialize CLI manager with command handlers."""
        self.command_handlers = {
            "train": TrainCommandHandler(),
            "predict": PredictCommandHandler(),
            "data-length": DataLengthCommandHandler(),
            "generate-video": VideoCommandHandler(),
        }

    def create_parser(self) -> argparse.ArgumentParser:
        """Create and configure the main argument parser.

        Returns:
            Configured ArgumentParser instance with all subcommands
        """
        parser = argparse.ArgumentParser(
            description="FINKI-NeRF Command Line Interface (CLI)",
            formatter_class=argparse.RawDescriptionHelpFormatter
        )

        # Global options
        parser.add_argument(
            "--log-level",
            default="INFO",
            choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL", "NOTSET"],
            help="Set the logging level. Default is INFO.",
        )

        parser.add_argument(
            "--model-name",
            type=str,
            default="finki_nerf",
            help="Name for saving/loading the trained model. Default is finki_nerf.",
        )

        # Create subcommands
        subparsers = parser.add_subparsers(dest="command", required=True, help="Available commands")

        self._create_train_parser(subparsers)
        self._create_predict_parser(subparsers)
        self._create_data_length_parser(subparsers)
        self._create_video_parser(subparsers)

        return parser

    def _create_train_parser(self, subparsers):
        """Create argument parser for training commands."""
        train_parser = subparsers.add_parser(
            "train",
            help="Train a NeRF model on a dataset",
            formatter_class=argparse.RawDescriptionHelpFormatter
        )

        train_parser.add_argument(
            "--list-objects",
            action="store_true",
            help="List all objects in the dataset and exit.",
        )
        train_parser.add_argument(
            "--generate-data-for-object",
            type=str,
            metavar="OBJECT_NAME",
            help="Generate an HDF5 rays file for a specific object.",
        )
        train_parser.add_argument(
            "--data",
            type=str,
            help="Path to a preprocessed HDF5 rays file.",
        )
        train_parser.add_argument(
            "--data-set",
            type=str,
            choices=["Synthetic"],
            default="Synthetic",
            help="Dataset to use for training. Default is Synthetic.",
        )
        train_parser.add_argument(
            "--eager",
            action="store_true",
            help="Enable eager execution mode for debugging.",
        )
        train_parser.add_argument(
            "--batch-size",
            type=int,
            default=10000,
            help="Batch size for training. Default is 10000.",
        )
        train_parser.add_argument(
            "--sample-size",
            type=int,
            default=1000,
            help="Sample size for training. Default is 1000.",
        )
        train_parser.add_argument(
            "--epochs",
            type=int,
            default=10,
            help="Number of epochs per batch. Default is 10.",
        )

    def _create_predict_parser(self, subparsers):
        """Create argument parser for prediction commands."""
        predict_parser = subparsers.add_parser(
            "predict",
            help="Generate predictions (rendered images) from a trained model"
        )

        predict_parser.add_argument(
            "--create-boilerplate-from-object",
            type=str,
            help="Create a boilerplate to be used for generating images from a specific object.",
        )

    def _create_data_length_parser(self, subparsers):
        """Create argument parser for data inspection commands."""
        subparsers.add_parser(
            "data-length",
            help="Show the length of the rays saved in a h5 file."
        )

    def _create_video_parser(self, subparsers):
        """Create argument parser for simplified video generation commands."""
        video_parser = subparsers.add_parser(
            "generate-video (HAS ISSUES AND NOT WORKING YET)",
            help="Generate video sequences automatically from trained NeRF models",
            formatter_class=argparse.RawDescriptionHelpFormatter
        )

        # Required object argument
        video_parser.add_argument(
            "object",
            type=str,
            help="Object name to generate video for (required)"
        )

    def execute_command(self, args):
        """Execute the specified command with error handling.

        Args:
            args: Parsed command-line arguments

        The method routes the command to the appropriate handler and provides
        comprehensive error handling with user-friendly feedback.
        """
        if args.command in self.command_handlers:
            try:
                self.command_handlers[args.command].execute(args)
            except Exception as e:
                logger.error(f"Error executing command '{args.command}': {e}")
                sys.exit(1)
        else:
            logger.error(f"Unknown command: {args.command}")
            sys.exit(1)