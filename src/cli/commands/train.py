"""
Training command handler.

This module implements the training command which handles:
- Dataset exploration (--list-objects)
- Training data generation (--generate-data-for-object)
- Model training (--data)
"""

from logs import logger
from core import DataManager, TrainingManager
from .base import CommandHandler


class TrainCommandHandler(CommandHandler):
    """Handles all training-related CLI operations."""

    def execute(self, args):
        """Execute training command based on provided arguments.

        Args:
            args: Parsed command-line arguments containing training parameters

        The training command supports three main operations:
        1. List objects in dataset (--list-objects)
        2. Generate training data (--generate-data-for-object)
        3. Train model (--data)
        """
        logger.debug(
            "(CLI) Train Input: %s %s %s %s %s %s %s %s",
            args.model_name,
            args.list_objects,
            args.generate_data_for_object,
            args.data,
            args.data_set,
            args.eager,
            args.batch_size,
            args.epochs,
        )

        # Handle --list-objects
        if args.list_objects:
            DataManager.list_objects_in_dataset("train")
            return

        # Handle generate-data subcommand
        if args.generate_data_for_object:
            DataManager.generate_object_data(args.generate_data_for_object, args.batch_size)
            return

        # Train model
        if args.data:
            TrainingManager.train_model(
                model_name=f"{args.model_name}.weights.h5",
                data=args.data,
                batch_size=args.batch_size,
                sample_size=args.sample_size,
                epochs=args.epochs,
                eager=args.eager,
            )
            return

        logger.error("No valid training option provided. Use --help for usage information.")