"""
Training management and coordination utilities.

This module provides training-related classes:
- TrainingManager: Coordinates the training process
- DualModelCheckpoint: Custom checkpoint callback for coarse/fine models
"""

import os
import numpy as np
from logs import logger
from keras.callbacks import ModelCheckpoint
from model import NeRFModel, NeRFTrainer
from tools import load_pixels
import tensorflow as tf
from .models import ModelManager


class DualModelCheckpoint(ModelCheckpoint):
    """Custom checkpoint callback that saves both coarse and fine model weights."""

    def __init__(self, coarse_path: str, fine_path: str, **kwargs):
        """Initialize dual model checkpoint.

        Args:
            coarse_path: Path to save coarse model weights
            fine_path: Path to save fine model weights
            **kwargs: Additional arguments passed to parent ModelCheckpoint
        """
        super().__init__(filepath=fine_path, save_weights_only=True, **kwargs)
        self.coarse_path = coarse_path

    def on_epoch_end(self, epoch, logs=None):
        """Save both coarse and fine model weights at the end of each epoch.

        Args:
            epoch: Current epoch number
            logs: Training logs dictionary
        """
        self.model.coarse_model.save_weights(self.coarse_path)
        self.model.fine_model.save_weights(self.filepath)


class TrainingManager:
    """Manages the complete NeRF training process."""

    @staticmethod
    def train_model(model_name: str, data: str, batch_size: int, sample_size: int,
                   epochs: int, eager: bool) -> NeRFModel:
        """Train a NeRF model on the provided data.

        Args:
            model_name: Name for saving model checkpoints
            data: Path to HDF5 training data file
            batch_size: Number of training steps per epoch
            sample_size: Number of rays processed per training step
            epochs: Number of training epochs
            eager: Whether to enable TensorFlow eager execution

        Returns:
            Trained fine model

        Raises:
            FileNotFoundError: If data file doesn't exist
            ValueError: If parameters are invalid
        """
    

        # Validate inputs
        candidates = [
            data,
            f"{data}.h5",
            f"./outputs/{data}",
            f"./outputs/{data}.h5"
        ]

        for candidate in candidates:
            if os.path.exists(candidate):
                data = candidate
                break
        else:
            raise FileNotFoundError(f"Data file not found: {data}")

        if batch_size <= 0 or sample_size <= 0 or epochs <= 0:
            raise ValueError("Batch size, sample size, and epochs must be positive integers")

        logger.info(f"Starting training with: model={model_name}, data={data}, "
                   f"batch_size={batch_size}, sample_size={sample_size}, epochs={epochs}")

        # Create models
        coarse_model = ModelManager.create_model(eager)
        fine_model = ModelManager.create_model(eager)

        # Load existing weights if available
        ModelManager.load_model_if_exists(f"./outputs/coarse_{model_name}", coarse_model)
        ModelManager.load_model_if_exists(f"./outputs/fine_{model_name}", fine_model)

        # Set up checkpointing
        checkpoint = DualModelCheckpoint(
            coarse_path=f"./outputs/coarse_{model_name}",
            fine_path=f"./outputs/fine_{model_name}"
        )

        # Create data generator
        def pixel_data_generator():
            """Generator function for training data."""
            while True:
                for rays, colors, _ in load_pixels(data, batch_size=sample_size, shuffle=True):
                    yield rays, colors

        # Set up trainer
        trainer = NeRFTrainer(coarse_model, fine_model)
        trainer.compile(optimizer=tf.keras.optimizers.Adam(5e-4, clipnorm=1.0))

        # Train the model
        trainer.fit(
            pixel_data_generator(),
            steps_per_epoch=batch_size,
            epochs=epochs,
            callbacks=[checkpoint]
        )

        logger.info("Training completed successfully")
        return fine_model