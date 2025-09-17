"""
Model management and initialization utilities.

This module provides the ModelManager class which handles:
- NeRF model creation with standard configurations
- Model weight loading and validation
- Model initialization and compilation
"""

import os
import sys
from logs import logger
import numpy as np
from model import NeRFModel, NeRFModelOptions


class ModelManager:
    """Manages NeRF model creation, loading, and initialization."""

    @staticmethod
    def create_model(eager: bool = False) -> NeRFModel:
        """Create a NeRF model with standard configuration.

        Args:
            eager: Whether to enable TensorFlow eager execution

        Returns:
            Initialized and compiled NeRF model
        """
        options = NeRFModelOptions()
        options.set_neurons_per_layer([256, 128])
        options.set_eager(eager)
        options.set_hidden_layers(8, 4)
        model = NeRFModel(options)
        dummy_input = np.zeros((1, 1, 1, 2, 3), dtype=np.float32)
        _ = model(dummy_input)
        model.compile(optimizer="adam", loss="mse")
        return model

    @staticmethod
    def load_model_if_exists(model_path: str, model: NeRFModel) -> bool:
        """Load model weights if the file exists, with user confirmation.

        Args:
            model_path: Path to the model weights file
            model: Model instance to load weights into

        Returns:
            True if weights were loaded, False otherwise

        Note:
            Will prompt user for confirmation if model file exists.
            Exits the program if user chooses not to load existing model.
        """
        if os.path.exists(model_path):
            response = input(f"Model `{model_path}` already exists, load and continue training? (Y/N): ").lower()
            if response == "y":
                model.load_weights(model_path)
                return True
            else:
                print("Exiting...")
                sys.exit(0)
        return False