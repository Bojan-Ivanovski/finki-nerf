"""
Prediction command handler.

This module implements the prediction command which handles:
- Single image generation from trained NeRF models
- Camera position selection and management
- Model loading and validation
"""

import os
from composer.synthentic_worker import SyntheticObject
from composer import generate_camera_options, create_boilerplate
from data import SYNTHETIC_TRAIN_DATA_PATH
from tools import generate_image_from_boilerplate
from logs import logger
from core import ModelManager
from .base import CommandHandler


class PredictCommandHandler(CommandHandler):
    """Handles single image prediction from trained NeRF models."""

    def execute(self, args):
        """Execute prediction command to generate a single novel view.

        Args:
            args: Parsed command-line arguments containing prediction parameters

        The prediction process:
        1. Load trained coarse and fine models
        2. Generate or select camera positions
        3. Create boilerplate ray data
        4. Render the image using the models
        5. Clean up temporary files
        """
        logger.debug("(CLI) Predict Input: %s", args.create_boilerplate_from_object)

        # Load models
        coarse_model = ModelManager.create_model(False)
        fine_model = ModelManager.create_model(False)

        coarse_path = f"./outputs/coarse_{args.model_name}.weights.h5"
        fine_path = f"./outputs/fine_{args.model_name}.weights.h5"

        # Validate model files exist
        if not os.path.exists(coarse_path):
            logger.error(f"Coarse model not found: {coarse_path}")
            return
        if not os.path.exists(fine_path):
            logger.error(f"Fine model not found: {fine_path}")
            return

        # Load model weights
        coarse_model.load_weights(coarse_path)
        fine_model.load_weights(fine_path)

        # Generate camera positions
        if args.create_boilerplate_from_object:
            obj = SyntheticObject(SYNTHETIC_TRAIN_DATA_PATH, args.create_boilerplate_from_object)
            positions = list(obj.list_frames())[:5]
        else:
            positions = generate_camera_options()

        # Interactive camera selection
        print(f"Available positions: {positions}")
        pick_frame = input("Pick a camera position (1,2,3,4,5): ")

        try:
            selected_frame = positions[int(pick_frame) - 1]

            # Generate boilerplate and render image
            create_boilerplate(selected_frame, 800, 800)
            generate_image_from_boilerplate(coarse_model, fine_model, filename=f"{selected_frame}.h5")

            # Clean up temporary boilerplate file
            os.remove(f"{selected_frame}.h5")

        except (ValueError, IndexError):
            logger.error("Invalid camera position selection.")