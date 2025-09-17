"""
Data management and processing utilities.

This module provides the DataManager class which handles:
- Dataset exploration and validation
- Training data generation from synthetic objects
- Data format validation and error handling
"""

from composer.synthentic_worker import SyntheticObject, SyntheticDataset
from data import SYNTHETIC_TRAIN_DATA_PATH
from logs import logger
from tools import save_pixels


class DataManager:
    """Manages data operations including dataset exploration and generation."""

    @staticmethod
    def list_objects_in_dataset(data_type: str):
        """List all available objects in the specified dataset.

        Args:
            data_type: Type of dataset to query (e.g., "train")

        Raises:
            RuntimeError: If dataset cannot be accessed or listed
        """
        try:
            dataset = SyntheticDataset(data_type)
            print("Objects in dataset:", dataset)
        except Exception as e:
            raise RuntimeError(f"Failed to list objects in dataset '{data_type}': {e}")

    @staticmethod
    def generate_object_data(object_to_generate: str, batch_size: int):
        """Generate HDF5 training data for a specific object.

        This method processes all frames of an object, extracts ray-pixel pairs,
        and saves them to an HDF5 file for training.

        Args:
            object_to_generate: Name of the object to generate data for
            batch_size: Number of pixels to process in each batch

        Raises:
            ValueError: If batch_size is not positive
            RuntimeError: If object data generation fails
        """
        if batch_size <= 0:
            raise ValueError("Batch size must be a positive integer")

        try:
            obj = SyntheticObject(SYNTHETIC_TRAIN_DATA_PATH, object_to_generate)
            total_frames = len(obj.frames)
            logger.info(f"Generating data for object '{object_to_generate}' with {total_frames} frames")

            for n_frame, frame in enumerate(obj.list_frames()):
                pixels = []
                for n_pixel, pixel in enumerate(frame.list_pixels()):
                    pixels.append(pixel)
                    if len(pixels) % batch_size == 0:
                        logger.info(
                            "(Frame) %d / %d (Pixel) Processed %d / %d",
                            n_frame + 1,
                            total_frames,
                            n_pixel + 1,
                            frame.get_image_pixel_size(),
                        )
                        save_pixels(pixels, filename=f"./outputs/{object_to_generate}.h5")
                        pixels.clear()

                # Process remaining pixels in the frame
                if frame.get_image_pixel_size() % batch_size != 0:
                    logger.info(
                        "(Frame) %d / %d (Pixel) Processed %d / %d",
                        n_frame + 1,
                        total_frames,
                        n_pixel + 1,
                        frame.get_image_pixel_size(),
                    )
                    save_pixels(pixels, filename=f"{object_to_generate}.h5")
                    pixels.clear()

            logger.info(f"Data generation completed for '{object_to_generate}'")
        except Exception as e:
            raise RuntimeError(f"Failed to generate object data for '{object_to_generate}': {e}")