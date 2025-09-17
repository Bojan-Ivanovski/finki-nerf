import os
import json

import numpy as np

from logs import logger
from skimage import io
from data import (
    SYNTHETIC_TEST_DATA_PATH,
    SYNTHETIC_TRAIN_DATA_PATH,
    SYNTHETIC_VAL_DATA_PATH,
)
from tools.pixel import FramePixel, Ray


class SyntheticDataset:
    def __init__(self, dataset_type="train"):
        dataset_types = {}
        dataset_types["train"] = SYNTHETIC_TRAIN_DATA_PATH
        dataset_types["test"] = SYNTHETIC_TEST_DATA_PATH
        dataset_types["val"] = SYNTHETIC_VAL_DATA_PATH
        logger.info("Initializing SyntheticDataset with dataset type: %s", dataset_type)
        self.path = dataset_types.get(dataset_type, "")
        logger.debug("Using synthetic data path: %s", self.path)

    def list_objects(self):
        directories = os.listdir(self.path)
        for name in directories:
            yield SyntheticObject(self.path, name)

    def __str__(self) -> str:
        out = "[ " + ", ".join([str(obj) for obj in self.list_objects()]) + " ]"
        return out


class SyntheticObject:
    def __init__(self, object_path, name):
        logger.info("Initializing SyntheticObject: %s", name)
        self.name = name
        self.path = os.path.join(object_path, name)
        logger.debug("Using synthetic data path: %s", self.path)
        with open(os.path.join(self.path, "metadata.json"), "r") as file:
            self.metadata: dict = json.load(file)
        self.camera_fov = self.metadata.get("camera_angle_x", None)
        self.frames = self.metadata.get("frames", [])

    def list_frames(self):
        for frame in self.frames:
            frame_path = os.path.join(self.path, frame["file_path"] + ".png")
            transformation_matrix = frame["transform_matrix"]
            yield SyntheticFrame(frame_path, transformation_matrix, self.camera_fov)

    def __repr__(self) -> str:
        return f"{self.name}"

    def __str__(self) -> str:
        return f"{self.name}"


class SyntheticFrame:
    def __init__(self, frame_path, transform_matrix, camera_fov):
        logger.info("Initializing SyntheticFrame from image: %s", frame_path)
        self.path = frame_path
        self.camera_fov = camera_fov
        transform_matrix = transform_matrix[:-1]
        self.camera_origin = tuple(camera_info[-1] for camera_info in transform_matrix)
        self.camera_orientation = tuple(
            tuple(world_info[:-1]) for world_info in transform_matrix
        )
        logger.debug("Camera origin: %s", list(self.camera_origin))
        logger.debug("Camera rotation matrix: %s", list(self.camera_orientation))
        if "boilerplate" not in self.path:
            img = io.imread(self.path) / 255.0  # normalize to [0,1]
            if img.shape[-1] == 4:  # RGBA
                rgb, alpha = img[..., :3], img[..., 3:]
                img = rgb * alpha + (1.0 - alpha)
            self.image = img.astype(np.float32)

    def get_image_pixel_size(self):
        return self.image.shape[0] * self.image.shape[1]

    def list_pixels(self):
        for y in range(self.image.shape[0]):
            for x in range(self.image.shape[1]):
                ray = Ray(self.camera_origin)
                pixel = FramePixel(x, y, self.image[x, y][:3])
                pixel_direction = pixel.calculate_pixel_direction(
                    self.camera_fov, self.image.shape[1], self.image.shape[0]
                )
                ray.compute_direction(pixel_direction, self.camera_orientation)
                pixel.assign_ray(ray)
                yield pixel

    def generate_pixels(self, width, height):
        for y in range(height):
            for x in range(width):
                ray = Ray(self.camera_origin)
                pixel = FramePixel(x, y, np.array([0.0,0.0,0.0]))
                pixel_direction = pixel.calculate_pixel_direction(
                    self.camera_fov, width, height
                )
                ray.compute_direction(pixel_direction, self.camera_orientation)
                pixel.assign_ray(ray)
                yield pixel

    def __str__(self):
        return self.path
    
    def __repr__(self):
        return self.path


