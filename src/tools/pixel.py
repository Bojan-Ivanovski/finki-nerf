from .ray import Ray
from logs import logger
import numpy as np


class FramePixel:
    def __init__(self, x, y, ground_truth):
        self.x = x
        self.y = y
        self.ground_truth = ground_truth
        self.ray = Ray((0, 0, 0))

    def calculate_pixel_direction(self, camera_fov, image_w, image_h):
        """
        Calculate the pixel direction into world space based on the pixel position and camera orientation.
        """
        logger.debug(
            "Calculating pixel direction with camera_fov: %s and image dimensions: (%d, %d)",
            camera_fov,
            image_w,
            image_h,
        )

        center_of_image_x = image_w / 2
        center_of_image_y = image_h / 2

        logger.debug(
            "Center of image: center_of_image_x = %f, center_of_image_y = %f",
            center_of_image_x,
            center_of_image_y,
        )

        focal_x = image_w / (2 * np.tan(camera_fov / 2))
        focal_y = focal_x / image_w * image_h

        logger.debug("Focal lengths: focal_x = %f, focal_y = %f", focal_x, focal_y)

        world_x = (self.x - center_of_image_x) / focal_x
        world_y = (self.y - center_of_image_y) / focal_y
        world_z = 1

        norm = np.sqrt(world_x**2 + world_y**2 + world_z**2)
        world_x /= norm
        world_y /= norm
        world_z /= norm

        return (world_x, world_y, world_z)

    def __repr__(self):
        return (
            f"FramePixel at ({self.x}, {self.y}) with ground truth: {self.ground_truth}"
        )

    def assign_ray(self, ray: Ray):
        self.ray = ray
