from typing import Tuple, Generator
import numpy as np
from logs import logger


class Ray:
    def __init__(self, origin: Tuple):
        self.origin = np.array(origin)
        self.direction = np.array((0, 0, 0))
        self.min_step = 2
        self.max_step = 6
        self.step_size = 0.0625

    def compute_direction(self, pixel_direction: Tuple, camera_orientation: Tuple):
        self.direction = np.array(camera_orientation) @ np.array(pixel_direction)
        logger.debug("Computed ray direction: %s", self.direction)

    def list_points(self) -> Generator[np.ndarray, None, None]:
        for step in np.arange(self.min_step, self.max_step, self.step_size):
            point = self.origin + step * self.direction
            yield point

    def __repr__(self):
        return f"Ray : {self.origin} + t * {self.direction}"
