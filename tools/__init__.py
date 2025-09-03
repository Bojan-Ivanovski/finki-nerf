from typing import List
import numpy as np
import os
import h5py
from .ray import Ray


def save_pixels(pixels: List, filename="pixel_rays.h5"):
    mode = "a" if os.path.exists(filename) else "w"
    with h5py.File(filename, mode) as f:
        if "rays" not in f:
            maxshape = (None, 11)
            dset = f.create_dataset(
                "rays",
                shape=(0, 11),
                maxshape=maxshape,
                dtype="float32",
                chunks=True,
            )
        else:
            dset = f["rays"]

        rays = []
        for pixel in pixels:
            row = np.concatenate(
                [
                    pixel.ray.origin,
                    pixel.ray.direction,
                    pixel.ground_truth.astype(np.float32),
                    np.array([pixel.x, pixel.y], dtype=np.float32),
                ]
            )
            rays.append(row)

        rays = np.vstack(rays)

        old_size = dset.shape[0]  # pyright: ignore
        new_size = old_size + rays.shape[0]
        dset.resize((new_size, 11))  # pyright: ignore
        dset[old_size:new_size] = rays  # pyright: ignore


def load_pixels(filename="pixel_rays.h5", batch_size=1024, bg_keep_prob=0.5):
    with h5py.File(filename, "r") as f:
        dset = f["rays"]
        total = dset.shape[0]  # pyright: ignore

        for i in range(0, total, batch_size):
            batch = dset[i : i + batch_size]  # pyright: ignore

            is_foreground = np.any(batch[:, 6:9] != 0, axis=1)  # pyright: ignore
            is_background = ~is_foreground
            bg_mask = np.random.rand(batch.shape[0]) < bg_keep_prob
            keep_mask = is_foreground | (is_background & bg_mask)
            batch = batch[keep_mask]  # pyright: ignore

            if batch.shape[0] == 0:  # pyright: ignore
                continue

            rays = []
            for origin, direction in zip(batch[:, 0:3], batch[:, 3:6]):  # pyright: ignore
                ray = Ray(origin)
                ray.direction = direction

                points = np.array(list(ray.list_points()))

                directions = np.repeat(
                    direction[np.newaxis, :], points.shape[0], axis=0
                )

                stacked = np.stack([points, directions], axis=1)
                rays.append(stacked)

            rays = np.stack(rays, axis=0)

            yield {
                "rays": rays,
                "colors": batch[:, 6:9] / 255.0,  # pyright: ignore
                "coords": batch[:, 9:11],  # pyright: ignore
            }

