from typing import List
import numpy as np
import os
import h5py
from .ray import Ray
import matplotlib.pyplot as plt
from keras import Model
from .ray import Ray
from .pixel import FramePixel

def save_pixels(pixels: List[FramePixel], filename="pixel_rays.h5"):
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


def load_pixels(filename="pixel_rays.h5", batch_size=1024, shuffle=True, remove_bg = False):
    with h5py.File(filename, "r") as f:
        dset = f["rays"]
        total = dset.shape[0]  # pyright: ignore

        indices = np.arange(total)
        if shuffle:
            np.random.shuffle(indices)

        for i in range(0, total, batch_size):
            batch_indices = indices[i : i + batch_size]
            batch = dset[np.sort(batch_indices)]  # pyright: ignore

            if remove_bg:
                mask = np.any(batch[:, 6:9] != 0, axis=1)  # pyright: ignore
                batch = batch[mask]  # pyright: ignore

                if batch.shape[0] == 0:  # pyright: ignore
                    continue

            rays = []
            for origin, direction in zip(batch[:, 0:3], batch[:, 3:6]):  # pyright: ignore
                ray = Ray(origin)
                ray.direction = direction

                points = np.array(list(ray.list_random_points()))
                directions = np.repeat(
                    direction[np.newaxis, :], points.shape[0], axis=0
                )

                stacked = np.stack([points, directions], axis=1)
                rays.append(stacked)

            rays = np.stack(rays, axis=0)

            yield {
                "rays": rays,
                "colors": batch[:, 6:9] / 255,  # pyright: ignore
                "coords": batch[:, 9:11],       # pyright: ignore
            }


def generate_image_from_boilerplate(model : Model, filename="blank_image.h5", batch_size=10000):
    print("Generating image from rays...")

    with h5py.File(filename, "r") as f:
        dset = f["rays"]
        width = int(np.max(dset[:, 9]) + 1)
        height = int(np.max(dset[:, 10]) + 1)
        total_rays = dset.shape[0]

    img = np.zeros((height, width, 3), dtype=np.float32)
    processed = 0


    for batch in load_pixels(filename, batch_size=batch_size, shuffle=False):
        rays = batch["rays"]        # (B, S, 2, 3)
        coords = batch["coords"].astype(int)
        pred = model.predict(rays, verbose=0) 
        
        # Handle hierarchical model output (returns dict) vs single model output
        if isinstance(pred, dict):
            # Use fine network output for final image
            colors = pred['fine_rgb']
        else:
            colors = pred

        for (x, y), color in zip(coords, colors):
            if 0 <= x < width and 0 <= y < height:
                img[y, x] = color  # keep values in [0,1]

        processed += len(coords)
        print(f"Predicted: {processed}/{total_rays}")

    img = np.clip(img, 0, 1)  # ensure no values out of range

    plt.imshow(img)
    plt.axis("off")
    plt.show()
    return img


def points_to_ndc(points, W, H, f, near):
    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2]

    x_ndc = (2 * f / W) * (x / -z)
    y_ndc = (2 * f / H) * (y / -z)
    z_ndc = 1 + 2 * near / (z - near)
    
    return np.stack([x_ndc, y_ndc, z_ndc], axis=-1)