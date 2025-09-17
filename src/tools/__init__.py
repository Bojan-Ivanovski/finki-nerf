from typing import List
import numpy as np
import os
import h5py
from .ray import Ray
import matplotlib.pyplot as plt
from keras import Model
from .ray import Ray
from .pixel import FramePixel
import tensorflow as tf
from skimage import io 

def get_dataset_length(filename="pixel_rays.h5") -> int:
    with h5py.File(filename, "r") as f:
        length = f["rays"].shape
    return length

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

def load_pixels(filename="pixel_rays.h5", batch_size=1024, shuffle=True):
    with h5py.File(filename, "r") as f:
        dset = f["rays"]
        total = dset.shape[0]

        indices = np.arange(total)
        if shuffle:
            np.random.shuffle(indices)

        for i in range(0, total, batch_size):
            batch_indices = indices[i : i + batch_size]
            batch = dset[np.sort(batch_indices)]

            origins = batch[:, 0:3]      
            directions = batch[:, 3:6]   
            colors = batch[:, 6:9]       
            coords = batch[:, 9:11]      

            rays = np.stack([origins, directions], axis=1)

            yield rays.astype(np.float32), colors.astype(np.float32), coords.astype(int)


def sample_pdf(bins, weights, N_importance, det=False):
    weights = tf.reshape(weights, [tf.shape(weights)[0], -1])  
    weights += 1e-5

    pdf = weights / tf.reduce_sum(weights, axis=-1, keepdims=True)  
    cdf = tf.cumsum(pdf, axis=-1)                                
    cdf = tf.concat([tf.zeros_like(cdf[:, :1]), cdf], axis=-1)      

    if det:
        u = tf.linspace(0.0, 1.0, N_importance)
        u = tf.expand_dims(u, 0)  
        u = tf.tile(u, [tf.shape(cdf)[0], 1])  
    else:
        u = tf.random.uniform([tf.shape(cdf)[0], N_importance])

    inds = tf.searchsorted(cdf, u, side='right')
    below = tf.maximum(0, inds - 1)
    above = tf.minimum(tf.shape(cdf)[1] - 1, inds)

    inds_g = tf.stack([below, above], axis=-1) 

    cdf_g = tf.gather(cdf, inds_g, batch_dims=1)   
    bins_g = tf.gather(bins, inds_g, batch_dims=1) 

    denom = cdf_g[..., 1] - cdf_g[..., 0]
    denom = tf.where(denom < 1e-5, tf.ones_like(denom), denom)
    t = (u - cdf_g[..., 0]) / denom

    samples = bins_g[..., 0] + t * (bins_g[..., 1] - bins_g[..., 0]) 
    return samples


def generate_image_from_boilerplate(coarse_model, fine_model, filename="blank_image.h5",
                                      batch_size=10000, N_coarse=64, N_fine=128, near=2.0, far=6.0):
    with h5py.File(filename, "r") as f:
        dset = f["rays"]
        width = int(np.max(dset[:, 9]) + 1)
        height = int(np.max(dset[:, 10]) + 1)
        total_rays = dset.shape[0]

    img = np.zeros((height, width, 3), dtype=np.float32)
    processed = 0

    for rays, _, coords in load_pixels(filename, batch_size=batch_size, shuffle=False):
        ray_origins, ray_dirs = np.split(rays, 2, axis=1) 
        ray_origins, ray_dirs = np.split(rays, 2, axis=1) 
        ray_origins = np.squeeze(ray_origins, axis=1)      
        ray_dirs    = np.squeeze(ray_dirs, axis=1)   
        t_vals = np.linspace(0.0, 1.0, N_coarse, dtype=np.float32)
        z_vals_coarse = near * (1.0 - t_vals) + far * t_vals  
        z_vals_coarse = np.broadcast_to(z_vals_coarse[None, :], [ray_origins.shape[0], N_coarse])

        pts_coarse = ray_origins[:, None, :] + ray_dirs[:, None, :] * z_vals_coarse[..., None]

        dirs_coarse = np.repeat(ray_dirs[:, None, :], N_coarse, axis=1)

        coarse_input = np.stack([pts_coarse, dirs_coarse], axis=2) 

        coarse_rgb, coarse_weights, z_vals_coarse_out = coarse_model.predict(coarse_input, verbose=0)
        bins = 0.5 * (z_vals_coarse_out[:, 1:] + z_vals_coarse_out[:, :-1])  
        z_vals_fine = sample_pdf(bins, coarse_weights[:, 1:-1, 0], N_fine, det=False).numpy()  
        z_vals_all = np.sort(np.concatenate([z_vals_coarse_out, z_vals_fine], axis=-1), axis=-1)

        pts_fine = ray_origins[:, None, :] + ray_dirs[:, None, :] * z_vals_all[..., None]

        dirs_fine = np.repeat(ray_dirs[:, None, :], z_vals_all.shape[1], axis=1)
        fine_input = np.stack([pts_fine, dirs_fine], axis=-2)

        fine_rgb, _, _ = fine_model.predict(fine_input, verbose=0)
        for (x, y), color in zip(coords, fine_rgb):
            if 0 <= x < width and 0 <= y < height:
                img[x, y] = color

        processed += len(coords)
        print(f"Rendered: {processed}/{total_rays}")

    if not os.path.exists("outputs"):
        os.mkdir("outputs")
        
    iteration = len([f for f in os.listdir("outputs") if f.endswith(".png")])
    
    img_uint8 = (np.clip(img, 0, 1) * 255).astype(np.uint8)
    io.imsave(f"outputs/prediction{iteration}.png", img_uint8)
    print(f"Saved rendered image to outputs/prediction{iteration}.png")
    return img
