from .synthentic_worker import SyntheticFrame
import numpy as np
from tools.ray import Ray
from tools.pixel import FramePixel
from tools import save_pixels

def generate_camera_options(num_options=5, radius=5.0, height=1.5, camera_fov=np.radians(60)):
    frames = []
    for i in range(num_options):
        angle = np.random.uniform(0, 2 * np.pi)

        cam_x = radius * np.cos(angle)
        cam_y = height
        cam_z = radius * np.sin(angle)
        origin = (cam_x, cam_y, cam_z)

        forward = np.array([0, 0, 0]) - np.array(origin)
        forward /= np.linalg.norm(forward)

        up = np.array([0, 1, 0])
        right = np.cross(up, forward)
        right /= np.linalg.norm(right)
        true_up = np.cross(forward, right)

        rotation_matrix = np.stack([right, true_up, forward], axis=1)  

        transform_matrix = np.column_stack([rotation_matrix, np.array(origin)])
        transform_matrix = np.vstack([transform_matrix, [0, 0, 0, 1]])  

        frame = SyntheticFrame(frame_path=f"boilerplate_{i}", transform_matrix=transform_matrix, camera_fov=camera_fov)
        frames.append(frame)
    return frames

def create_boilerplate(
    frame : SyntheticFrame,
    width: int = None,
    height: int = None,
    batch_size=10000
):
    print("Generating pixel rays boilerplate...")
    filename = frame.path+".h5"
    H, W = height, width
    pixel_gen = frame.generate_pixels(width, height)
    total_pixels = H * W
    pixels_batch = []

    for idx, pixel in enumerate(pixel_gen):
        pixels_batch.append(pixel)

        if len(pixels_batch) == batch_size or (idx + 1) == total_pixels:
            save_pixels(pixels_batch, filename)
            print(f"Saved rays: {idx + 1}/{total_pixels}")
            pixels_batch = []

    print(f"Boilerplate HDF5 file created from frame: {filename}")