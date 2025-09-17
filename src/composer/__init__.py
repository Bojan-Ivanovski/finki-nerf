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

def interpolate_camera_poses(pose1, pose2, t):
    """Interpolate between two camera poses using spherical linear interpolation.

    Args:
        pose1: First camera pose (4x4 transformation matrix)
        pose2: Second camera pose (4x4 transformation matrix)
        t: Interpolation parameter [0, 1]

    Returns:
        Interpolated pose (4x4 transformation matrix)
    """
    # Extract positions
    pos1 = pose1[:3, 3]
    pos2 = pose2[:3, 3]

    # Linear interpolation for position
    interp_pos = (1 - t) * pos1 + t * pos2

    # SLERP for rotation (simplified - using linear interpolation on rotation matrices)
    rot1 = pose1[:3, :3]
    rot2 = pose2[:3, :3]

    # Simple linear interpolation for rotation (could use proper SLERP for better results)
    interp_rot = (1 - t) * rot1 + t * rot2

    # Orthogonalize the interpolated rotation matrix
    u, _, vt = np.linalg.svd(interp_rot)
    interp_rot = np.dot(u, vt)

    # Construct interpolated pose
    interp_pose = np.eye(4)
    interp_pose[:3, :3] = interp_rot
    interp_pose[:3, 3] = interp_pos

    return interp_pose

def generate_smooth_camera_path(path_type="orbit", num_frames=30, radius=5.0, height=1.5,
                               start_angle=0, end_angle=2*np.pi, camera_fov=np.radians(60)):
    """Generate smooth camera path for video rendering.

    Args:
        path_type: Type of camera path ("orbit", "spiral", "linear", "custom")
        num_frames: Number of frames to generate
        radius: Radius of orbit/spiral
        height: Camera height
        start_angle: Starting angle for orbit
        end_angle: Ending angle for orbit
        camera_fov: Camera field of view

    Returns:
        List of SyntheticFrame objects representing camera path
    """
    frames = []

    if path_type == "orbit":
        # Circular orbit around the origin
        for i in range(num_frames):
            t = i / (num_frames - 1) if num_frames > 1 else 0
            angle = start_angle + t * (end_angle - start_angle)

            cam_x = radius * np.cos(angle)
            cam_y = height
            cam_z = radius * np.sin(angle)
            origin = np.array([cam_x, cam_y, cam_z])

            # Look at center
            forward = -origin / np.linalg.norm(origin)
            up = np.array([0, 1, 0])
            right = np.cross(up, forward)
            right /= np.linalg.norm(right)
            true_up = np.cross(forward, right)

            rotation_matrix = np.stack([right, true_up, forward], axis=1)
            transform_matrix = np.column_stack([rotation_matrix, origin])
            transform_matrix = np.vstack([transform_matrix, [0, 0, 0, 1]])

            frame = SyntheticFrame(
                frame_path=f"video_frame_{i:04d}",
                transform_matrix=transform_matrix,
                camera_fov=camera_fov
            )
            frames.append(frame)

    elif path_type == "spiral":
        # Spiral path (changing height and radius)
        for i in range(num_frames):
            t = i / (num_frames - 1) if num_frames > 1 else 0
            angle = start_angle + t * (end_angle - start_angle) * 2  # More rotations for spiral

            # Vary radius and height for spiral effect
            current_radius = radius * (1 - 0.3 * t)  # Spiral inward
            current_height = height + 2 * t  # Rise up

            cam_x = current_radius * np.cos(angle)
            cam_y = current_height
            cam_z = current_radius * np.sin(angle)
            origin = np.array([cam_x, cam_y, cam_z])

            # Look at center
            center = np.array([0, height, 0])  # Look at original height level
            forward = center - origin
            forward /= np.linalg.norm(forward)

            up = np.array([0, 1, 0])
            right = np.cross(up, forward)
            right /= np.linalg.norm(right)
            true_up = np.cross(forward, right)

            rotation_matrix = np.stack([right, true_up, forward], axis=1)
            transform_matrix = np.column_stack([rotation_matrix, origin])
            transform_matrix = np.vstack([transform_matrix, [0, 0, 0, 1]])

            frame = SyntheticFrame(
                frame_path=f"video_frame_{i:04d}",
                transform_matrix=transform_matrix,
                camera_fov=camera_fov
            )
            frames.append(frame)

    elif path_type == "linear":
        # Linear path between two points
        start_pos = np.array([radius, height, 0])
        end_pos = np.array([-radius, height, 0])

        for i in range(num_frames):
            t = i / (num_frames - 1) if num_frames > 1 else 0
            origin = (1 - t) * start_pos + t * end_pos

            # Look at center
            forward = np.array([0, 0, 0]) - origin
            forward /= np.linalg.norm(forward)

            up = np.array([0, 1, 0])
            right = np.cross(up, forward)
            right /= np.linalg.norm(right)
            true_up = np.cross(forward, right)

            rotation_matrix = np.stack([right, true_up, forward], axis=1)
            transform_matrix = np.column_stack([rotation_matrix, origin])
            transform_matrix = np.vstack([transform_matrix, [0, 0, 0, 1]])

            frame = SyntheticFrame(
                frame_path=f"video_frame_{i:04d}",
                transform_matrix=transform_matrix,
                camera_fov=camera_fov
            )
            frames.append(frame)

    return frames

def generate_interpolated_path(keyframes, num_frames):
    """Generate smooth interpolated path between keyframes.

    Args:
        keyframes: List of SyntheticFrame objects as keyframes
        num_frames: Total number of frames to generate

    Returns:
        List of interpolated SyntheticFrame objects
    """
    if len(keyframes) < 2:
        raise ValueError("Need at least 2 keyframes for interpolation")

    frames = []
    segment_length = (num_frames - 1) / (len(keyframes) - 1)

    for i in range(num_frames):
        # Find which segment we're in
        segment_idx = int(i / segment_length)
        if segment_idx >= len(keyframes) - 1:
            segment_idx = len(keyframes) - 2

        # Local interpolation parameter within segment
        local_t = (i - segment_idx * segment_length) / segment_length
        local_t = np.clip(local_t, 0, 1)

        # Interpolate between keyframes
        pose1 = keyframes[segment_idx].transform_matrix
        pose2 = keyframes[segment_idx + 1].transform_matrix

        interp_pose = interpolate_camera_poses(pose1, pose2, local_t)

        frame = SyntheticFrame(
            frame_path=f"video_frame_{i:04d}",
            transform_matrix=interp_pose,
            camera_fov=keyframes[0].camera_fov
        )
        frames.append(frame)

    return frames