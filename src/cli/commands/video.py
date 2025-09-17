"""
Simplified video generation command handler.

This module implements the video generation command which handles:
- Automatic circular camera path generation around objects
- Interactive starting point selection (like predict command)
- Batch frame rendering with progress tracking
- Python-based video creation using OpenCV
- Automatic cleanup and organization
"""

import os
import numpy as np
try:
    import cv2
except ImportError:
    cv2 = None
from composer.synthentic_worker import SyntheticObject
from composer import create_boilerplate, generate_camera_options
from data import SYNTHETIC_TRAIN_DATA_PATH
from tools import generate_image_from_boilerplate
from logs import logger
from core import ModelManager
from .base import CommandHandler


class VideoCommandHandler(CommandHandler):
    """Handles simplified automatic video generation from trained NeRF models."""

    def execute(self, args):
        """Execute simplified video generation command.

        Args:
            args: Parsed command-line arguments containing object name

        The simplified video generation process:
        1. Load trained models
        2. Get starting point from user (like predict command)
        3. Automatically generate circular camera path around object
        4. Render frames in sequence
        5. Create video using Python/OpenCV
        6. Clean up temporary files
        """
        logger.info(f"Starting automatic video generation for object: {args.object}")

        # Check OpenCV availability
        if cv2 is None:
            logger.error("OpenCV (cv2) is required for video generation. Install with: pip install opencv-python")
            return

        # Load models
        coarse_model = ModelManager.create_model(False)
        fine_model = ModelManager.create_model(False)

        coarse_path = f"./outputs/coarse_{args.model_name}.weights.h5"
        fine_path = f"./outputs/fine_{args.model_name}.weights.h5"

        # Validate model files
        if not os.path.exists(coarse_path):
            logger.error(f"Coarse model not found: {coarse_path}")
            return
        if not os.path.exists(fine_path):
            logger.error(f"Fine model not found: {fine_path}")
            return

        # Load model weights
        coarse_model.load_weights(coarse_path)
        fine_model.load_weights(fine_path)

        # Get starting point from user (like predict command)
        starting_position = self._get_starting_position(args.object)
        if not starting_position:
            return

        # Generate circular camera path around object
        camera_frames = self._generate_circular_path(starting_position)

        # Create output directory for video frames
        video_output_dir = f"outputs/video_{args.object}"
        os.makedirs(video_output_dir, exist_ok=True)

        # Render frames
        frame_files = self._render_frames(camera_frames, coarse_model, fine_model, video_output_dir)

        # Create video using Python
        video_path = self._create_video_python(frame_files, video_output_dir, args.object)

        # Clean up frame files
        self._cleanup_frames(frame_files)

        logger.info(f"Video created successfully: {video_path}")

    def _get_starting_position(self, object_name):
        """Get starting camera position from user (like predict command).

        Args:
            object_name: Name of the object to generate video for

        Returns:
            Selected camera frame or None if selection failed
        """
        try:
            # Generate camera positions from object
            obj = SyntheticObject(SYNTHETIC_TRAIN_DATA_PATH, object_name)
            positions = list(obj.list_frames())[:5]
        except Exception as e:
            logger.warning(f"Could not load object {object_name}: {e}")
            # Fallback to generic camera options
            positions = generate_camera_options()

        # Interactive camera selection (same as predict command)
        print(f"Available starting positions: {positions}")
        pick_frame = input("Pick a starting camera position (1,2,3,4,5): ")

        try:
            selected_frame = positions[int(pick_frame) - 1]
            logger.info(f"Selected starting position: {selected_frame}")
            return selected_frame
        except (ValueError, IndexError):
            logger.error("Invalid camera position selection.")
            return None

    def _generate_circular_path(self, starting_position, num_frames=30, radius_increment=0.1):
        """Generate smooth circular camera path starting from a real dataset position.

        Args:
            starting_position: Real dataset camera frame to start from
            num_frames: Number of frames in the video (default: 30)
            radius_increment: How much to move the camera each frame

        Returns:
            List of camera frames for smooth circular motion
        """
        logger.info(f"Generating smooth circular path with {num_frames} frames from real starting position")

        # Import the SyntheticFrame class
        from composer.synthentic_worker import SyntheticFrame

        # Extract the real camera position and orientation from starting frame
        camera_origin = np.array(starting_position.camera_origin)
        camera_orientation = np.array(starting_position.camera_orientation)
        camera_fov = starting_position.camera_fov

        logger.info(f"Starting from real position: {camera_origin}")

        camera_frames = []

        for i in range(num_frames):
            # Calculate circular motion angle
            angle = (2 * np.pi * i) / num_frames

            # Create smooth circular motion around the object center
            # Assume object is roughly at origin, adjust camera position accordingly
            radius = np.linalg.norm(camera_origin[:2])  # Distance from origin in XZ plane

            # Gradually move in circle
            new_x = radius * np.cos(angle)
            new_z = radius * np.sin(angle)
            new_y = camera_origin[1]  # Keep same height

            new_camera_pos = np.array([new_x, new_y, new_z])

            # Look at origin (object center)
            target = np.array([0.0, 0.0, 0.0])
            up = np.array([0.0, 1.0, 0.0])

            # Compute camera coordinate system
            forward = target - new_camera_pos
            forward = forward / np.linalg.norm(forward)

            right = np.cross(forward, up)
            right = right / np.linalg.norm(right)

            up_actual = np.cross(right, forward)
            up_actual = up_actual / np.linalg.norm(up_actual)

            # Create transformation matrix
            transform_matrix = [
                [right[0], up_actual[0], -forward[0], new_camera_pos[0]],
                [right[1], up_actual[1], -forward[1], new_camera_pos[1]],
                [right[2], up_actual[2], -forward[2], new_camera_pos[2]],
                [0.0, 0.0, 0.0, 1.0]
            ]

            # Create frame path for this position
            frame_path = f"outputs/boilerplate_video_frame_{i:04d}"

            # Create SyntheticFrame with the new position but same FOV as real frame
            frame = SyntheticFrame(frame_path, transform_matrix, camera_fov)

            camera_frames.append(frame)
            logger.debug(f"Frame {i}: Camera at {new_camera_pos}")

        return camera_frames

    def _render_frames(self, camera_frames, coarse_model, fine_model, video_output_dir):
        """Render all frames in the camera sequence.

        Args:
            camera_frames: List of camera frame objects
            coarse_model: Loaded coarse NeRF model
            fine_model: Loaded fine NeRF model
            video_output_dir: Directory to save rendered frames

        Returns:
            List of successfully rendered frame file paths
        """
        frame_files = []

        for i, frame in enumerate(camera_frames):
            logger.info(f"Rendering frame {i+1}/{len(camera_frames)}")

            boilerplate_file = None
            try:
                # Use create_boilerplate with the REAL dataset frame (not synthetic)
                create_boilerplate(frame, 800, 800)
                boilerplate_file = f"{frame.path}.h5"

                # Render the frame using the real dataset frame data
                img = generate_image_from_boilerplate(
                    coarse_model,
                    fine_model,
                    filename=boilerplate_file,
                    batch_size=10000  # Fixed batch size
                )

                # The function automatically saves to outputs/prediction{iteration}.png
                # We need to find which file it created and move it
                import glob
                prediction_files = sorted(glob.glob("outputs/prediction*.png"))
                if prediction_files:
                    # Get the most recent prediction file
                    src_file = prediction_files[-1]
                    dst_file = f"{video_output_dir}/frame_{i:04d}.png"

                    if os.path.exists(src_file):
                        os.rename(src_file, dst_file)
                        frame_files.append(dst_file)
                        logger.debug(f"Moved {src_file} to {dst_file}")
                    else:
                        logger.warning(f"Expected rendered image not found: {src_file}")
                else:
                    logger.warning(f"No prediction files found after rendering frame {i}")

            except Exception as e:
                logger.error(f"Failed to render frame {i}: {e}")
                continue
            finally:
                # Always clean up boilerplate file, even if there was an error
                if boilerplate_file and os.path.exists(boilerplate_file):
                    try:
                        os.remove(boilerplate_file)
                        logger.debug(f"Cleaned up boilerplate: {boilerplate_file}")
                    except OSError:
                        logger.warning(f"Could not remove boilerplate file: {boilerplate_file}")

        return frame_files

    def _create_video_python(self, frame_files, output_dir, object_name, fps=24):
        """Create video from rendered frames using Python/OpenCV.

        Args:
            frame_files: List of frame file paths
            output_dir: Directory to save the video
            object_name: Name of the object (for filename)
            fps: Frames per second (default: 24)

        Returns:
            Path to created video file
        """
        if not frame_files:
            logger.error("No frames to create video from")
            return None

        video_filename = f"{output_dir}/video_{object_name}.mp4"

        try:
            # Read first frame to get dimensions
            first_frame = cv2.imread(frame_files[0])
            if first_frame is None:
                logger.error(f"Could not read first frame: {frame_files[0]}")
                return None

            height, width, _ = first_frame.shape

            # Create video writer
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video_writer = cv2.VideoWriter(video_filename, fourcc, fps, (width, height))

            logger.info(f"Creating video with {len(frame_files)} frames at {fps} fps")

            # Write frames to video
            for frame_file in frame_files:
                frame = cv2.imread(frame_file)
                if frame is not None:
                    video_writer.write(frame)
                else:
                    logger.warning(f"Could not read frame: {frame_file}")

            # Release video writer
            video_writer.release()
            logger.info(f"Video created successfully: {video_filename}")
            return video_filename

        except Exception as e:
            logger.error(f"Failed to create video using Python/OpenCV: {e}")
            return None

    def _cleanup_frames(self, frame_files):
        """Clean up frame files after video creation.

        Args:
            frame_files: List of frame file paths to clean up
        """
        for frame_file in frame_files:
            try:
                os.remove(frame_file)
            except OSError:
                logger.warning(f"Could not remove frame file: {frame_file}")
        logger.info(f"Cleaned up {len(frame_files)} frame files")