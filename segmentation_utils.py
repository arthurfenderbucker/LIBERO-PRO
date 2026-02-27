"""
Utility functions for extracting and processing segmentation data from LIBERO environments.

Usage example:
    from segmentation_utils import SegmentationExtractor
    
    # Create extractor
    extractor = SegmentationExtractor(
        output_dir="data/segmentations",
        save_arrays=True,
        save_videos=True
    )
    
    # Initialize for an episode
    extractor.initialize_episode(env, task_id, episode_id)
    
    # In action loop:
    extractor.process_observation(obs, timestep=t)
    
    # At episode end:
    extractor.save_episode()
"""

import json
import logging
import pathlib
from typing import Dict, List, Optional, Tuple

import imageio
import numpy as np


class SegmentationExtractor:
    """Helper class to extract and save segmentation data from LIBERO environments."""
    
    def __init__(
        self,
        output_dir: str = "data/segmentations",
        camera_names: List[str] = None,
        save_arrays: bool = True,
        save_videos: bool = True,
        save_mappings: bool = True,
        video_fps: int = 10,
    ):
        """
        Initialize the segmentation extractor.
        
        Args:
            output_dir: Base directory for saving segmentation data
            camera_names: List of camera names to extract segmentation from
            save_arrays: Whether to save raw segmentation arrays as .npy files
            save_videos: Whether to save visualized segmentation as video files
            save_mappings: Whether to save object ID mappings as JSON
            video_fps: Frame rate for segmentation videos
        """
        self.output_dir = pathlib.Path(output_dir)
        self.camera_names = camera_names or ["agentview", "robot0_eye_in_hand"]
        self.save_arrays = save_arrays
        self.save_videos = save_videos
        self.save_mappings = save_mappings
        self.video_fps = video_fps
        
        # Create output directories
        self.output_dir.mkdir(parents=True, exist_ok=True)
        if self.save_arrays:
            (self.output_dir / "arrays").mkdir(exist_ok=True)
        if self.save_videos:
            (self.output_dir / "videos").mkdir(exist_ok=True)
        if self.save_mappings:
            (self.output_dir / "mappings").mkdir(exist_ok=True)
        
        # Episode-specific state
        self.current_task_id = None
        self.current_episode_id = None
        self.env = None
        self.segmentation_mapping = None
        self.instance_to_id = None
        self.robot_seg_id = None
        self.video_buffers = {}  # Camera name -> list of frames
        
    def initialize_episode(self, env, task_id: int, episode_id: int):
        """
        Initialize extraction for a new episode.
        
        Args:
            env: LIBERO SegmentationRenderEnv instance
            task_id: Task ID
            episode_id: Episode ID
        """
        self.current_task_id = task_id
        self.current_episode_id = episode_id
        self.env = env
        
        # Extract segmentation mapping from environment
        self.segmentation_mapping = env.segmentation_id_mapping.copy()
        self.instance_to_id = env.instance_to_id.copy()
        self.robot_seg_id = env.segmentation_robot_id
        
        # Reset video buffers
        self.video_buffers = {cam: [] for cam in self.camera_names}
        
        # Save mapping once per episode
        if self.save_mappings:
            self._save_mapping()
        
        logging.info(f"Initialized segmentation extraction for task {task_id}, episode {episode_id}")
        logging.info(f"  Objects in scene: {list(self.segmentation_mapping.values())}")
    
    def process_observation(
        self, 
        obs: Dict, 
        timestep: int,
        save_individual_frames: bool = False
    ) -> Dict[str, np.ndarray]:
        """
        Process observation and extract segmentation data.
        
        Args:
            obs: Observation dictionary from environment
            timestep: Current timestep
            save_individual_frames: Whether to save each frame as individual array file
        
        Returns:
            Dictionary mapping camera names to segmentation arrays
        """
        segmentation_data = {}
        
        for cam_name in self.camera_names:
            seg_key = f"{cam_name}_image_segmentation"
            
            if seg_key not in obs:
                logging.warning(f"Segmentation key '{seg_key}' not found in observation")
                continue
            
            # Extract and rotate segmentation (match RGB preprocessing)
            seg_array = np.ascontiguousarray(obs[seg_key][::-1, ::-1])
            segmentation_data[cam_name] = seg_array
            
            # Save individual frame if requested
            if self.save_arrays and save_individual_frames:
                array_filename = (
                    self.output_dir / "arrays" /
                    f"seg_task{self.current_task_id:02d}_ep{self.current_episode_id:03d}_t{timestep:04d}_{cam_name}.npy"
                )
                np.save(array_filename, seg_array)
            
            # Add to video buffer
            if self.save_videos:
                seg_rgb = self.segmentation_to_rgb(seg_array)
                self.video_buffers[cam_name].append(seg_rgb)
        
        return segmentation_data
    
    def get_objects_in_view(self, seg_array: np.ndarray) -> Dict[int, Tuple[str, int]]:
        """
        Get list of objects visible in a segmentation array.
        
        Args:
            seg_array: Segmentation array
        
        Returns:
            Dictionary mapping segmentation ID to (object_name, pixel_count)
        """
        unique_seg_ids = np.unique(seg_array)
        objects_in_view = {}
        
        for seg_id in unique_seg_ids:
            if seg_id == 0:  # Background
                continue
            
            pixel_count = int(np.sum(seg_array == seg_id))
            
            if seg_id == self.robot_seg_id + 1:
                objects_in_view[int(seg_id)] = ("robot", pixel_count)
            elif seg_id - 1 in self.segmentation_mapping:
                obj_name = self.segmentation_mapping[seg_id - 1]
                objects_in_view[int(seg_id)] = (obj_name, pixel_count)
            else:
                objects_in_view[int(seg_id)] = (f"unknown_{seg_id}", pixel_count)
        
        return objects_in_view
    
    def get_object_masks(self, seg_array: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Separate segmentation into individual object masks.
        
        Args:
            seg_array: Segmentation array
        
        Returns:
            Dictionary mapping object names to binary masks
        """
        if self.env is None:
            raise RuntimeError("Environment not initialized. Call initialize_episode() first.")
        
        return self.env.get_segmentation_instances(seg_array)
    
    def segmentation_to_rgb(self, seg_array: np.ndarray, random_colors: bool = False) -> np.ndarray:
        """
        Convert segmentation to colorized RGB image for visualization.
        
        Args:
            seg_array: Segmentation array
            random_colors: Whether to use random colors (default: deterministic rainbow)
        
        Returns:
            RGB image (H, W, 3) with uint8 values
        """
        if self.env is None:
            raise RuntimeError("Environment not initialized. Call initialize_episode() first.")
        
        return self.env.segmentation_to_rgb(seg_array, random_colors=random_colors)
    
    def save_episode(self, task_name: str = "", success: bool = False):
        """
        Save all accumulated data for the current episode.
        
        Args:
            task_name: Human-readable task name for filename
            success: Whether the episode was successful
        """
        if self.current_task_id is None:
            logging.warning("No episode to save")
            return
        
        suffix = "success" if success else "failure"
        task_segment = task_name.replace(" ", "_") if task_name else f"task{self.current_task_id:02d}"
        
        # Save videos
        if self.save_videos:
            for cam_name, frames in self.video_buffers.items():
                if len(frames) == 0:
                    continue
                
                video_filename = (
                    self.output_dir / "videos" /
                    f"seg_{task_segment}_ep{self.current_episode_id:03d}_{cam_name}_{suffix}.mp4"
                )
                
                try:
                    imageio.mimwrite(
                        video_filename,
                        [np.asarray(x) for x in frames],
                        fps=self.video_fps,
                    )
                    logging.info(f"Saved segmentation video: {video_filename.name}")
                except Exception as e:
                    logging.error(f"Error saving segmentation video: {e}")
        
        # Reset episode state
        self.current_task_id = None
        self.current_episode_id = None
        self.video_buffers = {}
    
    def _save_mapping(self):
        """Save object ID mapping to JSON file."""
        mapping_filename = (
            self.output_dir / "mappings" /
            f"mapping_task{self.current_task_id:02d}_ep{self.current_episode_id:03d}.json"
        )
        
        mapping_data = {
            "task_id": int(self.current_task_id),
            "episode_id": int(self.current_episode_id),
            "segmentation_mapping": {int(k): v for k, v in self.segmentation_mapping.items()},
            "instance_to_id": self.instance_to_id,
            "robot_seg_id": int(self.robot_seg_id),
            "objects": list(self.segmentation_mapping.values()),
        }
        
        with open(mapping_filename, 'w') as f:
            json.dump(mapping_data, f, indent=2)
        
        logging.info(f"Saved segmentation mapping: {mapping_filename.name}")


def analyze_segmentation_sequence(
    segmentation_arrays: List[np.ndarray],
    segmentation_mapping: Dict[int, str],
    robot_seg_id: int
) -> Dict:
    """
    Analyze a sequence of segmentation arrays to extract statistics.
    
    Args:
        segmentation_arrays: List of segmentation arrays (one per timestep)
        segmentation_mapping: Mapping from seg ID to object name
        robot_seg_id: Segmentation ID for the robot
    
    Returns:
        Dictionary with analysis results
    """
    num_frames = len(segmentation_arrays)
    object_visibility = {}  # Object name -> list of (frame_idx, pixel_count)
    
    for frame_idx, seg_array in enumerate(segmentation_arrays):
        unique_ids = np.unique(seg_array)
        
        for seg_id in unique_ids:
            if seg_id == 0:  # Background
                continue
            
            pixel_count = int(np.sum(seg_array == seg_id))
            
            if seg_id == robot_seg_id + 1:
                obj_name = "robot"
            elif seg_id - 1 in segmentation_mapping:
                obj_name = segmentation_mapping[seg_id - 1]
            else:
                obj_name = f"unknown_{seg_id}"
            
            if obj_name not in object_visibility:
                object_visibility[obj_name] = []
            object_visibility[obj_name].append((frame_idx, pixel_count))
    
    # Compute statistics
    statistics = {
        "num_frames": num_frames,
        "objects_detected": list(object_visibility.keys()),
        "object_statistics": {}
    }
    
    for obj_name, visibility_data in object_visibility.items():
        frames_visible = len(visibility_data)
        avg_pixels = np.mean([count for _, count in visibility_data])
        max_pixels = max([count for _, count in visibility_data])
        min_pixels = min([count for _, count in visibility_data])
        
        statistics["object_statistics"][obj_name] = {
            "frames_visible": frames_visible,
            "visibility_ratio": frames_visible / num_frames,
            "avg_pixel_count": float(avg_pixels),
            "max_pixel_count": int(max_pixels),
            "min_pixel_count": int(min_pixels),
        }
    
    return statistics


def visualize_object_tracking(
    seg_array: np.ndarray,
    rgb_image: np.ndarray,
    object_name: str,
    instance_to_id: Dict[str, int],
    alpha: float = 0.5
) -> np.ndarray:
    """
    Create visualization overlaying a specific object's segmentation on RGB image.
    
    Args:
        seg_array: Segmentation array
        rgb_image: RGB image (H, W, 3)
        object_name: Name of object to highlight
        instance_to_id: Mapping from object name to segmentation ID
        alpha: Transparency of overlay (0=transparent, 1=opaque)
    
    Returns:
        RGB image with segmentation overlay
    """
    if object_name not in instance_to_id:
        logging.warning(f"Object '{object_name}' not found in scene")
        return rgb_image
    
    seg_id = instance_to_id[object_name]
    
    # Create binary mask for this object
    mask = (seg_array == seg_id).astype(np.float32)
    
    # Create colored overlay (green for the object)
    overlay = rgb_image.copy()
    overlay[mask > 0] = [0, 255, 0]
    
    # Blend overlay with original image
    result = (alpha * overlay + (1 - alpha) * rgb_image).astype(np.uint8)
    
    return result
