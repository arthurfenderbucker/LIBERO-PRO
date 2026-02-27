#!/usr/bin/env python3
"""
Script to load LIBERO scenes from different task suites, spawn simulations,
and save the first frame for visual comparison of initial configurations and assets.
"""

import logging
import os
import pathlib
import sys
from typing import List

import imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch
import tyro

# Add LIBERO-PRO to path
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from libero.libero import benchmark
from libero.libero import get_libero_path
from libero.libero.envs import OffScreenRenderEnv


LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
LIBERO_ENV_RESOLUTION = 256


def load_custom_task_suite(suite_name: str):
    """
    Load a custom task suite directly from the filesystem.
    This is used for task suites that are not registered in the benchmark dict.
    
    Returns a dictionary with task information needed for environment creation.
    """
    # Path to LIBERO-PRO directories
    libero_pro_path = pathlib.Path(__file__).resolve().parent.parent
    bddl_dir = libero_pro_path / "libero" / "libero" / "bddl_files" / suite_name
    init_dir = libero_pro_path / "libero" / "libero" / "init_files" / suite_name
    
    if not bddl_dir.exists():
        raise FileNotFoundError(f"BDDL directory not found: {bddl_dir}")
    if not init_dir.exists():
        raise FileNotFoundError(f"Init states directory not found: {init_dir}")
    
    # Get all BDDL files
    bddl_files = sorted(bddl_dir.glob("*.bddl"))
    
    tasks = []
    for bddl_file in bddl_files:
        task_name = bddl_file.stem
        # Find corresponding init file
        init_file = init_dir / f"{task_name}.pruned_init"
        
        if not init_file.exists():
            logging.warning(f"Init file not found for {task_name}, skipping")
            continue
        
        # Extract language from filename
        language = " ".join(task_name.split("_"))
        
        tasks.append({
            "name": task_name,
            "language": language,
            "problem_folder": suite_name,
            "bddl_file": bddl_file.name,
            "init_file": init_file.name,
            "bddl_path": bddl_file,
            "init_path": init_file,
        })
    
    return {
        "name": suite_name,
        "n_tasks": len(tasks),
        "tasks": tasks,
    }


def capture_scene_images(
    task_suite_names: List[str],
    output_dir: str = "scene_comparisons",
    resolution: int = 256,
    seed: int = 7,
    num_steps_wait: int = 10,
) -> None:
    """
    Capture first frame images from LIBERO task suites for visual comparison.
    
    Args:
        task_suite_names: List of task suite names to load (e.g., ['libero_object_temp_x0.1'])
        output_dir: Directory to save output images
        resolution: Camera resolution for rendering
        seed: Random seed for reproducibility
        num_steps_wait: Number of steps to wait for objects to stabilize in sim
    """
    output_path = pathlib.Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    logging.info(f"Output directory: {output_path.resolve()}")
    logging.info(f"Task suites to process: {task_suite_names}")
    
    benchmark_dict = benchmark.get_benchmark_dict()
    
    for suite_name in task_suite_names:
        logging.info(f"\n{'='*80}")
        logging.info(f"Processing task suite: {suite_name}")
        logging.info(f"{'='*80}")
        
        # Try to load from benchmark dict, otherwise load from filesystem
        is_custom = False
        try:
            task_suite = benchmark_dict[suite_name]()
            num_tasks = task_suite.n_tasks
        except KeyError:
            logging.info(f"Task suite '{suite_name}' not in benchmark dict, loading from filesystem...")
            try:
                custom_suite = load_custom_task_suite(suite_name)
                num_tasks = custom_suite["n_tasks"]
                is_custom = True
            except FileNotFoundError as e:
                logging.error(f"Error loading custom suite: {e}")
                continue
        
        logging.info(f"Number of tasks in suite: {num_tasks}")
        
        # Process each task in the suite
        for task_id in range(num_tasks):
            if is_custom:
                task_info = custom_suite["tasks"][task_id]
                task_description = task_info["language"]
                logging.info(f"\nTask {task_id}: {task_description}")
                
                # Load initial states directly
                try:
                    init_states = torch.load(task_info["init_path"])
                    logging.info(f"  Found {len(init_states)} initial states")
                except Exception as e:
                    logging.error(f"  Error loading initial states: {e}")
                    continue
                
                # Create a simple task object for _get_libero_env
                class SimpleTask:
                    def __init__(self, task_info):
                        self.language = task_info["language"]
                        self.problem_folder = task_info["problem_folder"]
                        self.bddl_file = task_info["bddl_file"]
                        self.bddl_path = task_info["bddl_path"]  # Full path
                
                task = SimpleTask(task_info)
            else:
                task = task_suite.get_task(task_id)
                task_description = task.language
                logging.info(f"\nTask {task_id}: {task_description}")
                
                # Get initial states for this task
                try:
                    initial_states = task_suite.get_task_init_states(task_id)
                    init_states = initial_states
                    logging.info(f"  Found {len(init_states)} initial states")
                except Exception as e:
                    # Try loading from LIBERO-PRO directory instead
                    logging.warning(f"  Failed to load from default path, trying LIBERO-PRO directory...")
                    try:
                        libero_pro_path = pathlib.Path(__file__).resolve().parent.parent
                        init_states_path = libero_pro_path / "libero" / "libero" / "init_files" / task.problem_folder / task.init_states_file
                        logging.info(f"  Trying: {init_states_path}")
                        init_states = torch.load(init_states_path)
                        logging.info(f"  Found {len(init_states)} initial states")
                    except Exception as e2:
                        logging.error(f"  Error loading initial states from LIBERO-PRO: {e2}")
                        continue
            
            # Initialize environment
            try:
                env, _ = _get_libero_env(task, resolution, seed)
            except Exception as e:
                logging.error(f"  Error initializing environment: {e}")
                continue
            
            # Use the first initial state
            if len(init_states) == 0:
                logging.warning(f"  No initial states available for task {task_id}")
                continue
            
            try:
                # Reset environment
                env.reset()
                
                # Set initial state
                obs = env.set_init_state(init_states[0])
                
                # Wait for objects to stabilize (important for LIBERO)
                for _ in range(num_steps_wait):
                    obs, _, _, _ = env.step(LIBERO_DUMMY_ACTION)
                
                # Extract images (both agentview and wrist camera)
                # IMPORTANT: rotate 180 degrees to match training preprocessing
                agentview_img = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
                # wrist_img = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
                
                # Save images
                task_name_clean = task_description.replace(" ", "_")
                agentview_filename = output_path / f"{suite_name}_task{task_id:02d}_{task_name_clean}_agentview.png"
                wrist_filename = output_path / f"{suite_name}_task{task_id:02d}_{task_name_clean}_wrist.png"
                
                imageio.imwrite(agentview_filename, agentview_img)
                # imageio.imwrite(wrist_filename, wrist_img)
                
                logging.info(f"  ✓ Saved images:")
                logging.info(f"    - {agentview_filename.name}")
                # logging.info(f"    - {wrist_filename.name}")
                
                # Clean up environment
                env.close()
                
            except Exception as e:
                logging.error(f"  Error capturing scene: {e}")
                import traceback
                traceback.print_exc()
                continue
    
    logging.info(f"\n{'='*80}")
    logging.info(f"Scene capture complete! Images saved to: {output_path.resolve()}")
    logging.info(f"{'='*80}")


def create_comparison_gifs(
    task_suite_names: List[str],
    output_dir: str = "scene_comparisons",
    frame_duration: float = 1.5,
) -> None:
    """
    Create GIFs combining images from the same task across different task suites.
    
    Args:
        task_suite_names: List of task suite names in the order they should appear
        output_dir: Directory containing the captured images
        frame_duration: Duration (in seconds) to display each frame
    """
    output_path = pathlib.Path(output_dir)
    gif_output_path = output_path / "gifs"
    gif_output_path.mkdir(parents=True, exist_ok=True)
    
    logging.info(f"\n{'='*80}")
    logging.info(f"Creating comparison GIFs...")
    logging.info(f"{'='*80}")
    
    # Find all unique tasks by scanning the first task suite's images
    if not task_suite_names:
        logging.error("No task suites provided")
        return
    
    # Get all task IDs and names from the first suite
    first_suite_images = sorted(output_path.glob(f"{task_suite_names[0]}_task*_agentview.png"))
    
    if not first_suite_images:
        logging.error(f"No images found for suite {task_suite_names[0]}")
        return
    
    # Extract task information
    tasks_info = []
    for img_path in first_suite_images:
        # Parse filename: {suite_name}_task{id}_{task_description}_agentview.png
        filename = img_path.stem  # Remove .png
        # Remove suite name prefix and _agentview suffix
        parts = filename.split('_task')
        if len(parts) < 2:
            continue
        task_part = parts[1]  # e.g., "00_pick_up_the_alphabet_soup_and_place_it_in_the_basket_agentview"
        task_id = task_part.split('_')[0]  # "00"
        # Reconstruct task description
        task_desc = '_'.join(task_part.split('_')[1:-1])  # Remove task_id and _agentview
        tasks_info.append((task_id, task_desc))
    
    logging.info(f"Found {len(tasks_info)} tasks to process")
    
    # Try to load a default font, fallback to default if not available
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 16)
    except:
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf", 16)
        except:
            font = ImageFont.load_default()
            logging.warning("Using default font as TrueType fonts not found")
    
    # Process each task
    for task_id, task_desc in tasks_info:
        logging.info(f"\nProcessing task {task_id}: {task_desc.replace('_', ' ')}")
        
        frames = []
        valid_suites = []
        
        # Collect images from each suite for this task
        for suite_name in task_suite_names:
            img_path = output_path / f"{suite_name}_task{task_id}_{task_desc}_agentview.png"
            
            if not img_path.exists():
                logging.warning(f"  Image not found for suite {suite_name}: {img_path.name}")
                continue
            
            # Load image
            img = Image.open(img_path)
            
            # Add text label with suite name
            draw = ImageDraw.Draw(img)
            
            # Add a semi-transparent background for the text
            text = suite_name
            # Get text bounding box
            bbox = draw.textbbox((0, 0), text, font=font)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]
            
            # Position at top center
            x = (img.width - text_width) // 2
            y = 10
            
            # Draw background rectangle
            padding = 10
            draw.rectangle(
                [x - padding, y - padding, x + text_width + padding, y + text_height + padding],
                fill=(0, 0, 0, 200)
            )
            
            # Draw text
            draw.text((x, y), text, font=font, fill=(255, 255, 255))
            
            # Convert back to array for imageio
            frames.append(np.array(img))
            valid_suites.append(suite_name)
        
        if not frames:
            logging.warning(f"  No frames found for task {task_id}, skipping")
            continue
        
        # Save as GIF
        gif_filename = gif_output_path / f"task{task_id}_{task_desc}.gif"
        imageio.mimsave(
            gif_filename,
            frames,
            duration=frame_duration,
            loop=0  # Infinite loop
        )
        
        logging.info(f"  ✓ Created GIF: {gif_filename.name}")
        logging.info(f"    Suites: {', '.join(valid_suites)}")
    
    logging.info(f"\n{'='*80}")
    logging.info(f"GIF creation complete! GIFs saved to: {gif_output_path.resolve()}")
    logging.info(f"{'='*80}")


def _get_libero_env(task, resolution, seed):
    """
    Initializes and returns the LIBERO environment.
    
    This function constructs the correct paths to the .bddl file for the given task
    and creates an OffScreenRenderEnv with the appropriate configuration.
    """
    task_description = task.language
    
    # Check if task has a full bddl_path (custom task) or need to construct it (benchmark task)
    if hasattr(task, 'bddl_path'):
        task_bddl_file = task.bddl_path
    else:
        task_bddl_file = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
        
        # If file doesn't exist, try LIBERO-PRO directory
        if not task_bddl_file.exists():
            libero_pro_path = pathlib.Path(__file__).resolve().parent.parent
            task_bddl_file = libero_pro_path / "libero" / "libero" / "bddl_files" / task.problem_folder / task.bddl_file
    
    logging.info(f"  BDDL file: {task_bddl_file}")
    
    if not task_bddl_file.exists():
        raise FileNotFoundError(f"BDDL file not found: {task_bddl_file}")
    
    env_args = {
        "bddl_file_name": task_bddl_file,
        "camera_heights": resolution,
        "camera_widths": resolution
    }
    env = OffScreenRenderEnv(**env_args)
    env.seed(seed)
    
    return env, task_description


def main(
    task_suite_names: List[str] = None,
    output_dir: str = "scene_comparisons",
    resolution: int = 256,
    seed: int = 7,
    num_steps_wait: int = 10,
    create_gifs: bool = True,
    gif_frame_duration: float = 10,
) -> None:
    """
    Main entry point for the script.
    
    Args:
        task_suite_names: List of task suite names to process. Defaults to test suites.
        output_dir: Directory to save output images
        resolution: Camera resolution for rendering
        seed: Random seed for reproducibility
        num_steps_wait: Number of steps to wait for objects to stabilize
        create_gifs: Whether to create comparison GIFs after capturing images
        gif_frame_duration: Duration (in seconds) to display each frame in GIFs
    """
    # Default test suites if none provided
    if task_suite_names is None:
        task_suite_names = [
            # "libero_object_temp_x0.1",
            # "libero_object_temp_x0.2",
            # "libero_object_temp_x0.3",
            # "libero_object_temp_x0.4",
            # "libero_object_temp_x0.5",
            # "libero_object_temp_y0.1",
            # "libero_object_temp_y0.2",
            # "libero_object_temp_y0.3",
            # "libero_object_temp_y0.4",
            # "libero_object_temp_y0.5",
            "libero_10",
            "libero_10_swap",

        ]
    
    capture_scene_images(
        task_suite_names=task_suite_names,
        output_dir=output_dir,
        resolution=resolution,
        seed=seed,
        num_steps_wait=num_steps_wait,
    )
    
    # Create comparison GIFs if requested
    if create_gifs:
        create_comparison_gifs(
            task_suite_names=task_suite_names,
            output_dir=output_dir,
            frame_duration=gif_frame_duration,
        )


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    tyro.cli(main)
