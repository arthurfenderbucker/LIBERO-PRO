#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
generate_init_states.py
- Read all .bddl files from the specified bddl_base_dir
- Generate num_inits initial states for each task
- Save as .pruned_init compressed files to output_dir

Usage example:
    python generate_init_states.py \
        --bddl_base_dir /path/to/bddl_dir \
        --output_dir /path/to/output_dir \
        --num_inits 50 \
        --height 128 \
        --width 128
"""

import os
import zipfile
import pickle
import numpy as np
from pathlib import Path
from tqdm import tqdm
import argparse

from libero.libero.envs import OffScreenRenderEnv


def generate_init_states(
    bddl_base_dir: str,
    output_dir: str,
    num_inits: int = 50,
    height: int = 128,
    width: int = 128,
):
    bddl_base_dir = Path(bddl_base_dir).resolve()
    output_dir = Path(output_dir).resolve()
    os.makedirs(output_dir, exist_ok=True)

    # Get all .bddl files
    bddl_files = list(bddl_base_dir.glob("*.bddl"))
    print(f"Found {len(bddl_files)} BDDL files")

    for bddl_file in tqdm(bddl_files, desc="Processing BDDL files"):
        task_base_name = bddl_file.stem
        print(f"\nStarting to process task: {task_base_name}")

        all_initial_states = []

        for i in tqdm(range(num_inits), desc=f"Generating initial states for {task_base_name}"):
            env = None
            try:
                env_args = {
                    "bddl_file_name": str(bddl_file),
                    "camera_heights": height,
                    "camera_widths": width,
                }
                env = OffScreenRenderEnv(**env_args)

                initial_state = env.get_sim_state()
                all_initial_states.append(initial_state)

            except Exception as e:
                print(f"  Error generating state {i+1}: {e}")

            finally:
                if env is not None and hasattr(env, 'close'):
                    env.close()

        output_filename = f"{task_base_name}.pruned_init"
        output_filepath = output_dir / output_filename

        try:
            with zipfile.ZipFile(output_filepath, 'w', zipfile.ZIP_DEFLATED) as zipf:
                all_initial_states = np.array(all_initial_states)
                pickled_states_list = pickle.dumps(all_initial_states)
                zipf.writestr("archive/data.pkl", pickled_states_list)
                zipf.writestr("archive/version", b"1")

            print(f"Successfully saved {len(all_initial_states)} states to: {output_filepath}")

        except Exception as e:
            print(f"Error saving state list: {e}")

    print("\nAll tasks processing complete!")


def parse_args():
    parser = argparse.ArgumentParser(description="Generate init states for LIBERO BDDL tasks.")
    parser.add_argument("--bddl_base_dir", type=str, required=True, help="Directory containing BDDL files.")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save .pruned_init files.")
    parser.add_argument("--num_inits", type=int, default=50, help="Number of init states to generate per task.")
    parser.add_argument("--height", type=int, default=128, help="Camera height.")
    parser.add_argument("--width", type=int, default=128, help="Camera width.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    generate_init_states(
        bddl_base_dir=args.bddl_base_dir,
        output_dir=args.output_dir,
        num_inits=args.num_inits,
        height=args.height,
        width=args.width,
    )