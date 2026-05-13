# LIBERO-PRO Fork Notes

This fork (`arthurfenderbucker/LIBERO-PRO`) extends the upstream
[`Zxy-MLlab/LIBERO-PRO`](https://github.com/Zxy-MLlab/LIBERO-PRO) with the
features needed by the [`policy_abstraction`](https://github.com/arthurfenderbucker/policy_abstraction)
project. Two commits ahead of `upstream/master`:

## Changes vs upstream

| File | Change | Purpose |
|---|---|---|
| `segmentation_utils.py` (new, 376 lines) | Segmentation rendering utilities | Adds per-object segmentation masks to env observations — required by the perception pipeline |
| `scripts/compare_libero_scenes.py` (new, 453 lines) | Scene comparison tool | Used to inspect/compare LIBERO scenes for the perturbation analysis |
| `perturbation.py` | Refactored (170 lines changed) | Cleaner perturbation API consumed by `policy_abstraction/envs/libero/perturbation.py` |
| `notebooks/generate_init_states.py` | Refactored (26 lines) | Consistent with the refactored `perturbation.py` |
| `config.yaml` (new) | LIBERO path config | Required so `from libero.libero import get_libero_path` resolves correctly when LIBERO-PRO is checked out under `third_party/` |
| `evaluation_config.yaml` | Absolute paths for `bddl_files_path`, `script_path`, `init_file_dir`, `ood_task_configs` | Lets the eval server find the perturbation OOD configs |
| `.gitignore` | Add `**/bddl_files/`, `**/init_files/`, `**/scene_comparisons` | Avoid committing the LFS-downloaded assets and run artifacts |

## Path configuration

`config.yaml` and `evaluation_config.yaml` currently contain absolute paths
hardcoded to the original author's machine. On a fresh checkout you must edit
both files to point at your local `third_party/LIBERO-PRO/...` and dataset
directories. A planned follow-up converts these to `${LIBERO_PRO_PATH}/...`
placeholders + an `os.path.expandvars` loader so no per-machine edits are
needed.

## Usage

Clone under `third_party/` and follow the install + path-config steps in
[`policy_abstraction/docs/install_libero_pro.md`](https://github.com/arthurfenderbucker/policy_abstraction/blob/main/docs/install_libero_pro.md).
