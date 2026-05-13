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
| `libero/libero/__init__.py` | `get_libero_path` applies `os.path.expandvars` + `os.path.expanduser` to each path read from `config.yaml` | Makes `config.yaml` portable across machines via `${LIBERO_PRO_PATH}/...` placeholders. The `config.yaml` itself lives inside the consumer repo (`policy_abstraction/src/policy_abstraction/envs/libero/configs/`) and is found via `LIBERO_CONFIG_PATH`. |
| `.gitignore` | Add `**/bddl_files/`, `**/init_files/`, `**/scene_comparisons` | Avoid committing the LFS-downloaded assets and run artifacts |

## Usage

Clone under `third_party/` and follow the install + path-config steps in
[`policy_abstraction/docs/install_libero_pro.md`](https://github.com/arthurfenderbucker/policy_abstraction/blob/main/docs/install_libero_pro.md).
