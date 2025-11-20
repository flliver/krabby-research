# Scripts

This directory contains helper scripts for the Parkour project. The most relevant one for day-to-day workflows is `publish_weights.py` which packages the latest training checkpoints into version-controlled assets.

## publish_weights.py

Publishes the latest checkpoints from `logs/<workflow>/<experiment>/...` into `parkour/assets/weights/` so they are easy to consume and track in version control. It writes both the weight file and a small JSON metadata file per experiment.

- Discovers experiments under: `parkour/logs/<workflow>/`
- Selects newest run directory (e.g., `2025-11-18_20-23-18`)
- Picks the newest checkpoint:
  - Prefer `model_*.pt`
  - Fallback to `exported*/policy.pt` or `policy.pt`
- Publishes to: `parkour/assets/weights/<experiment>.pt`
- Metadata: `parkour/assets/weights/<experiment>.json`

### Requirements

- Run inside this repository; the script resolves paths relative to its own location.
- Trained runs exist under `parkour/logs/<workflow>/<experiment>/`.
- Python 3.8+.

### Basic Usage

- Publish all latest weights (overwrite existing):

```bash
python3 parkour/scripts/publish_weights.py --force
```

- Dry-run (show what would be copied without writing):

```bash
python3 parkour/scripts/publish_weights.py --dry-run
```

- Publish specific experiments only:

```bash
python3 parkour/scripts/publish_weights.py \
  --experiments unitree_go2_parkour_student unitree_go2_parkour_teacher --force
```

- Publish to a different target directory:

```bash
python3 parkour/scripts/publish_weights.py --assets-dir /tmp/my_weights --force
```

- Use a different workflow (default is `rsl_rl`):

```bash
python3 parkour/scripts/publish_weights.py --workflow rsl_rl --force
```

### Outputs

- `parkour/assets/weights/unitree_go2_parkour_student.pt`
- `parkour/assets/weights/unitree_go2_parkour_student.json`
- `parkour/assets/weights/unitree_go2_parkour_teacher.pt`
- `parkour/assets/weights/unitree_go2_parkour_teacher.json`

The JSON includes: experiment name, source checkpoint path, run directory, publish timestamp, and file size.

### Consuming the Published Weights

You can pass the published paths directly to your demo/eval scripts via `--checkpoint`.

- Example (student play):

```bash
python3 parkour/scripts/rsl_rl/demo.py \
  --task Isaac-Extreme-Parkour-Student-Unitree-Go2-Play-v0 \
  --checkpoint parkour/assets/weights/unitree_go2_parkour_student.pt
```

- Example (teacher play):

```bash
python3 parkour/scripts/rsl_rl/demo.py \
  --task Isaac-Extreme-Parkour-Teacher-Unitree-Go2-Play-v0 \
  --checkpoint parkour/assets/weights/unitree_go2_parkour_teacher.pt
```

### Version Control

The `assets/weights/` directory is intended to be committed so downstream users have access to the current best weights.

```bash
git add parkour/assets/weights/*
git commit -m "Publish latest parkour weights"
```

### Exit Codes

- `0`: At least one experiment published
- `1`: Logs root not found
- `2`: No weights published (e.g., no runs or checkpoints)

### Troubleshooting

- "No experiments found": Check `parkour/logs/<workflow>/` exists and contains experiment folders.
- "No runs found": Ensure at least one timestamped run directory exists (e.g., `2025-11-18_20-23-18`).
- "No checkpoints found": Verify that `model_*.pt` or `exported*/policy.pt` exists in the latest run.
- Overwrite needed: Add `--force` if the destination file already exists.

## gen_house_01_usda.py

Generates a reproducible USD/USDA scene for the reference house environment (house, site landscaping, and fence). This script allows deterministic reconstruction of the asset without relying on a manually authored file.

### Features
- Authors hierarchy: `/House`, `/Site`, `/Fence` with sub-prims for walls, roof, door, windows, ground, walkway, trees, hedge, panels, gate.
- Creates and binds `UsdShade.Material` prims (UsdPreviewSurface) using MaterialBindingAPI.
- Optionally writes ASCII `.usda` for diff-friendly version control (`--ascii`).
- Sets stage metadata: Z-up axis, meters-per-unit = 1.

### Requirements
- Python environment providing the `pxr` USD modules (Omniverse / Isaac Sim or official USD build).
- Write permissions to the target output path.

### Usage

Generate binary USD (default):
```bash
python3 parkour/scripts/gen_house_01_usda.py --output ../assets/scenes/house_01_generated.usd --overwrite
```

Generate ASCII USDA (forced extension change):
```bash
python3 parkour/scripts/gen_house_01_usda.py --output ../assets/scenes/house_01_generated.usd --ascii --overwrite
```

If output already ends with `.usda` the `--ascii` flag is optional:
```bash
python3 parkour/scripts/gen_house_01_usda.py --output ../assets/scenes/house_01_generated.usda --overwrite
```

### Exit Codes
- `0`: Success (file written)
- `1`: Refused overwrite (existing file without `--overwrite`)
- `2`: Generation failure (file missing after build)

### Notes
- The script sets the default prim to `/House` for viewer convenience.
- Materials use consistent shader child name `PreviewSurface` (verifier tolerant to alternatives).

## verify_usda_equivalence.py

Compares a generated scene file against the authored reference to ensure structural and content parity (materials, meshes).

### Checks Performed
1. Prim path presence (ignores shader child naming differences: `PreviewSurface` vs `Shader`).
2. Material diffuse colors (and opacity where present).
3. Mesh vertex counts, face group counts, and world-space bounding boxes.

### Requirements
- Same USD Python environment as the generator (must import `pxr`).
- Access to both original and generated scene files.

### Usage
```bash
python3 parkour/scripts/verify_usda_equivalence.py \
  --original parkour/assets/house/house_model.usda \
  --generated parkour/assets/house/house_model_generated.usda
```

### Exit Codes
- `0`: Equivalent within tolerances
- `1`: Structural differences (missing/extra prims)
- `2`: Geometry or material mismatches
- `3`: Error opening stages or unexpected failure

### Typical Workflow
```bash
# 1. Generate scene (ASCII for readability)
python3 parkour/scripts/gen_house_01_usda.py --output parkour/assets/house/house_model_generated.usda --ascii --overwrite

# 2. Verify equivalence
python3 parkour/scripts/verify_usda_equivalence.py \
  --original parkour/assets/house/house_model.usda \
  --generated parkour/assets/house/house_model_generated.usda
```

### Troubleshooting
- Missing `pxr`: Activate your USD/Omniverse environment or install official USD Python build.
- Structural mismatches: Confirm you generated with the latest script version and did not manually edit the output.
- Material color differences: Ensure no post-processing modified shader inputs.
- Bounding box differences: Verify up-axis metadata (script sets Z-up) and that transforms were not applied afterward.

### Version Control Tips
- Prefer committing the ASCII `.usda` output for meaningful diffs.
- Include both generator and verifier scripts so others can reproduce and validate assets.

