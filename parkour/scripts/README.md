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
