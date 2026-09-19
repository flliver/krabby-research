# Krabby-Uno Isaac Sim scripts

## Generating the training plant

The training plant (the model the RL tasks load) is **generated**, not hand-authored:

- `generate_crab.py` (plain run) writes the **main asset** [`assets/crab.usda`](../crab.usda) — the **A15+B** geometry (15° outward splay of the front/rear leg mounts, outer yaw axes re-hinged to 2.5 in from the body ends).
- `generate_crab.py --legacy-golden` regenerates [`assets/variants/crab_simple__splay00_axis5p5in.usda`](../variants/crab_simple__splay00_axis5p5in.usda), the 2026-08-20 build (splay 0 / axes 5.5 in) kept byte-pinned as the **legacy golden**.
- [`assets/crab_simple.usda`](../crab_simple.usda) is the one hand-authored USDA: the 2026-08-09 campaign-baseline Cube model, kept as a historical reference (sha-pinned; the generator refuses to write it; no task loads it).
- `generate_crab.py --all-variants` regenerates `assets/variants/*.usda` and `assets/variants/MANIFEST.md`.
- Byte-pin tests: `tests/unit/test_crab_hex_usd_generation.py` — the checked-in USDAs must match the generator output byte for byte.
- Source of truth for every dimension: `parkour/parkour_tasks/parkour_tasks/crab_hex_forward_task/mdp/crab_hex_dimensions.py`.
- `extract_leg_profiles.py` regenerates `parkour/parkour_tasks/parkour_tasks/crab_hex_forward_task/mdp/crab_hex_leg_profiles.py` (the CAD leg-part outlines the generator turns into link meshes and mass properties) from `~/krabby/joint_specs/KrabV3-Legs.svg`; re-run it when the CAD changes, then regenerate the USDAs.

See [`docs/crab-hexapod-plant.md`](../../docs/crab-hexapod-plant.md) for plant selection (`KRABBY_PLANT` / `--plant`) and provenance.

## Legacy `crab_hex.usd` demos

The scripts below are **legacy demos** for the hand-rigged [`crab_hex.usd`](../crab_hex.usd) Blender-pipeline model (stage **`/World/KrabbyUno`**; see [`Krabby-Uno-USD-pipeline.md`](../Krabby-Uno-USD-pipeline.md) -- the joystick/HAL task loads `assets/crab_hex_ref.usd`, not this file). They do not target the generated training plant above.

### `squat.py` (legacy)

- **Full-leg squat:** all six legs in sync — **hip yaw**, **hip–femur prismatic**, **femur–tibia prismatic** (same joint paths as `simple_walk.py`).
- Run from the **Script Editor** (open the file or paste its contents). It schedules an **async** coroutine using `await app.next_update_async()` while the timeline plays.
- Do **not** call `omni.kit.app.get_app().update()` in a tight loop.

### `simple_walk.py` (legacy)

- **Self-contained** — paste the **entire** file into the Script Editor and run (no extra modules on `sys.path`).
- Open-loop **tripod** gait with **zero-mean** commands (stable over longer runs).
- Default: `walk_forward_steps(5)` when run as `__main__`. Call `walk_forward_steps(n)` or `stop_robot()` as needed.
- After editing the script, **paste the updated buffer** again before re-running.

### Quick run (legacy demos)

1. Load `assets/crab_hex.usd` in Isaac Sim.
2. Open the Script Editor, paste `squat.py` or `simple_walk.py`, execute.

Or run the file from disk if `__file__` resolves correctly in your Kit setup.
