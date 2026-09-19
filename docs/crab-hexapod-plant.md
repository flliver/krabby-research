# Crab hexapod plant of record (hardware geometry)

Task-agnostic reference for the simulated robot ("plant") that every crab-hex task trains and
evaluates on: which USD is the main asset, where its numbers come from, how to select an
alternative plant, and which checkpoints need which plant. Task/MDP/schedule details live in
[crab-hex-forward-policy-config.md](crab-hex-forward-policy-config.md) and the task README
([crab_hex_forward_task/README.md](../parkour/parkour_tasks/parkour_tasks/crab_hex_forward_task/README.md)).

Facts of record as of 2026-09-09.

## 1. The main asset

`assets/crab.usda` is the MAIN asset and the plant of record, the **A15+B** geometry:

- **A15** -- the front and rear leg mounts are shimmed 15 deg outward (row F toes toward -x,
  row R toward +x; symmetric about the transverse mid-plane, so the robot stays reversible).
- **B** -- the outer yaw axes are re-hinged to 2.5 in from the body ends, i.e. the outer mounts
  sit at x = +-0.2921 m instead of the legacy +-0.2159 m.
- The mid legs are unchanged.

What did **not** change versus the legacy golden: splay and re-hinge are rigid transforms of the
whole leg chain (hip plate, cam rotor, femur, tibia, footpad), so joint-local anchors, link
masses and inertias, joint limits, the cam mapping and the linkage geometry are all untouched.
The byte-pin test `test_splay_variant_touches_only_outer_leg_mount_lines` enforces that a splay
change touches only the outer-mount lines (orient / translate / localRot0) at a fixed axis position;
`test_variant_masses_unchanged` checks that a combined splay + re-hinge variant (20 deg, 2.5 in)
keeps the total mass.

`assets/variants/crab_simple__splay00_axis5p5in.usda` is the **legacy golden**: the 2026-08-20 measured-robot
build with splay 0 and outer axes 5.5 in from the body ends. It is kept byte-pinned because every
checkpoint from the 2026-08-20 measured-hardware rebuild up to the a15b lineage was trained on it
(see section 5; earlier heads used the hand-authored pre-generator models).

`assets/crab_simple.usda` is **not a plant any more**. On 2026-09-09 it was reverted to the
hand-authored Cube model of the 2026-08-09 campaign baseline (git `5ca0a8c`: 31 Cube prims with the
cam-shaft mechanism, before the measured-hardware rebuild replaced it with generated plywood-outline
meshes). It is kept as a historical reference of the robot as it stood at the 2026-08-09 Task-1
campaign baseline (the pre-experimentation May-2026 revision is the snapshot bundled under
`experiments/old-runs/`), sha-pinned by
`tests/unit/test_crab_hex_usd_generation.py`, never generated, and no task or plant name loads it.

## 2. Measured hardware

Source of truth for every number below (and for the USD generator and the sim configs):
[crab_hex_forward_task/mdp/crab_hex_dimensions.py](../parkour/parkour_tasks/parkour_tasks/crab_hex_forward_task/mdp/crab_hex_dimensions.py).
Values are stored in the units they were measured in (inches / pounds); metric values are derived
there. 2026-08-20 measurement session, user-confirmed against CAD.

| Item | Value | Constant |
|---|---|---|
| Body (x travel-direction, y lateral, z) | 28 x 48 x 12.5 in (0.7112 x 1.2192 x 0.3175 m) | `BODY_LENGTH_X_IN`, `BODY_WIDTH_Y_IN`, `BODY_HEIGHT_IN` |
| Body mass (all-inclusive) | 350 lb | `BODY_MASS_LB` |
| Leg mass, each (incl. both linear actuators) | 26.2 lb | `LEG_MASS_LB` |
| Outer yaw axis from body end (A15+B) | 2.5 in | `OUTER_LEG_AXIS_FROM_BODY_END_IN` |
| Outer-row splay (A15+B) | 15 deg | `OUTER_ROW_SPLAY_DEG` |
| Outer yaw axis from body end (legacy) | 5.5 in | `LEGACY_OUTER_LEG_AXIS_FROM_BODY_END_IN` |
| Outer-row splay (legacy) | 0 deg | `LEGACY_OUTER_ROW_SPLAY_DEG` |
| Femur hinge-to-hinge | 23.0 in (0.5842 m) | `FEMUR_HINGE_TO_HINGE_IN` |
| Tibia knee-to-toe | 32.5 in (0.8255 m) | `TIBIA_KNEE_TO_TOE_IN` |
| Yaw ROM | +-25 deg | `YAW_THROW_DEG` |
| Hip ROM (from vertical-up; 45 = raised, 150 = extended down) | 45-150 deg | `HIP_ROM_FROM_UP_DEG` |
| Knee ROM (interior femur-tibia angle) | 5-140 deg | `KNEE_INTERIOR_ROM_DEG` |

The per-link mass split, the hip-plate geometry, actuator parameters, friction and the sim-frame
joint-limit conventions are all in the same module; the docstring records the provenance of each
number (CAD vs as-built) and what superseded what. Do not copy numbers from this table into code;
import the module.

## 3. Spawn height

`KRABBY_HEX_SPAWN_Z=1.085` (m) is the articulation spawn height read by the scene config
(`_crab_robot_cfg()` in `crab_hex_scene_cfg.py`); the same value is used for training,
play and stance checks. It was set for the 2026-08-20 vertical-plate geometry and re-validated on
A15+B in the leg-mount morphology campaign: settled root height 1.0620 m on A15+B versus 1.0612 m
on the legacy golden, well inside the +-10 mm band the campaign set for keeping the spawn value
unchanged
([2026-09-02_1446_leg_mount_morphology/RESULTS.md](../parkour/parkour_tasks/parkour_tasks/crab_hex_forward_task/experiments/2026-09-02_1446_leg_mount_morphology/RESULTS.md)).

The pre-generator May-2026 snapshots used `KRABBY_HEX_SPAWN_Z=1.05` (see section 5).

## 4. Plant table and selection

Names are defined in `crab_hex_phases.PLANTS`
([crab_hex_forward_task/config/crab_hex/crab_hex_phases.py](../parkour/parkour_tasks/parkour_tasks/crab_hex_forward_task/config/crab_hex/crab_hex_phases.py)).

| Plant name | File | Splay (deg) | Outer axis (in) | Note |
|---|---|---|---|---|
| `A15+B`, `main` | `assets/crab.usda` | 15 | 2.5 | plant of record; default, nothing to set |
| `legacy_golden`, `golden` | `assets/variants/crab_simple__splay00_axis5p5in.usda` | 0 | 5.5 | 2026-08-20 build; every head between the 2026-08-20 rebuild and the a15b lineage. `golden` is the alias used by pre-2026-09-09 records and commands |
| `B` | `assets/variants/crab_simple__splay00_axis2p5in.usda` | 0 | 2.5 | re-hinge only |
| `A10` | `assets/variants/crab_simple__splay10_axis5p5in.usda` | 10 | 5.5 | splay only |
| `A15` | `assets/variants/crab_simple__splay15_axis5p5in.usda` | 15 | 5.5 | splay only |
| `A20` | `assets/variants/crab_simple__splay20_axis5p5in.usda` | 20 | 5.5 | splay only |
| `A10+B` | `assets/variants/crab_simple__splay10_axis2p5in.usda` | 10 | 2.5 | splay + re-hinge |
| `A20+B` | `assets/variants/crab_simple__splay20_axis2p5in.usda` | 20 | 2.5 | splay + re-hinge |

`assets/variants/crab_simple__splay15_axis2p5in.usda` also exists and is byte-identical to
`assets/crab.usda` (pinned by `test_main_asset_is_the_a15b_plant`). The variants are listed in
[assets/variants/MANIFEST.md](../assets/variants/MANIFEST.md) (generated).

**Selecting a plant**

- Training: `KRABBY_PLANT=<name>` in the environment; `activate_phase()` expands it (together
  with `KRABBY_PHASE`) into `KRABBY_HEX_USD_PATH` before the task config is imported. Nothing is
  needed for the main plant.
- Gait harness: `--plant <name>` on `eval_crab_hex_gait.py` or `run_gait_eval_suite.py`. The USD
  is read at config-import time, so a scenario manifest's env block cannot select it; the flag
  exports `KRABBY_PLANT` (only when it is not already set) before the task package import.

**Precedence:** an explicitly exported `KRABBY_HEX_USD_PATH` wins over `KRABBY_PLANT` / `--plant`,
which wins over the default (`assets/crab.usda`). If `KRABBY_HEX_USD_PATH` is already set, the
harness aborts after spawning when the spawned USD is not the one `--plant` asked for; unset it
first. `--plant` is applied with setdefault, so an already-exported `KRABBY_PLANT` naming a
different plant also wins over the flag and triggers the same abort (whose message names only
`KRABBY_HEX_USD_PATH`); unset both before passing a different `--plant`. Every eval `run_meta.json`
records both the requested plant and the USD actually spawned.

**Warning (joystick-task coupling):** the joystick/HAL task `Isaac-CrabHex-Joystick-v0`
(`extreme_parkour_task/config/hex/crab_hex_play_cfg.py`) reads the same `KRABBY_HEX_USD_PATH`
variable but defaults to a DIFFERENT model, `assets/crab_hex_ref.usd`. Never export
`KRABBY_HEX_USD_PATH` in a shell profile; set the plant per command so neither task silently
loads the other's robot.

## 5. Which heads need which plant

| Heads | Plant | How to select |
|---|---|---|
| A15+B lineage and later (2c head `crab_hex_flat_walk/2026-09-07_04-38-50/model_19996.pt`, 3a head `crab_hex_student/2026-09-08_05-54-01/model_24995.pt`, all `launch_phases.sh --plant A15+B` runs) | `A15+B` | nothing |
| Legacy-golden heads (2026-08-20 rebuild .. 2026-09-06): golden 30k `crab_hex_flat_walk/2026-09-02_00-42-53/model_29994.pt`, golden 20k `crab_hex_flat_walk/2026-09-01_15-10-31/model_19996.pt`, the era-B campaign heads under `experiments/2026-08-2*/head/` and `2026-08-31_*/head/` | `legacy_golden` | `--plant legacy_golden` / `KRABBY_PLANT=legacy_golden` |
| May-2026 bundled stage checkpoints under `crab_hex_forward_task/experiments/old-runs/<ts>/` | the pre-generator USD snapshot bundled with the 2026-05-19 and 2026-05-23 runs (byte-identical; the five later May bundles point at the 2026-05-23 copy), e.g. `experiments/old-runs/2026-05-23_10-15-21/crab_simple_2026-05-23_10-15-21.usda` | `KRABBY_HEX_USD_PATH=<that file>` plus their setting of record `KRABBY_HEX_SPAWN_Z=1.05` |
| Era-A heads (2026-08-04 .. 08-17): the v1 eval baselines under `experiments/eval/baselines/v1/` and the `experiments/2026-08-0*..2026-08-1*/head/` bundles | hand-authored pre-generator `crab_simple.usda` revisions (the cam-shaft model committed as `5ca0a8c`, then `1b42d8d` / `ba8d060`); no plant name resolves to them | not runnable in the current MDP: their `run_meta.json` record 1151-wide observations vs 1149 today and `runner.load` is strict -- records only. Reproduction needs `git show <rev>:assets/crab_simple.usda` as `KRABBY_HEX_USD_PATH` plus that era's code |

Checkpoint paths above are relative to `parkour/logs/rsl_rl/` (git-ignored); campaigns that baked a
checkpoint also keep it under `<campaign>/head/` in the experiments tree (probe-only campaigns have
none; see experiments/README.md).

Gait harness on a legacy head (run from `parkour/`, one scenario from a manifest):

```bash
"$KRABBY_ROOT/IsaacLab/isaaclab.sh" -p parkour_tasks/parkour_tasks/crab_hex_forward_task/scripts/eval_crab_hex_gait.py \
  --headless --scenario <id> --checkpoint logs/rsl_rl/crab_hex_flat_walk/2026-09-02_00-42-53/model_29994.pt \
  --policy-role auto --plant legacy_golden
# whole manifest, same plant:
python3 parkour_tasks/parkour_tasks/crab_hex_forward_task/scripts/run_gait_eval_suite.py --plant legacy_golden
```

Training on a named plant (single phase, and the campaign driver):

```bash
KRABBY_PHASE=legacy_golden_1a KRABBY_PLANT=legacy_golden python scripts/rsl_rl/train.py \
  --task Isaac-Crab-Hex-Flat-Walk-v0 --headless --num_envs 256 --max_iterations 5000
# --campaign-dir is resolved by run_phases.py inside the systemd unit, whose cwd is the REPO ROOT
# (launch_phases.sh --working-directory), not parkour/: pass an absolute path
parkour_tasks/parkour_tasks/crab_hex_forward_task/experiments/tools/launch_phases.sh \
  --campaign-dir "$PWD/parkour_tasks/parkour_tasks/crab_hex_forward_task/experiments/<campaign>" \
  --plant A15+B --phases 1a,2a,2b,2c,3a [--seed 3] [--from-checkpoint <pt>] [--continue]
```

A May-2026 snapshot head:

```bash
RUNS=parkour_tasks/parkour_tasks/crab_hex_forward_task/experiments/old-runs
KRABBY_HEX_USD_PATH="$RUNS/2026-05-23_10-15-21/crab_simple_2026-05-23_10-15-21.usda" KRABBY_HEX_SPAWN_Z=1.05 \
  <play or eval command> --checkpoint "$RUNS/2026-05-23_10-15-21/model_6000.pt"
```

Loading a head on the wrong plant does not fail: the joint set is identical, so the policy runs
and simply walks badly. Check the plant recorded in `run_meta.json` before trusting a comparison.

## 6. RSI-bank caveat

The reference-state-initialization banks (`rsi_bank_P0_null.npz`, `rsi_bank_C1..C4.npz` under
`experiments/2026-08-26_2200_gated_lineage/`, `rsi_bank_pg_r1..r5.npz` under
`experiments/2026-08-31_1414_gait_income_phaseout/`) were harvested on the legacy golden. They are
reused unchanged on A15+B by design (the joint set, joint limits and defaults are identical across
plants; only the outer mounts move). The formation preset of record points at
`rsi_bank_P0_null.npz` regardless of plant. Any bank harvested in the future should record the
plant it was harvested on.

The knobs (`config/crab_hex/crab_hex_env_cfg.py`): `KRABBY_RSI_FRAC=<fraction of resets>` arms RSI
(unset or `0` = off; the presets use `0.2`) and `KRABBY_RSI_BANK=<npz>` picks the bank. Every preset
sets both: the A15+B lineage and `legacy_golden_1a`/`2a` point at `rsi_bank_P0_null.npz`,
`legacy_golden_2b..2e` at `rsi_bank_pg_r1..r4.npz` (`crab_hex_phases.py`, `LEGACY_GOLDEN_BANKS`). If
`KRABBY_RSI_FRAC` is set by hand without `KRABBY_RSI_BANK`, the code default is
`experiments/2026-08-22_1200_gait_formation_v2/rsi_bank_setAB.npz` -- the original 165-state bank,
harvested 2026-08-22 on the legacy-golden geometry and superseded by `rsi_bank_E1.npz` and then
`rsi_bank_P0_null.npz`; it is not a bank of record, so always pass the bank explicitly.
`KRABBY_RSI_SPAWN_FIX=1` (set by no preset) places bank resets where `reset_root_state` places every
other reset instead of the original tile-centre placement.

## 7. Regeneration and tests

Generator: [assets/scripts/generate_crab.py](../assets/scripts/generate_crab.py) (stdlib only; loads
`crab_hex_dimensions.py` and the CAD-outline module `crab_hex_leg_profiles.py` by path, no Isaac needed).

```bash
python3 assets/scripts/generate_crab.py                  # assets/crab.usda (main asset, A15+B)
python3 assets/scripts/generate_crab.py --legacy-golden  # assets/variants/crab_simple__splay00_axis5p5in.usda (legacy golden)
python3 assets/scripts/generate_crab.py --all-variants   # assets/variants/*.usda + MANIFEST.md
python3 assets/scripts/generate_crab.py --splay-deg 10 --outer-axis-in 5.5 --out /tmp/x.usda   # ad hoc
```

Never hand-edit a USDA: change `crab_hex_dimensions.py` (numbers), `crab_hex_leg_profiles.py` via
`assets/scripts/extract_leg_profiles.py` (link outlines), or the generator, and regenerate all
three forms. [tests/unit/test_crab_hex_usd_generation.py](../tests/unit/test_crab_hex_usd_generation.py)
byte-pins the committed files to the generator (`test_committed_asset_matches_generator`,
`test_legacy_golden_variant_matches_generator`, `test_main_asset_is_the_a15b_plant`,
`test_committed_variants_match_their_regeneration`, `test_manifest_lists_every_plant`; the hand-authored `assets/crab_simple.usda` is sha-pinned by `test_hand_authored_crab_simple_is_the_campaign_baseline_cube_model`) and checks
the physical invariants (per-leg mass sums to 26.2 lb, total mass matches hardware, body keeps its
full mass, variants change only the outer-mount lines and no masses). Toe forward kinematics
against Isaac's settled footpads is covered by `tests/unit/test_crab_hex_foot_fk.py`.

```bash
python -m pytest tests/unit/test_crab_hex_usd_generation.py -q
```

## 8. Provenance

Campaign records, under `crab_hex_forward_task/experiments/` (record files are git-tracked; raw
artifacts stay on disk, git-ignored):

- [2026-08-20_1506_hardware_morphology/RESULTS.md](../parkour/parkour_tasks/parkour_tasks/crab_hex_forward_task/experiments/2026-08-20_1506_hardware_morphology/RESULTS.md) -- measured robot to generated plant (the legacy golden build).
- [2026-09-02_1446_leg_mount_morphology/PLAN.md](../parkour/parkour_tasks/parkour_tasks/crab_hex_forward_task/experiments/2026-09-02_1446_leg_mount_morphology/PLAN.md), [RESULTS.md](../parkour/parkour_tasks/parkour_tasks/crab_hex_forward_task/experiments/2026-09-02_1446_leg_mount_morphology/RESULTS.md) -- splay / axis variants, spawn-height validation.
- [2026-09-04_1105_morph_x_exposure/REPORT.md](../parkour/parkour_tasks/parkour_tasks/crab_hex_forward_task/experiments/2026-09-04_1105_morph_x_exposure/REPORT.md) -- morphology x obstacle exposure; A15+B chosen (user decision 2026-09-06).
- [2026-09-06_2130_a15b_lineage/CHANGELOG.md](../parkour/parkour_tasks/parkour_tasks/crab_hex_forward_task/experiments/2026-09-06_2130_a15b_lineage/CHANGELOG.md) -- A15+B lineage charter and bake.
- [2026-09-07_1330_phase_pipeline/CHARTER.md](../parkour/parkour_tasks/parkour_tasks/crab_hex_forward_task/experiments/2026-09-07_1330_phase_pipeline/CHARTER.md), [REPORT.md](../parkour/parkour_tasks/parkour_tasks/crab_hex_forward_task/experiments/2026-09-07_1330_phase_pipeline/REPORT.md) -- the phase pipeline run on A15+B (heads of record).
