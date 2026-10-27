# Omni RSI bank — provenance (plan D6 / Verification 5, step 10)

`rsi_bank_omni.npz` merges reference states harvested from BOTH heads of record named by the
sideways and forward campaigns: the forward 2c teacher (body +x, `source=0`) and the sideways 2a
teacher (body +y, left-leads, `source=1`). It carries the same six arrays the forward RSI reset
event (`crab_hex_forward_task/mdp/crab_hex_rsi.py::_load_bank`) reads --
`joint_pos (N,24)`, `joint_vel (N,24)`, `root_quat_w (N,4)`, `root_z (N,)`, `root_lin_vel_b (N,3)`,
`clock_phase (N,)`, all float32 -- plus `source` (int8) and `meta` (a JSON string). Verified
directly (no Isaac needed): loading the merged bank's six named keys via the same dict
comprehension `_load_bank` uses succeeds and every array's first dimension is 4000; the extra
`source`/`meta` keys are ignored by that loader exactly as the two extra keys in `rsi_bank_pg_r*`
already are.

## Commands as run

Both harvests ran serially (one Isaac process at a time), from `$R/parkour`, with every `KRABBY_*`
var unset first and `OMNI_KIT_ACCEPT_EULA=yes` exported, under `timeout 2400`. Neither run needed a
retry.

**Forward** (`--axis x --source 0`, `Isaac-Crab-Hex-Flat-Walk-Play-v0`, the 2c teacher head):
```bash
export OMNI_KIT_ACCEPT_EULA=yes KRABBY_PHASE=2c KRABBY_PLANT=A15+B KRABBY_HEADING=0:0 \
       KRABBY_FLAT_TERRAIN_FLAT_FRAC=1.0 KRABBY_STAND_FRAC=0 KRABBY_RSI_FRAC=0 KRABBY_DR_PUSH=0
$PY harvest_omni_rsi_bank.py --task Isaac-Crab-Hex-Flat-Walk-Play-v0 \
    --checkpoint parkour/logs/rsl_rl/crab_hex_flat_walk/2026-09-07_04-38-50/model_19996.pt \
    --axis x --source 0 --out rsi_bank_fwd_raw.npz --headless --num_envs 64 --frames 2000
```

**Sideways** (`--axis y --source 1`, `Isaac-Crab-Hex-Side-Walk-Play-v0`, the 2a teacher head):
```bash
export OMNI_KIT_ACCEPT_EULA=yes KRABBY_SIDE_PHASE=2a KRABBY_PLANT=A15+B \
       KRABBY_FLAT_TERRAIN_FLAT_FRAC=1.0 KRABBY_STAND_FRAC=0 KRABBY_DR_PUSH=0
$PY harvest_omni_rsi_bank.py --task Isaac-Crab-Hex-Side-Walk-Play-v0 \
    --checkpoint parkour/logs/rsl_rl/crab_hex_side_teacher/2026-09-12_03-04-37/model_9998.pt \
    --axis y --source 1 --out rsi_bank_side_raw.npz --headless --num_envs 64 --frames 2000
```
`KRABBY_HEADING` was NOT set in the sideways process (the 2a preset -- `ELEMENTS_5K`, not
`ELEMENTS_10K` -- does not set it either; only 2b+ does), matching the plan's instruction. The
sideways checkpoint (trained on `Isaac-Crab-Hex-Side-Teacher-v0`) loaded on
`Isaac-Crab-Hex-Side-Walk-Play-v0` on the first attempt -- **no fallback to
`Isaac-Crab-Hex-Side-Teacher-Play-v0` was needed.**

**Merge:**
```bash
$PY merge_rsi_banks.py rsi_bank_fwd_raw.npz rsi_bank_side_raw.npz \
    --out rsi_bank_omni.npz --per_source 2000 --seed 0
$PY merge_rsi_banks.py --check rsi_bank_omni.npz   # Verification 5
```

## Frame counts before/after filtering

`harvest_omni_rsi_bank.py`'s own filter is upright > 0.95, the AXIS-ALIGNED speed component >
0.15, `|wz| < 0.3`, episode step >= 100 (the harvester's "before/after"):

| source | task | frames rolled | num_envs | candidate env-steps (min_step + thin gate) | kept (harvest filter) |
|---|---|---:|---:|---:|---:|
| 0 forward | Isaac-Crab-Hex-Flat-Walk-Play-v0 | 2000 | 64 | 40443 | 32324 |
| 1 sideways | Isaac-Crab-Hex-Side-Walk-Play-v0 | 2000 | 64 | 40543 | 29794 |

`merge_rsi_banks.py` applies a SECOND, stricter filter the harvester does not (it only bounds the
axis-aligned component, not the cross-axis one): per frame, `atan2(vy_b, vx_b)` within +/-20 deg of
the axis its `source` tag implies, and `||v_b,xy|| in [0.15, 0.5]`. A real rollout's gait-cycle
wobble legitimately fails a nonzero fraction of this tighter band, so the merge script FILTERS
(not rejects) each raw bank to the passing subset before subsampling:

| source | harvest-filtered frames | pass angle+speed | pass rate | bad angle | bad speed | sampled |
|---|---:|---:|---:|---:|---:|---:|
| 0 forward | 32324 | 25713 | 79.5% | 6386 | 277 | 2000 |
| 1 sideways | 29794 | 22382 | 75.1% | 6407 | 1133 | 2000 |

Merged total: **4000 frames** (2000 per source, 50/50), shuffled with seed 0.

## Per-source stats (harvest-filtered candidate pool, PRE the merge's tighter angle/speed gate)

Computed over every thinned, episode-eligible env-step during the harvest roll (i.e. before the
harvester's own upright/speed/wz thresholds are applied) -- the natural distribution those filters
cut into. `wz` (root angular velocity z) is never itself stored in the bank (see
`crab_hex_rsi.py`: "ang vel not recorded in the probe; ~0 in steady walking"), so this is the only
place its distribution is captured.

| source | quantity | median | p10 | p90 | n |
|---|---|---:|---:|---:|---:|
| 0 forward | upright (`-projected_gravity_b[:,2]`) | 0.9934 | 0.9860 | 0.9984 | 40443 |
| 0 forward | speed along +x (m/s) | 0.2206 | 0.1086 | 0.3289 | 40443 |
| 0 forward | wz (rad/s) | -0.0016 | -0.1326 | 0.1304 | 40443 |
| 1 sideways | upright | 0.9996 | 0.9984 | 0.9999 | 40543 |
| 1 sideways | speed along +y (m/s) | 0.2458 | 0.0681 | 0.3960 | 40543 |
| 1 sideways | wz (rad/s) | 0.0018 | -0.1703 | 0.1736 | 40543 |

Both medians sit inside the merge's `[0.15, 0.5]` speed band and both wz medians are near zero
(consistent with the forward run's pinned `KRABBY_HEADING=0:0` and the sideways 2a preset, which
sets no heading steering at all). Full per-run filter/count/stats JSON is in each raw bank's
`meta` array and echoed in the harvest logs' `[harvest] meta=...` lines.

## Checkpoint sha256s

| head | path | sha256 |
|---|---|---|
| forward 2c teacher | `parkour/logs/rsl_rl/crab_hex_flat_walk/2026-09-07_04-38-50/model_19996.pt` | `ed43a09d3165b87dd651b718ce8ba01d866cf115f59239357677dbe0ad913018` |
| sideways 2a teacher | `parkour/logs/rsl_rl/crab_hex_side_teacher/2026-09-12_03-04-37/model_9998.pt` | `27a51c2b7cc9104a83b6645f0c37f1a9d033e7f11bcfd0a8d855a5e092dc05f5` |

The sideways sha256 matches the one already recorded in
`crab_hex_sideways_task/experiments/2026-09-11_0414_sideways_formation/head/README.md`, confirming
the correct checkpoint file.

## Merged bank sha256 (`sha256sum rsi_bank_omni.npz`)

```
98ce8420192efcbf24c37c07fa521c91e00a264d54ecb1d73b41b1d9901c46c6  rsi_bank_omni.npz
```

## Verification 5 check output

```
[check] rsi_bank_omni.npz: {"n": 4000, "n_bad_angle": 0, "n_bad_speed": 0, "n_bad": 0,
"max_angle_dev_deg": 19.996627807617188, "speed_min": 0.15050201713761327,
"speed_max": 0.4998421482248337, "ok": true}
[check] PASS -- 4000 frames, max angle deviation 20.00 deg, speed range [0.151, 0.500]
```

`merge_rsi_banks.py --check <bank>` re-runs this same assertion standalone against any bank file
(raw or merged) and is what a bad bank would fail loudly against -- e.g. it correctly rejected an
early bug in this script (a version that hard-asserted 100% of RAW harvest frames pass the
angle/speed band, which no real rollout does; fixed by filtering before subsampling instead of
rejecting the whole source, see `merge_rsi_banks.py`'s docstring).

## Open item for a later step: `bundle.yaml`'s pin (CLOSED — the pin below now sits in `bundle.yaml`; kept as the step-10 record)

`bundle.yaml` (already present in this campaign dir, written by another step's agent) carries a
placeholder:
```yaml
banks:
  - dest: rsi_bank_omni.npz
    sha256: TODO_PIN_AFTER_STEP_10
```
Per this task's scope, step 10 may not edit existing files, so `bundle.yaml` was left untouched.
The real value to pin there is the merged bank's sha256 above:
`98ce8420192efcbf24c37c07fa521c91e00a264d54ecb1d73b41b1d9901c46c6`.
