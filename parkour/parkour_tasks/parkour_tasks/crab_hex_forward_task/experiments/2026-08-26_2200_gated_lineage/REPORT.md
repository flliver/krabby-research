<!-- paths-note -->
> **Paths note (2026-09-09):** this campaign moved from `sim_fine_tuning/2026-08-26_2200_gated_lineage/` to `parkour/parkour_tasks/parkour_tasks/crab_hex_forward_task/experiments/2026-08-26_2200_gated_lineage/`. Absolute paths below (`/home/.../sim_fine_tuning/...`, `parkour/logs/rsl_rl/...`) name raw artifacts that stay on disk untracked; the tracked checkpoint of record is `head/` (see `bundle.yaml`) and the eval summaries are in place / under `evals/`.

## BASELINE control — round 0 (r0_001_control)
- probe @2k: clock income 0.457 | failure tail 0.134 | ep len 955.441 | trends clock +0.009 fail -0.011
- flat canary: tripod 0.494 | completion 0.970 | tracking 0.449
- sanity vs historical formation calibration (formed: clock 0.40–0.43 / fail 0.15–0.26): WITHIN band
>>> ENTRY baseline round0

## ELEMENT decision — turning @ round 0
| metric | candidate | control | margin |
|---|---|---|---|
| clock income @2k | 0.351 | 0.457 | 0.77x |
| failure tail @2k | 0.409 | 0.134 | +0.275 |
| ep len @2k | 823.407 | 955.441 | -132 |
- **DECISION: RETRY** — clock 0.77x < 0.85 or failure +0.275 > +0.10 (one-element delta vs control)
>>> ENTRY decision turning round0 RETRY

## ELEMENT decision — episode40 @ round 0
| metric | candidate | control | margin |
|---|---|---|---|
| clock income @2k | 0.373 | 0.457 | 0.82x |
| failure tail @2k | 0.318 | 0.134 | +0.183 |
| ep len @2k | 1487.306 | 955.441 | +532 |
- **DECISION: RETRY** — clock 0.82x < 0.85 or failure +0.183 > +0.10 (one-element delta vs control)
>>> ENTRY decision episode40 round0 RETRY

## ELEMENT decision — yaw_income @ round 0
| metric | candidate | control | margin |
|---|---|---|---|
| clock income @2k | 0.356 | 0.457 | 0.78x |
| failure tail @2k | 0.311 | 0.134 | +0.177 |
| ep len @2k | 861.335 | 955.441 | -94 |
- **DECISION: RETRY** — clock 0.78x < 0.85 or failure +0.177 > +0.10 (one-element delta vs control)
>>> ENTRY decision yaw_income round0 RETRY

## ELEMENT decision — goalvel_income @ round 0
| metric | candidate | control | margin |
|---|---|---|---|
| clock income @2k | 0.454 | 0.457 | 0.99x |
| failure tail @2k | 0.166 | 0.134 | +0.031 |
| ep len @2k | 926.319 | 955.441 | -29 |
- **DECISION: PASS** — clock 0.99x >= 0.95 and failure +0.031 <= +0.05
>>> ENTRY decision goalvel_income round0 PASS

## ELEMENT decision — terrain50 @ round 0
| metric | candidate | control | margin |
|---|---|---|---|
| clock income @2k | 0.240 | 0.457 | 0.52x |
| failure tail @2k | 0.693 | 0.134 | +0.559 |
| ep len @2k | 562.380 | 955.441 | -393 |
- **DECISION: RETRY** — clock 0.52x < 0.85 or failure +0.559 > +0.10 (one-element delta vs control)
>>> ENTRY decision terrain50 round0 RETRY

## ELEMENT decision — terrain_recal @ round 0
| metric | candidate | control | margin |
|---|---|---|---|
| clock income @2k | nan | 0.454 | nanx |
| failure tail @2k | nan | 0.166 | +nan |
| ep len @2k | nan | 926.319 | +nan |
- **DECISION: RETRY** — training crashed
>>> ENTRY decision terrain_recal round0 RETRY


## CORRECTION — terrain_recal round-0 RETRY was a crash artifact
The probe crashed at env construction: `hurdle_height_range` in the recal2b2
geometry preset was a tuple of two strings, but the terrain generator `eval()`s
a single comma-expression string (TypeError before any training). Fixed in
`crab_hex_env_cfg.py` (single expression string, same numbers); no formation
evidence exists for this element yet. terrain_recal re-queued at ROUND 0.
Orchestrator also patched: crashes now HALT (never recorded as RETRY), and
mid-round state survives restarts (persistent queue).
>>> ENTRY correction terrain_recal crash-artifact requeued
## INFRA note — 2026-08-27 afternoon outage (08:51-19:18)
Three trainer OOM-kills were diagnosed to infrastructure, not configs: (1) two
relaunches overlapped a dying Isaac's teardown; (2) test cgroups under 40G count
boot page cache; (3) a SPORADIC Isaac boot memory balloon (37.8 GB inside
SimulationApp._start_app, before any task code — baseline config booted clean
minutes later). The recal2b2 geometry is exonerated: it trains at 1.5 s/it once
booted (its only real bug was the hurdle_height_range TypeError, fixed).
Mitigations now active: launcher waits for Isaac teardown; orchestrator retries
a crashed run ONCE when the log shows no Python error (real errors still HALT).
>>> ENTRY infra boot-balloon diagnosis + retry mitigation
## ELEMENT decision — terrain_recal @ round 0
| metric | candidate | control | margin |
|---|---|---|---|
| clock income @2k | 0.429 | 0.454 | 0.94x |
| failure tail @2k | 0.259 | 0.166 | +0.093 |
| ep len @2k | 863.047 | 926.319 | -63 |
| canary t/c/tr | 0.256/0.410/0.480 | 0.494/0.970/0.449 | ratio 0.42 |
- escalation history: ambiguous at 2k (clock 0.94x, fail +0.093); continued to 5k
- **DECISION: RETRY (AMBIGUOUS)** — canary ratio 0.42 < 0.85 after escalation — conservative retry next round
>>> ENTRY decision terrain_recal round0 RETRY (AMBIGUOUS)

## ELEMENT decision — terrain_curriculum @ round 0
| metric | candidate | control | margin |
|---|---|---|---|
| clock income @2k | 0.363 | 0.454 | 0.80x |
| failure tail @2k | 0.334 | 0.166 | +0.168 |
| ep len @2k | 829.938 | 926.319 | -96 |
- **DECISION: RETRY** — clock 0.80x < 0.85 or failure +0.168 > +0.10 (one-element delta vs control)
>>> ENTRY decision terrain_curriculum round0 RETRY

## ELEMENT decision — safety_pack @ round 0
| metric | candidate | control | margin |
|---|---|---|---|
| clock income @2k | 0.438 | 0.454 | 0.96x |
| failure tail @2k | 0.189 | 0.166 | +0.023 |
| ep len @2k | 913.137 | 926.319 | -13 |
- **DECISION: PASS** — clock 0.96x >= 0.95 and failure +0.023 <= +0.05
>>> ENTRY decision safety_pack round0 PASS

## ELEMENT decision — clearance_pack @ round 0
| metric | candidate | control | margin |
|---|---|---|---|
| clock income @2k | 0.451 | 0.438 | 1.03x |
| failure tail @2k | 0.248 | 0.189 | +0.058 |
| ep len @2k | 888.830 | 913.137 | -24 |
| canary t/c/tr | 0.315/0.670/0.487 | 0.494/0.970/0.449 | ratio 0.64 |
- escalation history: ambiguous at 2k (clock 1.03x, fail +0.058); continued to 5k
- **DECISION: RETRY (AMBIGUOUS)** — canary ratio 0.64 < 0.85 after escalation — conservative retry next round
>>> ENTRY decision clearance_pack round0 RETRY (AMBIGUOUS)

## ELEMENT decision — speed_band @ round 0
| metric | candidate | control | margin |
|---|---|---|---|
| clock income @2k | 0.203 | 0.438 | 0.46x |
| failure tail @2k | 0.829 | 0.189 | +0.640 |
| ep len @2k | 437.772 | 913.137 | -475 |
- **DECISION: RETRY** — clock 0.46x < 0.85 or failure +0.640 > +0.10 (one-element delta vs control)
>>> ENTRY decision speed_band round0 RETRY

## ELEMENT decision — dr_push @ round 0
| metric | candidate | control | margin |
|---|---|---|---|
| clock income @2k | 0.411 | 0.438 | 0.94x |
| failure tail @2k | 0.242 | 0.189 | +0.052 |
| ep len @2k | 876.123 | 913.137 | -37 |
| canary t/c/tr | 0.264/0.970/0.154 | 0.494/0.970/0.449 | ratio 0.53 |
- escalation history: ambiguous at 2k (clock 0.94x, fail +0.052); continued to 5k
- **DECISION: RETRY (AMBIGUOUS)** — canary ratio 0.34 < 0.85 after escalation — conservative retry next round
>>> ENTRY decision dr_push round0 RETRY (AMBIGUOUS)

## ELEMENT decision — dr_masscom @ round 0
| metric | candidate | control | margin |
|---|---|---|---|
| clock income @2k | 0.403 | 0.438 | 0.92x |
| failure tail @2k | 0.233 | 0.189 | +0.044 |
| ep len @2k | 883.812 | 913.137 | -29 |
| canary t/c/tr | 0.000/0.050/0.182 | 0.494/0.970/0.449 | ratio 0.00 |
- escalation history: ambiguous at 2k (clock 0.92x, fail +0.044); continued to 5k
- **DECISION: RETRY (AMBIGUOUS)** — canary ratio 0.00 < 0.85 after escalation — conservative retry next round
>>> ENTRY decision dr_masscom round0 RETRY (AMBIGUOUS)

## CUMULATIVE check — round 0 (2 round-accepts)
- full set vs round control: clock 0.96x (>=0.85) | failure +0.055 (<=+0.10) | canary ratio 0.39 (>=0.85)
- set: goalvel_income, safety_pack
- **FAIL**
- ejecting most costly: **safety_pack** (marginal fail +0.023, clock 0.96x) -> retry next round; re-testing reduced set
>>> ENTRY cumcheck round0 EJECT safety_pack


## CORRECTION — safety_pack ejection was a measurement artifact
The cumulative backstop compared a 2k-trained full-set canary against the
5k-trained control canary — the 0.39 ratio was missing training time, not
element cost (both probe-axis gates passed: clock 0.96x, failure +0.055).
safety_pack is restored to the round-0 accepted set. The check now trains
the full set to 5k (duration-matched canary) and its passing checkpoint is
adopted directly as the bake, so the fair comparison adds no extra runtime.
>>> ENTRY correction safety_pack restored, cumcheck duration-matched
## CUMULATIVE check — round 0 (2 round-accepts)
- full set vs round control: clock 0.67x (>=0.85) | failure +0.159 (<=+0.10) | canary ratio 0.00 (>=0.85)
- set: goalvel_income, safety_pack
- **FAIL**
- ejecting most costly: **safety_pack** (marginal fail +0.023, clock 0.96x) -> retry next round; re-testing reduced set
>>> ENTRY cumcheck round0 EJECT safety_pack


## INFRA note — r0_016 cumcheck run salvaged (2026-08-28 16:20 outage)
The goal-vel-only backstop re-test COMPLETED all 5000 iterations and saved
model_4999, then died in Isaac teardown with a nonzero exit; the orchestrator
discarded the finished run and its 60 s retry pause overlapped the teardown,
OOM-killing the unit. Fixes: train() now trusts a final on-disk checkpoint over
the exit code; the infra retry waits for the dead trainer to fully clear. The
finished checkpoint is seeded for adoption on resume — no retraining needed.
>>> ENTRY infra r0_016 salvage + teardown-trust fixes
## CUMULATIVE check — round 0 (1 round-accepts)
- full set vs round control: clock 1.04x (>=0.85) | failure +0.083 (<=+0.10) | canary ratio 0.59 (>=0.85)
- set: goalvel_income
- **FAIL**
- ejecting most costly: **goalvel_income** (marginal fail +0.031, clock 0.99x) -> retry next round; re-testing reduced set
>>> ENTRY cumcheck round0 EJECT goalvel_income

## BAKE C1 — through iteration 5k
- configuration: baseline core + [none]
- env stack: {"KRABBY_APEX_W": "1.0", "KRABBY_CLOCK_W": "1.0", "KRABBY_FLAT_TERRAIN_DIFF": "0.05:0.2", "KRABBY_FLAT_TERRAIN_FLAT_FRAC": "0.8", "KRABBY_FLAT_TERRAIN_MODE": "light", "KRABBY_LIN_VEL_X": "0.0:0.35", "KRABBY_RSI_BANK": "/home/nickmagus/krabby/krabby-research/sim_fine_tuning/2026-08-26_2200_gated_lineage/rsi_bank_P0_null.npz", "KRABBY_RSI_FRAC": "0.2", "KRABBY_TRACK_L1_W": "-1.0", "KRABBY_TRACK_SIGMA2": "0.1"}
- checkpoint: 2026-08-27_01-06-53/model_4999.pt
- flat canary: tripod 0.494 | completion 0.970 | tracking 0.449
- all-obstacle eval: completion 0.650 | tripod 0.465
>>> ENTRY bake C1

## BAKE C1 — through iteration 5k
- configuration: baseline core + [none]
- env stack: {"KRABBY_APEX_W": "1.0", "KRABBY_CLOCK_W": "1.0", "KRABBY_FLAT_TERRAIN_DIFF": "0.05:0.2", "KRABBY_FLAT_TERRAIN_FLAT_FRAC": "0.8", "KRABBY_FLAT_TERRAIN_MODE": "light", "KRABBY_LIN_VEL_X": "0.0:0.35", "KRABBY_RSI_BANK": "/home/nickmagus/krabby/krabby-research/sim_fine_tuning/2026-08-26_2200_gated_lineage/rsi_bank_P0_null.npz", "KRABBY_RSI_FRAC": "0.2", "KRABBY_TRACK_L1_W": "-1.0", "KRABBY_TRACK_SIGMA2": "0.1"}
- checkpoint: 2026-08-27_01-06-53/model_4999.pt
- flat canary: tripod 0.494 | completion 0.970 | tracking 0.449
- all-obstacle eval: completion 0.680 | tripod 0.447
>>> ENTRY bake C1

## BAKE C1 — through iteration 5k
- configuration: baseline core + [none]
- env stack: {"KRABBY_APEX_W": "1.0", "KRABBY_CLOCK_W": "1.0", "KRABBY_FLAT_TERRAIN_DIFF": "0.05:0.2", "KRABBY_FLAT_TERRAIN_FLAT_FRAC": "0.8", "KRABBY_FLAT_TERRAIN_MODE": "light", "KRABBY_LIN_VEL_X": "0.0:0.35", "KRABBY_RSI_BANK": "/home/nickmagus/krabby/krabby-research/sim_fine_tuning/2026-08-26_2200_gated_lineage/rsi_bank_P0_null.npz", "KRABBY_RSI_FRAC": "0.2", "KRABBY_TRACK_L1_W": "-1.0", "KRABBY_TRACK_SIGMA2": "0.1"}
- checkpoint: 2026-08-27_01-06-53/model_4999.pt
- flat canary: tripod 0.494 | completion 0.970 | tracking 0.449
- all-obstacle eval: completion 0.680 | tripod 0.447
>>> ENTRY bake C1

## RSI refresh — bank rsi_bank_C1.npz adopted for round 1
>>> ENTRY rsi C1

## HALT — round-1 control crashed
>>> ENTRY halt control

## HALT — round-1 control crashed
>>> ENTRY halt control

## BASELINE control — round 1 (r1_019_control)
- probe @2k: clock income 0.509 | failure tail 0.082 | ep len 941.677 | trends clock +0.004 fail +0.002
- flat canary: tripod 0.450 | completion 1.000 | tracking 0.455
>>> ENTRY baseline round1

## ELEMENT decision — turning @ round 1
| metric | candidate | control | margin |
|---|---|---|---|
| clock income @2k | 0.471 | 0.509 | 0.92x |
| failure tail @2k | 0.155 | 0.082 | +0.073 |
| ep len @2k | 906.344 | 941.677 | -35 |
| canary t/c/tr | 0.435/0.990/0.500 | 0.450/1.000/0.455 | ratio 0.97 |
- escalation history: ambiguous at 2k (clock 0.92x, fail +0.073); continued to 5k
- **DECISION: PASS** — escalated canary ratio 0.97 >= 0.85 of control
>>> ENTRY decision turning round1 PASS

## ELEMENT decision — episode40 @ round 1
| metric | candidate | control | margin |
|---|---|---|---|
| clock income @2k | 0.434 | 0.471 | 0.92x |
| failure tail @2k | 0.298 | 0.155 | +0.143 |
| ep len @2k | 1636.670 | 906.344 | +730 |
- **DECISION: RETRY** — clock 0.92x < 0.85 or failure +0.143 > +0.10 (one-element delta vs control)
>>> ENTRY decision episode40 round1 RETRY

## ELEMENT decision — yaw_income @ round 1
| metric | candidate | control | margin |
|---|---|---|---|
| clock income @2k | 0.460 | 0.471 | 0.98x |
| failure tail @2k | 0.202 | 0.155 | +0.047 |
| ep len @2k | 870.776 | 906.344 | -36 |
- **DECISION: PASS** — clock 0.98x >= 0.95 and failure +0.047 <= +0.05
>>> ENTRY decision yaw_income round1 PASS

## ELEMENT decision — terrain50 @ round 1
| metric | candidate | control | margin |
|---|---|---|---|
| clock income @2k | 0.473 | 0.460 | 1.03x |
| failure tail @2k | 0.131 | 0.202 | -0.071 |
| ep len @2k | 915.062 | 870.776 | +44 |
- **DECISION: PASS** — clock 1.03x >= 0.95 and failure -0.071 <= +0.05
>>> ENTRY decision terrain50 round1 PASS

## ELEMENT decision — terrain_recal @ round 1
| metric | candidate | control | margin |
|---|---|---|---|
| clock income @2k | 0.464 | 0.473 | 0.98x |
| failure tail @2k | 0.223 | 0.131 | +0.092 |
| ep len @2k | 866.382 | 915.062 | -49 |
| canary t/c/tr | 0.403/0.980/0.505 | 0.435/0.990/0.500 | ratio 0.93 |
- escalation history: ambiguous at 2k (clock 0.98x, fail +0.092); continued to 5k
- **DECISION: PASS** — escalated canary ratio 0.93 >= 0.85 of control
>>> ENTRY decision terrain_recal round1 PASS

## HALT — r1_025_terrain_curriculum crashed before producing a checkpoint (config/code error, not a formation verdict). terrain_curriculum stays queued at round 1; fix and relaunch.
>>> ENTRY HALT crash terrain_curriculum round1


## INFRA root cause CLOSED — the boot balloon (2026-08-27..29)
Kit's DerivedDataCache (inside isaac_venv, NOT ~/.cache) was corrupted by an OOM
kill mid-write; every boot whose index scan tripped the corruption entered a
disk-GC loop migrating multi-GB buckets in RAM until the kernel OOM-killed it —
which corrupted the cache further (self-sustaining; grew to 101 GB on disk).
Explains all balloon behavior: reboot immunity, ~/.cache-clear immunity,
config-independence, nondeterminism. Fixed by evicting the cache (regenerates);
clean-cache boot validated (2 iterations, 40 s, no balloon). terrain_curriculum's
five round-1 deaths were all this — the element is not implicated.
>>> ENTRY infra root-cause derived-data-cache corruption fixed
## ELEMENT decision — terrain_curriculum @ round 1
| metric | candidate | control | margin |
|---|---|---|---|
| clock income @2k | 0.463 | 0.464 | 1.00x |
| failure tail @2k | 0.187 | 0.223 | -0.036 |
| ep len @2k | 892.963 | 866.382 | +27 |
- **DECISION: PASS** — clock 1.00x >= 0.95 and failure -0.036 <= +0.05
>>> ENTRY decision terrain_curriculum round1 PASS

## ELEMENT decision — clearance_pack @ round 1
| metric | candidate | control | margin |
|---|---|---|---|
| clock income @2k | 0.394 | 0.463 | 0.85x |
| failure tail @2k | 0.368 | 0.187 | +0.181 |
| ep len @2k | 778.107 | 892.963 | -115 |
- **DECISION: RETRY** — clock 0.85x < 0.85 or failure +0.181 > +0.10 (one-element delta vs control)
>>> ENTRY decision clearance_pack round1 RETRY

## ELEMENT decision — speed_band @ round 1
| metric | candidate | control | margin |
|---|---|---|---|
| clock income @2k | 0.420 | 0.463 | 0.91x |
| failure tail @2k | 0.398 | 0.187 | +0.211 |
| ep len @2k | 752.196 | 892.963 | -141 |
- **DECISION: RETRY** — clock 0.91x < 0.85 or failure +0.211 > +0.10 (one-element delta vs control)
>>> ENTRY decision speed_band round1 RETRY

## ELEMENT decision — dr_push @ round 1
| metric | candidate | control | margin |
|---|---|---|---|
| clock income @2k | 0.444 | 0.463 | 0.96x |
| failure tail @2k | 0.187 | 0.187 | -0.000 |
| ep len @2k | 852.980 | 892.963 | -40 |
- **DECISION: PASS** — clock 0.96x >= 0.95 and failure -0.000 <= +0.05
>>> ENTRY decision dr_push round1 PASS

## ELEMENT decision — dr_masscom @ round 1
| metric | candidate | control | margin |
|---|---|---|---|
| clock income @2k | 0.447 | 0.444 | 1.01x |
| failure tail @2k | 0.212 | 0.187 | +0.025 |
| ep len @2k | 850.455 | 852.980 | -3 |
- **DECISION: PASS** — clock 1.01x >= 0.95 and failure +0.025 <= +0.05
>>> ENTRY decision dr_masscom round1 PASS

## ELEMENT decision — safety_pack @ round 1
| metric | candidate | control | margin |
|---|---|---|---|
| clock income @2k | 0.465 | 0.447 | 1.04x |
| failure tail @2k | 0.201 | 0.212 | -0.011 |
| ep len @2k | 876.721 | 850.455 | +26 |
- **DECISION: PASS** — clock 1.04x >= 0.95 and failure -0.011 <= +0.05
>>> ENTRY decision safety_pack round1 PASS

## ELEMENT decision — goalvel_income @ round 1
| metric | candidate | control | margin |
|---|---|---|---|
| clock income @2k | 0.439 | 0.465 | 0.94x |
| failure tail @2k | 0.261 | 0.201 | +0.060 |
| ep len @2k | 867.543 | 876.721 | -9 |
| canary t/c/tr | 0.434/0.920/0.487 | 0.403/0.980/0.505 | ratio 0.94 |
- escalation history: ambiguous at 2k (clock 0.94x, fail +0.060); continued to 5k
- **DECISION: PASS** — escalated canary ratio 0.94 >= 0.85 of control
>>> ENTRY decision goalvel_income round1 PASS

## CUMULATIVE check — round 1 (9 round-accepts)
- full set vs round control: clock 0.84x (>=0.85) | failure +0.112 (<=+0.10) | canary ratio 1.00 (>=0.85)
- set: dr_masscom, dr_push, goalvel_income, safety_pack, terrain50, terrain_curriculum, terrain_recal, turning, yaw_income
- **FAIL**
- ejecting most costly: **turning** (marginal fail +0.073, clock 0.92x) -> retry next round; re-testing reduced set
>>> ENTRY cumcheck round1 EJECT turning

## CUMULATIVE check — round 1 (8 round-accepts)
- full set vs round control: clock 0.75x (>=0.85) | failure +0.206 (<=+0.10) | canary ratio 0.98 (>=0.85)
- set: dr_masscom, dr_push, goalvel_income, safety_pack, terrain50, terrain_curriculum, terrain_recal, yaw_income
- **FAIL**
- ejecting most costly: **goalvel_income** (marginal fail +0.060, clock 0.94x) -> retry next round; re-testing reduced set
>>> ENTRY cumcheck round1 EJECT goalvel_income

## CUMULATIVE check — round 1 (7 round-accepts)
- full set vs round control: clock 0.99x (>=0.85) | failure +0.031 (<=+0.10) | canary ratio 0.96 (>=0.85)
- set: dr_masscom, dr_push, safety_pack, terrain50, terrain_curriculum, terrain_recal, yaw_income
- **PASS**
>>> ENTRY cumcheck round1 PASS

## BAKE C2 — through iteration 10k
- configuration: baseline core + [yaw_income@r1, terrain50@r1, terrain_recal@r1, terrain_curriculum@r1, dr_push@r1, dr_masscom@r1, safety_pack@r1]
- env stack: {"KRABBY_APEX_W": "1.0", "KRABBY_CLOCK_W": "1.0", "KRABBY_COLLISION_W": "-2.0", "KRABBY_DR_COM": "0.01", "KRABBY_DR_MASS": "-0.5:1.5", "KRABBY_DR_PUSH": "0.5", "KRABBY_EDGE_W": "-0.3", "KRABBY_FLAT_TERRAIN_CURRICULUM": "1", "KRABBY_FLAT_TERRAIN_DIFF": "0.20:0.70", "KRABBY_FLAT_TERRAIN_FLAT_FRAC": "0.5", "KRABBY_FLAT_TERRAIN_GEOM": "recal2b2", "KRABBY_FLAT_TERRAIN_MODE": "light", "KRABBY_LIN_VEL_X": "0.0:0.35", "KRABBY_RSI_BANK": "/home/nickmagus/krabby/krabby-research/sim_fine_tuning/2026-08-26_2200_gated_lineage/rsi_bank_C1.npz", "KRABBY_RSI_FRAC": "0.2", "KRABBY_STUMBLE_W": "-1.0", "KRABBY_TERRAIN_PROMOTE": "0.45:0.25", "KRABBY_TRACK_L1_W": "-1.0", "KRABBY_TRACK_SIGMA2": "0.1", "KRABBY_YAW_W": "0.2"}
- checkpoint: 2026-08-30_03-51-02/model_9998.pt
- flat canary: tripod 0.432 | completion 0.990 | tracking 0.456 (Δ vs prev bake -0.062/+0.02)
- all-obstacle eval: completion 0.770 | tripod 0.376
>>> ENTRY bake C2

## RSI refresh — bank rsi_bank_C2.npz adopted for round 2
>>> ENTRY rsi C2

## BASELINE control — round 2 (r2_036_control)
- probe @2k: clock income 0.515 | failure tail 0.103 | ep len 935.151 | trends clock +0.013 fail -0.006
- flat canary: tripod 0.431 | completion 1.000 | tracking 0.464
>>> ENTRY baseline round2

## ELEMENT decision — episode40 @ round 2
| metric | candidate | control | margin |
|---|---|---|---|
| clock income @2k | 0.422 | 0.515 | 0.82x |
| failure tail @2k | 0.280 | 0.103 | +0.177 |
| ep len @2k | 1641.285 | 935.151 | +706 |
- **DECISION: RETRY** — clock 0.82x < 0.85 or failure +0.057 > +0.10 (one-element delta vs control) [failure hazard-normalized for episode-length shift]
>>> ENTRY decision episode40 round2 RETRY

## ELEMENT decision — clearance_pack @ round 2
| metric | candidate | control | margin |
|---|---|---|---|
| clock income @2k | 0.487 | 0.515 | 0.95x |
| failure tail @2k | 0.162 | 0.103 | +0.059 |
| ep len @2k | 910.276 | 935.151 | -25 |
| canary t/c/tr | 0.418/0.990/0.470 | 0.431/1.000/0.464 | ratio 0.97 |
- escalation history: ambiguous at 2k (clock 0.95x, fail +0.059); continued to 5k
- **DECISION: PASS** — escalated canary ratio 0.97 >= 0.85 of control
>>> ENTRY decision clearance_pack round2 PASS

## ELEMENT decision — speed_band @ round 2
| metric | candidate | control | margin |
|---|---|---|---|
| clock income @2k | 0.405 | 0.487 | 0.83x |
| failure tail @2k | 0.344 | 0.162 | +0.182 |
| ep len @2k | 773.824 | 910.276 | -136 |
- **DECISION: RETRY** — clock 0.83x < 0.85 or failure +0.182 > +0.10 (one-element delta vs control)
>>> ENTRY decision speed_band round2 RETRY

## ELEMENT decision — turning @ round 2
| metric | candidate | control | margin |
|---|---|---|---|
| clock income @2k | 0.491 | 0.487 | 1.01x |
| failure tail @2k | 0.138 | 0.162 | -0.024 |
| ep len @2k | 927.842 | 910.276 | +18 |
- **DECISION: PASS** — clock 1.01x >= 0.95 and failure -0.024 <= +0.05
>>> ENTRY decision turning round2 PASS

## ELEMENT decision — goalvel_income @ round 2
| metric | candidate | control | margin |
|---|---|---|---|
| clock income @2k | 0.451 | 0.491 | 0.92x |
| failure tail @2k | 0.215 | 0.138 | +0.077 |
| ep len @2k | 854.209 | 927.842 | -74 |
| canary t/c/tr | 0.408/0.990/0.521 | 0.418/0.990/0.470 | ratio 0.98 |
- escalation history: ambiguous at 2k (clock 0.92x, fail +0.077); continued to 5k
- **DECISION: PASS** — escalated canary ratio 0.98 >= 0.85 of control
>>> ENTRY decision goalvel_income round2 PASS

## CUMULATIVE check — round 2 (3 round-accepts)
- full set vs round control: clock 0.89x (>=0.85) | failure +0.080 (<=+0.10) | canary ratio 0.96 (>=0.85)
- set: clearance_pack, dr_masscom, dr_push, goalvel_income, safety_pack, terrain50, terrain_curriculum, terrain_recal, turning, yaw_income
- **PASS**
>>> ENTRY cumcheck round2 PASS

## BAKE C3 — through iteration 15k
- configuration: baseline core + [yaw_income@r1, terrain50@r1, terrain_recal@r1, terrain_curriculum@r1, dr_push@r1, dr_masscom@r1, safety_pack@r1, clearance_pack@r2, turning@r2, goalvel_income@r2]
- env stack: {"KRABBY_APEX_W": "1.0", "KRABBY_CLEARANCE_W": "0.9", "KRABBY_CLOCK_W": "1.0", "KRABBY_COLLISION_W": "-2.0", "KRABBY_DR_COM": "0.01", "KRABBY_DR_MASS": "-0.5:1.5", "KRABBY_DR_PUSH": "0.5", "KRABBY_EDGE_W": "-0.3", "KRABBY_FLAT_TERRAIN_CURRICULUM": "1", "KRABBY_FLAT_TERRAIN_DIFF": "0.20:0.70", "KRABBY_FLAT_TERRAIN_FLAT_FRAC": "0.5", "KRABBY_FLAT_TERRAIN_GEOM": "recal2b2", "KRABBY_FLAT_TERRAIN_MODE": "light", "KRABBY_FOOT_CLEAR_FLAT": "1", "KRABBY_FOOT_CLEAR_MIN": "0.03", "KRABBY_FOOT_CLEAR_W": "1.0", "KRABBY_GOAL_VEL_W": "0.75", "KRABBY_HEADING": "-1.2:1.2", "KRABBY_LIN_VEL_X": "0.0:0.35", "KRABBY_RSI_BANK": "/home/nickmagus/krabby/krabby-research/sim_fine_tuning/2026-08-26_2200_gated_lineage/rsi_bank_C2.npz", "KRABBY_RSI_FRAC": "0.2", "KRABBY_STUMBLE_W": "-1.0", "KRABBY_SWING_MIN_CLEAR_W": "-0.4", "KRABBY_TERRAIN_PROMOTE": "0.45:0.25", "KRABBY_TRACK_L1_W": "-1.0", "KRABBY_TRACK_SIGMA2": "0.1", "KRABBY_YAW_W": "0.2"}
- checkpoint: 2026-08-30_15-22-57/model_14997.pt
- flat canary: tripod 0.414 | completion 0.990 | tracking 0.564 (Δ vs prev bake -0.018/+0.00)
- all-obstacle eval: completion 0.660 | tripod 0.380
>>> ENTRY bake C3

## RSI refresh — bank rsi_bank_C3.npz adopted for round 3
>>> ENTRY rsi C3

## BASELINE control — round 3 (r3_044_control)
- probe @2k: clock income 0.422 | failure tail 0.334 | ep len 816.328 | trends clock -0.000 fail +0.022
- flat canary: tripod 0.433 | completion 0.980 | tracking 0.525
>>> ENTRY baseline round3

## ELEMENT decision — episode40 @ round 3
| metric | candidate | control | margin |
|---|---|---|---|
| clock income @2k | 0.365 | 0.422 | 0.86x |
| failure tail @2k | 0.405 | 0.334 | +0.071 |
| ep len @2k | 1392.122 | 816.328 | +576 |
| canary t/c/tr | 0.424/1.000/0.514 | 0.433/0.980/0.525 | ratio 0.98 |
- escalation history: ambiguous at 2k (clock 0.86x, fail -0.096 [failure hazard-normalized for episode-length shift]); continued to 5k
- **DECISION: PASS** — escalated canary ratio 0.98 >= 0.85 of control
>>> ENTRY decision episode40 round3 PASS

## ELEMENT decision — speed_band @ round 3
| metric | candidate | control | margin |
|---|---|---|---|
| clock income @2k | 0.224 | 0.365 | 0.61x |
| failure tail @2k | 0.619 | 0.405 | +0.214 |
| ep len @2k | 991.385 | 1392.122 | -401 |
- **DECISION: RETRY** — clock 0.61x < 0.85 or failure +0.464 > +0.10 (one-element delta vs control) [failure hazard-normalized for episode-length shift]
>>> ENTRY decision speed_band round3 RETRY

## CUMULATIVE check — round 3 (1 round-accepts)
- full set vs round control: clock 0.69x (>=0.85) | failure +0.080 (<=+0.10) | canary ratio 1.02 (>=0.85)
- set: clearance_pack, dr_masscom, dr_push, episode40, goalvel_income, safety_pack, terrain50, terrain_curriculum, terrain_recal, turning, yaw_income
- **FAIL**
- ejecting most costly: **episode40** (marginal fail -0.096, clock 0.86x) -> retry next round; re-testing reduced set
>>> ENTRY cumcheck round3 EJECT episode40

## CUMULATIVE check — all round acceptances ejected; baking without new elements
>>> ENTRY cumcheck round3 emptied

## BAKE C4 — through iteration 20k
- configuration: baseline core + [yaw_income@r1, terrain50@r1, terrain_recal@r1, terrain_curriculum@r1, dr_push@r1, dr_masscom@r1, safety_pack@r1, clearance_pack@r2, turning@r2, goalvel_income@r2]
- env stack: {"KRABBY_APEX_W": "1.0", "KRABBY_CLEARANCE_W": "0.9", "KRABBY_CLOCK_W": "1.0", "KRABBY_COLLISION_W": "-2.0", "KRABBY_DR_COM": "0.01", "KRABBY_DR_MASS": "-0.5:1.5", "KRABBY_DR_PUSH": "0.5", "KRABBY_EDGE_W": "-0.3", "KRABBY_FLAT_TERRAIN_CURRICULUM": "1", "KRABBY_FLAT_TERRAIN_DIFF": "0.20:0.70", "KRABBY_FLAT_TERRAIN_FLAT_FRAC": "0.5", "KRABBY_FLAT_TERRAIN_GEOM": "recal2b2", "KRABBY_FLAT_TERRAIN_MODE": "light", "KRABBY_FOOT_CLEAR_FLAT": "1", "KRABBY_FOOT_CLEAR_MIN": "0.03", "KRABBY_FOOT_CLEAR_W": "1.0", "KRABBY_GOAL_VEL_W": "0.75", "KRABBY_HEADING": "-1.2:1.2", "KRABBY_LIN_VEL_X": "0.0:0.35", "KRABBY_RSI_BANK": "/home/nickmagus/krabby/krabby-research/sim_fine_tuning/2026-08-26_2200_gated_lineage/rsi_bank_C3.npz", "KRABBY_RSI_FRAC": "0.2", "KRABBY_STUMBLE_W": "-1.0", "KRABBY_SWING_MIN_CLEAR_W": "-0.4", "KRABBY_TERRAIN_PROMOTE": "0.45:0.25", "KRABBY_TRACK_L1_W": "-1.0", "KRABBY_TRACK_SIGMA2": "0.1", "KRABBY_YAW_W": "0.2"}
- checkpoint: 2026-08-31_01-12-23/model_19996.pt
- flat canary: tripod 0.436 | completion 0.990 | tracking 0.495 (Δ vs prev bake +0.022/+0.00)
- all-obstacle eval: completion 0.530 | tripod 0.399
>>> ENTRY bake C4

## RSI refresh — bank rsi_bank_C4.npz adopted for round 4
>>> ENTRY rsi C4

## SEARCH COMPLETE — elements FAILED (out of scope to fix): speed_band, episode40
>>> ENTRY search failed-elements

## SCHEDULE CONFIRMATION (seed 3) — canary 0.606/0.990/0.433 vs primary bake 0.436/0.990 — CONFIRMED | obstacle 0.480
>>> ENTRY confirmation

## GRADUATION CANDIDATE — schedule: [yaw_income@5k, terrain50@5k, terrain_recal@5k, terrain_curriculum@5k, dr_push@5k, dr_masscom@5k, safety_pack@5k, clearance_pack@10k, turning@10k, goalvel_income@10k] | failed: [speed_band, episode40] | head /home/nickmagus/krabby/krabby-research/parkour/logs/rsl_rl/crab_hex_flat_walk/2026-08-31_01-12-23/model_19996.pt — run graduation battery
>>> ENTRY graduation


## GRADUATED (user decision, 2026-08-31) — confirmation head is the reference of record
Head: `parkour/logs/rsl_rl/crab_hex_flat_walk/2026-08-31_10-46-19/model_19996.pt`
(seed-3 schedule replay, 20k). Gates: flat canary 0.606/0.990/0.433 PASS;
all-obstacle-light 0.64 PASS (tripod 0.577); recal-terrain 0.480 WAIVED (-0.020,
measured at promoted difficulty); failure tail 0.292 WAIVED (<0.25 gate conflicts
with the promotion rule's designed equilibrium; non-failure terminations dominate);
turn vx-proxy ~0.47-0.53 at +-0.8 (yaw-rate aggregate not recorded by harness);
goal_idx gate VOID (40 s premise failed out of the curriculum).
Schedule of record: 7 elements @5k, 3 @10k, speed_band + episode40 FAILED.
>>> ENTRY graduated confirm-head reference-of-record
