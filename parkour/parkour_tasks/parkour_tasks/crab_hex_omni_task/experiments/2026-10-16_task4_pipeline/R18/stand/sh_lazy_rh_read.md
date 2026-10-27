# Stand height -- K1-K6 read (PLAN.md H7)

Generated 2026-10-21T15:49:00-04:00 by `scripts/sh_read.py` (sha256 1a62ef5ad64d; CPU, read-only). Env seeds es1, es2; height rows hwalkstop__omni -> ['K1', 'K4', 'K6'], hraise__omni -> ['K2', 'K4']. Statistics as pinned in the module docstring.

**Control table: 14 / 14 cells land, 0 mismatch(es), 0 pending. Exit 4.**

**R_h stand-up premise (sh_table_core.rh_floor: STOOD >= 0.9 and falls <= 1 per 100 on each gated top ['table_0p635', 'table_0p6985', 'table_0p762'] pooled over the gated poses ['knee_folded', 'random']; the reported cells are listed, never a miss):** MISS in es1 top table_0p635

**K3 cells:** GATED (6; the K3 line and the R_h floor read these only): table_0p635__knee_folded, table_0p635__random, table_0p6985__knee_folded, table_0p6985__random, table_0p762__knee_folded, table_0p762__random. REPORTED (9; run and read, never gated): table_0p4572__knee_folded, table_0p4572__rod_min, table_0p4572__random, table_0p635__rod_min, table_0p6985__rod_min, table_0p762__rod_min, ground__knee_folded, ground__rod_min, ground__random.

**K3 stand checkpoints (the one-checkpoint rule reads the gated cells only):**

- Amask_old es1: gated cells' stand checkpoint ['769cf2533140']; reported cells' ['769cf2533140']
- Amask_old es2: gated cells' stand checkpoint ['769cf2533140']; reported cells' ['769cf2533140']
- Amask es1: gated cells' stand checkpoint ['769cf2533140']; reported cells' ['769cf2533140']
- Amask es2: gated cells' stand checkpoint ['769cf2533140']; reported cells' ['769cf2533140']
- SH_T4R18_sh_s17 es1: gated cells' stand checkpoint ['e629abd5cef3']; reported cells' ['e629abd5cef3']
- SH_T4R18_sh_s17 es2: gated cells' stand checkpoint ['e629abd5cef3']; reported cells' ['e629abd5cef3']

**h_max re-check (R_h top bin None):** not read
  Reported beside it (not the decision statistic): within 0.02 m of h_max, -- N (PD -- N, None rows).

## Heads

| head | K1 | K2 | K3 | K4 | K5 | K6 | all PASS |
|---|---|---|---|---|---|---|---|
| SH_T4R18_sh_s17 | PASS | PASS | PASS | FAIL | INVALID | FAIL | False |

- SH_T4R18_sh_s17: residual deadband 0.25 (one per head; absent record = 0); trained at 0.25 (null = not recorded)

## Gates

| gate | role | expected | observed | lands | values |
|---|---|---|---|---|---|
| K1 TRACKING | R_h | PASS | PASS | yes | median es1 0.002347; p90 es1 0.005712; median es1 0.001901; p90 es1 0.004129; median es1 0.001511; p90 es1 0.003566; median es1 0.001381; p90 es1 0.003569; median es1 0.001518; p90 es1 0.003885; median es2 0.002708; p90 es2 0.006043; median es2 0.001515; p90 es2 0.004877; median es2 0.001676; p90 es2 0.003519; median es2 0.001414; p90 es2 0.003548; median es2 0.001585; p90 es2 0.003616 |
| K1 TRACKING | R_old | FAIL | FAIL | yes | **median es1 0.2952 (line 0.02)**; **p90 es1 0.3469 (line 0.04)**; **median es1 0.1433 (line 0.02)**; **p90 es1 0.1458 (line 0.04)**; median es1 0.001703; p90 es1 0.003603; **median es1 0.08737 (line 0.02)**; **p90 es1 0.08934 (line 0.04)**; **median es1 0.2359 (line 0.02)**; **p90 es1 0.2849 (line 0.04)**; **median es2 0.2956 (line 0.02)**; **p90 es2 0.3465 (line 0.04)**; **median es2 0.1432 (line 0.02)**; **p90 es2 0.1472 (line 0.04)**; median es2 0.001585; p90 es2 0.003523; **median es2 0.08694 (line 0.02)**; **p90 es2 0.08954 (line 0.04)**; **median es2 0.2336 (line 0.02)**; **p90 es2 0.2848 (line 0.04)** |
| K1 TRACKING | Amask_old | FAIL | FAIL | yes | **median es1 0.2941 (line 0.02)**; **p90 es1 0.3469 (line 0.04)**; **median es1 0.1426 (line 0.02)**; **p90 es1 0.1456 (line 0.04)**; median es1 0.002129; p90 es1 0.004839; **median es1 0.08732 (line 0.02)**; **p90 es1 0.08998 (line 0.04)**; **median es1 0.2371 (line 0.02)**; **p90 es1 0.2857 (line 0.04)**; **median es2 0.2941 (line 0.02)**; **p90 es2 0.3463 (line 0.04)**; **median es2 0.142 (line 0.02)**; **p90 es2 0.1465 (line 0.04)**; median es2 0.001712; p90 es2 0.003739; **median es2 0.08734 (line 0.02)**; **p90 es2 0.08988 (line 0.04)**; **median es2 0.2336 (line 0.02)**; **p90 es2 0.286 (line 0.04)** |
| K1 TRACKING | posture | -- | PASS | reported | median es1 0.002608; p90 es1 0.005795; median es1 0.001748; p90 es1 0.003741; median es1 0.001201; p90 es1 0.003796; median es1 0.001617; p90 es1 0.00393; median es1 0.001657; p90 es1 0.003734; median es2 0.00262; p90 es2 0.005829; median es2 0.001747; p90 es2 0.004433; median es2 0.001697; p90 es2 0.004112; median es2 0.001748; p90 es2 0.004648; median es2 0.001526; p90 es2 0.003647 |
| K1 TRACKING | Amask_q | -- | FAIL | reported | **median es1 0.0238 (line 0.02)**; p90 es1 0.03229; median es1 0.01145; p90 es1 0.01787; median es1 0.00169; p90 es1 0.004817; **median es1 0.03729 (line 0.02)**; **p90 es1 0.06556 (line 0.04)**; **median es1 0.08745 (line 0.02)**; **p90 es1 0.1018 (line 0.04)**; **median es2 0.02322 (line 0.02)**; p90 es2 0.03086; median es2 0.01099; p90 es2 0.01735; median es2 0.001789; p90 es2 0.004632; **median es2 0.03811 (line 0.02)**; **p90 es2 0.0632 (line 0.04)**; **median es2 0.08786 (line 0.02)**; **p90 es2 0.1009 (line 0.04)** |
| K1 TRACKING | SH_T4R18_sh_s17 | -- | PASS | head | median es1 0.002469; p90 es1 0.005197; median es1 0.001525; p90 es1 0.004795; median es1 0.001532; p90 es1 0.003286; median es1 0.001812; p90 es1 0.00392; median es1 0.001532; p90 es1 0.003844; median es2 0.002467; p90 es2 0.006463; median es2 0.0017; p90 es2 0.005334; median es2 0.001439; p90 es2 0.003592; median es2 0.001541; p90 es2 0.004183; median es2 0.001383; p90 es2 0.00393 |
| K2 RAISE / LOWER | R_h | PASS | PASS | yes | reach es1 -1.175; stand falls es1 0; reach es2 -1.17; stand falls es2 0 |
| K2 RAISE / LOWER | R_old | FAIL | FAIL | yes | **reach es1 inf (line 0)**; stand falls es1 0; **reach es2 inf (line 0)**; stand falls es2 0 |
| K2 RAISE / LOWER | Amask_old | FAIL | FAIL | yes | **reach es1 inf (line 0)**; stand falls es1 0; **reach es2 inf (line 0)**; stand falls es2 0 |
| K2 RAISE / LOWER | posture | -- | PASS | reported | reach es1 -1.17; stand falls es1 0; reach es2 -1.165; stand falls es2 0 |
| K2 RAISE / LOWER | Amask_q | -- | FAIL | reported | **reach es1 inf (line 0)**; stand falls es1 0; **reach es2 inf (line 0)**; stand falls es2 0 |
| K2 RAISE / LOWER | SH_T4R18_sh_s17 | -- | PASS | head | reach es1 -1.177; stand falls es1 0; reach es2 -1.175; stand falls es2 0 |
| K3 STAND-UP | R_h | PASS | PASS | yes | stood tables es1 0.9833; falls tables es1 0.8333; stood tables es2 0.9783; falls tables es2 0.3333 |
| K3 STAND-UP | posture | -- | PASS | reported | stood tables es1 0.995; falls tables es1 0; stood tables es2 0.9783; falls tables es2 0.1667 |
| K3 STAND-UP | Amask | -- | FAIL | reported | **stood tables es1 0.13 (line 0.9333)**; **falls tables es1 1.333 (line 1)**; **stood tables es2 0.1617 (line 0.9283)**; **falls tables es2 1.667 (line 1)** |
| K3 STAND-UP | S1 | FAIL | FAIL | yes | **stood tables es1 0 (line 0.9333)**; **falls tables es1 3.667 (line 1)**; **stood tables es2 0 (line 0.9283)**; **falls tables es2 3.167 (line 1)** |
| K3 STAND-UP | SH_T4R18_sh_s17 | -- | PASS | head | stood tables es1 0.9883; falls tables es1 0.6667; stood tables es2 0.9783; falls tables es2 0.3333 |
| K4 RESTORE + HAND-BACK | R_h | PASS | PASS | yes | miss es1 7; exit falls es1 0; onset es1 -1.333; miss es2 3; exit falls es2 0; onset es2 -1.315 |
| K4 RESTORE + HAND-BACK | Amask | -- | PASS | reported | miss es1 0; exit falls es1 1; onset es1 -0.145; miss es2 0; exit falls es2 0; onset es2 -0.116 |
| K4 RESTORE + HAND-BACK | no_restore | FAIL | FAIL | yes | **miss es1 788 (line 7)**; exit falls es1 0; onset es1 -1.56; **miss es2 784 (line 3)**; exit falls es2 0; onset es2 -1.54 |
| K4 RESTORE + HAND-BACK | SH_T4R18_sh_s17 | -- | FAIL | head | **miss es1 13 (line 7)**; exit falls es1 0; onset es1 -1.339; **miss es2 10 (line 3)**; exit falls es2 0; onset es2 -1.31 |
| K5 NON-REGRESSION @1.064 | R_h | PASS | PASS | yes | G1 PASS; G2 PASS; G3 PASS; G4 PASS; G5 PASS |
| K5 NON-REGRESSION @1.064 | Amask | PASS | PASS | yes | G1 PASS; G2 PASS; G3 PASS; G4 PASS; G5 PASS |
| K5 NON-REGRESSION @1.064 | SH_T4R18_sh_s17 | -- | INVALID | head | **G1 INVALID (line PASS)**; **G2 INVALID (line PASS)**; **G3 INVALID (line PASS)**; **G4 INVALID (line PASS)**; **G5 INVALID (line PASS)** |
| K6 HOMING + STILLNESS | R_h | PASS | PASS | yes | homed es1 0.995; settled es1 1; homed es1 0.81; settled es1 0.82; homed es1 1; settled es1 1; homed es1 0.99; settled es1 0.99; homed es1 0.885; settled es1 0.895; still steps es1 0.1852; still taps es1 0.1287; homed es2 0.99; settled es2 1; homed es2 0.75; settled es2 0.75; homed es2 1; settled es2 1; homed es2 0.99; settled es2 0.99; homed es2 0.85; settled es2 0.85; still steps es2 0.1809; still taps es2 0.1597 |
| K6 HOMING + STILLNESS | posture | FAIL | FAIL | yes | **homed es1 0 (line 0.945)**; **settled es1 0 (line 0.95)**; **homed es1 0 (line 0.7315)**; **settled es1 0 (line 0.7432)**; **homed es1 0 (line 0.95)**; **settled es1 0 (line 0.95)**; **homed es1 0 (line 0.94)**; **settled es1 0 (line 0.94)**; **homed es1 0.005 (line 0.835)**; **settled es1 0.005 (line 0.845)**; still steps es1 0; still taps es1 0; **homed es2 0 (line 0.94)**; **settled es2 0 (line 0.95)**; **homed es2 0 (line 0.6634)**; **settled es2 0 (line 0.6634)**; **homed es2 0 (line 0.95)**; **settled es2 0 (line 0.95)**; **homed es2 0 (line 0.94)**; **settled es2 0 (line 0.94)**; **homed es2 0.005 (line 0.7985)**; **settled es2 0.005 (line 0.7985)**; still steps es2 0.2024; still taps es2 0.3036 |
| K6 HOMING + STILLNESS | Amask | -- | FAIL | reported | **homed es1 0 (line 0.945)**; **settled es1 0 (line 0.95)**; **homed es1 0 (line 0.7315)**; **settled es1 0.299 (line 0.6994)**; homed es1 1; settled es1 1; **homed es1 0 (line 0.94)**; **settled es1 0 (line 0.94)**; **homed es1 0 (line 0.835)**; **settled es1 0.2172 (line 0.8221)**; **still steps es1 0.8075 (line 0.5)**; **still taps es1 2.06 (line 1)**; **homed es2 0 (line 0.94)**; **settled es2 0 (line 0.95)**; **homed es2 0 (line 0.6634)**; **settled es2 0.3434 (line 0.6211)**; homed es2 1; settled es2 1; **homed es2 0 (line 0.94)**; **settled es2 0 (line 0.94)**; **homed es2 0 (line 0.7995)**; **settled es2 0.215 (line 0.773)**; **still steps es2 0.7134 (line 0.5)**; **still taps es2 1.673 (line 1)** |
| K6 HOMING + STILLNESS | SH_T4R18_sh_s17 | -- | FAIL | head | homed es1 0.95; settled es1 0.96; homed es1 0.83; settled es1 0.88; homed es1 0.99; settled es1 0.99; homed es1 1; settled es1 1; homed es1 0.89; settled es1 0.89; still steps es1 0.1797; still taps es1 0.1442; **homed es2 0.925 (line 0.94)**; settled es2 0.96; homed es2 0.84; settled es2 0.86; homed es2 0.99; settled es2 1; homed es2 1; settled es2 1; homed es2 0.88; settled es2 0.88; still steps es2 0.1652; still taps es2 0.1357 |

## Pending

- none

## INVALID cells

- K5 x SH_T4R18_sh_s17: SH_T4R18_sh_s17z suite@1: p2r_mode.s1_checkpoint_sha256 b5a237cc98d5 != REFS 968cc5509899; SH_T4R18_sh_s17z suite@2: p2r_mode.s1_checkpoint_sha256 b5a237cc98d5 != REFS 968cc5509899; SH_T4R18_sh_s17z walkstop__omni@1: p2r_mode.s1_checkpoint_sha256 b5a237cc98d5 != REFS 968cc5509899; SH_T4R18_sh_s17z walkstop__omni@2: p2r_mode.s1_checkpoint_sha256 b5a237cc98d5 != REFS 968cc5509899; SH_T4R18_sh_s17z f

## Forces per height bin (reported; foot-load statics = the cap statistic, PD torque alongside)

| role | bin | rows | h_meas | F_hip 6-leg mean foot-load (N) | p90 | PD reading (N) | worst hip leg (foot-load) | F_knee foot-load / PD (N) | over cap |
|---|---|---|---|---|---|---|---|---|---|

## Reads

- R_h es1 hwalkstop__omni: ok `/home/nickmagus/krabby/krabby-research/parkour/outputs/omni_plan/phase2/lr/SH_RH_T4R18/hwalkstop__omni/seed001/2026-10-21_19-23-39`
- R_h es1 hraise__omni: ok `/home/nickmagus/krabby/krabby-research/parkour/outputs/omni_plan/phase2/lr/SH_RH_T4R18/hraise__omni/seed001/2026-10-21_19-37-32`
- R_h es2 hwalkstop__omni: ok `/home/nickmagus/krabby/krabby-research/parkour/outputs/omni_plan/phase2/lr/SH_RH_T4R18/hwalkstop__omni/seed002/2026-10-21_19-30-37`
- R_h es2 hraise__omni: ok `/home/nickmagus/krabby/krabby-research/parkour/outputs/omni_plan/phase2/lr/SH_RH_T4R18/hraise__omni/seed002/2026-10-21_19-42-02`
- R_old es1 hwalkstop__omni: ok `/home/nickmagus/krabby/krabby-research/parkour/outputs/omni_plan/phase2/lr/SH_ROLD/hwalkstop__omni/seed001/2026-10-13_23-48-58`
- R_old es1 hraise__omni: ok `/home/nickmagus/krabby/krabby-research/parkour/outputs/omni_plan/phase2/lr/SH_ROLD/hraise__omni/seed001/2026-10-14_00-02-38`
- R_old es2 hwalkstop__omni: ok `/home/nickmagus/krabby/krabby-research/parkour/outputs/omni_plan/phase2/lr/SH_ROLD/hwalkstop__omni/seed002/2026-10-13_23-55-50`
- R_old es2 hraise__omni: ok `/home/nickmagus/krabby/krabby-research/parkour/outputs/omni_plan/phase2/lr/SH_ROLD/hraise__omni/seed002/2026-10-14_00-07-06`
- Amask_old es1 hwalkstop__omni: ok `/home/nickmagus/krabby/krabby-research/parkour/outputs/omni_plan/phase2/lr/SH_AMOLD/hwalkstop__omni/seed001/2026-10-14_00-11-31`
- Amask_old es1 hraise__omni: ok `/home/nickmagus/krabby/krabby-research/parkour/outputs/omni_plan/phase2/lr/SH_AMOLD/hraise__omni/seed001/2026-10-14_00-25-21`
- Amask_old es2 hwalkstop__omni: ok `/home/nickmagus/krabby/krabby-research/parkour/outputs/omni_plan/phase2/lr/SH_AMOLD/hwalkstop__omni/seed002/2026-10-14_00-18-26`
- Amask_old es2 hraise__omni: ok `/home/nickmagus/krabby/krabby-research/parkour/outputs/omni_plan/phase2/lr/SH_AMOLD/hraise__omni/seed002/2026-10-14_00-29-52`
- posture es1 hwalkstop__omni: ok `/home/nickmagus/krabby/krabby-research/parkour/outputs/omni_plan/phase2/lr/SH_POST/hwalkstop__omni/seed001/2026-10-13_23-26-27`
- posture es1 hraise__omni: ok `/home/nickmagus/krabby/krabby-research/parkour/outputs/omni_plan/phase2/lr/SH_POST/hraise__omni/seed001/2026-10-13_23-40-07`
- posture es2 hwalkstop__omni: ok `/home/nickmagus/krabby/krabby-research/parkour/outputs/omni_plan/phase2/lr/SH_POST/hwalkstop__omni/seed002/2026-10-13_23-33-17`
- posture es2 hraise__omni: ok `/home/nickmagus/krabby/krabby-research/parkour/outputs/omni_plan/phase2/lr/SH_POST/hraise__omni/seed002/2026-10-13_23-44-33`
- Amask_q es1 hwalkstop__omni: ok `/home/nickmagus/krabby/krabby-research/parkour/outputs/omni_plan/phase2/lr/SH_AMq/hwalkstop__omni/seed001/2026-10-13_22-18-32`
- Amask_q es1 hraise__omni: ok `/home/nickmagus/krabby/krabby-research/parkour/outputs/omni_plan/phase2/lr/SH_AMq/hraise__omni/seed001/2026-10-14_00-41-14`
- Amask_q es2 hwalkstop__omni: ok `/home/nickmagus/krabby/krabby-research/parkour/outputs/omni_plan/phase2/lr/SH_AMq/hwalkstop__omni/seed002/2026-10-14_00-34-23`
- Amask_q es2 hraise__omni: ok `/home/nickmagus/krabby/krabby-research/parkour/outputs/omni_plan/phase2/lr/SH_AMq/hraise__omni/seed002/2026-10-14_00-45-41`
- Amask es1 hwalkstop__omni: ok `/home/nickmagus/krabby/krabby-research/parkour/outputs/omni_plan/phase2/lr/SH_AMq/hwalkstop__omni/seed001/2026-10-13_22-18-32`
- Amask es1 hraise__omni: ok `/home/nickmagus/krabby/krabby-research/parkour/outputs/omni_plan/phase2/lr/SH_AMq/hraise__omni/seed001/2026-10-14_00-41-14`
- Amask es2 hwalkstop__omni: ok `/home/nickmagus/krabby/krabby-research/parkour/outputs/omni_plan/phase2/lr/SH_AMq/hwalkstop__omni/seed002/2026-10-14_00-34-23`
- Amask es2 hraise__omni: ok `/home/nickmagus/krabby/krabby-research/parkour/outputs/omni_plan/phase2/lr/SH_AMq/hraise__omni/seed002/2026-10-14_00-45-41`
- no_restore es1 hwalkstop__omni: ok `/home/nickmagus/krabby/krabby-research/parkour/outputs/omni_plan/phase2/lr/SH_NR/hwalkstop__omni/seed001/2026-10-13_23-04-00`
- no_restore es1 hraise__omni: ok `/home/nickmagus/krabby/krabby-research/parkour/outputs/omni_plan/phase2/lr/SH_NR/hraise__omni/seed001/2026-10-13_23-17-35`
- no_restore es2 hwalkstop__omni: ok `/home/nickmagus/krabby/krabby-research/parkour/outputs/omni_plan/phase2/lr/SH_NR/hwalkstop__omni/seed002/2026-10-13_23-10-50`
- no_restore es2 hraise__omni: ok `/home/nickmagus/krabby/krabby-research/parkour/outputs/omni_plan/phase2/lr/SH_NR/hraise__omni/seed002/2026-10-13_23-22-02`
- S1 es1 hwalkstop__omni: missing `None`
- S1 es1 hraise__omni: missing `None`
- S1 es2 hwalkstop__omni: missing `None`
- S1 es2 hraise__omni: missing `None`
- SH_T4R18_sh_s17 es1 hwalkstop__omni: ok `/home/nickmagus/krabby/krabby-research/parkour/outputs/omni_plan/phase2/lr/SH_T4R18_sh_s17/hwalkstop__omni/seed001/2026-10-21_18-16-44`
- SH_T4R18_sh_s17 es1 hraise__omni: ok `/home/nickmagus/krabby/krabby-research/parkour/outputs/omni_plan/phase2/lr/SH_T4R18_sh_s17/hraise__omni/seed001/2026-10-21_18-30-38`
- SH_T4R18_sh_s17 es2 hwalkstop__omni: ok `/home/nickmagus/krabby/krabby-research/parkour/outputs/omni_plan/phase2/lr/SH_T4R18_sh_s17/hwalkstop__omni/seed002/2026-10-21_18-23-41`
- SH_T4R18_sh_s17 es2 hraise__omni: ok `/home/nickmagus/krabby/krabby-research/parkour/outputs/omni_plan/phase2/lr/SH_T4R18_sh_s17/hraise__omni/seed002/2026-10-21_18-35-11`

Full per-clause values, per-bin reads and stamp paths: the JSON.
