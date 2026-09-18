# CHANGELOG — 2026-09-16_1805_omni_formation

### 2026-09-16 23:10 — campaign start: plant A15+B, seed 3, phases 1a, 1b, 1c, 2a, 2b
> NOTIFY: seed3_1a_001_A15pB: soft gate FAILED on hold 'fwd' — tripod 0.452 (need >= 0.3; by hold: back 0.267, diag 0.341, fwd 0.452, lat_l 0.376, move_turn 0.268, stand 0.170, turn_l90 0.234, turn_r90 0.254), completion 0.090 (need >= 0.7) — campaign HALTED (relaunch with --continue to override, or fix and --redo)

### 2026-09-16 23:40 — phase 1a gate_1a: model_999.pt (seed3_1a_001_A15pB)
> NOTIFY: phase 1a gate_1a — campaign HALTED (see REPORT)

### 2026-09-16 23:43 — phase 1a: gate_1a overridden by --continue — model_999.pt carried into 1b unretrained (see CHARTER); record status -> ok

### 2026-09-17 00:13 — phase 1b ok: model_1998.pt (seed3_1b_002_A15pB)
> NOTIFY: seed3_1c_003_A15pB: formation gate FAILED — walking-hold tripod 0.215 (need >= 0.4; by hold: back 0.853, diag 0.832, fwd 0.315, lat_l 0.850, move_turn 0.215, stand 0.000, turn_l90 0.225, turn_r90 0.051), completion 0.000 (need >= 0.7), yaw-progress rate turn_l90 0.005, turn_r90 0.020 rad/s (net turn_l90 0.005, turn_r90 0.004) (need >= 0.05 rad/s on each), per-direction speed_ratio lat_l 0.028, diag 0.147, back 0.330, move_turn -0.318 (need >= 0.3 on each of lat_l, diag, back, move_turn; failing: lat_l, diag, move_turn) on slow__omni — campaign HALTED (relaunch with --continue to override, or fix and --redo)

### 2026-09-17 02:20 — phase 1c gate_1c: model_6997.pt (seed3_1c_003_A15pB)
> NOTIFY: phase 1c gate_1c — campaign HALTED (see REPORT)
> NOTIFY: seed3_1c_004_A15pB: formation gate FAILED — walking-hold tripod 0.489 (need >= 0.4; by hold: back 0.579, diag 0.489, fwd 0.531, lat_l 0.543, move_turn 0.508, stand 0.576, turn_l90 0.542, turn_r90 0.579), completion 0.600 (need >= 0.7), yaw-progress rate turn_l90 0.092, turn_r90 0.098 rad/s (net turn_l90 0.092, turn_r90 0.081) (need >= 0.05 rad/s on each), per-direction speed_ratio lat_l 0.412, diag 0.413, back 0.504, move_turn 0.443 (need >= 0.3 on each of lat_l, diag, back, move_turn; failing: none) on slow__omni, training track_ratio_mean 0.568 (need >= 0.6 over the last 500 its) — campaign HALTED (relaunch with --continue to override, or fix and --redo)

### 2026-09-18 04:37 — phase 1c gate_1c: model_4999.pt (seed3_1c_004_A15pB)
> NOTIFY: phase 1c gate_1c — campaign HALTED (see REPORT)

### 2026-09-18 09:40 — phase 1c: gate_1c overridden by --continue — model_4999.pt carried into 2a unretrained (see CHARTER); record status -> ok
> NOTIFY: seed3_2a_005_A15pB: facing-precision gate could not be evaluated — final heading error turn_l90 n/a, turn_r90 n/a on slow__omni; campaign HALTED (inspect the eval log, then relaunch with --continue to override, or fix and --redo)

### 2026-09-18 12:28 — phase 2a gate_facing: model_9998.pt (seed3_2a_005_A15pB)
> NOTIFY: phase 2a gate_facing — campaign HALTED (see REPORT)

### 2026-09-18 12:58 — phase 2a: gate_facing overridden by --continue — model_9998.pt carried into 2b unretrained (see CHARTER); record status -> ok

### 2026-09-18 15:02 — phase 2b ok: model_14997.pt (seed3_2b_006_A15pB)
> NOTIFY: phases 2a, 2b complete for seed3 — see REPORT.md
