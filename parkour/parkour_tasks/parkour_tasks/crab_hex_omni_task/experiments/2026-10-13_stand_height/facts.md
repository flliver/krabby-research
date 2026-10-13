# Stand height -- step 1 facts (CPU, existing raws)

Generated 2026-10-13T14:54:40-04:00 by `scripts/sh_facts.py` (posture map: `scripts/sh_kin.py`). Full record with methods: `facts.json`.

## STOP flags

| flag | value | evidence (short) |
|---|---|---|
| `stop_model_error` | **True** | pooled four walkstop stamps, settled stands at h_meas 1.0625 m: six-leg mean measured tau_hip 172.4 N*m, tau_knee -74.4 N*m; exploration static model (no leg weight) at the measured posture 116.3 / -103.4 N*m -> errors -32.5 % hip / -39.0 % knee; with leg weight 100.3 / -101.5 -> -41.8 % / -36.4 %. Tolerance 10 % on the joint torques: exceeded on BOTH joints by both models. The split between hip and knee is controller-dependent (internal horizontal stance forces), but the weight-fixed sum tau_hip - tau_knee is 12-22 % above statics on every stamp, which no foot force can produce (per unit foot load: measured hip + |knee| slope sum 0.714 m vs the geometric 0.584 m) -- an unexplained term in the sim torque reading that lands entirely on F_hip. |
| `stop_h_max_low` | **True** | measured six-leg mean |F_hip| 2064 N (pooled) already exceeds the 2000 N cap at the walking height; model scaled to the measurement (x1.162) -> h_max 1.020 m (< 1.10). Unscaled exploration model h_max 1.22 m; leg-weight model unscaled 1.342 m; foot-load-based model 1.346 m. |
| `stop_coupling_sign_ambiguous` | **False** | dLh/dth, dLk/dtk and dLk/dth (signed) match central finite differences of the rod-length functions to 1.0e-10 / 1.4e-10 / 1.6e-10 m/rad and the torch linkage module (knee_hip_coupling) to 2.8e-17; dLh/dth in [0.0501, 0.1204] and dLk/dtk in [0.0062, 0.0762] m/rad over the whole ROM grid (both > 0, the action term's own feedforward relies on it); coupling dLk/dth in [0.0081, 0.0869] m/rad over the stance range h_min..1.552 m (> 0; it crosses zero only at 0.685 m, below h_min, near the kinematic floor); the six-leg mean measured tau_hip > 0 and tau_knee (left conv.) < 0 on >= 1.0000 / 0.9453 of settled rows per stamp (loaded legs only: >= 0.958 / 0.887) -> F_hip pushes, F_knee pulls; with the coupling > 0 the knee pull ADDS to the hip rod force (44 % of it at 1.064). |

## (a) Posture map q(h)

- Anchor: h_model = h - H_OFFSET, H_OFFSET = -0.0009 m (tilted-pad rigid model gives 1.0649 m at a = 0; the flat-pad reading gives 1.0561 m). q(1.064) = (0.0, 0.0). Toe reach U0 = 0.3742 m.
- Kinematic floor (stance hip at the -45 deg stop, constant reach): 0.6700 m. Top of the constant-reach family: 1.552 m (knee_stop_lo).
- **h_min = 0.719 m** = max(floor + 0.02 = 0.6900, hip room 0.125 rad = 0.7188) -> binding: hip_room; q(h_min) = (-2.748, +0.371).
- q over 0.670-1.552 m (1 mm grid): hip strictly rising True; knee strictly falling up to 1.499 m (it turns only there, far above any h_max candidate), so q is monotone and smooth on [h_min, h_max]; inside +-4.8: True; max second difference 6.1e-05 / 6.1e-05 units per mm^2.
- Limits (action units): hip stop -3.250 / +4.080; hip rod window -3.510 / +5.060; knee rod window (left) -7.032 / +5.534; raw clip +-4.8.

| h (m) | hip (deg) | knee L (deg) | q_hip | q_knee L | hip room (rad) | toe rise hip-only (cm) | toe rise R lift (cm) |
|---|---|---|---|---|---|---|---|
| 0.670 | -45.00 | 52.12 | -3.250 | +0.525 | -0.000 | 2.3 | 4.9 |
| 0.700 | -40.53 | 49.82 | -2.938 | +0.424 | 0.078 | 4.6 | 6.6 |
| 0.719 | -37.81 | 48.33 | -2.748 | +0.371 | 0.126 | 4.6 | 7.1 |
| 0.750 | -33.54 | 45.86 | -2.450 | +0.302 | 0.200 | 4.7 | 7.8 |
| 0.800 | -27.06 | 41.76 | -1.998 | +0.222 | 0.313 | 5.1 | 8.8 |
| 0.850 | -21.02 | 37.53 | -1.576 | +0.166 | 0.419 | 5.5 | 9.5 |
| 0.900 | -15.33 | 33.16 | -1.179 | +0.124 | 0.518 | 5.9 | 10.0 |
| 0.950 | -9.94 | 28.64 | -0.803 | +0.088 | 0.612 | 6.3 | 10.4 |
| 1.000 | -4.78 | 23.98 | -0.443 | +0.052 | 0.702 | 6.6 | 10.6 |
| 1.030 | -1.78 | 21.11 | -0.233 | +0.029 | 0.754 | 6.7 | 10.6 |
| 1.064 | 1.56 | 17.77 | +0.000 | +0.000 | 0.813 | 6.8 | 10.7 |
| 1.100 | 5.03 | 14.15 | +0.242 | -0.034 | 0.873 | 7.0 | 10.7 |
| 1.150 | 9.78 | 8.94 | +0.574 | -0.086 | 0.956 | 7.0 | 10.6 |
| 1.200 | 14.49 | 3.49 | +0.903 | -0.145 | 1.038 | 7.1 | 10.4 |
| 1.250 | 19.21 | -2.24 | +1.232 | -0.210 | 1.121 | 7.1 | 10.3 |
| 1.300 | 23.98 | -8.31 | +1.566 | -0.280 | 1.204 | 7.0 | 10.0 |
| 1.350 | 28.89 | -14.81 | +1.908 | -0.349 | 1.290 | 6.9 | 9.7 |
| 1.400 | 34.01 | -21.87 | +2.265 | -0.414 | 1.379 | 6.7 | 9.3 |

## (b) Forces (coupled linkage conversion)

| stamp | rows | h_meas | tau_hip | tau_knee | 6-leg mean abs F_hip | p90 | worst hip leg | 6-leg mean abs F_knee | worst knee leg |
|---|---|---|---|---|---|---|---|---|---|
| SN_Amask_s17/es1 | 66168 | 1.0622 | 158.5 | -88.2 | 2051 | 2104 | FL 2534 | 1242 | FL 1594 |
| SN_Amask_s17/es2 | 65397 | 1.0623 | 158.3 | -88.4 | 2053 | 2107 | MR 2580 | 1245 | FL 1616 |
| R_P2R_RSTg_s18/es1 | 69055 | 1.0625 | 185.3 | -61.4 | 2073 | 2138 | RR 2208 | 1102 | FR 1201 |
| R_P2R_RSTg_s18/es2 | 63274 | 1.0631 | 187.3 | -59.6 | 2081 | 2144 | FL 2272 | 1055 | RR 1129 |
| **pooled** | | 1.0625 | 172.4 | -74.4 | **2064** | | | 1161 | |
| model (exploration, no leg weight) | | | 116.3 | -103.4 | 1776 | | | 1438 | |
| model (with leg weight) | | | 100.3 | -101.5 | 1624 | | | 1411 | |

- Model vs measured joint torques: exploration model hip -32.5 %, knee -39.0 %; with leg weight hip -41.8 %, knee -36.4 % (tolerance 10 %).
- Diagnostic: per-leg regression of torque on foot load -- hip slope 0.533 m vs geometric lever 0.309; knee slope -0.181 vs -0.275; hip + abs(knee) 0.714 vs the geometric 0.584 m that any foot force must satisfy; tau_hip - tau_knee per stamp {"SN_Amask_s17": [246.8, 246.8], "R_P2R_RSTg_s18": [246.7, 246.9]} N*m vs model 219.7 (no leg weight) / 201.7 -> an unexplained term in the sim torque reading, not a missing foot force; F_hip tracks this sum, so the excess lands on the hip rod force (needs a Kit-side check).
- Exploration reference reproduced at 1.064: F_hip 1785 N, F_knee -1423 N (with leg weight 1633 / -1396).
- h_max by curve (2000 N cap): {"exploration_model_unscaled": 1.22, "exploration_model_scaled": 1.02, "leg_weight_model_unscaled": 1.342, "leg_weight_model_scaled": 1.021, "foot_load_model": 1.346}

| h (m) | F_hip model | F_hip scaled | F_hip leg-wt | F_hip leg-wt scaled | F_hip foot-load model | F_knee model | F_knee scaled |
|---|---|---|---|---|---|---|---|
| 0.700 | 1150 | 1337 | 1028 | 1307 | 1025 | 1116 | 901 |
| 0.720 | 1174 | 1365 | 1051 | 1335 | 1047 | 1217 | 983 |
| 0.750 | 1223 | 1422 | 1097 | 1394 | 1093 | 1333 | 1076 |
| 0.800 | 1320 | 1535 | 1190 | 1512 | 1186 | 1452 | 1172 |
| 0.850 | 1421 | 1652 | 1286 | 1635 | 1282 | 1508 | 1218 |
| 0.900 | 1518 | 1764 | 1378 | 1751 | 1374 | 1524 | 1230 |
| 0.950 | 1607 | 1868 | 1463 | 1859 | 1458 | 1511 | 1220 |
| 1.000 | 1689 | 1963 | 1541 | 1959 | 1536 | 1480 | 1195 |
| 1.030 | 1735 | 2017 | 1585 | 2015 | 1580 | 1455 | 1175 |
| 1.064 | 1785 | 2075 | 1633 | 2076 | 1628 | 1423 | 1149 |
| 1.100 | 1836 | 2135 | 1682 | 2137 | 1676 | 1384 | 1118 |
| 1.150 | 1905 | 2214 | 1747 | 2220 | 1742 | 1325 | 1070 |
| 1.200 | 1972 | 2293 | 1811 | 2302 | 1805 | 1260 | 1018 |
| 1.250 | 2040 | 2371 | 1875 | 2384 | 1870 | 1191 | 961 |
| 1.300 | 2109 | 2451 | 1941 | 2468 | 1935 | 1115 | 900 |
| 1.350 | 2181 | 2535 | 2011 | 2555 | 2004 | 1031 | 832 |
| 1.400 | 2257 | 2624 | 2083 | 2648 | 2077 | 934 | 754 |

## (c) Table and ground starts

- Primary (h_max 1.02): top_min 0.412 m, top_max (train) 0.756 m (restore bound 0.805, h_max - 0.264 = 0.756); eval max 0.710; K3 tops [0.4572, 0.635, 0.6858].
- If h_max came from the foot-load model (1.346): top_max (train) 0.805, eval max 0.805, K3 tops [0.4572, 0.635, 0.762].

| pose | min table top (legs >= 0.02 m above floor) | lowest leg part | ground: first contact | ground spawn root z = h_meas | below h_min |
|---|---|---|---|---|---|
| rod_min | 0.162 | femur+hip_plate | femur+hip_plate | 0.311 | True |
| knee_folded | 0.412 | tibia_or_pad | tibia_or_pad | 0.561 | True |
| random folds (9x9 grid) | 0.162-0.651 | | {'femur+hip_plate': 14, 'tibia_or_pad': 67} | 0.311-0.800 | 81% of the grid |

- ground starts: h_meas at spawn (root z - floor) is below h_min for rod_min, knee_folded and most random folds (the rest spawn above it but below 1.064), so the limiter state starts at the measured height and both q(h) and the limiter CLAMP to [h_min, h_max] (q is evaluated at max(h, h_min)). rod_min rests on the hip plates' bottoms (the femur pivot hull sits at the same z): the hip-contact > 500 N clause fires at spawn -> the PLAN H5 wrapper; knee_folded rests on its toes (pads). Random folds: 17 % of the grid rest on the plates, the rest on the toes; spawn root z must be computed per draw (lowest leg-plane hull point, sn_table_core.tibia_hull / plate / femur stadia, + 0.005 m).
