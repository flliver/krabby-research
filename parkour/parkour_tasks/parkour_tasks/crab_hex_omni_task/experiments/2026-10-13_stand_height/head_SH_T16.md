# Stand height -- K1-K6 read (PLAN.md H7)

Generated 2026-10-14T21:10:34-04:00 by `scripts/sh_read.py` (sha256 1a62ef5ad64d; CPU, read-only). Env seeds es1, es2; height rows hwalkstop__omni -> ['K1', 'K4', 'K6'], hraise__omni -> ['K2', 'K4']. Statistics as pinned in the module docstring.

**Control table: 14 / 14 cells land, 0 mismatch(es), 0 pending. Exit 4.**

**R_h stand-up premise (sh_table_core.rh_floor: STOOD >= 0.9 and falls <= 1 per 100 on each gated top ['table_0p635', 'table_0p6985', 'table_0p762'] pooled over the gated poses ['knee_folded', 'random']; the reported cells are listed, never a miss):** MISS in es1 top table_0p635

**K3 cells:** GATED (6; the K3 line and the R_h floor read these only): table_0p635__knee_folded, table_0p635__random, table_0p6985__knee_folded, table_0p6985__random, table_0p762__knee_folded, table_0p762__random. REPORTED (9; run and read, never gated): table_0p4572__knee_folded, table_0p4572__rod_min, table_0p4572__random, table_0p635__rod_min, table_0p6985__rod_min, table_0p762__rod_min, ground__knee_folded, ground__rod_min, ground__random.

**K3 stand checkpoints (the one-checkpoint rule reads the gated cells only):**

- Amask_old es1: gated cells' stand checkpoint ['769cf2533140']; reported cells' ['769cf2533140']
- Amask_old es2: gated cells' stand checkpoint ['769cf2533140']; reported cells' ['769cf2533140']
- Amask es1: gated cells' stand checkpoint ['769cf2533140']; reported cells' ['769cf2533140']
- Amask es2: gated cells' stand checkpoint ['769cf2533140']; reported cells' ['769cf2533140']
- SH_T16 es1: gated cells' stand checkpoint ['230cc4973bea']; reported cells' ['230cc4973bea']
- SH_T16 es2: gated cells' stand checkpoint ['230cc4973bea']; reported cells' ['230cc4973bea']

**h_max re-check (R_h top bin 1.221-1.346):** ok -- foot-load six-leg mean |F_hip| 1912 N vs cap 2000 N (PD reading 2457 N, 122689 rows, mean h 1.279 m)
  Reported beside it (not the decision statistic): within 0.02 m of h_max, 1994 N (PD 2582 N, 65948 rows).

## Heads

| head | K1 | K2 | K3 | K4 | K5 | K6 | all PASS |
|---|---|---|---|---|---|---|---|
| SH_T16 | PASS | PASS | PASS | FAIL | PASS | PASS | False |

- SH_T16: residual deadband 0.25 (one per head; absent record = 0); trained at 0.25 (null = not recorded)

## Gates

| gate | role | expected | observed | lands | values |
|---|---|---|---|---|---|
| K1 TRACKING | R_h | PASS | PASS | yes | median es1 0.003; p90 es1 0.006289; median es1 0.001849; p90 es1 0.004715; median es1 0.001751; p90 es1 0.00364; median es1 0.001598; p90 es1 0.003591; median es1 0.001519; p90 es1 0.003572; median es2 0.002487; p90 es2 0.005868; median es2 0.001767; p90 es2 0.004354; median es2 0.001554; p90 es2 0.003171; median es2 0.001817; p90 es2 0.004027; median es2 0.001524; p90 es2 0.003604 |
| K1 TRACKING | R_old | FAIL | FAIL | yes | **median es1 0.2952 (line 0.02)**; **p90 es1 0.3469 (line 0.04)**; **median es1 0.1433 (line 0.02)**; **p90 es1 0.1458 (line 0.04)**; median es1 0.001703; p90 es1 0.003603; **median es1 0.08737 (line 0.02)**; **p90 es1 0.08934 (line 0.04)**; **median es1 0.2359 (line 0.02)**; **p90 es1 0.2849 (line 0.04)**; **median es2 0.2956 (line 0.02)**; **p90 es2 0.3465 (line 0.04)**; **median es2 0.1432 (line 0.02)**; **p90 es2 0.1472 (line 0.04)**; median es2 0.001585; p90 es2 0.003523; **median es2 0.08694 (line 0.02)**; **p90 es2 0.08954 (line 0.04)**; **median es2 0.2336 (line 0.02)**; **p90 es2 0.2848 (line 0.04)** |
| K1 TRACKING | Amask_old | FAIL | FAIL | yes | **median es1 0.2941 (line 0.02)**; **p90 es1 0.3469 (line 0.04)**; **median es1 0.1426 (line 0.02)**; **p90 es1 0.1456 (line 0.04)**; median es1 0.002129; p90 es1 0.004839; **median es1 0.08732 (line 0.02)**; **p90 es1 0.08998 (line 0.04)**; **median es1 0.2371 (line 0.02)**; **p90 es1 0.2857 (line 0.04)**; **median es2 0.2941 (line 0.02)**; **p90 es2 0.3463 (line 0.04)**; **median es2 0.142 (line 0.02)**; **p90 es2 0.1465 (line 0.04)**; median es2 0.001712; p90 es2 0.003739; **median es2 0.08734 (line 0.02)**; **p90 es2 0.08988 (line 0.04)**; **median es2 0.2336 (line 0.02)**; **p90 es2 0.286 (line 0.04)** |
| K1 TRACKING | posture | -- | PASS | reported | median es1 0.002608; p90 es1 0.005795; median es1 0.001748; p90 es1 0.003741; median es1 0.001201; p90 es1 0.003796; median es1 0.001617; p90 es1 0.00393; median es1 0.001657; p90 es1 0.003734; median es2 0.00262; p90 es2 0.005829; median es2 0.001747; p90 es2 0.004433; median es2 0.001697; p90 es2 0.004112; median es2 0.001748; p90 es2 0.004648; median es2 0.001526; p90 es2 0.003647 |
| K1 TRACKING | Amask_q | -- | FAIL | reported | **median es1 0.0238 (line 0.02)**; p90 es1 0.03229; median es1 0.01145; p90 es1 0.01787; median es1 0.00169; p90 es1 0.004817; **median es1 0.03729 (line 0.02)**; **p90 es1 0.06556 (line 0.04)**; **median es1 0.08745 (line 0.02)**; **p90 es1 0.1018 (line 0.04)**; **median es2 0.02322 (line 0.02)**; p90 es2 0.03086; median es2 0.01099; p90 es2 0.01735; median es2 0.001789; p90 es2 0.004632; **median es2 0.03811 (line 0.02)**; **p90 es2 0.0632 (line 0.04)**; **median es2 0.08786 (line 0.02)**; **p90 es2 0.1009 (line 0.04)** |
| K1 TRACKING | SH_T16 | -- | PASS | head | median es1 0.00248; p90 es1 0.005665; median es1 0.00143; p90 es1 0.004174; median es1 0.00166; p90 es1 0.003826; median es1 0.001736; p90 es1 0.003949; median es1 0.001471; p90 es1 0.003641; median es2 0.002596; p90 es2 0.005668; median es2 0.00168; p90 es2 0.004214; median es2 0.001466; p90 es2 0.003833; median es2 0.001641; p90 es2 0.004061; median es2 0.001388; p90 es2 0.003251 |
| K2 RAISE / LOWER | R_h | PASS | PASS | yes | reach es1 -1.165; stand falls es1 0; reach es2 -1.17; stand falls es2 0 |
| K2 RAISE / LOWER | R_old | FAIL | FAIL | yes | **reach es1 inf (line 0)**; stand falls es1 0; **reach es2 inf (line 0)**; stand falls es2 0 |
| K2 RAISE / LOWER | Amask_old | FAIL | FAIL | yes | **reach es1 inf (line 0)**; stand falls es1 0; **reach es2 inf (line 0)**; stand falls es2 0 |
| K2 RAISE / LOWER | posture | -- | PASS | reported | reach es1 -1.17; stand falls es1 0; reach es2 -1.165; stand falls es2 0 |
| K2 RAISE / LOWER | Amask_q | -- | FAIL | reported | **reach es1 inf (line 0)**; stand falls es1 0; **reach es2 inf (line 0)**; stand falls es2 0 |
| K2 RAISE / LOWER | SH_T16 | -- | PASS | head | reach es1 -1.17; stand falls es1 0; reach es2 -1.16; stand falls es2 0 |
| K3 STAND-UP | R_h | PASS | PASS | yes | stood tables es1 0.9833; falls tables es1 0.8333; stood tables es2 0.9783; falls tables es2 0.3333 |
| K3 STAND-UP | posture | -- | PASS | reported | stood tables es1 0.995; falls tables es1 0; stood tables es2 0.9783; falls tables es2 0.1667 |
| K3 STAND-UP | Amask | -- | FAIL | reported | **stood tables es1 0.13 (line 0.9333)**; **falls tables es1 1.333 (line 1)**; **stood tables es2 0.1617 (line 0.9283)**; **falls tables es2 1.667 (line 1)** |
| K3 STAND-UP | S1 | FAIL | FAIL | yes | **stood tables es1 0 (line 0.9333)**; **falls tables es1 3.667 (line 1)**; **stood tables es2 0 (line 0.9283)**; **falls tables es2 3.167 (line 1)** |
| K3 STAND-UP | SH_T16 | -- | PASS | head | stood tables es1 0.99; falls tables es1 0.5; stood tables es2 0.98; falls tables es2 0.3333 |
| K4 RESTORE + HAND-BACK | R_h | PASS | PASS | yes | miss es1 1; exit falls es1 0; onset es1 -1.305; miss es2 1; exit falls es2 0; onset es2 -1.305 |
| K4 RESTORE + HAND-BACK | Amask | -- | PASS | reported | miss es1 0; exit falls es1 1; onset es1 -0.125; miss es2 0; exit falls es2 0; onset es2 -0.116 |
| K4 RESTORE + HAND-BACK | no_restore | FAIL | FAIL | yes | **miss es1 788 (line 1)**; exit falls es1 0; onset es1 -1.54; **miss es2 784 (line 1)**; exit falls es2 0; onset es2 -1.54 |
| K4 RESTORE + HAND-BACK | SH_T16 | -- | FAIL | head | miss es1 1; exit falls es1 0; onset es1 -1.305; **miss es2 2 (line 1)**; exit falls es2 0; onset es2 -1.305 |
| K5 NON-REGRESSION @1.064 | R_h | PASS | PASS | yes | G1 PASS; G2 PASS; G3 PASS; G4 PASS; G5 PASS |
| K5 NON-REGRESSION @1.064 | Amask | PASS | PASS | yes | G1 PASS; G2 PASS; G3 PASS; G4 PASS; G5 PASS |
| K5 NON-REGRESSION @1.064 | SH_T16 | -- | PASS | head | G1 PASS; G2 PASS; G3 PASS; G4 PASS; G5 PASS |
| K6 HOMING + STILLNESS | R_h | PASS | PASS | yes | homed es1 1; settled es1 1; homed es1 0.93; settled es1 0.93; homed es1 0.98; settled es1 0.98; homed es1 1; settled es1 1; homed es1 0.995; settled es1 0.995; still steps es1 0.1651; still taps es1 0.1473; homed es2 1; settled es2 1; homed es2 0.97; settled es2 0.97; homed es2 0.99; settled es2 0.99; homed es2 0.98; settled es2 0.98; homed es2 0.955; settled es2 0.955; still steps es2 0.1625; still taps es2 0.1473 |
| K6 HOMING + STILLNESS | posture | FAIL | FAIL | yes | **homed es1 0 (line 0.95)**; **settled es1 0 (line 0.95)**; **homed es1 0 (line 0.879)**; **settled es1 0 (line 0.879)**; **homed es1 0 (line 0.93)**; **settled es1 0 (line 0.93)**; **homed es1 0 (line 0.95)**; **settled es1 0 (line 0.95)**; **homed es1 0.005 (line 0.945)**; **settled es1 0.005 (line 0.945)**; still steps es1 0; still taps es1 0; **homed es2 0 (line 0.95)**; **settled es2 0 (line 0.95)**; **homed es2 0 (line 0.92)**; **settled es2 0 (line 0.92)**; **homed es2 0 (line 0.94)**; **settled es2 0 (line 0.94)**; **homed es2 0 (line 0.93)**; **settled es2 0 (line 0.93)**; **homed es2 0.005 (line 0.905)**; **settled es2 0.005 (line 0.905)**; still steps es2 0.2024; still taps es2 0.3036 |
| K6 HOMING + STILLNESS | Amask | -- | FAIL | reported | **homed es1 0 (line 0.95)**; **settled es1 0 (line 0.95)**; **homed es1 0 (line 0.879)**; **settled es1 0.299 (line 0.8239)**; homed es1 1; settled es1 1; **homed es1 0 (line 0.95)**; **settled es1 0 (line 0.95)**; **homed es1 0 (line 0.945)**; **settled es1 0.2172 (line 0.9356)**; **still steps es1 0.8075 (line 0.5)**; **still taps es1 2.06 (line 1)**; **homed es2 0 (line 0.95)**; **settled es2 0 (line 0.95)**; **homed es2 0 (line 0.92)**; **settled es2 0.3434 (line 0.8686)**; homed es2 1; settled es2 1; **homed es2 0 (line 0.93)**; **settled es2 0 (line 0.93)**; **homed es2 0 (line 0.905)**; **settled es2 0.215 (line 0.8899)**; **still steps es2 0.7134 (line 0.5)**; **still taps es2 1.673 (line 1)** |
| K6 HOMING + STILLNESS | SH_T16 | -- | PASS | head | homed es1 1; settled es1 1; homed es1 1; settled es1 1; homed es1 0.99; settled es1 1; homed es1 0.99; settled es1 0.99; homed es1 0.995; settled es1 0.995; still steps es1 0.1586; still taps es1 0.1876; homed es2 0.995; settled es2 0.995; homed es2 0.98; settled es2 0.98; homed es2 1; settled es2 1; homed es2 0.99; settled es2 0.99; homed es2 0.995; settled es2 1; still steps es2 0.1352; still taps es2 0.1746 |

## Pending

- none

## INVALID cells

- none

## Forces per height bin (reported; foot-load statics = the cap statistic, PD torque alongside)

| role | bin | rows | h_meas | F_hip 6-leg mean foot-load (N) | p90 | PD reading (N) | worst hip leg (foot-load) | F_knee foot-load / PD (N) | over cap |
|---|---|---|---|---|---|---|---|---|---|
| R_h | 0.719-0.844 | 71117 | 0.7771 | 1140 | 1243 | 1515 | ML 1336 | 1336 / 1323 | False |
| R_h | 0.844-0.970 | 65297 | 0.9062 | 1394 | 1507 | 1783 | ML 1518 | 1482 / 1288 | False |
| R_h | 0.970-1.095 | 50606 | 1.063 | 1609 | 1674 | 2082 | ML 1813 | 1372 / 1009 | False |
| R_h | 1.095-1.221 | 66669 | 1.149 | 1738 | 1803 | 2233 | ML 1903 | 1281 / 1057 | False |
| R_h | 1.221-1.346 | 122689 | 1.279 | 1912 | 2018 | 2457 | MR 2068 | 1107 / 1125 | False |
| R_h | outside [h_min, h_max] | 39948 | 1.295 | 1907 | 2050 | 2484 | MR 2232 | 1024 / 1167 | False |
| R_old | 0.970-1.095 | 704244 | 1.063 | 1621 | 1686 | 2081 | MR 1717 | 1375 / 1081 | False |
| Amask_old | 0.970-1.095 | 727695 | 1.062 | 1613 | 1668 | 2052 | MR 2074 | 1403 / 1234 | False |
| posture | 0.719-0.844 | 68404 | 0.7744 | 1192 | 1322 | 1640 | ML 1423 | 1240 / 1608 | False |
| posture | 0.844-0.970 | 53449 | 0.911 | 1436 | 1535 | 1821 | ML 1632 | 1450 / 1628 | False |
| posture | 0.970-1.095 | 48058 | 1.063 | 1675 | 1761 | 2125 | ML 1978 | 1331 / 1329 | False |
| posture | 1.095-1.221 | 58693 | 1.149 | 1778 | 1856 | 2277 | ML 1897 | 1251 / 1254 | False |
| posture | 1.221-1.346 | 121743 | 1.284 | 1938 | 2053 | 2472 | ML 2222 | 1092 / 1208 | False |
| posture | outside [h_min, h_max] | 31407 | 1.304 | 1947 | 2084 | 2495 | ML 2448 | 1013 / 1337 | False |
| Amask_q | 0.719-0.844 | 1864 | 0.7876 | 1223 | 1311 | 1417 | RR 3062 | 1298 / 2134 | False |
| Amask_q | 0.844-0.970 | 81 | 0.8603 | 1303 | 1415 | 1559 | RR 3554 | 1540 / 2242 | False |
| Amask_q | 0.970-1.095 | 46535 | 1.062 | 1615 | 1669 | 2053 | FL 2031 | 1407 / 1253 | False |
| Amask_q | 1.095-1.221 | 21398 | 1.167 | 1843 | 1930 | 2371 | FR 4381 | 1557 / 1094 | False |
| Amask_q | 1.221-1.346 | 4381 | 1.245 | 1964 | 2080 | 2503 | FR 5146 | 1525 / 1262 | False |
| Amask | 0.719-0.844 | 1864 | 0.7876 | 1223 | 1311 | 1417 | RR 3062 | 1298 / 2134 | False |
| Amask | 0.844-0.970 | 81 | 0.8603 | 1303 | 1415 | 1559 | RR 3554 | 1540 / 2242 | False |
| Amask | 0.970-1.095 | 46535 | 1.062 | 1615 | 1669 | 2053 | FL 2031 | 1407 / 1253 | False |
| Amask | 1.095-1.221 | 21398 | 1.167 | 1843 | 1930 | 2371 | FR 4381 | 1557 / 1094 | False |
| Amask | 1.221-1.346 | 4381 | 1.245 | 1964 | 2080 | 2503 | FR 5146 | 1525 / 1262 | False |
| no_restore | 0.719-0.844 | 70662 | 0.7774 | 1139 | 1247 | 1520 | MR 1268 | 1333 / 1298 | False |
| no_restore | 0.844-0.970 | 65204 | 0.9061 | 1394 | 1498 | 1788 | ML 1539 | 1473 / 1250 | False |
| no_restore | 0.970-1.095 | 43863 | 1.063 | 1613 | 1676 | 2085 | RR 1802 | 1373 / 1019 | False |
| no_restore | 1.095-1.221 | 59463 | 1.149 | 1741 | 1809 | 2231 | ML 2009 | 1286 / 1073 | False |
| no_restore | 1.221-1.346 | 114265 | 1.282 | 1918 | 2029 | 2466 | ML 2066 | 1103 / 1112 | False |
| no_restore | outside [h_min, h_max] | 38049 | 1.292 | 1906 | 2046 | 2485 | MR 2169 | 1023 / 1119 | False |
| SH_T16 | 0.719-0.844 | 77922 | 0.7696 | 1123 | 1243 | 1443 | MR 1234 | 1349 / 1562 | False |
| SH_T16 | 0.844-0.970 | 61857 | 0.907 | 1397 | 1519 | 1786 | ML 1468 | 1475 / 1275 | False |
| SH_T16 | 0.970-1.095 | 53288 | 1.063 | 1612 | 1675 | 2073 | ML 1662 | 1380 / 1075 | False |
| SH_T16 | 1.095-1.221 | 69848 | 1.148 | 1745 | 1820 | 2220 | FR 1792 | 1292 / 1100 | False |
| SH_T16 | 1.221-1.346 | 127274 | 1.281 | 1919 | 2027 | 2464 | MR 2087 | 1104 / 1116 | False |
| SH_T16 | outside [h_min, h_max] | 38536 | 1.328 | 1959 | 2051 | 2536 | FL 2126 | 1021 / 1183 | False |

## Reads

- R_h es1 hwalkstop__omni: ok `/home/nickmagus/krabby/krabby-research/parkour/outputs/omni_plan/phase2/lr/SH_RH/hwalkstop__omni/seed001/2026-10-13_22-41-45`
- R_h es1 hraise__omni: ok `/home/nickmagus/krabby/krabby-research/parkour/outputs/omni_plan/phase2/lr/SH_RH/hraise__omni/seed001/2026-10-13_22-55-24`
- R_h es2 hwalkstop__omni: ok `/home/nickmagus/krabby/krabby-research/parkour/outputs/omni_plan/phase2/lr/SH_RH/hwalkstop__omni/seed002/2026-10-13_22-48-35`
- R_h es2 hraise__omni: ok `/home/nickmagus/krabby/krabby-research/parkour/outputs/omni_plan/phase2/lr/SH_RH/hraise__omni/seed002/2026-10-13_22-59-42`
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
- SH_T16 es1 hwalkstop__omni: ok `/home/nickmagus/krabby/krabby-research/parkour/outputs/omni_plan/phase2/lr/SH_T16/hwalkstop__omni/seed001/2026-10-14_20-45-26`
- SH_T16 es1 hraise__omni: ok `/home/nickmagus/krabby/krabby-research/parkour/outputs/omni_plan/phase2/lr/SH_T16/hraise__omni/seed001/2026-10-14_20-59-08`
- SH_T16 es2 hwalkstop__omni: ok `/home/nickmagus/krabby/krabby-research/parkour/outputs/omni_plan/phase2/lr/SH_T16/hwalkstop__omni/seed002/2026-10-14_20-52-18`
- SH_T16 es2 hraise__omni: ok `/home/nickmagus/krabby/krabby-research/parkour/outputs/omni_plan/phase2/lr/SH_T16/hraise__omni/seed002/2026-10-14_21-03-35`

Full per-clause values, per-bin reads and stamp paths: the JSON.
