# Stand height -- K1-K6 read (PLAN.md H7)

Generated 2026-11-09T17:17:24-05:00 by `scripts/sh_read.py` (sha256 1a62ef5ad64d; CPU, read-only). Env seeds es1, es2; height rows hwalkstop__omni -> ['K1', 'K4', 'K6'], hraise__omni -> ['K2', 'K4']. Statistics as pinned in the module docstring.

**Control table: 12 / 14 cells land, 2 mismatch(es), 0 pending. Exit 4.**

**Mismatches (step 4: STOP, notify, ask):**

- K5 NON-REGRESSION @1.064 x R_h (R_h (height-aware R)): INVALID (a listed cell cannot be read)
- K5 NON-REGRESSION @1.064 x Amask (A-mask s17): INVALID (a listed cell cannot be read)

**R_h stand-up premise (sh_table_core.rh_floor: STOOD >= 0.9 and falls <= 1 per 100 on each gated top ['table_0p635', 'table_0p6985', 'table_0p762'] pooled over the gated poses ['knee_folded', 'random']; the reported cells are listed, never a miss):** MISS in es1 top table_0p635

**K3 cells:** GATED (6; the K3 line and the R_h floor read these only): table_0p635__knee_folded, table_0p635__random, table_0p6985__knee_folded, table_0p6985__random, table_0p762__knee_folded, table_0p762__random. REPORTED (9; run and read, never gated): table_0p4572__knee_folded, table_0p4572__rod_min, table_0p4572__random, table_0p635__rod_min, table_0p6985__rod_min, table_0p762__rod_min, ground__knee_folded, ground__rod_min, ground__random.

**K3 stand checkpoints (the one-checkpoint rule reads the gated cells only):**

- Amask_old es1: gated cells' stand checkpoint ['769cf2533140']; reported cells' ['769cf2533140']
- Amask_old es2: gated cells' stand checkpoint ['769cf2533140']; reported cells' ['769cf2533140']
- Amask es1: gated cells' stand checkpoint ['769cf2533140']; reported cells' ['769cf2533140']
- Amask es2: gated cells' stand checkpoint ['769cf2533140']; reported cells' ['769cf2533140']
- SH_RSD_TDR_sh_s19 es1: gated cells' stand checkpoint ['89f4428b559f']; reported cells' ['89f4428b559f']
- SH_RSD_TDR_sh_s19 es2: gated cells' stand checkpoint ['89f4428b559f']; reported cells' ['89f4428b559f']

**h_max re-check (R_h top bin 1.221-1.346):** ok -- foot-load six-leg mean |F_hip| 1912 N vs cap 2000 N (PD reading 2465 N, 116468 rows, mean h 1.278 m)
  Reported beside it (not the decision statistic): within 0.02 m of h_max, 1994 N (PD 2607 N, 55798 rows).

## Heads

| head | K1 | K2 | K3 | K4 | K5 | K6 | all PASS |
|---|---|---|---|---|---|---|---|
| SH_RSD_TDR_sh_s19 | PASS | PASS | PASS | FAIL | INVALID | PASS | False |

- SH_RSD_TDR_sh_s19: residual deadband 0.25 (one per head; absent record = 0); trained at 0.25 (null = not recorded)

## Gates

| gate | role | expected | observed | lands | values |
|---|---|---|---|---|---|
| K1 TRACKING | R_h | PASS | PASS | yes | median es1 0.002646; p90 es1 0.006076; median es1 0.002147; p90 es1 0.005174; median es1 0.001382; p90 es1 0.003514; median es1 0.001452; p90 es1 0.003396; median es1 0.001504; p90 es1 0.004282; median es2 0.002745; p90 es2 0.006766; median es2 0.002069; p90 es2 0.005563; median es2 0.001592; p90 es2 0.003498; median es2 0.001585; p90 es2 0.003895; median es2 0.001716; p90 es2 0.004248 |
| K1 TRACKING | R_old | FAIL | FAIL | yes | **median es1 0.2952 (line 0.02)**; **p90 es1 0.3469 (line 0.04)**; **median es1 0.1433 (line 0.02)**; **p90 es1 0.1458 (line 0.04)**; median es1 0.001703; p90 es1 0.003603; **median es1 0.08737 (line 0.02)**; **p90 es1 0.08934 (line 0.04)**; **median es1 0.2359 (line 0.02)**; **p90 es1 0.2849 (line 0.04)**; **median es2 0.2956 (line 0.02)**; **p90 es2 0.3465 (line 0.04)**; **median es2 0.1432 (line 0.02)**; **p90 es2 0.1472 (line 0.04)**; median es2 0.001585; p90 es2 0.003523; **median es2 0.08694 (line 0.02)**; **p90 es2 0.08954 (line 0.04)**; **median es2 0.2336 (line 0.02)**; **p90 es2 0.2848 (line 0.04)** |
| K1 TRACKING | Amask_old | FAIL | FAIL | yes | **median es1 0.2941 (line 0.02)**; **p90 es1 0.3469 (line 0.04)**; **median es1 0.1426 (line 0.02)**; **p90 es1 0.1456 (line 0.04)**; median es1 0.002129; p90 es1 0.004839; **median es1 0.08732 (line 0.02)**; **p90 es1 0.08998 (line 0.04)**; **median es1 0.2371 (line 0.02)**; **p90 es1 0.2857 (line 0.04)**; **median es2 0.2941 (line 0.02)**; **p90 es2 0.3463 (line 0.04)**; **median es2 0.142 (line 0.02)**; **p90 es2 0.1465 (line 0.04)**; median es2 0.001712; p90 es2 0.003739; **median es2 0.08734 (line 0.02)**; **p90 es2 0.08988 (line 0.04)**; **median es2 0.2336 (line 0.02)**; **p90 es2 0.286 (line 0.04)** |
| K1 TRACKING | posture | -- | PASS | reported | median es1 0.002608; p90 es1 0.005795; median es1 0.001748; p90 es1 0.003741; median es1 0.001201; p90 es1 0.003796; median es1 0.001617; p90 es1 0.00393; median es1 0.001657; p90 es1 0.003734; median es2 0.00262; p90 es2 0.005829; median es2 0.001747; p90 es2 0.004433; median es2 0.001697; p90 es2 0.004112; median es2 0.001748; p90 es2 0.004648; median es2 0.001526; p90 es2 0.003647 |
| K1 TRACKING | Amask_q | -- | FAIL | reported | **median es1 0.0238 (line 0.02)**; p90 es1 0.03229; median es1 0.01145; p90 es1 0.01787; median es1 0.00169; p90 es1 0.004817; **median es1 0.03729 (line 0.02)**; **p90 es1 0.06556 (line 0.04)**; **median es1 0.08745 (line 0.02)**; **p90 es1 0.1018 (line 0.04)**; **median es2 0.02322 (line 0.02)**; p90 es2 0.03086; median es2 0.01099; p90 es2 0.01735; median es2 0.001789; p90 es2 0.004632; **median es2 0.03811 (line 0.02)**; **p90 es2 0.0632 (line 0.04)**; **median es2 0.08786 (line 0.02)**; **p90 es2 0.1009 (line 0.04)** |
| K1 TRACKING | SH_RSD_TDR_sh_s19 | -- | PASS | head | median es1 0.00238; p90 es1 0.005722; median es1 0.001658; p90 es1 0.004215; median es1 0.001682; p90 es1 0.003763; median es1 0.001503; p90 es1 0.003295; median es1 0.001627; p90 es1 0.004368; median es2 0.002756; p90 es2 0.006221; median es2 0.001693; p90 es2 0.004474; median es2 0.001601; p90 es2 0.004191; median es2 0.001853; p90 es2 0.003971; median es2 0.001596; p90 es2 0.003604 |
| K2 RAISE / LOWER | R_h | PASS | PASS | yes | reach es1 -1.16; stand falls es1 0; reach es2 -1.16; stand falls es2 0 |
| K2 RAISE / LOWER | R_old | FAIL | FAIL | yes | **reach es1 inf (line 0)**; stand falls es1 0; **reach es2 inf (line 0)**; stand falls es2 0 |
| K2 RAISE / LOWER | Amask_old | FAIL | FAIL | yes | **reach es1 inf (line 0)**; stand falls es1 0; **reach es2 inf (line 0)**; stand falls es2 0 |
| K2 RAISE / LOWER | posture | -- | PASS | reported | reach es1 -1.17; stand falls es1 0; reach es2 -1.165; stand falls es2 0 |
| K2 RAISE / LOWER | Amask_q | -- | FAIL | reported | **reach es1 inf (line 0)**; stand falls es1 0; **reach es2 inf (line 0)**; stand falls es2 0 |
| K2 RAISE / LOWER | SH_RSD_TDR_sh_s19 | -- | PASS | head | reach es1 -1.17; stand falls es1 0; reach es2 -1.165; stand falls es2 0 |
| K3 STAND-UP | R_h | PASS | PASS | yes | stood tables es1 0.9833; falls tables es1 0.8333; stood tables es2 0.9783; falls tables es2 0.3333 |
| K3 STAND-UP | posture | -- | PASS | reported | stood tables es1 0.995; falls tables es1 0; stood tables es2 0.9783; falls tables es2 0.1667 |
| K3 STAND-UP | Amask | -- | FAIL | reported | **stood tables es1 0.13 (line 0.9333)**; **falls tables es1 1.333 (line 1)**; **stood tables es2 0.1617 (line 0.9283)**; **falls tables es2 1.667 (line 1)** |
| K3 STAND-UP | S1 | FAIL | FAIL | yes | **stood tables es1 0 (line 0.9333)**; **falls tables es1 3.667 (line 1)**; **stood tables es2 0 (line 0.9283)**; **falls tables es2 3.167 (line 1)** |
| K3 STAND-UP | SH_RSD_TDR_sh_s19 | -- | PASS | head | stood tables es1 0.99; falls tables es1 0.6667; stood tables es2 0.975; falls tables es2 0.5 |
| K4 RESTORE + HAND-BACK | R_h | PASS | PASS | yes | miss es1 4; exit falls es1 0; onset es1 -1.305; miss es2 5; exit falls es2 0; onset es2 -1.305 |
| K4 RESTORE + HAND-BACK | Amask | -- | PASS | reported | miss es1 0; exit falls es1 1; onset es1 -0.125; miss es2 0; exit falls es2 0; onset es2 -0.116 |
| K4 RESTORE + HAND-BACK | no_restore | FAIL | FAIL | yes | **miss es1 788 (line 4)**; exit falls es1 0; onset es1 -1.54; **miss es2 784 (line 5)**; exit falls es2 0; onset es2 -1.54 |
| K4 RESTORE + HAND-BACK | SH_RSD_TDR_sh_s19 | -- | FAIL | head | **miss es1 5 (line 4)**; exit falls es1 0; onset es1 -1.305; **miss es2 6 (line 5)**; exit falls es2 0; onset es2 -1.305 |
| K5 NON-REGRESSION @1.064 | R_h | PASS | INVALID | NO: INVALID (a listed cell cannot be read) | **G1 INVALID (line PASS)**; **G2 INVALID (line PASS)**; **G3 INVALID (line PASS)**; **G4 INVALID (line PASS)**; **G5 INVALID (line PASS)** |
| K5 NON-REGRESSION @1.064 | Amask | PASS | INVALID | NO: INVALID (a listed cell cannot be read) | **G1 INVALID (line PASS)**; **G2 INVALID (line PASS)**; **G3 INVALID (line PASS)**; **G4 INVALID (line PASS)**; **G5 INVALID (line PASS)** |
| K5 NON-REGRESSION @1.064 | SH_RSD_TDR_sh_s19 | -- | INVALID | head | G1 PASS; **G2 INVALID (line PASS)**; **G3 INVALID (line PASS)**; **G4 INVALID (line PASS)**; **G5 FAIL (line PASS)** |
| K6 HOMING + STILLNESS | R_h | PASS | PASS | yes | homed es1 1; settled es1 1; homed es1 0.67; settled es1 0.67; homed es1 1; settled es1 1; homed es1 0.99; settled es1 0.99; homed es1 0.825; settled es1 0.83; still steps es1 0.2101; still taps es1 0.1535; homed es2 0.995; settled es2 1; homed es2 0.66; settled es2 0.66; homed es2 0.99; settled es2 0.99; homed es2 1; settled es2 1; homed es2 0.87; settled es2 0.87; still steps es2 0.2026; still taps es2 0.1355 |
| K6 HOMING + STILLNESS | posture | FAIL | FAIL | yes | **homed es1 0 (line 0.95)**; **settled es1 0 (line 0.95)**; **homed es1 0 (line 0.576)**; **settled es1 0 (line 0.576)**; **homed es1 0 (line 0.95)**; **settled es1 0 (line 0.95)**; **homed es1 0 (line 0.94)**; **settled es1 0 (line 0.94)**; **homed es1 0.005 (line 0.7703)**; **settled es1 0.005 (line 0.7759)**; still steps es1 0; still taps es1 0; **homed es2 0 (line 0.945)**; **settled es2 0 (line 0.95)**; **homed es2 0 (line 0.5653)**; **settled es2 0 (line 0.5653)**; **homed es2 0 (line 0.94)**; **settled es2 0 (line 0.94)**; **homed es2 0 (line 0.95)**; **settled es2 0 (line 0.95)**; **homed es2 0.005 (line 0.82)**; **settled es2 0.005 (line 0.82)**; still steps es2 0.2024; still taps es2 0.3036 |
| K6 HOMING + STILLNESS | Amask | -- | FAIL | reported | **homed es1 0 (line 0.95)**; **settled es1 0 (line 0.95)**; **homed es1 0 (line 0.576)**; **settled es1 0.299 (line 0.5378)**; homed es1 1; settled es1 1; **homed es1 0 (line 0.94)**; **settled es1 0 (line 0.94)**; **homed es1 0 (line 0.7713)**; **settled es1 0.2172 (line 0.7509)**; **still steps es1 0.8075 (line 0.5)**; **still taps es1 2.06 (line 1)**; **homed es2 0 (line 0.945)**; **settled es2 0 (line 0.95)**; **homed es2 0 (line 0.5653)**; **settled es2 0.3434 (line 0.5255)**; homed es2 1; settled es2 1; **homed es2 0 (line 0.95)**; **settled es2 0 (line 0.95)**; **homed es2 0 (line 0.82)**; **settled es2 0.215 (line 0.7949)**; **still steps es2 0.7134 (line 0.5)**; **still taps es2 1.673 (line 1)** |
| K6 HOMING + STILLNESS | SH_RSD_TDR_sh_s19 | -- | PASS | head | homed es1 0.995; settled es1 1; homed es1 0.74; settled es1 0.76; homed es1 1; settled es1 1; homed es1 1; settled es1 1; homed es1 0.885; settled es1 0.93; still steps es1 0.186; still taps es1 0.1392; homed es2 1; settled es2 1; homed es2 0.81; settled es2 0.82; homed es2 1; settled es2 1; homed es2 0.99; settled es2 1; homed es2 0.93; settled es2 0.955; still steps es2 0.1875; still taps es2 0.1484 |

## Pending

- none

## INVALID cells

- K5 x R_h: [R_h -> R] G1: [R] suite@1: code_sha / eval_sha / dirty differ from REFS_RSD_TDR's (('07e02578eb29463957561f87b338bf286275de1b437583b3d6c05c6deff4cb7b', '1d021625844bb372556c26389fdb0ce610e42e1bfeae2dcd15e7ebc0ffabb7ff', False) vs ('de3772e26a3ecdb8ceffa5804b5a80ef63021324e11e00cc220a4727fbbebd99', '9fe964d7f46d1116d1d2983b0ece43f2430695cc35c9f33fd4cdc28f169ed5b8', False)); [R_h -> R] G1: [R] P2R_
- K5 x Amask: [Amask -> SN_Amask_s17] G1: [SN_Amask_s17] suite@1: code_sha / eval_sha / dirty differ from REFS_RSD_TDR's (('14301cd2f2ba60804a76cd037b741f9c97f5a47d2516ea8445531f2cc7ae27cc', '1d021625844bb372556c26389fdb0ce610e42e1bfeae2dcd15e7ebc0ffabb7ff', False) vs ('de3772e26a3ecdb8ceffa5804b5a80ef63021324e11e00cc220a4727fbbebd99', '9fe964d7f46d1116d1d2983b0ece43f2430695cc35c9f33fd4cdc28f169ed5b8', False));
- K5 x SH_RSD_TDR_sh_s19: [SH_RSD_TDR_sh_s19 -> SH_RSD_TDR_sh_s19z] G2: [R] walkstop__omni@1: code_sha / eval_sha / dirty differ from REFS_RSD_TDR's (('07e02578eb29463957561f87b338bf286275de1b437583b3d6c05c6deff4cb7b', '1d021625844bb372556c26389fdb0ce610e42e1bfeae2dcd15e7ebc0ffabb7ff', False) vs ('de3772e26a3ecdb8ceffa5804b5a80ef63021324e11e00cc220a4727fbbebd99', '9fe964d7f46d1116d1d2983b0ece43f2430695cc35c9f33fd4cdc28f169

## Forces per height bin (reported; foot-load statics = the cap statistic, PD torque alongside)

| role | bin | rows | h_meas | F_hip 6-leg mean foot-load (N) | p90 | PD reading (N) | worst hip leg (foot-load) | F_knee foot-load / PD (N) | over cap |
|---|---|---|---|---|---|---|---|---|---|
| R_h | 0.719-0.844 | 67888 | 0.7811 | 1156 | 1249 | 1552 | ML 1329 | 1331 / 1227 | False |
| R_h | 0.844-0.970 | 52272 | 0.9047 | 1400 | 1547 | 1816 | ML 1515 | 1437 / 1202 | False |
| R_h | 0.970-1.095 | 52204 | 1.063 | 1614 | 1668 | 2081 | ML 1705 | 1374 / 996.9 | False |
| R_h | 1.095-1.221 | 64329 | 1.149 | 1741 | 1810 | 2246 | FL 1830 | 1274 / 1044 | False |
| R_h | 1.221-1.346 | 116468 | 1.278 | 1912 | 2027 | 2465 | FL 1983 | 1105 / 1141 | False |
| R_h | outside [h_min, h_max] | 31735 | 1.307 | 1920 | 2039 | 2532 | ML 2229 | 1008 / 1071 | False |
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
| SH_RSD_TDR_sh_s19 | 0.719-0.844 | 75651 | 0.7806 | 1152 | 1253 | 1498 | ML 1434 | 1367 / 1431 | False |
| SH_RSD_TDR_sh_s19 | 0.844-0.970 | 61948 | 0.9079 | 1400 | 1526 | 1801 | ML 1576 | 1456 / 1241 | False |
| SH_RSD_TDR_sh_s19 | 0.970-1.095 | 60557 | 1.062 | 1623 | 1687 | 2066 | ML 1806 | 1388 / 1049 | False |
| SH_RSD_TDR_sh_s19 | 1.095-1.221 | 67302 | 1.149 | 1746 | 1822 | 2249 | ML 1919 | 1272 / 1058 | False |
| SH_RSD_TDR_sh_s19 | 1.221-1.346 | 119432 | 1.281 | 1918 | 2023 | 2467 | RR 2009 | 1102 / 1141 | False |
| SH_RSD_TDR_sh_s19 | outside [h_min, h_max] | 35203 | 1.321 | 1952 | 2064 | 2549 | ML 2379 | 1008 / 1151 | False |

## Reads

- R_h es1 hwalkstop__omni: ok `/home/nickmagus/krabby/krabby-research/parkour/outputs/omni_plan/phase2/lr/SH_RH_RSD_TDR/hwalkstop__omni/seed001/2026-11-09_21-49-51`
- R_h es1 hraise__omni: ok `/home/nickmagus/krabby/krabby-research/parkour/outputs/omni_plan/phase2/lr/SH_RH_RSD_TDR/hraise__omni/seed001/2026-11-09_22-03-10`
- R_h es2 hwalkstop__omni: ok `/home/nickmagus/krabby/krabby-research/parkour/outputs/omni_plan/phase2/lr/SH_RH_RSD_TDR/hwalkstop__omni/seed002/2026-11-09_21-56-31`
- R_h es2 hraise__omni: ok `/home/nickmagus/krabby/krabby-research/parkour/outputs/omni_plan/phase2/lr/SH_RH_RSD_TDR/hraise__omni/seed002/2026-11-09_22-07-30`
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
- SH_RSD_TDR_sh_s19 es1 hwalkstop__omni: ok `/home/nickmagus/krabby/krabby-research/parkour/outputs/omni_plan/phase2/lr/SH_RSD_TDR_sh_s19/hwalkstop__omni/seed001/2026-11-09_20-50-28`
- SH_RSD_TDR_sh_s19 es1 hraise__omni: ok `/home/nickmagus/krabby/krabby-research/parkour/outputs/omni_plan/phase2/lr/SH_RSD_TDR_sh_s19/hraise__omni/seed001/2026-11-09_21-04-04`
- SH_RSD_TDR_sh_s19 es2 hwalkstop__omni: ok `/home/nickmagus/krabby/krabby-research/parkour/outputs/omni_plan/phase2/lr/SH_RSD_TDR_sh_s19/hwalkstop__omni/seed002/2026-11-09_20-57-17`
- SH_RSD_TDR_sh_s19 es2 hraise__omni: ok `/home/nickmagus/krabby/krabby-research/parkour/outputs/omni_plan/phase2/lr/SH_RSD_TDR_sh_s19/hraise__omni/seed002/2026-11-09_21-08-32`

Full per-clause values, per-bin reads and stamp paths: the JSON.
