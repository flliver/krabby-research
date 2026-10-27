# PPO construction -- verdict support (scripts/verdict.py)

## Runs

| ref | bed | arm | seed | hand L | hand R | run | fall share | trip thr | tripped | dose 1000 | dose final | sat15 cam/rod 1000 (reported) | DEAD | wiring | LR floor share | lr_applied | feet 2500 -> 2999 | jitter | fft | rod odd L/R | flags |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| C__A_s11 | C | A | 11 | NA |  | not FAST | -- | 0.150 |  | yes | -- | 0.27/0.41 | no | yes | 0.59 | 2.13e-05 | -- -> -- | -- | -- | --/-- | KILLED dose at it 1000: {"chain": "C__A", "at": 1000, "result": "FAIL", "kill": "dose", "why": ["DOSE at it 1000 {'ok': False, 'mu_med_cam': 0.9730458790713018, 'mu_med_rod': 1.2661859564261861, 'sat15_cam': 0.27420967596002144, 'sat15_rod': 0.41066722970197694}"], "dead": {"dead": false, "track_1000": 0.7383215067410233, ; EXCLUDED: killed for dose at it 1000 under the pre-amendment rule; passes the amended median rule (CHARTER 4.3, 2026-10-01): superseded -- its rerun under a fresh reference counts, not a stop |
| C__A_s11_r2 | C | A | 11 | FAST |  | FAST | 0.040 | 0.150 |  | yes | no | 0.27/0.41 | no | yes | 0.67 | 1.84e-05 | -- -> 0.240 | 0.069 | 0.98 | 2.06/-- | dose lost late (final window); the class still counts |
| C__A_s12 | C | A | 12 | FAST |  | FAST | 0.000 | 0.150 |  | yes | yes | 0.28/0.38 | no | yes | 0.64 | 2.10e-05 | -- -> 0.215 | 0.054 | 0.89 | 2.09/-- |  |
| C__A_s13 | C | A | 13 | FAST |  | FAST | 0.015 | 0.150 |  | yes | no | 0.14/0.44 | no | yes | 0.66 | 1.95e-05 | -- -> 0.250 | 0.051 | 0.85 | 2.31/-- | dose lost late (final window); the class still counts |
| C__C0_s11 | C | C0 | 11 | SLOW |  | not FAST | 0.000 | 0.150 |  | ref | ref | 0.88/0.88 | -- | -- | 0.35 | 3.24e-05 | -- -> 0.088 | 0.061 | 1.28 | 0.65/-- |  |
| C__C0_s12 | C | C0 | 12 | FAST |  | FAST | 0.030 | 0.150 |  | ref | ref | 0.85/0.85 | -- | -- | 0.42 | 2.97e-05 | -- -> 0.193 | 0.079 | 1.25 | 1.31/-- |  |
| C__C0_s13 | C | C0 | 13 | INTERMEDIATE |  | not FAST | 0.020 | 0.150 |  | ref | ref | 0.87/0.87 | -- | -- | 0.36 | 3.06e-05 | -- -> 0.151 | 0.085 | 1.34 | 1.36/-- |  |
| C_lr__A_s11 | C_lr | A | 11 | SLOW | INTERMEDIATE | not FAST | 0.020 | 0.150 |  | yes | yes | 0.14/0.38 | no | yes | 0.64 | 2.09e-05 | 0.129 -> 0.127 | 0.066 | 1.07 | 0.77/1.51 |  |
| C_lr__A_s12 | C_lr | A | 12 | FAST | SLOW | not FAST | 0.025 | 0.150 |  | yes | yes | 0.10/0.36 | no | yes | 0.58 | 2.14e-05 | 0.132 -> 0.134 | 0.063 | 1.03 | 1.67/0.79 |  |
| C_lr__A_s13 | C_lr | A | 13 | SLOW | SLOW | not FAST | 0.003 | 0.150 |  | yes | no | 0.09/0.45 | no | yes | 0.62 | 2.23e-05 | 0.077 -> 0.088 | 0.043 | 0.95 | 0.65/1.14 | dose lost late (final window); the class still counts |
| C_lr__B_s11 | C_lr | B | 11 | FAST | INTERMEDIATE | not FAST | 0.003 | 0.150 |  | yes | yes | 0.00/0.27 | no | yes | 0.53 | 2.59e-05 | 0.182 -> 0.194 | 0.104 | 1.01 | 1.85/1.67 |  |
| C_lr__B_s12 | C_lr | B | 12 | FAST | FAST | FAST | 0.000 | 0.150 |  | yes | yes | 0.03/0.23 | no | yes | 0.53 | 2.61e-05 | 0.209 -> 0.226 | 0.135 | 1.18 | 2.32/1.96 |  |
| C_lr__B_s13 | C_lr | B | 13 | FAST | INTERMEDIATE | not FAST | 0.003 | 0.150 |  | yes | yes | 0.03/0.34 | no | yes | 0.50 | 2.92e-05 | 0.165 -> 0.183 | 0.121 | 1.08 | 1.95/1.64 |  |
| C_lr__B_s14 | C_lr | B | 14 | FAST | INTERMEDIATE | not FAST | 0.000 | 0.150 |  | yes | yes | 0.00/0.24 | no | yes | 0.58 | 2.34e-05 | 0.161 -> 0.188 | 0.121 | 1.30 | 1.95/1.12 |  |
| C_lr__B_s15 | C_lr | B | 15 | INTERMEDIATE | FAST | not FAST | 0.003 | 0.150 |  | yes | yes | 0.01/0.32 | no | yes | 0.57 | 2.54e-05 | 0.178 -> 0.206 | 0.102 | 1.01 | 1.74/1.96 |  |
| C_lr__C0_s11 | C_lr | C0 | 11 | SLOW | SLOW | not FAST | 0.035 | 0.150 |  | ref | ref | 0.89/0.84 | -- | -- | 0.37 | 3.56e-05 | 0.082 -> 0.083 | 0.077 | 1.37 | 0.60/0.42 |  |
| C_lr__C0_s12 | C_lr | C0 | 12 | SLOW | SLOW | not FAST | 0.037 | 0.150 |  | ref | ref | 0.87/0.81 | -- | -- | 0.38 | 3.89e-05 | 0.085 -> 0.089 | 0.104 | 1.48 | 0.69/0.59 |  |
| C_lr__C0_s13 | C_lr | C0 | 13 | SLOW | SLOW | not FAST | 0.013 | 0.150 |  | ref | ref | 0.86/0.89 | -- | -- | 0.41 | 3.63e-05 | 0.067 -> 0.071 | 0.065 | 1.31 | 0.38/0.25 |  |
| C_lr__C0_s14 | C_lr | C0 | 14 | SLOW | SLOW | not FAST | 0.070 | 0.150 |  | ref | ref | 0.88/0.85 | -- | -- | 0.35 | 3.56e-05 | 0.079 -> 0.079 | 0.065 | 1.29 | 0.40/0.30 |  |
| C_lr__C0_s15 | C_lr | C0 | 15 | SLOW | SLOW | not FAST | 0.075 | 0.150 |  | ref | ref | 0.85/0.86 | -- | -- | 0.39 | 3.55e-05 | 0.081 -> 0.081 | 0.078 | 1.41 | 0.50/0.70 |  |
| C_lr_rsi__A_s11 | C_lr_rsi | A | 11 | SLOW | FAST | not FAST | 0.003 | 0.150 |  | yes | yes | 0.11/0.37 | no | yes | 0.63 | 2.07e-05 | 0.158 -> 0.161 | 0.046 | 0.90 | 0.61/2.07 |  |
| C_lr_rsi__A_s12 | C_lr_rsi | A | 12 | SLOW | SLOW | not FAST | 0.013 | 0.150 |  | yes | yes | 0.13/0.43 | no | yes | 0.57 | 2.42e-05 | 0.102 -> 0.094 | 0.052 | 0.99 | 0.35/0.25 |  |
| C_lr_rsi__A_s13 | C_lr_rsi | A | 13 | SLOW | FAST | not FAST | 0.005 | 0.150 |  | yes | yes | 0.06/0.34 | no | yes | 0.65 | 2.40e-05 | 0.158 -> 0.164 | 0.056 | 1.07 | 0.87/1.68 |  |
| C_lr_rsi__B_s11 | C_lr_rsi | B | 11 | FAST | INTERMEDIATE | not FAST | 0.005 | 0.150 |  | yes | yes | 0.00/0.31 | no | yes | 0.58 | 2.36e-05 | 0.175 -> 0.195 | 0.109 | 0.94 | 1.88/1.44 |  |
| C_lr_rsi__B_s12 | C_lr_rsi | B | 12 | FAST | INTERMEDIATE | not FAST | 0.000 | 0.150 |  | yes | yes | 0.00/0.31 | no | yes | 0.52 | 2.67e-05 | 0.208 -> 0.195 | 0.122 | 1.13 | 1.92/1.88 |  |
| C_lr_rsi__B_s13 | C_lr_rsi | B | 13 | SLOW | FAST | not FAST | 0.000 | 0.150 |  | yes | yes | 0.00/0.38 | no | yes | 0.51 | 2.92e-05 | 0.170 -> 0.193 | 0.116 | 1.11 | 1.40/1.99 |  |
| C_lr_rsi__C0_s11 | C_lr_rsi | C0 | 11 | SLOW | SLOW | not FAST | 0.013 | 0.150 |  | ref | ref | 0.80/0.82 | -- | -- | 0.36 | 3.75e-05 | 0.086 -> 0.090 | 0.087 | 1.46 | 0.47/1.06 |  |
| C_lr_rsi__C0_s12 | C_lr_rsi | C0 | 12 | SLOW | SLOW | not FAST | 0.045 | 0.150 |  | ref | ref | 0.90/0.82 | -- | -- | 0.31 | 3.89e-05 | 0.071 -> 0.070 | 0.087 | 1.43 | 0.79/0.90 |  |
| C_lr_rsi__C0_s13 | C_lr_rsi | C0 | 13 | SLOW | SLOW | not FAST | 0.048 | 0.150 |  | ref | ref | 0.86/0.81 | -- | -- | 0.32 | 3.79e-05 | 0.062 -> 0.068 | 0.118 | 1.63 | 0.70/0.78 |  |

## Excluded runs (never counted)

- C__A_s11: killed for dose at it 1000 under the pre-amendment rule; passes the amended median rule (CHARTER 4.3, 2026-10-01): superseded -- its rerun under a fresh reference counts, not a stop

## Tripwire

- C: reference 0.020, trips above 0.150
- C_lr: reference 0.037, trips above 0.150
- C_lr_rsi: reference 0.045, trips above 0.150

## Two-direction beds (VOID > UNLOCK > PARTIAL > NO-FLIP)

| bed | arm | verdict | arm FAST | C0 FAST | reasons | extend | Q better | adopt (MQ) | 4a trigger |
|---|---|---|---|---|---|---|---|---|---|
| C_lr | A | **PARTIAL** | 0/3 | 0/5 | C_lr__A_s11: hand(s) R INTERMEDIATE; C_lr__A_s12: a FAST hand in a run not counted FAST (the other hand not FAST) |  | no | no |  |
| C_lr | B | **PARTIAL** | 1/5 | 0/5 | C_lr__B_s11: hand(s) R INTERMEDIATE; C_lr__B_s11: a FAST hand in a run not counted FAST (the other hand not FAST); C_lr__B_s13: hand(s) R INTERMEDIATE; C_lr__B_s13: a FAST hand in a run not counted FAST (the other hand not FAST); C_lr__B_s14: hand(s) R INTERMEDIATE; C_lr__B_s14: a FAST hand in a run not counted FAST (the other hand not FAST); C_lr__B_s15: hand(s) L INTERMEDIATE; C_lr__B_s15: a FAST hand in a run not counted FAST (the other hand not FAST) |  | no | no |  |
| C_lr_rsi | A | **PARTIAL** | 0/3 | 0/3 | C_lr_rsi__A_s11: a FAST hand in a run not counted FAST (the other hand not FAST); C_lr_rsi__A_s13: a FAST hand in a run not counted FAST (the other hand not FAST) |  | yes | yes |  |
| C_lr_rsi | B | **PARTIAL** | 0/3 | 0/3 | C_lr_rsi__B_s11: hand(s) R INTERMEDIATE; C_lr_rsi__B_s11: a FAST hand in a run not counted FAST (the other hand not FAST); C_lr_rsi__B_s12: hand(s) R INTERMEDIATE; C_lr_rsi__B_s12: a FAST hand in a run not counted FAST (the other hand not FAST); C_lr_rsi__B_s13: a FAST hand in a run not counted FAST (the other hand not FAST) |  | no | no |  |

## C-bed guard

- **PASS**: A FAST 3/3 >= 2/3; C bed not reliable at fresh seeds (C0 1/3) -- under the literal 4.7 reading the guard is UNDEFINED (user's call)
- C__A: Q better no; adopt for motion quality no; 4a trigger --

## Pilot

- PASS (C0 not FAST at s11): continue the blocks; the C-bed reliability is read at three seeds 

## Outcome

- UNLOCK (step 3, the 4c trigger): none (4c needs the C-bed guard PASS or the user's approval: guard PASS)
- UNLOCK of a step-4 arm (4c under plan step 12's carry rule; listed apart): none
- INC0: not run
- LR-floor confound (4a trigger): none

