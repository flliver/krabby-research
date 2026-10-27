# Turn band, t3x and side-first -- the pre-registered reading (CHARTER 13)

## Turn band (in-band +-0.5 rad = the read; +-1.0 rad = 0.4 rad beyond the trained band, descriptive)

| head | turn heading_final (rad) | turn facing_closed | turn yaw_progress | turn_mid heading_final | turn_mid facing_closed |
|---|---|---|---|---|---|
| P1B6__B_t3_s16 | 0.225 | 0.547 | 0.021 | 0.949 | 0.057 |
| P1B6__B_t3_s17 | 0.181 | 0.638 | 0.026 | 0.852 | 0.154 |
| P1B6__B_t3_s18 | 0.161 | 0.678 | 0.028 | 0.739 | 0.259 |
| P1B6__C0_t3_s16 | 0.102 | 0.801 | 0.040 | 0.401 | 0.613 |
| P1B6__C0_t3_s17 | 0.082 | 0.828 | 0.046 | 0.186 | 0.813 |
| P1B6__C0_t3_s18 | 0.111 | 0.779 | 0.040 | 0.299 | 0.704 |
| FB6k__B_s16 | 0.310 | 0.378 | 0.016 | 0.461 | 0.540 |
| FB6k__B_s17 | 0.345 | 0.316 | 0.013 | 0.842 | 0.155 |
| FB6k__B_s18 | 0.217 | 0.562 | 0.023 | 0.711 | 0.298 |
| W_p1b6 | 0.071 | 0.855 | 0.064 | 0.079 | 0.920 |

Sampled-action suite (diagnostic only, never gated):

- P1B6__B_t3_s16: turn facing_closed 0.107, yaw_progress 0.014
- P1B6__B_t3_s17: turn facing_closed 0.165, yaw_progress 0.021
- P1B6__B_t3_s18: turn facing_closed 0.203, yaw_progress 0.026
- P1B6__C0_t3_s16: turn facing_closed 0.377, yaw_progress 0.049

## t3x: **INCOMPLETE** -- missing: a t3x head missing or NA, in-band turn reads -- notify


## Side-first c4 gate: **INCOMPLETE** -- c4 reads missing or NA: seeds [16, 17, 18] -- stop and ask

## C0 trigger: run the C0 side-first tails = False (B FAST/FAST hands at c6x 0/3)

## Side-first final gate: **PENDING** -- side-first B c6/c6x reads missing or NA

