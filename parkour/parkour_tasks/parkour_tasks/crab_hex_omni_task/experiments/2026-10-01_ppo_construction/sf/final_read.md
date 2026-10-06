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
| P1B6__B_t3x_s16 | 0.124 | 0.753 | 0.038 | 0.855 | 0.148 |
| P1B6__B_t3x_s17 | 0.134 | 0.731 | 0.032 | 0.523 | 0.473 |
| P1B6__B_t3x_s18 | 0.131 | 0.737 | 0.032 | 0.801 | 0.201 |
| P1B6__C0_t3x_s16 | 0.125 | 0.756 | 0.039 | 0.598 | 0.410 |
| SF__B_rest_s16 | 0.036 | 0.928 | 0.117 | 0.059 | 0.942 |
| SF__B_rest_s16_c6 | 0.098 | 0.803 | 0.092 | 0.493 | 0.510 |
| SF__B_rest_s17 | 0.076 | 0.849 | 0.095 | 0.088 | 0.912 |
| SF__B_rest_s17_c6 | 0.116 | 0.769 | 0.085 | 0.922 | 0.077 |
| SF__B_rest_s18 | 0.045 | 0.910 | 0.092 | 0.084 | 0.915 |
| SF__B_rest_s18_c6 | 0.120 | 0.763 | 0.086 | 0.120 | 0.880 |
| SF__C0_rest_s16 | 0.076 | 0.846 | 0.049 | 0.108 | 0.889 |
| SF__C0_rest_s16_c6 | 0.113 | 0.769 | 0.036 | 0.434 | 0.566 |

Sampled-action suite (diagnostic only, never gated):

- P1B6__B_t3_s16: turn facing_closed 0.107, yaw_progress 0.014
- P1B6__B_t3_s17: turn facing_closed 0.165, yaw_progress 0.021
- P1B6__B_t3_s18: turn facing_closed 0.203, yaw_progress 0.026
- P1B6__C0_t3_s16: turn facing_closed 0.377, yaw_progress 0.049

## t3x: **TAILS CANDIDATE** -- B FAST/FAST 2/3 at 8998 and the in-band turn settles: running the P1 tails from these heads is the user's call (stop and ask)

- P1B6__B_t3x_s16: 5999 INTE/SLOW 0.147/0.127; 6100 INTE/INTE 0.146/0.139; 7500 FAST/INTE 0.181/0.155; head FAST/FAST 0.190/0.213
- P1B6__B_t3x_s17: 5999 SLOW/INTE 0.128/0.150; 6100 SLOW/INTE 0.132/0.135; 7500 INTE/FAST 0.158/0.192; head INTE/FAST 0.145/0.202
- P1B6__B_t3x_s18: 5999 FAST/INTE 0.188/0.171; 6100 FAST/FAST 0.181/0.192; 7500 FAST/FAST 0.174/0.204; head FAST/FAST 0.167/0.191
- P1B6__C0_t3x_s16: 5999 SLOW/SLOW 0.119/0.100; 6100 SLOW/SLOW 0.119/0.104; 7500 SLOW/SLOW 0.123/0.104; head SLOW/SLOW 0.123/0.099

## Side-first c4 gate: **CONTINUE** -- stroke lost in 0/3 seeds at c4

## C0 trigger: run the C0 side-first tails = True (B FAST/FAST hands at c6x 3/3)

## Side-first final gate: **PENDING** -- the C0 side-first tails are triggered and not read yet

