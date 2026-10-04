# FB6k + INC0 -- the pre-registered reading (CHARTER 11)

**INC0 gate: ACTION_RATE ALONE INSUFFICIENT** -- INC0 FAST/FAST 0/3: action_rate 0 alone does not flip C_lr by 6000 continuous iterations; the construction attribution is read together with FB6k, whose from-scratch B runs test the fork == continuous premise of the x6k comparison

- INC0 s11: FAST/SLOW
- INC0 s12: INTERMEDIATE/FAST
- INC0 s13: FAST/INTERMEDIATE

| seed | final | extended | hands L / R (feet) | fall | dose cam/rod | FAST/FAST counted | suite@5999 fwd / back / side / diag ratio |
|---|---|---|---|---|---|---|---|
| B s16 | FB6k__B_s16 | False | FAST 0.203 / FAST 0.202 | 0.000 | 0.54/0.88 | True | 0.996 / 0.956 / 0.856 / 0.917 |
| B s17 | FB6k__B_s17 | False | FAST 0.203 / FAST 0.198 | 0.000 | 0.52/0.88 | True | 0.933 / 0.917 / 0.859 / 0.822 |
| B s18 | FB6k__B_s18 | False | FAST 0.226 / FAST 0.187 | 0.000 | 0.51/0.84 | True | 0.968 / 0.933 / 0.874 / 0.964 |
| C0 s16 | FB6k__C0_s16 | False | SLOW 0.096 / SLOW 0.111 | 0.020 | 7.59/9.15 | False | 0.691 / 0.715 / 0.437 / 0.546 |
| C0 s17 | FB6k__C0_s17 | False | SLOW 0.106 / SLOW 0.085 | 0.025 | 6.84/9.22 | False | 0.579 / 0.605 / 0.443 / 0.483 |
| C0 s18 | FB6k__C0_s18 | False | SLOW 0.101 / SLOW 0.098 | 0.013 | 8.34/9.42 | False | 0.645 / 0.687 / 0.450 / 0.582 |

Tripwire: reference 0.020, threshold 0.150
Extensions the rule triggers: none

- fwd: B mean 0.966 vs C0 mean 0.638 -> holds True, clearly inferior False
- back: B mean 0.935 vs C0 mean 0.669 -> holds True, clearly inferior False

**FB6k gate: PASS** -- B FAST/FAST 3/3, C0 0/3, no B trip, fwd and back hold

- B s16 trajectory: 3000: FAST/INTE 0.197/0.131; 4500: FAST/FAST 0.210/0.195; head: FAST/FAST 0.203/0.202
- B s17 trajectory: 3000: INTE/INTE 0.161/0.149; 4500: FAST/FAST 0.178/0.232; head: FAST/FAST 0.203/0.198
- B s18 trajectory: 3000: FAST/SLOW 0.181/0.117; 4500: FAST/INTE 0.211/0.172; head: FAST/FAST 0.226/0.187
- C0 s16 trajectory: 3000: SLOW/SLOW 0.096/0.101; 4500: SLOW/SLOW 0.104/0.121; head: SLOW/SLOW 0.096/0.111
- C0 s17 trajectory: 3000: SLOW/SLOW 0.107/0.085; 4500: SLOW/SLOW 0.108/0.080; head: SLOW/SLOW 0.106/0.085
- C0 s18 trajectory: 3000: SLOW/SLOW 0.102/0.094; 4500: SLOW/SLOW 0.103/0.093; head: SLOW/SLOW 0.101/0.098
