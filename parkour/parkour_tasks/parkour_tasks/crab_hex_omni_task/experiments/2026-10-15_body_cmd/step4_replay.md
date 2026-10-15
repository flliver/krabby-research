# Step 4 — offline replay of the body yaw-term set

Shipped pure functions (`crab_hex_omni_yaw_window.py`), window 100 steps, weights window 0.2 / tent 2.0; envelope {'w_min': 0.05, 'ceiling': 0.168, 'w_max': 0.21, 'cap': [[0.0, 0.21], [0.22, 0.105], [0.3, 0.065]]}; cap80 = 0.0840. Traces: {'turn': 400, 'move': 400, 'stand': 200, 'translate': 1000}; S1 turn traces' mean rate 0.1680, move traces' 0.0953 rad/s.

**Verdict: PASS**

| healthy | income / step |
|---|---|
| healthy_turn@ceiling | 0.4455 |
| healthy_move@cap80 | 0.2771 |
| healthy_synth@w_min | 0.2418 |
| healthy_synth@ceiling | 0.4778 |

| degenerate | income / step | vs | share | margin | pass |
|---|---|---|---|---|---|
| over_spin@w_min | -0.1150 | healthy_synth@w_min | -0.476 | 0.4 | True |
| stand_on_turn@ceiling | 0.0087 | healthy_turn@ceiling | 0.020 | 0.1 | True |
| stand_on_turn@w_min | 0.0775 | healthy_synth@w_min | 0.321 | 0.4 | True |
| step_in_place@w_min | 0.0766 | healthy_synth@w_min | 0.317 | 0.4 | True |
| step_in_place@ceiling | 0.0073 | healthy_turn@ceiling | 0.016 | 0.1 | True |

Reported only: straight-row yaw income of S1's wobble 1.1311 / step; with a 0.01 rad/s drift 1.1221 / step.
