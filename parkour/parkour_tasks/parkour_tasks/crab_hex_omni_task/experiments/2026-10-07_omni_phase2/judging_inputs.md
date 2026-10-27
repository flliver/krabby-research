# Omni phase 2 -- the pre-registered reading (CHARTER 6)

| head | verdict | gain | field_hard (pooled) | field_light | ring_obst | why |
|---|---|---|---|---|---|---|
| P2__A1_s18 | **PASS** | True | 0.745 | 0.995 | 0.995 | gain, no flat loss |
| P2__A0_s18 | **PASS** | True | 0.715 | 1.000 | 1.000 | gain, no flat loss |
| P2__S1_s16 | **PASS** | True | 0.830 | 0.980 | 1.000 | gain, no flat loss |
| P2__S1_s17 | **PASS** | True | 0.730 | 0.985 | 1.000 | gain, no flat loss |
| P2__S1_s18 | **COST** | True | 0.800 | 0.995 | 1.000 | err_back worse by 0.0573 |
| P2__E1_s18 | **NOT RUN** | None | -- | -- | -- | dropped by the user 2026-10-06 (CHARTER 4 replay gate FAIL) |
| P2__F1_s18 | **NO GAIN** | False | 0.695 | 0.985 | 1.000 | field_hard 0.695 vs need 0.700; field_light ok True; ring_obst ok True |

**Recipe read (6.2) on S1: ADOPTABLE** {"pass": 2, "hand_loss": 0, "gain": 3, "shared_cost": []}
**p2b trigger (6.5) on S1:** {"trigger": true, "gain": 3, "deficit": {"16": true, "17": true, "18": true}, "pending": false}

**CHARTER 7 candidates:** S1_s16, S1_s17, S1_s18, A1_s18

**m-screen (CHARTER 7):** {"S1_s16": {"16000": "NO", "17500": "NO", "19000": "NO"}, "S1_s17": {"16000": "NO", "17500": "NO", "19000": "NO"}, "S1_s18": {"16000": "NO", "17500": "NO", "19000": "NO"}, "A1_s18": {"16000": "NO", "17500": "NO", "19000": "NO"}}
