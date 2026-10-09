# Omni phase 2, stage p2s -- the pre-registered reading (CHARTER 11 / P2S_PLAN.md)

| head | verdict | stand | gain | onset (delta vs REFS) | field_hard n=400 (REFS) | cost | label | why |
|---|---|---|---|---|---|---|---|---|
| C0R_s18 | **NO EFFECT** | NO EFFECT | False | 0.160 (+0.020) | 0.785 (0.812) | -- |  | stand NO EFFECT |
| P2__ST2_s18 | **OBSTACLE LOSS** | PARTIAL | True | 0.720 (+0.580) | 0.618 (0.812) | -- |  | field_hard 0.6175 (n=400) < REFS 0.8125 - max(0.05, 2 SE 0.0623) |
| P2__ST1_s18 | **PENDING** | None | None | -- (--) | -- (--) | -- |  | no record (not run) |
| P2__ST2_s16 | **PENDING** | None | None | -- (--) | -- (--) | -- | lever + continuation | no record (not run) |
| P2__ST2_s17 | **PENDING** | None | None | -- (--) | -- (--) | -- | lever + continuation | no record (not run) |

**STILL clauses (suite stand, value; '!' = fails), each head beside C0R's own:**
- C0R_s18: es1 steps 2.67!, clearance 0.113!, off 0.341!, hover 0, taps 2.08!, ks 1.65! | es2 steps 2.67!, clearance 0.113!, off 0.339!, hover 0, taps 2!, ks 1.66!
- P2__ST2_s18: es1 steps 0.167, clearance 0.00533, off 0, hover 0, taps 16.1!, ks 0.405 | es2 steps 0.167, clearance 0.0054, off 0, hover 0, taps 15.3!, ks 0.404
- P2__ST1_s18: es1 -- | es2 --
- P2__ST2_s16: es1 -- | es2 --
- P2__ST2_s17: es1 -- | es2 --

**Attribution at s18 (vs C0R):**
- P2__ST2_s18: **NOT ATTRIBUTED** {"head reads STILL or PARTIAL (COST included)": false, "C0R reads NO EFFECT (implemented: C0R is not STILL)": true, "no FLAT FAIL vs C0R": true, "field_hard 0.6175 >= C0R 0.7850 - max(0.05, 2 SE 0.0636)": false}
- P2__ST1_s18: **PENDING** ["the head's verdict is not read"]

**Recipe read (ST2, CHARTER 6.2 rules, STILL -> PASS / PARTIAL -> gain / COST -> COST):** **PENDING** {"why": "a seed is missing, INVALID EVAL or PENDING", "dose": "ST2", "heads": {"16": "P2__ST2_s16", "17": "P2__ST2_s17", "18": "P2__ST2_s18"}, "mapped": {"16": {"p2s_verdict": "PENDING", "verdict": "PENDING", "gain": null, "cost": []}, "17": {"p2s_verdict": "PENDING", "verdict": "PENDING", "gain": null, "cost": []}, "18": {"p2s_verdict": "OBSTACLE LOSS", "verdict": "OBSTACLE LOSS", "gain": true, "cost": []}}}
