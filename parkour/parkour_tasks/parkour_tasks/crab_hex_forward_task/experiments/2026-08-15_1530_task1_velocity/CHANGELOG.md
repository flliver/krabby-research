<!-- paths-note -->
> **Paths note (2026-09-09):** this campaign moved from `sim_fine_tuning/2026-08-15_1530_task1_velocity/` to `parkour/parkour_tasks/parkour_tasks/crab_hex_forward_task/experiments/2026-08-15_1530_task1_velocity/`. Absolute paths below (`/home/.../sim_fine_tuning/...`, `parkour/logs/rsl_rl/...`) name raw artifacts that stay on disk untracked; the tracked checkpoint of record is `head/` (see `bundle.yaml`) and the eval summaries are in place / under `evals/`.

# Task-1 velocity-era campaign (disciplined restart)

Protocol (per M18 TASK-1-REWARD-SHAPING.md s3 + campaign SOP):
- ONE change per run; replay gate before every screen; 3k from-scratch screens
  (seed 1, 256 envs); scored vs B0 with the Task-0 harness (flat_walk_forward);
  every change gets a before/after CHANGELOG row; reverted attempts documented.
- Certification gates AMENDED: full battery + per-hold tracking deficit < 0.1 m/s
  + training tracking income recorded.
- Effective reward stack at campaign start = position-era bake (unbake commit cc30e3c);
  instruments registered at 0.0 behind env vars.

## B0 baseline (adopted, no GPU): fromscratch_velact_short (geometry campaign)
| change | reward@3k | tracking deficit (lo/mid/hi) | 1-dir ratio | completion | slip | tripod |
|--------|-----------|------------------------------|-------------|------------|------|--------|
| B0     | 21.0      | +0.25 / +0.08 / -0.10        | 0.006       | 0.90       | 0.079| 0.0    |

## Change queue (audit-verdict order)
C1 tracking restoration (linear |v_err| penalty; fallback arm: + sigma^2 0.25)
C2 air-time threshold recalibration from cam kinematics (return stroke = 2.14 rad/omega)
C3 stance-count band penalty {3,4} (also prices fall-and-spin)
C4+ spin instruments one at a time, re-gated on tracking-era traces
ESC CPG action space (structural; separate plan)

## C1 — linear tracking penalty (KRABBY_TRACK_L1_W=-0.5): PASS (all gates)
| change | reward@3k | tracking deficit (lo/mid/hi) | 1-dir ratio | completion | slip | tripod |
|--------|-----------|------------------------------|-------------|------------|------|--------|
| B0     | 21.0      | +0.25 / +0.08 / -0.10        | 0.006       | 0.90       | 0.079| 0.0    |
| C1     | 21.0      | **+0.08 / +0.03 / -0.01**    | 0.005       | **1.00**   | 0.079| 0.0    |
First command-following policy of the velocity era. Mechanism confirmed: L1's constant
gradient carried the policy into the narrow exp well (tracking-exp income 0.76/1.25 vs
era's 0.25 flatline, sigma^2 still 0.02). Speed modulation comes from legs, not shaft
cadence (shaft |w| ~5.6 at all holds). C1 retained via env var for subsequent screens;
bake decision deferred to campaign end per SOP.

## C2 — air-time threshold 0.05 -> 0.20 s (cam-derived; KRABBY_AIRTIME_THRESH): PASS
| change | reward@3k | tracking deficit (lo/mid/hi) | tripod | completion | slip | tippy |
|--------|-----------|------------------------------|--------|------------|------|-------|
| C1     | 21.0      | +0.08 / +0.03 / -0.01        | 0.0    | 1.00       | 0.079| ~0.15 |
| C2     | **27.1**  | +0.02 / -0.01 / -0.08        | **0.573**| 0.90     | 0.113| 0.238 |
FIRST tripod of the velocity era — exceeds the position-era reference (0.517) at 3k.
Mechanism differed from prediction: micro-taps did NOT drop (19% vs 11% mass <=3 steps);
paying cam-length swings properly reorganized the gait into alternating tripod phasing
instead. WATCH-ITEMS for C3+: tippy 23.8%, slip 11.3%, one fall. Shafts still oscillate
(ratio 0.007) — spin remains a C4+/CPG objective. Stack now = L1 -0.5 + thresh 0.20.

## C3 — stance-count band {3,4}: REFUTED AT REPLAY GATE (no GPU spent)
Position-era reference: 61% of steady steps outside {3,4}; C2: 42%; oscillator: 69%.
The band assumes 50%-duty alternation; this mechanism's tripod runs ~0.66 duty with
legitimate 4-6-contact overlap. Term skipped per Task 1 s2.3/NOTE discipline. If a
fall-pricing term is needed later (positive spin terms reintroduced), derive the band
from the certified reference's own count distribution and drop the steady-mask
dependence (fallen traces have no steady steps to score).

## C1+C2 20k adoption run: CERTIFICATION FAIL (basin roulette)
model_19999: tripod 0.0 (never consolidated — same config+seed as the 0.573 screen; GPU
nondeterminism), completion 0.8, slip 16.7%, tippy 27.9%, high-hold deficit -0.14 (over
gate). Tracking substantially held. Income telemetry identical to the screen — the basin
difference is phase coordination, invisible in reward magnitudes.

## C3b proposal (recovery): tripod crossing term as consolidation lock
Replay on the C2 screen's tripod traces: 16.3/min @0.15 weight, 158 crossings/min — vs
~0 on every non-tripod family ever traced. The v1-v5-refuted term finally has a gait
that earns it: self-reinforcing once the basin is entered, inert otherwise. Plan:
enable reward_tripod_schedule 0.15 on C1+C2 -> 4-seed x 3k selection (pick tripod
formers) -> resume best +17k -> re-certify. (~overnight GPU)

## C3b — tripod crossing lock @0.15: FAIL (income 0.0000 all run — basin never entered)
Confirms the v-series theorem in the velocity era: the term is a lock, not a creator.
Tracking regressed (deficits -0.09/-0.17/-0.30), reward 19.5 < gate. REVERTED.
Basin-entry conclusion after 7 attempts across 2 campaigns: entry is not
reward-addressable — only SELECTION (multi-seed) or STRUCTURE (CPG) remain.
Task-1 flat-walk list status: 2.1 C2-PASS, 2.2 baked, 2.3 refuted+lock-only,
2.5 C5 re-sweep REMAINING, 2.6 C1-PASS.

## C6 part 1 — mirror map velocity-era consistency check (offline): PASS
Involution exact; mirrored-vs-original action stats on C2 traces match within 2-5% of
channel scale (cam/hip); knee residual 12% = the documented -0.07/+0.10 roll-balance
asymmetry (known approximation). No velocity-era sign error. Part 2 queued behind C5a:
C6a ablation (KRABBY_SYM_LOSS_COEF=0) and C6b strengthen (=1.0) on the C1+C2 stack.

## C5a — mechanical power @-0.001: FAIL, REVERTED
Reward 11.4 (<21.7 gate), tippy-tap 41.5% (policy shortened steps to cut power: -13%,
1096 W), tripod 0, tracking held. Replay-gate blind spot: replay prices FIXED behavior;
in-training the term reshaped behavior toward cheap-and-degraded (lit review s5's
predicted failure regime). Energy half of Task-1 s2.5 closed NEGATIVE for flat-walk.
Smoothness half (action_rate/delta_torques): position-era values stay; no velocity-era
evidence implicates them — C5b skipped per discipline (no implicated suspect).
Task-1 flat-walk exploration COMPLETE: every item screened, passed, or refuted.

## C6a — symmetry ablation (SYM=0): reward-gate FAIL, best-in-era stepping
| change | reward@3k | deficits | tripod | completion | slip | tippy |
|--------|-----------|----------|--------|------------|------|-------|
| C2 (sym 0.5) | 27.1 | .02/.01/.08 | 0.573 | 0.90 | 0.113| 0.238 |
| C6a (sym 0)  | 19.9 | .07/.05/.02 | 0.032 | 1.00 | **0.036**| **0.057** |
Symmetry OFF -> cleanest stepping of the era but no tripod. The loss appears implicated
in both tripod formation (+) and step roughness (-); single-roll basin variance caveat.

## C6b — symmetry 1.0: FAIL (tracking regressed, deficits to -0.40; reward 16.6)
Dose curve complete: sym 0 = cleanest stepping (slip 3.6%) no tripod; 0.5 = sweet spot
(tripod 0.573, reward 27.1); 1.0 = harmful (fights knee asymmetry + breaks tracking).
Baked 0.5 default CONFIRMED by its own audit.

# ===== TASK-1 EXPLORATION COMPLETE =====
| item | disposition |
|------|-------------|
| C1 linear tracking penalty -0.5 | PASS — command-following restored |
| C2 air-time thresh 0.20 (cam-derived) | PASS — first velocity-era tripod 0.573 |
| C3 stance band {3,4} | REFUTED at replay (0.66-duty overlap) |
| C3b tripod crossing lock 0.15 | FAIL — lock, not creator; basin never entered |
| C5a mechanical power -0.001 | FAIL — induces tippy-tap (replay blind spot documented) |
| C5b action-rate re-sweep | SKIPPED — no implicated suspect |
| C6 symmetry audit | map consistent; 0.5 confirmed optimal of {0, 0.5, 1.0} |
Standing candidate stack: position-era rewards + C1 (-0.5 L1) + C2 (0.20 thresh),
sym 0.5. Best single-run results: tripod 0.573 / tracking deficits <=0.08 (C2 roll);
adoption blocked ONLY by basin-entry stochasticity (20k roll failed certification).
Remaining paths to a certified reference: multi-seed selection OR CPG action space.

## C6a-20k (user-directed): symmetry-off long run for A/B vs the C1+C2 20k
C6a's 3k gait was best-in-era stepping (slip 3.6%, tippy 5.7%, completion 1.0, tracking
good). 20k from-scratch at SYM=0 on the C1+C2 stack; compare against fromscratch_C1C2_20k
(sym 0.5, certification FAIL: tripod 0.0, slip 16.7%). Mid-gate at 10k.

## C6a-20k final: DOMINATES the sym-0.5 20k on every axis; first 20k to pass tracking gate
| @20k | C6a (sym 0) | C1C2 (sym 0.5) |
|------|-------------|----------------|
| reward | 23.5 | 20.0 |
| tracking | +.04/-.02/-.06 PASS | -.14 FAIL |
| slip | 4.1% | 16.7% |
| tippy | 9.9% | 27.9% |
| completion | 0.8 | 0.8 |
| tripod | 0.0 | 0.0 |
Gaps vs full certification: completion 0.8 (<0.9; two falls) and tripod 0 (no 20k run of
the era has held tripod; the C2 3k screen remains the only sighting). Checkpoint:
fromscratch_C6a_20k/logs/rsl_rl/crab_hex_flat_walk/2026-08-16_12-51-51/model_19999.pt

## BAKED (user decision 2026-08-16): C2 checkpoint adopted as velocity-era reference
- penalty_tracking_error_l1 = -0.5 default (C1); flat-walk air-time threshold = 0.20 (C2)
- scenarios_v1.yaml flat_walk_forward -> C2 model_2999 (sha 7500414c...)
- Known gaps documented (slip 11.3%, tippy 23.8%, 3k-scale, basin-dependent tripod)
- Next: curriculum carry-up from this checkpoint through bridge/2b1 -> 2b2

## Flat extension (user-directed): resume C2 model_2999 +17k to 20k-equivalent
Process note: the 2b2 carry-up was started from the 3k base prematurely — the maturity
step should have come first (both prior successful carry-ups used 20k bases). The C2
checkpoint has never been extended; resuming preserves its tripod basin (r4c precedent)
vs the from-scratch 20k re-roll that lost it. Mid-run basin checks at +5k/+10k.
2b2 lineage state: paused at batch 3 best-seen (failure 0.372); will redo bridge->2b2
from the matured flat checkpoint.

## Flat extension complete: consolidation worked
3k->20k trajectory: tripod 0.573->0.654, slip 11.3->6.9%, tippy 23.8->13.2%, tracking
+-0.02, completion 0.9->0.5->0.4->0.7 (mid-run dip transitional; 3 late high-hold falls
remain). model_19998 is the era-best tripod policy. High hold (0.543) still weakest —
NOTE: bridge commands 0.45-0.85, faster than flat; watch bridge failure rate.
Carry-up v2 from model_19998 starting.

## Carry-up v2 verdict: STOPPED — maturity/plasticity trade-off surfaced
Bridge 0.44 & 2b1 0.59 (vs v1 0.79/0.93): matured base transfers walking far better.
BUT 2b2: failure 0.52->0.62 across 2 batches with clearance FROZEN at 0.013 (v1's
immature base reached 0.08-0.15 — it learned to lift; the consolidated tripod rhythm
does not modify its swing for obstacles). Same rigidity as the flat high-speed falls.
v1 plateaued from weakness; v2 from rigidity. The carry-up base wants INTERMEDIATE
maturity or explicit lift-plasticity help at 2b2 — decision fork for the user.

## Carry-up v3 (user-approved): intermediate-maturity base (model_8000, tripod 0.639)
Third point on the maturity-plasticity curve: 5k consolidation vs v1's 0k and v2's 17k.

## v3 CLOSED at ceiling (3000/3000): gates unmet
Batches 5-6: failure ~0.51 (best-seen remains batch 3: 0.435), clearance climbed to the
end (0.121 vs 0.15 gate — lifting nearly learned). Verdict: the intermediate base learns
terrain skills but cannot simultaneously reach gate-level stability. All three carry-up
lineages closed. Successor: sim_fine_tuning/2026-08-17_1500_phased_flat (approved plan).
