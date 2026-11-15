# Traversable rough-terrain teacher: slopes, hills, steps, potholes, gravel, rubble (no stopping)

> **Audit trail (cap 3).**
> - Pass 1: an evidence inventory, three premise probes (curriculum, gates, friction) and two opus·xhigh auditors
>   on split lenses. Blocking findings, all fixed:
>   - completion counted stuck episodes, so "success" now means traversed;
>   - the friction setup was under-specified, so a friction contract now fixes it;
>   - the early read could not detect a stall, so it is report-only and the remedy is removed;
>   - no check measured the realised dose, so dose tests and an eval smoke were added.
> - Your decisions in pass 1: settled 4 and 5.
> - Pass 2: a fresh opus·xhigh auditor and an independent fable·high auditor. Blocking findings, all fixed:
>   - a 4.1 m crossing radius measured walking pace, so every radius is now ≤ 3.2 m and a calibration check is
>     added;
>   - stock tiles were spawning at the wrong height, so the spawn now reads the tile centre level and is smoked;
>   - Settled 4 and step 7 disagreed on the one-gated-kind case, so that case is now a stop-and-ask.
> - Pass 3 (cap): the same pair of auditors. Blocking findings, all fixed with narrow restatements:
>   - R had dropped v1.2's turn lines, so R now uses the full decisive zero-kg set;
>   - the wave tiles and pass 2's "shift" option left cliffs at tile seams, so the seam rule applies and the shift
>     option is removed;
>   - pass 2's spawn check covered the eval path only, so the training smoke now checks spawn height too.
>
> - Pass 4 (an extra pass you approved; the last): the same pair of auditors. Blocking findings, fixed:
>   - the 0.04 effect floor would have swallowed the turn lines, so it is now scoped to shares;
>   - the stock pyramid delivered ≈ half its labelled slope on the eval path, so the slope tile is a
>     constant-gradient frustum with dose assertions on the eval track.
>
>   **These pass-4 fixes were not re-audited.** The pass-3 fixes were read and stood.

## Context

The construction-site teacher closed **not accepted** (2026-10-31). Its hazard curriculum never reached the
evaluation band: levels stayed near 2 against an eval band of 5.4–9, and both teachers crossed 0 / 200 there. It
also tried to teach stopping at walls, which was never learned. This campaign is the smaller follow-on you asked
for:
- terrain the robot can always keep walking over: slopes, hills, steps up and down, potholes, gravel and rubble;
- no walls and no obstruction that forces a stop.

Starting point: **TDR**, the omni teacher of record (`policy/rsd_9_p3b_rodspeed_dr/model_28991.pt`, sha 03c3e472),
merged to `origin/main` at 10:15 today (`3f944c4b`).
- TDR = T\* fine-tuned for 3000 iterations (1.22 h) with rod-speed DR U(0.6, 1.35).
- It trains on the wall-edged "field": 3–9 cm bars, slabs and pits, 20 columns × 10 levels of 10 m tiles, half of
  them flat.
- It uses obs layout v1 with a 9 × 12 scan at 0.3 m.
- Its curriculum promotes when tracked / commanded distance > 0.7 and demotes when it is < 0.4. Episodes with
  under 1 m of commanded travel neither promote nor demote. Levels are not saved in checkpoints.

Literature review (12 / 12 claims verified, 111 references, fable final check applied):
`lit-review-rough-terrain.md`, filed at step 0. What it changes here:
- **Blind is enough for most of this terrain.** Slopes, hills, gravel, rubble and steps below ≈ 0.12–0.14 m are
  blind-solvable. Wider potholes, larger drops and taller step-ups separate scan-using policies. So this campaign
  buys **robustness on traversable terrain**, not perception.
- **Envelope sized to Krabby** (walking root height 1.05 m, drop reach 0.46 m, rise reach 0.82 m):
  - slopes up to 20°, bounded by friction;
  - steps up to 0.30 m and down to 0.35 m;
  - pothole depth up to 0.35 m and width up to ≈ 0.6 m;
  - gravel as low friction plus 2–5 cm roughness. No particle simulation.
- **Layout:** centre-spawn tiles where every heading meets the feature.
- **Warm start, not from scratch:** fine-tune from TDR with the old terrains kept in the mix.

The site campaign stalled because it required a 3 m "crossing" before promotion, gave each kind 1/20 of the
columns, and reset levels on resume. This plan keeps main's ratio rule, gives each kind 2/20, and trains in one
run.

The reusable code sits on the unmerged site branch (`worktree-sen0581-cameras`): the ledge, pothole and rubble
generators, the kind raster and per-kind telemetry. A test merge onto `origin/main` conflicts in two files, but
the branch also carries the camera code and layout v2, so the plan **ports a trimmed file set** instead of merging.

Campaign dir: `crab_hex_omni_task/experiments/2026-11-10_rough_terrain/`, on a new branch from `origin/main`.
Code is default-off; commits are local; nothing is pushed or baked without your yes. The GPU is local, so budgets
are in wall-clock hours.

## Settled (from you, 2026-11-10)

1. No walls, blockers or obstructions that require stopping. The terrain is slopes, hills, ledges and steps,
   potholes, gravel and rubble.
2. The literature review informs the design (done).
3. The DR on main (rod-speed DR, TDR as teacher of record) is the base and is kept in training.
4. **G is judged per kind**:
   - at least 2 gated kinds gain, and none regresses;
   - bands above the blind envelope are report-only;
   - bands are chosen on an independent eval seed (es3) and G is read on es1–2.
5. **Perception extras are set aside to P3**: the scan-ablation knob port, the `_flat` twin rows, the site
   recorder port and the new teacher's band-above report rows.
6. **After the pass-3 cap:**
   - one more audit pass, then stop whatever it finds;
   - the plan size is kept as is;
   - the iteration-1500 early read is a single log line, not a script;
   - TDR's three above-envelope baseline rows in step 6 are kept (report only).

## Design

- **Terrain set: 20 columns.**
  - 3 flat and 3 TDR field (retention, 30 %).
  - 2 each for seven kinds. Each ladder spans levels 0–9:

    | Kind | Ladder (d = 0 → 1) | Source / layout |
    |---|---|---|
    | slopes | 0 → 20° constant gradient | constant-gradient frustum, up (sunk pad) and down (raised pad); rim at the pad edge (≤ 2.0 m in every heading); slope band to the outer flat at level 0; pad depth or height ≤ 0.8 m |
    | hills | smooth waves, realised max slope 0 → 20° | stock Isaac Lab wave terrain plus a spawn pad |
    | steps_up | 0.02 → 0.30 m | site `ledge_up`, sunk pad |
    | steps_down | 0.02 → 0.35 m | site `ledge_down`, raised pad |
    | potholes | depth 0.03 → 0.35 m, side 0.15 → 0.60 m | site `pits` scatter band, narrowed to r 2.0–2.7 m |
    | gravel | roughness 0.02 → 0.05 m, μ per the friction contract | rough tile |
    | rubble | 0.02 → 0.15 m cells | site `rubble` band, narrowed to r 2.0–2.7 m |

  - Rims are vertical.
  - Slopes and hills keep TDR's friction. The pyramid-axis bound needs μ ≥ √2·tan 20° = 0.52; TDR's effective μ is
    0.8–1.5.
- **Friction contract** (one setup, the same in training and eval).
  - Under a ground combine of "multiply", ground μ stays 1.0. **Every shape of an env** gets its effective μ,
    drawn together, with static = dynamic and restitution unchanged:
    - non-gravel envs: (b + 1) / 2 of TDR's 64 buckets b ~ U(0.6, 2.0), which is TDR's effective distribution;
    - gravel envs: U(0.2, 0.5), fixed per env.
  - The gravel bands differ in roughness only.
  - The privileged friction input is fed 2·μ_eff − 1, TDR's own mapping, so non-gravel inputs keep TDR's
    distribution and gravel reads −0.6 to 0.
  - **Eval regime per row family, identical for both teachers:**
    - retention rows and non-gravel kind rows run TDR's regime (knob off);
    - gravel rows run the contract (knob on, not stripped).
- **Curriculum.** Main's ratio rule is unchanged (`KRABBY_TERRAIN_PROMOTE 0.7:0.4`, 1 m floor), with per-kind level
  and vote telemetry from the site code.
  - The rule promotes episodes that end in a fall and never requires crossing. In TDR, levels saturated and wrapped
    within ≈ 300 iterations. So **level is never read as competence**; the gates read eval rows.
  - The run starts at levels 0–2 and is one continuous process.
- **Training.** Resume TDR on TDR's recipe with three changes: the terrain set and friction contract above,
  rod-speed DR 0.6:1.35 with no ramp, and **6000 iterations (≈ 2.5 h)**. Payload 0–180 kg, the command pool,
  std 0.80, seed 19 and layout v1 stay as they are. The seed is the same, but the config differs, so this is a new
  sample, not a replay.
- **Eval.** Deterministic rows on the field_hard 8-hold, 80 s schedule, 100 episodes per row, paired on env_index.
  - Per kind, three bands: d 0–0.3, 0.3–0.6 and 0.6–1.0.
  - **Success (the one kind-row statistic)** = schedule complete AND crossed.
  - Crossed = the maximum root distance from the spawn ≥ the kind's crossing radius:
    - pad edge + 1.5 m for steps (3.2 m);
    - 3.2 m for potholes and rubble (band outer edge 2.7 m + 0.5 m);
    - for slopes, the rim + ≥ 1.0 m (≤ 3.2 m);
    - 3.0 m for hills and gravel.
    - Every radius is ≤ 3.2 m, which is ≤ 70 % of the schedule's 4.66 m commanded reach, so crossing tests
      traversal, not walking pace.
    - It is computed from the raw `root_pos_w` the harness saves. No recorder is needed.
  - "Stuck" = complete AND not crossed, reported per row.
  - The legacy retention rows keep plain completion; there TDR walks 4.3–4.8 m.
  - **Blind envelope:** a step, drop or pothole band whose bottom exceeds 0.14 m is above the envelope. Those are the
    top bands of steps_up (0.19–0.30 m), steps_down (0.22–0.35 m) and potholes (depth 0.22–0.35 m).
- **Left out** (say if you want any):
  - the layout-v2 scan;
  - a looser pitch termination;
  - a critic-only warm-up or KL anchor;
  - windowed or ratchet promotion;
  - a second training seed;
  - terraces and snags;
  - plus settled 5's set-aside items.

## Steps

0. **Set up.**
   - Create a worktree and branch `feat/rough-terrain` from `origin/main` `3f944c4b`, and the campaign dir.
   - File the literature review, this PLAN and a CHARTER log.
   → haiku · medium — exact file moves and one git command; the paths are given.
1. **Port the trimmed site code onto the branch.**
   - Take the files at the site head, `2a3e3c51`, not the commits: site terrains, the omni site cfgs, the generator
     8-tuple path, `site_raster`, spawn pad level, and the per-kind ledger and telemetry.
   - Leave out layout v2, blockers and corridor slack, site DR (main's rod-speed DR covers rod scale), the critic
     group, the camera-rig commits, the scan-ablation knob and the recorder.
   - Reconcile with main's rod-speed DR (pinned method text, `_cam_scale` name).
   - Port the CPU tests; the full unit suite must stay at main's failure set.
   → opus · high — cross-module port into the policy-of-record path; a silent change to TDR's path is the risk.
2. **New terrain kinds, ladders and the friction contract.**
   - Build the slope tile as a constant-gradient frustum: h = g · clip(r_out − r, 0, r_out − r_rim), up and down,
     with the outer flat at 0. The stock pyramid is not used: its hyperbolic plateau halved the realised dose.
   - Wire the stock wave generator as an omni sub-terrain, plus a spawn pad.
     - Every new tile gives the spawn its centre level, as the site pads do.
     - Waves use ≥ 2 waves per tile, with the amplitude tapered to 0 at the seam ring.
     - Every tile's walkable level is 0 wherever max(|x|, |y|) ≥ 4.4 m (the site seam rule), so tile borders carry
       no wall or cliff.
   - Retune the potholes and rubble ladders.
   - Implement the friction contract and the 20-column mix knob.
   - CPU tests on the **realised** height fields per kind at levels 0, 4 and 9:
     - slopes:
       - rim distance ≤ 2.0 m in every heading;
       - median gradient from the rim to the crossing radius ≥ 0.9 × the table value;
       - max gradient along the commanded eval track (0,0) → (2.0,0) → (2.0,1.76) → (3.41,3.17) ≥ 0.9 × the table
         value;
     - wave max gradient;
     - step height, pothole depth and side, rubble height, and roughness amplitude, each within ±10 % of the table
       and ≤ cap;
     - every heading from the spawn meets the feature within 3 m;
     - the spawn level equals the realised centre height of each tile, and the lowest walkable level keeps the root
       above the −0.25 m termination with ≥ 0.5 m margin;
     - the walkable level is 0 ± 0.05 m on the seam ring of every tile;
     - knobs off → config dump identical to main.
   → opus · high — the geometry and the friction model the result rests on.
3. **Telemetry.** Log per-kind mean level, votes, fall share and crossing share. It serves a G-fail diagnosis. At
   iteration 1500 the watcher logs one notification line with each kind's fall share against iterations 300–500
   (settled 6: no separate script; **report only; no stop, no remedy**).
   → sonnet · high — specified; the step-5 smoke shows the telemetry logs.
4. **Eval rows and the read-out.**
   - A manifest generator for 7 kinds × 3 bands.
   - Retention rows reuse TDR's G1 / R manifests: field_hard, field_light, ring_obst, field_turn, suite, side22,
     field_light at 90 and 135 kg, and the rod-speed ×0.7 L090 field_light cell.
   - The read-out computes success and stuck from raw `root_pos_w`. It keys rows by checkpoint sha256 in
     `run_meta.json` (never by row-name prefix: RSDT_72d63d51 is the no-DR control, not TDR) and computes the gates
     below.
   - Validation:
     - it reproduces TDR's G1 numbers to 1e-12;
     - its crossed rule reproduces the site read-out's completed_and_crossed on saved T\* site rows (at the site
       radii).
   - Radius calibration on saved band-free rows (TDR field_hard, field_light and the ×0.7 field_light cell, es1–2):
     ≥ 0.95 of completers must reach every kind's radius. Otherwise that radius is lowered before step 6.
   → opus · high — the read-out decides the verdict; a wrong statistic is quiet.
5. **GPU smokes and neutrality** (≈ 0.7 h).
   - TDR on its saved `field_light` L090 es1 row with every new knob off, at the new build: bitwise equal, or the
     class rule from the eval process-state note.
   - A 50-iteration training smoke on the full mix. It must assert:
     - every kind builds;
     - the realised column count and env share per kind;
     - every shape's μ per env (gravel 0.2–0.5, others TDR's distribution);
     - the privileged friction input = 2·μ_eff − 1;
     - one slide or tilt contact check on a μ 0.2 and a μ 1.5 env;
     - telemetry logs for every kind;
     - the obs width stays 1149;
     - per kind in the mixed grid, the root height above the local surface at the first step after reset is within
       ± 0.05 m of the spawn height, and no env terminates within its first 2 s.
   - A 2–4-episode eval-harness smoke per kind × band, asserting:
     - the realised heights or scan range and μ_eff in the eval env (the gravel knob is not stripped);
     - the three slope assertions from step 2, repeated on the realised eval tiles;
     - root height above the local surface at step 1 within ± 0.05 m of the configured spawn height (1.085 m,
       `crab_hex_scene_cfg.py:167`);
     - no episode terminating in the first 2 s.
     - It also reports the fall share per kind × band.
   - The code and eval sha are frozen from here through step 9.
   → main — queue and watcher; results go to the CHARTER.
6. **TDR baseline** (≈ 3.5 h).
   - TDR at **es3** on all 21 kind × band rows, for band choice.
   - TDR retention rows: field_hard es1–4 and the rest at es1–2, at the new build.
   → main — queue.
7. **Freeze the gates (STOP 0), before any new-teacher number.**
   - Per kind, the gate band is the lowest **in-envelope** band where TDR's es3 success is ≤ 0.80.
   - A kind with no such band is "solved": it enters R on its highest in-envelope band.
   - Bands above the envelope are report-only.
   - k = 2 (Settled 4).
   - With 0 gated kinds: close per the decision rules.
   - With exactly 1 gated kind: G as settled cannot pass, so stop and ask before step 8.
   - Write `thresholds_rough_v1.json` and record the TDR baseline table.
   → main (opus) — the call the verdict rests on.
   ✓ fable · high — independent check of the gate bands and statistics against the baseline rows.
8. **Train** (≈ 2.5 h).
   - Resume TDR for 6000 iterations on the mix, through the detached queue and a log-sentinel watcher.
   - The iteration-1500 notification line (step 3): report only.
   → main — queue, watcher, decide-record-notify.
9. **Gate read** (≈ 4 h of rows).
   - Both teachers at es1–2 on every gate band and solved-kind R band.
   - The new teacher on the retention rows: field_hard es1–4, the rest at es1–2.
   - Then the read-out and the decision rule.
   → opus · high — runs the read-out and writes the verdict entry.
   ✓ fable · high — independent re-derivation of the verdict from the rows.
10. **Close.**
    - REPORT, CHANGELOG, LOG and memory; local commit; no push.
    - If accepted: propose the follow-on phase (student re-distil, stand re-pair, bake) for your approval.
    → sonnet · medium — records written from the CHARTER and read-out.

## Gates (frozen at step 7, read at step 9)

| Gate | Rows (n) | Pass line |
|---|---|---|
| G. gain, per kind | Each gated kind's gate band, es1–2 (200 per kind); success statistic | ≥ k gated kinds each gain new − TDR ≥ max(0.10, 3·SE) with CI lower bound ≥ 0.05. No gated kind is below TDR − max(0.05, 2·SE) |
| R. retention | field_hard es1–4 (400); field_light, ring_obst, field_turn es1–2; field_light at 90 and 135 kg es1–2; rod-speed ×0.7 L090 field_light es1–2 (completion). Solved kinds' highest in-envelope band, es1–2 (success) | new ≥ TDR − max(0.05, 2·SE) |
| R. tracking and turning | suite, 0 kg, es1–2 | v1.2's full decisive zero-kg set, `zero_kg_decisive_stats(fwd/side/back/diag, "track_err")`, exactly as TDR's G1 R read it: track_err per translate category, the move_turn ratio drop, survival per translate category, and the turn facing_closed and yaw_progress drops. Each line: new worse than TDR by ≤ max(R, 3·SE), with that line's v1.2 R and drift term 0 |
| R. side | side22 | FAST / FAST |
| report only | TDR above-envelope bands (step 6); stuck share; foot slip per kind (`slip_metrics`); fall classes | none |

Differences of 0.04 or less in a success or completion share are never read as effects. The R tracking-and-turning
lines use only their own v1.2 margins.

**Decision rules.**

| Outcome | Action |
|---|---|
| G ∧ R | Accept as the rough-terrain teacher; propose the follow-on phase (re-distil, stand re-pair, bake) |
| G passes, R fails | HALT, you decide (a retention remedy such as a critic warm-up or KL anchor is a new run) |
| G fails | Not accepted; CPU diagnosis; HALT, you decide |
| No gated kind (every kind solved in-envelope), decided at step 7 | Report "TDR already handles this terrain"; steps 8–9 do not run; close |
| Exactly 1 gated kind, decided at step 7 | Stop and ask before step 8 (k = 2 cannot be met) |

## Budget (wall-clock, local GPU)

| Item | Hours |
|---|---|
| Steps 1–4: code, tests, read-out (agent time, no GPU) | ≈ 4–6 |
| Step 5: smokes, eval smoke, neutrality | ≈ 0.7 |
| Step 6: TDR baseline (≈ 39 rows at ≈ 0.09 h) | ≈ 3.5 |
| Step 8: training, 6000 iterations | ≈ 2.5 |
| Step 9: gate rows (≈ 45) | ≈ 4 |
| **Total** | **≈ 15–17 h; ≈ 11 h GPU** |

## Later phases (not run; planned on acceptance)

- **P2. Pair and bake.** Student re-distil (≈ 9.6 h), stand re-pair (≈ 7 h), bake proposal.
- **P3. Perception.** The scan-ablation knob, `_flat` twins, the recorder and the band-above rows (settled 5). Then
  v2 by zero-init surgery, or a v2 student, if scan-limited rows show.
- **P4. Lost-foothold levers.** Looser pitch termination, ≥ 4 loaded feet, stance slowing, for the above-envelope
  bands.
- **P5. Loose ground** (user, 2026-11-12; not scoped). The gravel here is rigid low-μ rough ground. Candidate levers:
  support-failure or sinkage events, step-to-step and load-dependent footing, ground compliance DR, an RFT force
  hook. Details in CHARTER, 2026-11-12.

## Verification

- CPU:
  - ported and new tests;
  - realised-dose tests;
  - main's unit-suite failure set unchanged;
  - the read-out reproduces TDR's G1 numbers and the site completed_and_crossed;
  - config dumps identical with knobs off.
- GPU (step 5):
  - TDR neutrality row;
  - the training smoke (shares, all-shape μ, friction-input mapping, contact check, telemetry, width);
  - the per-kind × band eval smoke.

Routing: strong · roster ok (opus = Opus 5.5, fable = Fable 5.2) · planned on opus · ceiling: opus · 1× haiku·medium, 1× sonnet·high, 1× sonnet·medium, 4× opus·high, 5× main, 2× fable·high check, fable retry on 2 fails
Effort enforcement: per call

## Left for implementation (deferred audit findings; check each)

1. **Gravel roughness.** Main's knob loop forces `noise_range = (0.02, 0.02)` on every sub-terrain
   (`crab_hex_env_cfg.py:368`). Build the gravel 0.02 → 0.05 m roughness separately, and dose-test it.
2. **Difficulty mapping.** TDR's `FLAT_TERRAIN_DIFF 0.20:1.0` reaches stock tiles raw, so their level 0 = d 0.2.
   Run the step-2 dose tests on the composed training cfg. State which knob sets each eval band: site kinds use
   `KRABBY_SITE_DIFF`, stock kinds use `FLAT_TERRAIN_DIFF`.
3. **Pothole band.** The narrowed band rejects large yawed pits silently. Assert the placed cover per angular
   sector, and retune the yaw or cover.
4. **Crossed distance** is horizontal (xy), measured from the spawn.
5. **Gravel μ draws** come from a seeded generator, so env_index pairing holds across the two teachers' processes.
6. **Not adopted from the lit review:** the payload cap of 135 kg on the top rungs, and an initial std below 0.80.
7. **R margins** use drift term 0 (teacher vs teacher). If an R line fails, read it against v1.2's drift term
   before the HALT. Read the ×0.7 cell against its 0.07 stressed-cell resolution.
8. **PYTHONPATH.** Check `/proc/<pid>/environ` of the new worktree's queue before steps 5 and 8.
9. **Smoke length.** The 50-iteration smoke ends few episodes. Force resets, or lengthen it, so the per-kind
   telemetry and spawn assertions have data.
10. **Slopes rows** state the up / inverted share.
11. **Termination reasons.** Report them on the slope and hill rows: the 0.5 rad tilt limit sits 0.15 rad above
    a 20° incline.
12. **Read-out keys rows by checkpoint sha256** (RSDT_72d63d51 is the no-DR control).
13. **Friction input in eval.** The privileged friction input acts through training only; teacher eval runs the
    history encoder.
14. **Borderline band picks.** If TDR's es3 pick is borderline (0.74–0.86), record it. A report-only next band is
    not run (settled 5).
15. **"Every heading meets the feature"** for scattered pits = pit cover per angular sector.
16. **Slopes, hills and gravel.** "Crossed" at 3.0 m is ≈ 64 % of the commanded pace. Report the r_max distribution
    beside success on those rows.
17. **Flat-ground reward terms.** `parkour_isaaclab/envs/mdp/rewards.py:564, :596, :1331` take the ground as root z
    plus a fixed offset. Check whether they are live in TDR's recipe, or swapped for the terrain-aware twins
    (`omni_clearance.py:117`), before training on slope and hill tiles.
18. **Spawn pads against the 1.46 m toe reach.** The slope and hill pads must hold the stance without toes on the
    incline or inside a rising wave. Blend the hills pad into the wave.
19. **R via `gate_read.zero_kg_check`.** Pass `its_past = 0` (drift off) explicitly; the judge fails closed on a
    missing drift. Its field-row completion lines use v1.2's looser margin; the plan's retention row
    max(0.05, 2·SE) is the decisive one. Continuation drift on yaw_progress (≈ 0.012 at 6000 iterations against
    R 0.0066) can trigger an R HALT by itself; read it per item 7.
20. **Training smoke 2 s check.** Scope it to spawn faults: compare against the knobs-off termination rate, since
    μ 0.2 gravel envs may fall legitimately.
