# Side bisect: root cause of the slow omni side gait (2026-09-30)

The user asked: "We seem to have eliminated most reasons why the lr runs would always result in a slow gait. Run an
analysis on what we have tested so far and aim for the root cause of the slow gaits."

This is CPU-only work on existing checkpoints, eval `raw/episode_*.npz` files and tfevents. Nothing was trained and
nothing touched the GPU. Four probe passes fed it (symmetry decomposition, network representation, training dynamics,
actuation), plus three checks of my own. Where the probes disagree, section 2.10 says which one I trust and why.

**Sources.**
- Records: [CHARTER.md](CHARTER.md) (arms and pre-registered rules), [GATE.md](GATE.md), [ANALYSIS.md](ANALYSIS.md),
  [CHECKS.md](CHECKS.md), [../2026-09-28_side_speed_investigation/REPORT.md](../2026-09-28_side_speed_investigation/REPORT.md),
  and the three closed stroke/centring campaigns.
- Data: per-arm classifier JSONs in `parkour/outputs/omni_plan/side_bisect/arms/`; eval stamps; training runs
  X01–X19 under `parkour/logs/rsl_rl/crab_hex_omni_flat_walk/2026-09-29_*` and `2026-09-30_*`.
- Probe scripts and small outputs are copied to [scripts/rootcause/](scripts/rootcause/) (`sym/`, `rep/`, `dyn/`,
  `act/`, and the independent checker's `check/`). Large caches were not copied.
- **Independently checked** by a fable·high pass (`scripts/rootcause/check/`): every pooled odd/even/static number in
  section 3 was recomputed from the raw npz and matched to two decimals; the code citations, the tfevents numbers and
  the student probe reproduced. Its two blocking findings and ten minor ones are applied below, and section 1 is
  written at the level the data supports.

---

## 1. The root cause, at the level the data supports

**What is established (high confidence).**
- Sideways travel on this robot is a hip/knee rod stroke locked to the tripod gait phase (section 2.1). Under the exact
  left-right mirror, a stroke that pushes at speed is direction-flipped and phase-locked: at a fixed clock phase the
  same leg strokes the opposite way on the two hands (section 2.2). The static odd part, the lean, also pushes, but
  only at the 0.06–0.10 m/s the slow gaits reach, and without a stance sweep.
- **In every arm where one network is trained on both hands, that consistent direction-flipped stroke never appears.**
  Its pooled odd RMS in the rod commands is 0.15–0.93, against 1.50–2.16 in every fast gait (section 3). The stepping
  itself is intact (the even part is as large as in fast gaits), and direction rides a static carrier (a knee-bias lean
  or corner-cam spin), which is what makes the lean-and-shuffle.
- "Both hands" covers two families: real −y episodes (LRfa, LR4, LR4_dwell, C_lr, C_lr_10, C_lr_rsi, C_lr_10_rsi,
  C_lr_rsi33, W) and the −y images the 4-image mirror loss adds to a +y-only run (O, R, C_band, C_rsi). These are
  mechanistically different (an RL signal vs a detached-target regularizer), so "one network serving both directions"
  is an umbrella over two mechanisms with one symptom.
- One-direction networks with no −y image form the stroke by about it 600 (4 of 5 runs FAST, 1 INTERMEDIATE by
  0.002).
- Mirror canonicalization supplies the sign flip outside the network and walks FAST on both hands with no training
  (4 of 4 heads). So nothing in the plant, the sim, the mirror maps or the −y hand itself is the obstacle.
- What does NOT change the outcome: reward economy, band, RSI on/off, fast-gait RSI banks at 0.2 and 0.33, episode
  length, within-episode reversals, mirror type, and network capacity (a fresh network of the deployed architecture
  fits the two-hand fast policy to 99 % sign agreement in 40 CPU epochs).

**Why PPO does not build the stroke in a two-hand network is NOT established.** Three candidates fit all the data,
each at medium confidence:
- **(i) The interaction-order barrier.** From rest, the stroke's sign must come from the command, so the first stroke
  needs a direction × phase interaction in the network. Its gradient cancels across hands on any direction-independent
  feature; the even stepping and an additive carrier get same-signed gradients from both hands and win the race. Once
  a stroke exists, direction is legible in the proprioceptive state (the distilled student reads it there, not from the
  command), so this is a formation barrier only.
- **(ii) Rate × early lock-in.** A hand-specific stroke gets at most 40 % of the samples while the hand-agnostic shuffle
  plus an additive carrier gets all of them; 70 % of rod means are already past the clip at it 300, before the classes
  separate (it 540), so the first solution found is the one kept. The two-direction fa arms are not mirror-equivariant
  (M_y residual 0.25–0.40 vs 0.04–0.05 for LR4), so nothing forced them to share features across hands; they were free
  to learn C's +y stroke on their +y rows alone and did not, which (i) as stated does not explain.
- **(iii) For the phantom arms only,** the mirror loss pulls the +y output toward its −y image throughout training (O's
  symmetry loss rises from 0.037 to 0.070 while the fa arms end at 0.030–0.037), a standing conflict with +y RL.

The deciding pair for the real-−y family, **C vs C_lr**, changes three things: it adds side_r to the pool, halves the
+y share (80 % → 40 %), and makes the direction visible. Section 5 gives the zero-code controls that split (i) from
(ii): **C_40** (one hand at a 40 % share) and **C_fa** (one hand, direction visible).

**What this means for the fix.** Whichever of (i)–(iii) is right, the fix is to relieve the network of the two-hand
problem: canonical training (the network only ever sees +y states, the flip is applied outside it), or a policy
switch, or distillation from the canonical controller. Canonical deploy already working is consistent with every
candidate, because it removes the two-hand requirement rather than explaining it.

---

## 2. The causal chain

### 2.1 Sideways thrust is a phase-locked rod stroke; fore-aft thrust is the cams (confidence: high)

- **Where the speed comes from** (ACTUATION, `act/kin.py`):
  - In canonical C's fast lat_l gait (0.215 m/s), the rod radial stroke gives 0.189 m/s, cam yaw 0.014 and
    loaded-foot slip 0.013.
  - In W's forward gait (0.203 m/s), cam yaw gives 0.175 m/s and the rods 0.011.
- **The stroke amplitude** (how far each foot's radial reach swings in time with the tripod cycle) separates the
  classes perfectly over 27 hand rows: FAST 0.077–0.113 m, SLOW/DEAD 0.010–0.057 m. It correlates with feet speed at
  r = 0.94.
- **Why the hardware treats the two axes differently.** The two actuator types:

  | Actuator | Control | Limits | Mechanism | Code |
  |---|---|---|---|---|
  | Hip and knee rods | position targets | slew-limited lead screws, 28 / 33 mm/s | a constant command gives a posture | `crab_hex_forward_task/mdp/crab_hex_dimensions.py:165,173` |
  | Cams | velocity channels, ±π rad/s | ±25° throw | Whitworth quick-return, so a constant command gives a periodic yaw stroke | `config/crab_hex/agents/parkour_mdp_cfg.py:74,89-104`; `mdp/crab_hex_cam_mapping.py:105-117`; `crab_hex_dimensions.py:131` |

### 2.2 Under the exact left-right mirror, only the direction-flipped phase-locked part can push sideways (confidence: high)

- **The map M_y** (`crab_hex_forward_task/mdp/crab_hex_mirror.py:49-56`; `crab_hex_omni_task/mdp/crab_hex_omni_mirror.py:36-39,149-150`):
  - legs swap left and right;
  - the joint signs are cam −1, Body_Hip −1, Hip_Femur +1 and Femur_Tibia −1;
  - the clock slots flip, because the tripod sets swap, which moves the schedule by π.
- **Split into two parts.** Split any gait pattern into an even part (equal to its own M_y image) and an odd part.
  - The even part pushes equally in both lateral directions, so its net push is zero.
  - Net lateral travel must therefore come from the odd part.
- **Physically,** at a fixed clock phase the same leg has to stroke the opposite way on the two hands.
- **Measured on the canonical fast heads** (a map/sim exactness check, since R is M_y of L through the same
  network, so the cross-hand cosines are −1.00 / +1.00 by construction; the physical content is the loading vs stroke
  contrast):
  - Foot loading is the same on both hands (correlation +0.80 to +0.97), while the radial stroke reverses (−0.64 to
    −0.92).
  - The right hand equals the M_π-mirrored left hand to within 0.05–0.13, against split-half noise of 0.10–0.24.
  - The canonical L rows are bit-identical to C's driver rows, so the "27 hand rows" of section 2.1 contain 8
    duplicates.

### 2.3 So a two-direction policy must output sign(vy) × g(phase) on the rods, and hardware supplies this for fore-aft but not for lateral (confidence: high for the requirement, medium for the fore-aft contrast)

- **The requirement.** A one-direction policy needs only g(phase). A two-direction policy needs the product.
- **Fore-aft gets it from the cams.** In W:
  - The leading corner pair's cams spin continuously: FL/FR at −2.4/+2.4 rad/s going forward, RL/RR at +1.7/−1.8 rad/s
    going back. They contribute the least thrust of the six legs (0.017–0.018 m/s each forward), so they act as the
    direction carrier.
  - The middle cams, which push most, flip their timing with direction at a fixed phase (action correlation −1.0; per-leg
    yaw profile −0.89 to −1.00, `act/out_foreaft.txt`).
  - Flipping only the forward-command slot moves the rods mostly statically: 0.30–0.40 static against 0.09–0.19
    phase-locked in t3/c6 (`rep/probe_cflip.txt`). So the fore-aft reversal is carried by the cams' own dynamics and
    state feedback, not by a learned command × phase gate.
- **The rods have no equivalent.** A first-order rod bias is a posture, and that posture is the lean.
- **Status.** This contrast explains why W's back tracks like its fwd (back 0.784, fwd 0.785) in the same networks
  that fail sideways. It rests on 3 checkpoints and has not been tested by intervention.

### 2.4 Every network that must serve both directions lacks the odd stroke, and every fast one has it (confidence: high, descriptive)

- The signature table is in section 3.
- **Separation.** The *consistent* (pooled) rod-command odd RMS separates the classes with a gap of 0.57 (1.6×):
  FAST 1.50–2.16, INTERMEDIATE 1.46–1.53, SLOW/DEAD 0.15–0.93. Its Spearman correlation with feet_ss is 0.81 over all
  rows. In joint-position space the same statistic is 0.069–0.158 rad slow vs 0.203–0.269 rad fast (1.3×, the same
  ratio as the sweep); the action-space gap is inflated by saturated actions. It is the same physical fact as the sweep,
  measured in the policy's commands instead of the feet.
- **Stepping is intact.** The even stepping in slow rows is as large as in fast ones (0.82–2.30 against 1.41–1.83),
  so slow gaits are the fast stepping minus the stroke.
- **The residual odd part is mostly not a weak stroke.** Its cosine to the fast stroke is 0.07–0.49 (checker: |cos|
  0.02–0.34 against C and C_mo_s2, while those two fast heads agree with each other at 0.84). The exceptions are O and
  R, at 0.42–0.76.
- **The result is robust** with 12, 24 or 36 phase bins and with the clock or a data-derived phase. The pooled
  split-half statistic is the reliable one (halves agree within 0.03). Per episode, the medians are 2.2–2.3 fast vs
  0.93–1.31 slow above a phase-shuffle floor of 0.55–0.68 (itself an underestimate for autocorrelated bang-bang
  actions): slow episodes carry odd energy that does not cohere across episodes.

### 2.5 "Serving both directions" covers real −y data and the mirror loss's phantom −y rows (confidence: medium-high; confounded for the phantom arms)

**Real −y.** LRfa, LR4, LR4_dwell, C_lr, C_lr_10, C_lr_rsi, C_lr_10_rsi, C_lr_rsi33 and W. All are SLOW or DEAD on
every hand evaluated (W's right hand reads 0.101 at 0.22).
- The deciding pair is **C vs C_lr**. They share the fa mirror, RSI off and band 0.35. The change adds side_r to the
  pool, halves the +y share (80 % → 40 %) and makes the direction visible (`cmd_obs_scalar` removed; env.yaml diff
  lines 734, 2387–2390). C is FAST at 2 seeds; C_lr is SLOW on both hands. Three things move at once, which is why
  section 5 adds C_40 and C_fa.

**Phantom −y** (O, R, C_band, C_rsi: +y-only, 4-image mirror).
- **How the rows reach the policy.** The mirror loss (`parkour/scripts/rsl_rl/modules/ppo_with_extractor.py:432-456`)
  runs the actor on the M_y, M_x and M_xM_y images of each minibatch row. It pulls their means toward the mirrored,
  **detached** means of the original rows (449–452), with coefficient 0.5 (453–454; `agent.yaml:41-43`).
- **What those rows are.** M_y and R_z(π) turn a +y state into a −y state: slot 9 ×−1, the clock ×−1, legs swapped
  (`HEAD_SIGNS_LR`, `crab_hex_omni_mirror.py:149-150`). So the shared weights are trained to output the −y gait
  without any RL on −y.
- **The critic never sees them** (`use_data_augmentation: false`, `agent.yaml:40`).
- **The phantom rows alone halve the direction-specific part of the +y rod pattern.** Direction-specific fraction:
  O 0.42 and C_band 0.44–0.46, against 0.65–0.80 in fast +y runs (`rep/selfmirror.txt`).
- **The mirror loss is in standing conflict with +y RL.** O's symmetry loss rises all run, from 0.040 to 0.071. In
  the fa arms, fast and slow alike, it ends near 0.033.
- **The confound (H6).** The O → C_mo change also triples the M_x image's weight and zeroes slots 8–9.
  - The M_x tie (a residual; smaller = tighter) does not order with class: C's tie is tighter than O's (0.024–0.058 vs
    0.067–0.073) and C_mo_s1's looser (0.083–0.093), both FAST (`rep/probe_h6.txt`). So a tight tie is not necessary;
    H6 stays open for O, and only a no-mirror +y arm (ANALYSIS step 5c) separates the two.
  - **H6 is irrelevant to the lr runs.** LRfa and the C_lr family carry exactly C's fa mirror weight and are still
    slow.

### 2.6 Why PPO builds the carrier and the stepping but not the stroke (confidence: medium; inference — candidate (i) of section 1)

**The argument.**
- The stroke flips with direction. Its gradient on any direction-independent phase feature therefore has opposite
  signs on the two hands, or on a state and its M_y image, and averages toward zero.
- The even stepping gets the same-signed gradient from both hands. So does an additive direction channel (lean, cam
  bias), whose sign flips together with the command.
- The stroke can only grow through an interaction: a unit whose response to phase depends on direction. That path
  gets a weaker and noisier signal, and it loses the race.

**What supports it (consistent, not proof):**
- **The interaction is learnable when it is supervised** (`rep/distill_cpu*.txt`). A fresh network of the exact
  deployed architecture, direction visible, regresses the two-hand fast policy to:
  - 98.9–99.0 % held-out sign agreement;
  - the full stroke, 0.585 against the teacher's 0.583;
  - in 40 epochs (about 20 s), with 2 seeds.

  So capacity is not the problem. PPO's signal is.
- **One-direction networks form the stroke on schedule.** C and C_mo separate from the slow runs from it about 600.
- **Static direction carriers are present in slow rows**, at 0.26–2.47 action units, carried in one of three ways
  (cross-hand cosine of the static odd part):
  - flipped with the hand (LRfa, C_lr_rsi, LR4, LR4_dwell: −0.89 to −1.00);
  - partly flipped (C_lr: −0.39 to −0.59);
  - or a fixed handed posture shared by both hands (C_lr_10, C_lr_10_rsi: +0.77 to +0.86).
- **The command is used strongly from the start.** The slot-9 first-layer column norm is 2.2–2.9× the median proprio
  column by it 100, and about 3× at the end, in every direction-visible arm (`rep/wp_table.txt`). This refutes the
  "weak direction input" hypothesis (H1a). Part of the growth is generic, though: the one-direction arms grow their
  speed slot 10 by 1.4–1.75× at it 100.

**The chicken-and-egg (my check, `rep/zz_gate_student_probe.py`).** The working distilled two-hand student does not
gate on the command either:
- Its first-layer command swing is 0.27 of the within-hand spread (1 % of units > 1). The slow nets range 0.25–0.38.
- Its steady-state response to flipping the command slot alone is 0.067 of phase-locked rod change (0.138 in the first
  0.5 s of a hold). The slow nets range 0.05–0.17.
- It reads direction from the **state**: hand mean difference 0.79 of the spread, with 38 % of units above 1.

So once a stroke exists, direction is legible in the proprioceptive state and the gait sustains itself. From rest,
only the command carries direction, so the stroke's first appearance needs the direction × phase interaction. That is
a formation barrier.

Fast-gait RSI does not get around it, and the mechanism does not explain why. RSI supplies direction-legible states,
and the distilled student shows a stroke can be continued from state alone. Yet RSI episodes decay to the shuffle
within about 2 s in every RSI arm. The checker adds that RSI starts underperform normal spawns in *every* arm,
including the fast C_mo (edge reach 0.870 vs 0.980; failure 0.127 vs 0.021), so "RSI never takes" is evidence that
RSI is inert at init std 1.5, not evidence for this mechanism or for an absorbing basin. This is the least-supported
step of the chain.

### 2.7 The shuffle forms first and becomes absorbing (confidence: medium)

**It forms first.**
- Most slow runs reach 95 % of their final track ratio at it 186–876 (median about 450) and end at 0.46–0.60.
  C_lr_rsi33 is the exception: it climbs to it about 1700 (crossing 0.53 at it 960, ending at 0.582) inside the slow
  class, without forming the stroke (sweep 0.027).
- Fast runs pass through the shuffle's ceiling (about 0.53) at it 450–600 and keep climbing, to 0.75–0.87.
- The shuffle is the *quicker* solution: over its 100–350, O leads C_mo s3 (same economy) by 0.06–0.10.
- **The lean is a generic early solution, even in one direction.** Among fast heads, knee bias and speed go
  together:

  | Head | Knee bias | Speed (m/s) |
  |---|---|---|
  | C | 0.00 | 0.213 |
  | C_mo_s2 | 0.02 | 0.202 |
  | C_s2 | 0.22 | 0.194 |
  | C_mo_s1 | 0.36 | 0.182 |
  | C_mo s3 (INT) | 0.30 | 0.159 |

  In one direction the stroke competes on equal terms and usually wins.

**It becomes absorbing.**
- **Posture.** The lean parks both rows at their stroke ends: the trailing reach at 0.54–0.56 m, the leading reach
  about 0.10 m. The body sits 2–6 cm lower.
- **Later gains** are spread over smoothing (action_rate plus delta_torques: 30–70 % of the gain from [450, 550) to
  [2500, 3000), `dyn/gain.py`) and `penalty_foot_idle_when_forward` (+0.56 to +1.02 in the two-direction slow runs).
  The smoothing share is the same in fast runs (0.30–0.55), so it does not mark an absorbing basin. 4 of 12 slow runs
  give tracking back after their peak.
- **Saturation cannot select, and is present at formation.** Two statistics from `rep/probe_sat_mirror.txt`: the plain
  clipped share of rod means is 0.70 at it 300 and 0.84 at it 600; the share pinned past 1+2σ is 0.01–0.06 at it 300,
  0.15–0.27 at 600, 0.44–0.57 at 1500 and 0.62–0.72 at 2999. Both are identical in fast and slow runs. With 70 % of rod
  means past the clip before the classes separate (it 540), a lock is present *at* formation, which weakens a "forms
  first, then locks" sequence and supports candidate (ii) as much as (i).
- **RSI never takes, in any arm.** In C_lr_10_rsi the RSI-start failure rate stays at 0.15–0.19 for all 3000
  iterations while other episodes improve from 0.16 to 0.04. In C_lr_rsi and C_lr_rsi33, RSI episodes reach the edge
  less often than normal spawns (0.176 vs 0.235 overall; 0.245 vs 0.381 over its 2500–3000). The same deficit appears
  in the FAST C_mo (0.870 vs 0.980), so it is a property of RSI under this noise, not of the shuffle.

### 2.8 Reward does not select the class (confidence: high that it is not the cause; medium on penalty levers)

- **C and C_lr train on the same reward** and split FAST and SLOW.
- **The side hold pays for the fast gait.** In the CPU replay (`dyn/replay.py`, eval holds, training weights, clock
  and apex terms excluded), the fast gait earns more per side-hold second:
  - +0.25 to +0.43 /s over O;
  - +0.29 to +0.32 /s over the two-direction nets.
- **The training-average margin is thin.** It is +3–12 % without foot_idle, and the heavy-tailed foot_idle can reverse
  it. action_rate gives back 35–57 % of the tracking gain. So escape pressure is weak, but it points the right way.
- **More stroke income cannot tip it.** The three income campaigns added 3–7× less than tracking already pays, and R
  (the sideways economy) is SLOW.

### 2.9 Canonicalization supplies the interaction outside the network (confidence: high)

- For vy < 0 the deployed policy is π(s) = M_y π₊(M_y s). M_y swaps and flips the legs and shifts the clock by π,
  which turns the head's g(phase) into the reversed stroke at the same physical phase.
- **The result:**
  - 4 of 4 fast +y heads are FAST on both hands, with |L−R| ≤ 0.006;
  - odd parts have cross-hand cosine −1.00;
  - reversals keep speed, with falls at each head's baseline rate.
- **This rules out** a sim or plant asymmetry, a −y hand that is physically harder, and M_y map bugs. The one thing a
  one-direction head lacks for two-hand walking is exactly the sign flip at fixed phase.

### 2.10 Where the probes disagree, and what I trust

| Disagreement | Trust | Why |
|---|---|---|
| Rod odd *fraction* vs odd *RMS* (the fraction misclassifies R: 0.39 on the clock phase, 0.79–0.83 on its own 0.415 Hz rhythm) | **Odd RMS** | It is an absolute amplitude, needs no phase choice for R, and separates with a 0.57 gap. |
| ANALYSIS H1 ("direction rides a static carrier") vs the symmetry probe (static carriers do not separate the classes) | **Symmetry probe** | Static odd rod is 0.07–1.05 in fast rows (C_mo_s1 1.03) and 0.26–2.47 in slow ones. The lean is about 0 in C_lr_10 R and LR4_dwell. Knee bias correlates with speed at only −0.28. The carrier is a symptom, the missing stroke the invariant. |
| REPRESENTATION: "the command never becomes a gate" (command-flip response 0.07–0.17 vs a 0.31–0.34 "target"; first-layer command swing ≤ 0.38) | **Neither input probe discriminates** | The 0.31 target is the canonical wrapper, which mirrors the whole state. The distilled bare student, which walks the fast two-hand policy in open loop, scores 0.067 and 0.27, inside the slow range. The defect shows in the output decomposition, not in how the command is read. |
| "No two-direction net forms the product" vs LR4's odd part flipping with the hand (cos −0.94 to −0.97) and C_lr_10 changing its even stepping with direction (cos 0.67–0.71) | **Soften to "forms it only weakly and in the wrong shape"** | Odd RMS is 0.22–0.79 in two-direction rows, ≤ 53 % of the smallest fast value, and mostly not the fast stroke's shape. |
| Fore-aft: ACTUATION (the middle cams flip, so PPO learned a product) vs REPRESENTATION (the flip comes from cam-state feedback, not a command gate) | **Both, reconciled** | The fore-aft product exists and is carried through the cams' own dynamics. It is not formed from the command. |
| DYNAMICS "saturation locks the basin" vs ANALYSIS H2 "saturation is identical fast/slow" | **Both** | Saturation is 0.01–0.06 at formation (it 300), so it cannot select. It reaches 0.62–0.72 by the end, so it plausibly locks. The early-checkpoint probe uses final-eval states (off-distribution). |
| ACTUATION says "the back corner pair spins" in W fwd | **Corrected** | `act/out_foreaft.txt` with legs FL..RR shows the front pair spinning in fwd and the rear pair in back, i.e. the leading pair. |

---

## 3. The signature: symmetry decomposition across stamps

**Method.**
- Hold window [4, 10) s; classifier-valid episodes alive through 10 s (65–100 per row).
- 24 phase bins on the clock phase, or on a data-derived phase for W (clock hidden) and the sideways pins; each
  episode's mean is removed.
- "Odd" is the M_π-odd part of the 12 rod-command channels (clipped actions): O = (A − S·A[P](φ+π))/2. RMS values are
  in action units summed over channels.
- Cells are 0.22 / 0.25 m/s. L is +y and R is −y.
- Source: `sym/condensed.md`, `sym/final_tables.md`; the C_lr_rsi33 rows are from `sym/zz_rsi33lr.py` on its both-hands
  stamp (`parkour/outputs/omni_plan/side_bisect/lr/C_lr_rsi33/side22_25lr__omni/seed001/2026-09-30_17-42-44/`), with
  feet_ss and sweep taken from the classifier in `queue.log`.

| Stamp (trains) | Hand | Class | feet_ss | Sweep (m) | **Rod ODD RMS** | Rod EVEN RMS | Static odd rod | \|lean\| (m) |
|---|---|---|---|---|---|---|---|---|
| C (+y) | L | FAST | .217/.214 | .083/.083 | **1.90/2.03** | 1.44/1.54 | .18/.20 | .03/.05 |
| C_s2 (+y) | L | FAST | .195/.210 | .082/.090 | **1.57/1.74** | 1.70/1.82 | .63/.63 | .05/.05 |
| C_mo_s2 (+y) | L | FAST | .203/.210 | .086/.087 | **2.09/2.12** | 1.57/1.60 | .29/.32 | .01/.03 |
| C_mo_s1 (+y) | L | FAST | .184/.185 | .076/.069 | **1.54/1.53** | 1.75/1.79 | 1.03/1.03 | .07/.07 |
| C_mo s3 (+y) | L | INT | .162/.173 | .056/.057 | **1.46/1.53** | 1.88/1.92 | .87/.85 | .11/.11 |
| canon ×4 (heads above, via M_y) | L and R | FAST | .182–.217 | .074–.086 | **1.50–2.10** | 1.44–1.82 | .17–1.05, same on both hands | ≤ .07 |
| sideways pins ×4 | L | FAST | .195–.218 | .076–.109 | **1.94–2.16** | 1.41–1.80 | .07–.17 | ≤ .04 |
| O (+y, phantom −y) | L | SLOW | .085/.094 | .042/.047 | **.69/.74** | 1.45/1.59 | 1.89/2.10 | .18/.20 |
| R (+y, phantom) | L | SLOW | .075/.078 | .023/.023 | **.83/.84** | 1.04/1.09 | 1.05/1.21 | .16/.18 |
| C_band (+y, phantom) | L | SLOW | .093/.099 | .049/.051 | **.93/.88** | 1.79/1.79 | 1.83/1.84 | .21/.22 |
| C_rsi (+y, phantom) | L | SLOW | .069/.074 | .034/.037 | **.24/.15** | .88/1.02 | .87/.97 | .08/.10 |
| LRfa (±y) | L | SLOW | .061/.065 | .036/.038 | **.65/.65** | 1.87/1.90 | 2.00/2.02 | .26/.26 |
| LRfa (±y) | R | SLOW | .086/.088 | .033/.034 | **.75/.77** | 1.64/1.71 | 1.74/1.84 | .17/.18 |
| LR4 (±y) | L | SLOW | .085/.091 | .034/.035 | **.60/.67** | 1.92/1.88 | .50/.58 | .12/.14 |
| LR4 (±y) | R | SLOW | .082/.096 | .033/.033 | **.63/.72** | 1.90/2.01 | .52/.52 | .12/.13 |
| LR4_dwell (±y) | L | DEAD | .047/.058 | .029/.029 | **.30/.30** | .82/.95 | .93/.88 | .04/.06 |
| LR4_dwell (±y) | R | DEAD | .055/.028 | .029/.026 | **.30/.22** | .90/.83 | .86/1.01 | .05/.03 |
| C_lr (±y) | L | SLOW | .072/.077 | .034/.035 | **.59/.61** | 1.87/1.91 | .52/.55 | .19/.21 |
| C_lr (±y) | R | SLOW | .063/.073 | .033/.039 | **.45/.46** | 1.60/1.78 | 1.99/1.96 | .21/.22 |
| C_lr_10 (±y) | L | SLOW | .085/.085 | .050/.054 | **.56/.72** | 1.75/2.11 | .97/1.09 | .16/.16 |
| C_lr_10 (±y) | R | SLOW | .069/.074 | .039/.041 | **.74/.78** | 2.00/2.08 | .70/.59 | .01/.00 |
| C_lr_rsi (±y) | L | SLOW | .085/.085 | .035/.035 | **.30/.30** | 1.65/1.71 | 2.14/2.30 | .27/.27 |
| C_lr_rsi (±y) | R | SLOW | .069/.075 | .034/.037 | **.55/.43** | 1.51/1.62 | 2.47/2.44 | .21/.21 |
| C_lr_10_rsi (±y) | L | DEAD | .041/.042 | .043/.046 | **.46/.45** | 2.19/2.30 | .32/.26 | .07/.07 |
| C_lr_10_rsi (±y) | R | SLOW | .078/.081 | .043/.043 | **.55/.57** | 1.99/2.07 | 1.26/1.31 | .15/.16 |
| **C_lr_rsi33 (±y), new** | L | **SLOW** | .094/.103 | .027/.028 | **.79/.76** | 2.01/2.03 | .67/.86 | .21/.22 |
| **C_lr_rsi33 (±y), new** | R | **SLOW** | .069/.072 | .048/.050 | **.55/.70** | 2.01/2.04 | .67/.74 | .17/.18 |
| W (full omni, clock hidden) | L | SLOW | .096/.103 | .034/.035 | **.43/.41** | 2.05/2.04 | 1.96/2.06 | .25/.26 |

**Separation over all rows** (`sym/final_tables.md:100-109`):

| Statistic | FAST | INT | SLOW/DEAD | Separates? |
|---|---|---|---|---|
| **Rod ODD RMS** | 1.50–2.16 | 1.46–1.53 | 0.15–0.93 | **yes, gap 0.57 (1.6×); Spearman vs feet_ss 0.81** |
| Radial stroke amplitude (m; ACTUATION, 0.22 m/s) | 0.077–0.113 | 0.083 | 0.010–0.057 | yes; r = 0.94 with feet speed |
| Sweep (m) | 0.069–0.109 | 0.056–0.057 | 0.023–0.054 | yes, gap 0.015 (1.3×) |
| Foot-y odd RMS (m) | 0.170–0.270 | 0.185–0.196 | 0.033–0.123 | yes, gap 0.047 |
| Rod odd fraction (π) | 0.42–0.70 | 0.38–0.39 | 0.02–0.39 | only via R's phase choice |
| Rod EVEN RMS | 1.41–1.83 | 1.88–1.92 | 0.82–2.30 | no (Spearman −0.02) |
| Static odd rod / static share of odd / cam-spin static | overlap | | | no |
| Lean | −0.01 to 0.07 | 0.11 | −0.01 to 0.27 | no (it rules FAST out above ≈ 0.11, but does not define SLOW) |

The odd-stroke statistic reads the split in the policy's commands rather than in foot kinematics, and it reads the
same whether the carrier is a lean (C_lr_rsi), cam spin (LR4) or neither (C_lr_10 R). It is the same physical fact as
the sweep (1.3× in joint space); its mechanism content is that the stepping (even part, 0.17–0.27 rad in joint space)
is intact in every slow run, i.e. slow gaits are the fast stepping minus the stroke.

---

## 4. What is ruled out, and by what

| Candidate cause | Ruled out by |
|---|---|
| The plant or sim can't walk −y fast, or a M_y map bug | Canonical deploy: 4 of 4 heads FAST on both hands, \|L−R\| ≤ 0.006, odd parts exactly mirrored. The LR4 hands mirror within noise. Unit tests `tests/unit/test_crab_hex_omni_mirror.py`. |
| Network capacity or architecture | CPU distillation: the exact architecture fits the two-hand fast policy to 99 % sign agreement and the full stroke in 40 epochs, at 2 seeds (open loop). |
| Weak direction input (H1a) | Slot-9 weights at 2.2–2.9× the median proprio column by it 100 in every visible arm. The working student uses the command no more than the slow nets do. |
| Reward pricing or economy | C vs C_lr on the same reward. R SLOW. Three income campaigns. The replay: the fast gait already pays +0.25 to +0.43 /s more per side second. |
| Lateral band or cadence | C_band SLOW; C_lr SLOW with band 0.35. |
| RSI on/off; +y bank; two-direction fast-gait banks at 0.2 and 0.33 | C_rsi, LRfa, C_lr SLOW. C_lr_rsi SLOW. C_lr_10_rsi DEAD/SLOW. C_lr_rsi33 SLOW on both hands (L 0.094/0.103, R 0.069/0.072; odd RMS 0.55–0.79). C_mo FAST with RSI on. |
| Within-episode reversals; episode length | LR4_dwell (40-s single command) DEAD; C_lr_10 (10-s) SLOW. |
| Mirror type in two-direction training | LRfa (fa) and LR4 (4-image) both SLOW. |
| The body lean as the cause | LR4 (0.12) and C_lr_10 R (0.00) are slow without it. Knee bias vs speed r = −0.28, against 0.94 for the stroke. |
| Saturation or std as the selector | Rod-mean saturation 0.01–0.06 at it 300 and 0.15–0.27 at it 600, the same fast and slow. Std, entropy, adaptive LR and surrogate are the same in both classes. |
| Half the data per direction | O has all of the +y data and is SLOW. |
| Critic interference (for O) | O's critic never sees mirrored rows (`agent.yaml:40`). |
| Clock cadence law | W is clock-free and SLOW; C is clocked and FAST. |
| The ring world, relay, heading slots | O and C_mo share them and split. |

**Not ruled out:**
- H6 (M_x weight): affects only the phantom-arm explanation.
- A visible direction slot in a one-direction run: C_fa never ran. In 1b, slot 9 carries the same number as slot 10,
  so harm is implausible, but it is untested.
- The exact learning mechanism in link 6.

---

## 5. Untested predictions, and the cheapest decisive tests

**Predictions the root cause makes that have not been tested:**

| # | Prediction | Status |
|---|---|---|
| P1 | C_lr_rsi33 FAILS on both hands | **Confirmed while this was written** (queue.log 13:46, VERDICT FAIL). L 0.094/0.103, R 0.069/0.072; sweep 0.027–0.050; odd RMS 0.55–0.79; lean 0.17–0.22. Training ended at track ratio 0.583 (early read 0.522). |
| P2 | C_lr_10_rsi33 is SLOW or DEAD | training now (X20, launched 13:46) |
| P3 | Canonical two-direction training (C_lr_canon) is FAST on both hands, with odd RMS ≥ 1.4 | not run |
| P4 | A two-direction arm given the interaction as an input feature is FAST while its additive control is SLOW (test T2) | not run |
| P10 | C_40 (one hand at a 40 % share) is FAST by it ~600 under candidate (i); late or SLOW under (ii) | not run |
| P5 | In C_lr the odd stroke *never* forms: odd RMS stays ≤ about 0.9 at every checkpoint, with the static carrier present from the first moving checkpoint. C's odd RMS climbs past 1.4 between its 400 and 1000 (test T1). | not run |
| P6 | C_fa (+y only, fa mirror, direction visible) is FAST, ≈ C_mo | not run (drafted in `scripts/candidates_v4.yaml:256-268`) |
| P7 | A no-mirror +y arm (5c) is FAST | not run |
| P8 | The additive ±1 sign slot (5b) alone is SLOW. A weak prediction: the working student does not gate on the command either. | not run |
| P9 | The distilled two-hand student stays FAST under ~1000 its of two-direction PPO, i.e. the barrier is formation. The root cause holds either way; this prediction only decides whether a bare-network fix can be kept by PPO. | not run |

### Controls first: C_fa and C_40 (zero code, about 1.25 GPU-h each)

- **C_fa** (`scripts/candidates_v4.yaml:256-268`): +y only, fa mirror, direction visible, RSI 0.2, band 0.25. It
  answers whether a visible direction slot alone reforms the shuffle in a one-direction run. Predicted FAST (≈ C_mo).
  Any interpretation of a two-direction arm with the direction visible needs this first.
- **C_40**: C's overlay with `KRABBY_OMNI_CMD_POOL=stand:0.6,side_l:0.4` (one hand, scalar obs, the +y share of
  C_lr). It separates candidate (i) from (ii): under (i) the stroke forms on schedule (FAST by it about 600); under
  (ii) it forms late or not at all. The overlay is `KRABBY_OMNI_CMD_POOL=stand:0.6,side_l:0.4`,
  `KRABBY_OMNI_DIR_BAND=''` and `KRABBY_OMNI_SLOTS=0.6:0.4:0:0` (the pool's projection; the dry run refuses a
  mismatch), on C's other knobs.

### T1: checkpoint-series eval, formation vs collapse (zero code, about 0.5 GPU-h)

**Design.**
- Evaluate C (X02) on the lat_l `side22_25` row at model_200, 400, 600 and 1000 (its scalar obs cannot walk right).
- Evaluate C_lr (X14) on both hands (`scenarios_side_lr.yaml`) at the same four checkpoints.
- Use each arm's own driver-eval env through `eval_crab_hex_gait.py`. Add stochastic rollouts (`--sample-actions`, the
  training std ≈ 1.65) beside the deterministic ones for the early checkpoints, since a deterministic-mean eval of a
  checkpoint that trained under that noise is itself off-distribution.
- Then, on CPU, run the section 3 decomposition per checkpoint (`scripts/rootcause/sym/extract.py` + `sym.py`) and
  `rep/recon.py` for |μ| saturation.

**What it answers.** Whether the two-direction net *ever* forms the stroke, and how saturated it is while the gait
forms.

**Expected outcomes:**
- **Under (i) or (ii), formation barrier:** C_lr's odd RMS stays ≤ about 0.9 on both hands at every checkpoint; its
  static carrier and lean appear by the first checkpoint that walks (it about 200–400); C's odd RMS rises through 1.4 by
  it 600–1000, with lean ≤ 0.1.
- **Under "formed then lost" (collapse under two-direction PPO):** C_lr reaches C-like odd RMS (≥ 1.4) at some
  checkpoint and then decays. That would change the fix ranking: a distilled or warm-started bare network would not
  survive PPO, and only structural fixes would remain.

### T2: the interaction as a feature, with its additive control (about 60–100 lines + 2.5 GPU-h for the pair)

**Design.** C_lr exactly (pool stand 0.2 / side_l 0.4 / side_r 0.4, fa mirror, RSI off, band 0.35, direction visible,
1b from scratch, 3000 its, seed 3), plus two proprio channels: g(vy_b)·sin φ and g(vy_b)·cos φ, with
g = tanh(vy_b / 0.05).
- A linear readout of these channels *is* the fundamental of the direction-flipped stroke, sign(vy)·(a sin φ + b cos φ).
  The per-hand gradients on its weights add instead of cancelling, so the stroke becomes first-order, as in a
  one-direction net.
- **Mirror table entry: +1 under all images.** Under M_y both factors flip (slot 9 ×−1, clock ×−1); under M_x neither
  flips (`HEAD_SIGNS_FA` slots 9 and 15–16 are +1).
- **The additive control (ANALYSIS 5b):** the same arm with a ±1 sign(vy) slot (same mirror entry, no product). A
  FAST T2 is also predicted by "the direction input is too weak to become a gate"; the control separates the two.
- **Tests.** A CPU unit test that the new slots are invariant under all three images; knob-off identity; an
  `OVERLAY_DUMP_CHECKS` entry.
- **Implementation risk.** The actor slices the obs by index (`obs[:75]`, a 10×75 history view, the estimator slice
  `obs[207:216]`), and the live head is 89 dims. Either append the channels with every slice updated, or overwrite two
  slots that are structurally constant in this pool (candidate: cmd wz, slot 5, if the turn slot weight is 0; verify
  first). Pin the result with the test.

**Decision table** (same gate, both hands; read only after C_fa):

| T2 | Additive control | Reading |
|---|---|---|
| FAST | SLOW | the interaction-order barrier, candidate (i) |
| FAST | FAST | input strength / a first-order carrier suffices; (i) as stated is not it |
| SLOW | SLOW, with C_fa FAST | neither; favours lock-in, candidate (ii) |
| SLOW | any, with C_fa SLOW | uninterpretable: the visible slot itself reforms the shuffle |
| INTERMEDIATE | — | run seed 2 |

T2 is a mechanism test, not the omni fix: W is clock-free, so this feature has no clock to multiply in full omni.

**Order.** C_fa and C_40 first (zero code, as soon as C_lr_10_rsi33 releases the GPU), T1 alongside, then T2 with its
control. Canonical training (section 6) is the fix and can run at any point; it is expected FAST under every candidate,
so it validates the lever, not the mechanism.

---

## 6. The fix the root cause implies

**Canonical training.** For every env with a −y command, feed the network M_y(obs) and map its action back with M_y
(sticky flag, hysteresis h ≈ 0.08 m/s, reset on done). The network then only ever solves the one-direction problem,
where the stroke is g(phase) and forms by it about 600. The ±y interaction is applied outside it, exactly.

**Why the deploy result fits.** Canonical deploy on untrained +y heads already walks FAST on both hands. It is the
same construction without the training. It shows that everything a one-direction head lacks for two-hand walking is
the sign flip at fixed phase, which the wrapper supplies.

**Path** (ANALYSIS steps 3–4):
1. **C_lr_canon:** the C_lr overlay plus `KRABBY_OMNI_CANON=vy`.
   - The wrapper subclasses `ParkourRslRlVecEnvWrapper`.
   - Keep the fa mirror. M_y images would bring the phantom −y rows back.
   - Keep RSI off, or mirror the −y bank frames.
2. **FB_canon:** add fwd and back. The mirror applies M_x always, and M_y / R_z(π) only on rows with |vy_b| < h.
3. **P1_canon:** the 12k-it phase-1 recipe, compared against W on side (both hands), fwd, back, diag, turn and
   move_turn.

**Risks for full omni:**

| Risk | Detail and mitigation |
|---|---|
| **Reversals** | A flip re-mirrors the whole observation history at once, a new state distribution. Deploy on untrained heads was fine: speed held after rev_direct and rev_pause, falls at each head's baseline (1–19 % per 8-s hold). Canonical training puts flips in distribution. Watch the dead band around vy ≈ 0, where a stand-to-side transition flips on the first step. |
| **Diagonals** | The flag is sign(vy_b). At 45°, \|vy_b\| ≈ 0.7\|v\| ≫ h, so it is stable. Near-fore-aft diagonals and fwd/back rows must stay below h: the relay dead band lets fwd rows carry \|vy_b\| up to about 0.046, so h ≥ 0.08. Diagonals inherit the one-direction side gait's handedness only on the side component. The cams drive fore-aft, so they should compose, but this is untested. |
| **move_turn** | vy_b crosses zero mid-manoeuvre, forcing one flip after about asin(h/v): 13° at 0.35 m/s, 21° at 0.22 with h = 0.08. It is in distribution only if training is canonical. This compounds W's open move_turn issue ("nobody actually turns during move_turn"). |
| **Forward tripod handedness** | In the 4-image loss, the M_y image of forward rows carries the tripod-set handedness tie, and the mirror module notes that unconstrained training "breaks at random" without it. Canonical side rows must not get M_y images, so the masked mirror keeps M_y only for \|vy_b\| < h. Gate FB_canon on forward tripod and fwd/back ratios against the t3 record (fwd 0.558, back 0.524 at 3000 its). |
| **Clock-free W** | Canonicalization needs no clock, unlike the T2 feature or a "virtual cam". With the clock hidden, M_y's clock shift is moot and the canonical network must infer tripod phase from state. That is untested in canonical training. |
| **Deployment** | The exporter exports the bare actor, so the canonicalizer (obs mirror, action mirror, flag with hysteresis) must ship in the runtime. Alternatively, distill the canonical policy into a bare student. The student's target is discontinuous at vy ≈ 0, where it averages to the symmetric shuffle inside the dead band, so use DAgger for reversal states. Whether a distilled bare student survives PPO fine-tuning is P9. |
| **Real-robot asymmetry** | Any real left/right asymmetry becomes a hand-dependent error that training never sees. |
| **Survival cost** | C's bundle falls 9–14 % in the hold, against 0–1 % for W. The fast C_mo heads fall 2–5 %. Record side survival against W. |

**Fallbacks.**
- A policy switch: W for non-side commands, a canonical head for side-dominant ones. About 100 lines, eval only.
- Distillation.
- A sign-gated odd head, a = f_even(s) + g(vy)·f_odd(s). The lean also passes through the gate, so the risk is an
  INTERMEDIATE result, as with C_mo s3.

---

## 7. Uncertainties and caveats

- **Single-seed negatives.**
  - Every two-direction and phantom arm ran at seed 3 only. C_lr_10_rsi s2 was stopped at it about 1300 (early read
    0.535). O_s2 and LR4_s2 never ran.
  - The pattern still holds across 12 bisect configurations, W, and 18+ earlier two-direction omni stamps.
  - The positive side has 5 runs over 3 seeds (4 FAST, 1 INT by 0.002).
- **The mechanism (section 1, candidates i–iii) is inferred.** No per-sample gradients, advantages or action telemetry
  were logged during training, and no arm isolates the per-hand data rate (C_40 does). The RSI failure is not explained
  by any candidate: RSI starts underperform in fast arms too.
- **The probes are post hoc**, on deterministic eval rollouts. The early-checkpoint probes (weights, saturation, gate)
  use final-eval states, which are off-distribution for those checkpoints. T1 fixes that.
- **Phase choice.** The phase-binned decomposition uses the clock phase. W and the sideways pins use a data-derived
  phase; where both apply, the odd fractions agree within 0.01–0.03.
- **The distillation is open loop**, on canonical-C states. A bare two-hand net has not been shown to walk in closed
  loop, or to stay put under PPO (P9).
- **The fore-aft contrast (2.3)** rests on t3, c6 and W only. It has not been tested by intervention.
- **The phantom-row mechanism** is confounded with H6 (the M_x weight) and the scalar obs. None of these affects the lr
  conclusion.
- **C_fa has never run**, so a visible slot in a one-direction run is untested.
- **The reward replay** excludes the clock and apex terms. foot_idle is heavy-tailed, and its training-average sign
  flips between runs.
- **The per-leg split cannot be resolved.** By construction, the odd part has equal energy on the leading and trailing
  leg of each mirror pair, so this probe cannot say which row carries the stroke.

---

## 8. Independent check (fable·high, `scripts/rootcause/check/`)

- Reproduced: the detached mirror target and coefficients; the actuator contrast and rod slew enforcement
  (`parkour_actions.py:206-250`); the mirror table; every pooled odd/even/static number in section 3 for O, C,
  C_mo_s2, LR4, C_lr, canonical C and C_lr_rsi33 (to two decimals); the tfevents early reads, separation time (it 540),
  O's rising symmetry loss, X19's curve; the student gate probe; and the H1a refutation (slot-9 effective drive 2.9–4.2×
  the median proprio drive, equal to the clock slots').
- **Blocking 1, applied:** the draft stated the direction × phase interaction as *the* root cause; the data establish
  the missing stroke and its trigger, not why PPO misses it. Section 1 now separates the established part from three
  mechanism candidates, and the C vs C_lr pair is described as a three-way change with C_40 / C_fa as the controls.
- **Blocking 2, applied:** T2 alone is not decisive; C_fa precedes it and the additive control runs alongside.
- Minor 1–10 applied: the smoothing share, the RSI numbers and their generic deficit, the two saturation statistics,
  C_lr_rsi33's late plateau, the joint-space size of the signature and its per-episode floor, the canonical cosines as a
  map check, the M_x-tie ordering, the lean as a low-speed pusher, and the RSI "continuation" sentence.
