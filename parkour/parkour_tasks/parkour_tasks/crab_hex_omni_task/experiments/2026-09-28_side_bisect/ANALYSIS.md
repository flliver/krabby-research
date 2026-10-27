# Side bisect: analysis and proposed next steps (2026-09-30)

The user's tree after C_lr ended in BOTH_FAIL (CHARTER.md:495-499). This document analyses every sideways test so far and proposes the next steps. **Nothing has been launched. It waits for the user.**

Sources:
- Four analyst passes; their scripts and caches are copied to [scripts/analysis/](scripts/analysis/) (`evidence.py`, `probe_*.py`, `tb*.py`, `traj.py`, `all.json`).
- An independent fable·high critique recomputed every number and code citation. Its corrections are folded in below.
- A re-check of the cited code and records. Examples: the detached mirror target at `parkour/scripts/rsl_rl/modules/ppo_with_extractor.py:436-451`; the command slots at `mdp/crab_hex_omni_observations.py:385-404`; the stored `arms/C_lr_10_lr.json` matches the table below.

## 1. The answer

- **The slow side gait forms whenever one network has to produce both lateral directions.** That covers two cases:
  - it trains −y for real: LRfa, LR4, LR4_dwell, C_lr, C_lr_10 and W;
  - or it trains −y through the "phantom" −y rows of the 4-image mirror loss in a +y-only run: O, R, C_band, C_rsi.
  - All 9 bisect runs of either kind are SLOW or DEAD on every hand, and so is W.
- **Every +y-only run without phantom rows walks fast.** That is the fore-aft mirror plus scalar command obs: C at 2 seeds and C_mo at 3. Four of those five runs are FAST, and C_mo s3 misses FAST by 0.002. They walk 0.18–0.22 m/s, the same as the sideways-task heads.
- **The deciding pair is O vs C_mo.** They share the environment, RSI 0.2 and band 0.25. The only difference is the mirror and obs pair.
  - The pair removes the phantom rows.
  - It also triples the effective weight of the fore-aft (M_x) mirror image.
  - Every FAST run carries the fa mirror, so "phantom rows hurt" and "a strong M_x tie is needed" are confounded. A no-mirror +y-only arm would separate them (step 5c).
- **No sampling, episode, reward, RSI, band or mirror-type knob fixes a two-direction run.** C_band, C_rsi, R, LRfa, LR4, LR4_dwell (40-s single command), C_lr (C's whole bundle) and C_lr_10 (10-s single-command episodes) were all SLOW or DEAD.
- **The invariant of the slow gait is a short stance sweep, not the lean.**
  - The sweep is ≤ 0.054 m in every SLOW/DEAD row and ≥ 0.069 m in every FAST row.
  - The lean is common, but LR4 (0.11–0.13 m) and C_lr_10's right hand (−0.008 m) are slow without it.
  - Post-hoc probes suggest the direction rides on a static channel instead of a phase-flipped long stroke. That channel is either the body lean or a corner-cam bias.
- **The gait is chosen early and pays about the same as the fast one.**
  - Fast and slow runs look the same up to it ≈ 300. Slow runs reach 95 % of their final track ratio by it 175–675; fast runs keep climbing (95 % at it 821–2012).
  - Mean reward over its 2500–3000 is 67.0 for O against 66.1, 70.5 and 73.1 for the three C_mo seeds.
  - Equal returns do not prove that pricing can't tip it. The evidence that reward is not the lever is R plus the three failed stroke/centring campaigns, and those only ADDED income.
  - The penalties that price against the long stroke were never tested. The fast gait pays more in `reward_action_rate` and `penalty_foot_idle_when_forward`.
- **Recommended lever: mirror canonicalization.**
  - Every state with a −y command goes through the exact left-right map M_y before the policy. The network output is mapped back the same way.
  - The network then only ever sees +y side walking, which is the one setting that has walked fast.
  - The robot and MDP are exactly left-right symmetric, and the maps are tested and behaviourally correct (LR4's hands mirror each other within split-half noise).
  - It can be checked **today with zero training** on the existing fast heads, at about 0.5 GPU-h.
- **Main uncertainties:**
  - Every two-direction arm and every phantom-row arm ran at seed 3 only.
  - The mechanism probes use post-hoc thresholds.
  - A one-direction run with the direction visible (C_fa) has never been run.
  - Reversals under canonicalization are untested.
  - C's bundle falls more than W: C and C_s2 fall 9–14 % in the hold, against 0–1 % for W. The fast C_mo heads fall only 2–5 %, so the excess goes with RSI-off/band or with speed, not with the FAST class.
  - Deployment would need the canonicalizer in the runtime, because the exporter exports the bare actor.

## 2. Evidence

The evidence pass recomputed every value from the raw `episode_*.npz` files with the classifier's own `hold_stats` and `why_excluded` (`scripts/side22_classify.py`). The values match every stored `parkour/outputs/omni_plan/side_bisect/arms/*.json`.

Cells read "0.22 / 0.25", and "L | R" separates the hands. Training runs are X01–X15 under `parkour/logs/rsl_rl/crab_hex_omni_flat_walk/2026-09-29_*` and `2026-09-30_*`. Unless stated otherwise, runs use 40-s episodes with 10-s resamples.

| Arm (seed) | Directions trained | Mirror / cmd obs / RSI / band | Class L \| R | feet_ss L | feet_ss R | Sweep (m) | Lean_abs (m) | Cadence (Hz) | Stall | Early read |
|---|---|---|---|---|---|---|---|---|---|---|
| O (3) | +y, plus phantom −y | 4-img / visible / 0.2 / 0.25 | SLOW | 0.084 / 0.094 | — | 0.042 / 0.047 | 0.18 / 0.20 | 0.83 / 0.86 | 0% | 0.564 |
| R (3) | same, sideways reward economy | same | SLOW | 0.075 / 0.078 | — | 0.023 | 0.16 / 0.18 | 1.22 / 1.25 | 0% | 0.473 |
| C_band (3) | same | band 0.35 | SLOW | 0.093 / 0.100 | — | 0.049 / 0.052 | 0.20 / 0.21 | 0.92 / 0.97 | 0% | 0.561 |
| C_rsi (3) | same | RSI off | SLOW | 0.069 / 0.077 | — | 0.034 / 0.037 | 0.20 / 0.21 | 1.03 / 0.97 | **34–37%** | 0.498 |
| C_mo (3 / 2 / 1) | +y only | fa / scalar / 0.2 / 0.25 | INT / FAST / FAST | 0.162/0.173 · 0.203/0.210 · 0.184/0.185 | — | 0.056 · 0.086 · 0.069–0.076 | 0.11 · 0.005–0.03 · 0.07 | 0.97–1.03 · 0.86–0.92 · 0.81–0.89 | 0% | 0.654 · 0.700 · 0.765 |
| C (3 / 2) | +y only | fa / scalar / off / 0.35 | FAST / FAST | 0.215/0.214 · 0.194/0.209 | — | 0.083–0.085 · 0.082–0.090 | 0.03–0.05 · 0.05–0.06 | 0.83–0.89 | 0% | 0.671 · 0.702 |
| LRfa (3) | ±y | fa / visible / 0.2 (+y bank) / 0.25 | SLOW \| SLOW | 0.061 / 0.065 | 0.087 / 0.088 | 0.033–0.038 | L 0.25 \| R 0.16–0.18 | 0.93–1.11 | ≤1% | 0.536 |
| LR4 (3) | ±y | 4-img / visible / 0.2 / 0.25 | SLOW \| SLOW | 0.085 / 0.091 | 0.082 / 0.096 | 0.033–0.035 | 0.11–0.13 | 1.01–1.11 | ≤2% | 0.571 |
| LR4_dwell (3) | ±y, one command per 40-s episode | as LR4 | DEAD \| DEAD | 0.047 / 0.058 | 0.055 / 0.028 | 0.026–0.029 | 0.16–0.21 | 0.81–0.86 | **45–58%** | 0.491 |
| C_lr (3) | ±y | fa / visible / off / 0.35 | SLOW \| SLOW | 0.072 / 0.077 | 0.063 / 0.073 | 0.033–0.039 | 0.19–0.22 | L 1.39–1.50 \| R 0.92 | ≤4% | 0.524 |
| C_lr_10 (3) | ±y, one command per 10-s episode | as C_lr | SLOW \| SLOW | 0.085 / 0.085 | 0.067 / 0.073 | L 0.051–0.054 \| R 0.039–0.041 | L 0.15–0.16 \| R **−0.01** | 1.03–1.25 | ≤10% | 0.538 |
| W (reference, clock hidden, 12k its) | full omni | 4-img / visible | SLOW | 0.096 / 0.103 | 0.101 (0.22 only) | 0.033–0.035 | 0.25 | 1.22–1.25 | 0% | — |
| Sideways 1a references (4 stamps) | +y only (sideways task) | fore-aft | FAST | 0.196–0.218 / 0.193–0.208 | — | 0.077–0.109 | −0.01 to 0.04 | 0.78–0.86 | 0% | — |

Stall is the share of episodes with feet_ss < 0.03, plus the "static" exclusions. Falls inside the hold:
- C and C_s2: 9–14%.
- Sideways references: 9–21%.
- C_band: 10–14%.
- C_mo_s1 and C_mo_s2: 2–5%.
- O, R and LRfa: 2–8%.
- W: 0–1%.
- C_lr_10's left hand: 24–27%.

C_lr_10 trained on 10-s episodes. Its eval holds therefore run 4 s past anything it trained on, and its track_ratio is on a different basis from the other arms.

**What separates FAST from SLOW**
1. **Stance sweep is the only perfect separator at the row level.**
   - FAST: 0.069–0.109 m.
   - SLOW/DEAD: 0.023–0.054 m. C_lr_10's left hand is the new SLOW maximum.
   - INTERMEDIATE: C_mo s3 at 0.056–0.057.
2. **The early read separates perfectly at the run level.** Early read is track_ratio_mean over its 800–1000.
   - FAST: 0.671–0.765. SLOW/DEAD: 0.473–0.571. C_mo s3: 0.654.
   - After about it 560–600 the lowest fast run stays above the highest slow run.
   - The stance tibia-deviation term is a class correlate: ≤ 0.019 in fast runs and 0.023–0.061 in slow runs. LR4 sits at 0.023, next to the boundary.
3. **Lean rules FAST out but does not define SLOW.** Every row with lean ≥ 0.11 m is not FAST, yet LR4 and C_lr_10's right hand are slow with little or no lean. Within FAST runs, episodes that lean more are slower: r −0.27 to −0.69, and −0.77 in the INTERMEDIATE C_mo s3.
4. **Cadence only works one way.** Cadence above about 0.95 Hz is never FAST, but O, C_band and C_lr's right hand step at the FAST cadence (0.8–0.97 Hz) with a short stroke.
5. **Stalling is a separate failure mode.** It appears mainly with RSI off in a phantom run (C_rsi) and with long single-command episodes (LR4_dwell). C_lr_10's right hand also stalls 4–10 %.
   - Episodes that start leaning against the travel direction stall in 81% (C_rsi) and 100% (LR4_dwell) of cases.
   - The stand lean is bimodal (about ±0.15 m, 50/50) in O, C_rsi and LR4_dwell. It correlates with the frozen clock phase at |r| 0.44–0.81, with an arm-dependent sign; the bimodality is the load-bearing part.
   - The fast heads ignore the start lean: 0.218 vs 0.210.
   - Sources: `probe_stall.py`, `probe_stand.py`.
6. **Nothing in the optimizer separates the classes.**
   - Policy std is about 1.65 in every arm while the gait forms (its 300–600). It reaches the 2.0 cap (`scripts/rsl_rl/modules/crab_actor_critic_with_encoder.py:11`) only in the last ~900 its, and in 10 of 14 arms.
   - Entropy, adaptive LR, surrogate, priv_reg and the termination curves are the same in both groups.
   - Heads are 91–98% saturated (median |μ| 6.1–7.4 against clip ±1), fast and slow alike.

## 3. Mechanism ranking

**H1 (leading; consistent with every arm).** When the policy has to produce both lateral handednesses, it carries the direction in a static, first-order channel and walks a short, direction-agnostic rod shuffle. The static channel is the body lean, or a constant spin on the trailing corner cams (`probe_self.py`, `probe_w.py`). The alternative is a long stroke whose per-leg phase flips with direction.

- **Why the lateral axis and not fore-aft.** Fore-aft travel only needs a static cam-spin sign, which is first-order in the command. W's back tracks exactly like its fwd (0.784 vs 0.785, `policy/manifest.yaml`). "The phantom back rows in 1a did no harm" is an inference, untested: no fwd-only arm without M_x images exists. The lateral stroke runs on position-controlled, slew-limited rods and needs direction × gait phase.
- **Arm pairs that decide it:**
  - **O vs C_mo** (the pair only): phantom rows present → SLOW; removed → FAST at 2 seeds, INTERMEDIATE by 0.002 at the third.
  - **C_mo/C vs LRfa/C_lr** (same fa mirror; add real −y data): FAST → SLOW.
  - **LRfa vs LR4:** once −y is real, the mirror setting does not matter.
  - **LR4:** it is exactly mirror-equivariant (right vs mirrored left 0.36 against split-half noise 0.37, `probe_hand.py`) and still slow. So this is not a symmetry bug; it is the shared network.
  - **LRfa's two hands are neither the same nor mirrored in coordination** (R vs L 2.33, R vs M_y·L 3.79, noise 0.39), while its action means are closer to mirrored.
  - **C_lr_10 shares one coordination pattern across hands**, and its mean is also closer to direction-agnostic. That fits its unleaned right hand.
- **Mechanism of the phantom rows.** The mirror-loss target is detached (`ppo_with_extractor.py:449-451`). In a +y-only run the M_y images therefore train only the −y outputs, and that training reaches the +y gait through the shared weights.
- **Unresolved sub-mechanisms** (none of them tested):
  - **1a, weak direction input.** Slots 8–9 carry raw ±0.22–0.35, with no obs normalization (`empirical_normalization: false`).
  - **1b, direction × phase is hard to represent** under this PPO, whatever the input.
  - **1c, a formed long stroke collapses** once −y is trained.

**H2 (possible enabler, not the cause).** Saturated means (|μ| ≈ 6 against clip 1 at it 3000) might lock in the first solution.
- It is identical in fast and slow arms, so it cannot pick the class.
- Its timing is unsupported: std is about 1.65 when the gait forms (its 300–600).
- Whether the mean is already saturated then is unmeasured, because there is no action telemetry.

**H3 (amplifier).** The onset payoff picks the lean as the carrier. The lean is neither necessary (LR4, C_lr_10 R) nor sufficient to explain the class.

**H4 (explains stalls, not speed).** The frozen stand clock acts as a random direction bit (C_rsi, LR4_dwell).

**H5 (open).** Lateral lean is geometrically cheap.

**H6 (open, confounded with H1's phantom-row story).** The fa mirror triples the effective weight of the M_x image. Every FAST run carries fa, so "a strong M_x tie is needed" fits the data as well as "phantom rows harm". Step 5c separates them.

**Ruled out, with the arms that decide it:**

| Candidate cause | Deciding arms |
|---|---|
| Reward economy or pricing | R SLOW; C FAST on O's economy; three stroke/centring campaigns |
| Band / cadence | C_band SLOW, C_lr SLOW |
| RSI (on/off, +y-only bank) | C_rsi SLOW; C_lr/C_lr_10 SLOW with RSI off; C_mo FAST with RSI on |
| Within-episode reversals, episode structure | LR4_dwell DEAD, C_lr_10 SLOW |
| Mirror type in two-direction runs | LRfa and LR4 both SLOW |
| M_y map bug | LR4's mirror equivariance is within noise; unit tests (`tests/unit/test_crab_hex_omni_mirror.py:302,611,680`) |
| Ring world, relay, heading slots, critic interference | O and C_mo share them and split; O's critic never sees phantom rows (`use_data_augmentation: false`) |
| Half the data per direction | O has all the +y data and is slow |
| Clock cadence law as the steady-state cause | W is clock-free and slow; C is fast with the clock |
| Symmetric standing | LR4 stands centred and is slow; C stands near-centred and is fast |

**Honest uncertainty.**
- **Single-seed negatives.** Every two-direction and phantom negative ran at seed 3 only. The pattern still holds across 9 configurations, W, and the 18+ earlier two-direction omni stamps (CHARTER.md:402). The positive side has five runs over three seeds (s1, s2, s3).
- **Probes P1–P4 are post hoc.** The "static carrier" is a description that fits all of them, not a tested cause.
- **C_fa has never been run.** No one-direction run has had the direction slot visible, so H1's claim that a visible slot is harmless is an inference. It rests on GATE.md:147-150: in 1b the direction information is also in slots 6–7.

## 4. Proposed next steps (ranked)

Arm cost is about 1.25 GPU-h: about 1.15 h to train 3000 its, plus the driver eval and a both-hands eval of about 0.05 h.

The charter allows one Isaac process at a time. **Order of launch:** step 2 can start immediately, because it needs no code. Steps 1 and 3 are coded while it trains.

### Step 1. Zero-training canonicalization eval of the existing fast heads (about 0.5 GPU-h, about 150–200 lines)

**Question.**
- Does canonicalization give fast walking on both hands with no training?
- Is the M_y map plus the sim exact at deployment, not just inside a loss?
- What happens at a frame flip and at a reversal?

**Design.**
- **Hook.** Add a hook in the non-student branch of `crab_hex_forward_task/scripts/eval_crab_hex_gait.py` (lines about 856-860), using the omni `build_mirrors(env, obs_dim)` pair (`mdp/crab_hex_omni_mirror.py:314`; `CrabHexMirror.mirror_obs/mirror_actions`, `crab_hex_mirror.py:217,222`).
  - `obs_c = where(flag, lr.mirror_obs(obs), obs)`.
  - Run the estimator on `obs_c`.
  - `a = policy(obs_c)`.
  - `actions = where(flag, lr.mirror_actions(a), a)`.
- **The flag** comes from `cmd_term.vel_command_b[:,1]`. It cannot come from the obs, because C and C_mo zero slots 8–9.
- **Leave the harness's checks on the raw obs.** That covers the command-deviation check, `obs_cmd_vx` and the clock buffers.
- **Record the flag** in the npz and `run_meta`.
- **Two modes:**
  - `wave`: the flag is fixed per episode from the hold direction, so the mirror is exact, including the settle.
  - `cmd`: the flag follows the live command with hysteresis h = 0.08 m/s.
- **A new reversal manifest** of plain `schedule:` rows: stand 2 s → lat_l 8 s → lat_r 8 s → lat_l 8 s, with and without a 1-s stand before each reversal.
- **A CPU unit test:** the hook is an involution, and lat_l is bit-identical with the hook off.
- **Heads** (`heads.json`): C_mo_s2 (X08), C (X02), C_mo_s1 (X09) and C_s2 (X07). Each runs with its own driver-eval env (`KRABBY_OMNI_CMD_OBS=scalar`, clock visible).
- **Evals:** `scenarios_side_lr.yaml` in both modes, plus the reversal manifest in `cmd` mode.

**Outcomes.**
- **Expected:** `wave` lat_r within about 0.02–0.03 of lat_l, and FAST. This confirms the maps and gives a working two-hand fast side controller now.
- **lat_r slow or falling:** a sign-table bug or a sim asymmetry. Fix it before any canonical training.
- **`cmd` ≈ `wave`:** a flip at hold onset is harmless, consistent with P1 (fast heads ignore the start lean).
- **Reversals fall or stall:** training has to include reversals, and step 3 does.

**Risks.**
- The result is close to guaranteed by construction, so this is mainly a plumbing check and a deploy fallback, not a mechanism test.
- A +y-only head has never seen "moving −y, commanded +y", so its reversal behaviour is unknown.
- Eval holds cover only 0.22 and 0.25 m/s.

### Step 2. C_fa: one direction, fa mirror, direction visible (about 1.25 GPU-h, zero code)

**Question.** Does a visible direction slot hurt the one-direction fast gait? Every portable design (canonical-visible, full omni) shows the slot. This is also the direct falsifier for H1.

**Design.** Take the C_fa chain drafted in `scripts/candidates_v4.yaml:256-268` (1b from scratch, `KRABBY_OMNI_MIRROR_IMAGES=fa`, seed 3, eval `side22_25`). Fold it into the live `candidates.yaml` and dry-run it clean; the launch needs `--accept-candidates-change`. Use the same gate.

**Outcomes.**
- **FAST (predicted, ≈ C_mo; GATE.md:147-150):** the visible slot is harmless, and steps 3–4 use a visible direction. This head also becomes the clean fork source for step 5a.
- **SLOW:** H1 is falsified as stated. The visible slot alone reforms the shuffle, and full omni would need a different obs contract. Stop and re-plan before step 4.
- **INTERMEDIATE:** run seed 2.

**Risks.**
- It is low-information if FAST.
- It keeps RSI 0.2 with the +y bank, like C_mo.

### Step 3. Canonical two-direction training: C_lr_canon (about 1.3 GPU-h, about 250 lines + tests)

**Question.** This validates the lever, not the mechanism.
- Under canonicalization the network only ever receives +y states. So, from the network's view, this run is C_fa (step 2) plus mid-episode frame flips and re-mirrored history.
- FAST is expected if step 2 is FAST. It would confirm canonicalization as a working two-direction fix.
- SLOW would isolate reversal and flip transients as a separate cause.
- The mechanism claim rests on C_mo/C (FAST) vs LRfa/C_lr (SLOW), not on this arm.
- If step 1's reversal manifest already passes on the existing heads, this arm's marginal value shrinks. Gate its launch on step 1's reversal result.

**Design.** Take the C_lr overlay (CHARTER.md:425-427: pool `stand:0.2,side_l:0.4,side_r:0.4`, fa mirror, RSI off, band 0.35, direction visible) and add one knob, `KRABBY_OMNI_CANON=vy`. That makes it a single-knob contrast to a SLOW arm.
- **Wrapper.** Subclass `ParkourRslRlVecEnvWrapper` (`parkour/scripts/rsl_rl/vecenv_wrapper.py`, 171 lines). It mirrors the policy obs (the critic obs is the same group) and inverse-mirrors the actions per env.
  - Sticky flag with hysteresis h = 0.08.
  - The flag is reset from the command sign on `dones`.
- **Plumbing.** The knob goes in the omni knobs, `EVAL_CARRY` and `OVERLAY_DUMP_CHECKS`, with the wrapper chosen in `train.py` and in the step-1 harness.
- **Keep the fa mirror.** M_y images would reintroduce −y rows the canonical network never sees.
- **Keep RSI off.** The bank frames are +y only, so for −y episodes they would need mirroring.
- **Evals:** the driver's `side22_25` plus `scenarios_side_lr.yaml` through the canonical harness, plus the step-1 reversal manifest.

**Pre-register before launch.**
- Success: FAST on both hands, with |L−R| ≤ 0.03.
- Fail: SLOW on both hands. Before concluding, run seed 2 and the step-1 hand-agreement check on the trained head.
- Early read at it 800–1000 ≥ 0.60 counts as "forming FAST". It is a progress note, not a kill rule.

**Outcomes.**
- **FAST:** canonicalization works as a two-direction training fix. Go to step 4.
  - This is expected by construction if step 2 is FAST.
  - It does not split 1a, 1b and 1c; step 5 does that.
- **SLOW:** the added reversal and flip states reform the shuffle.
  - Next, run C_lr_10 + canon, which has no reversals.
  - If that is also SLOW, the cause is not network sharing. Re-plan.

**Risks.**
- Mid-episode flips re-mirror the whole 0.2-s history at once. That is a new state distribution.
- Deployment has to carry the canonicalizer.
- It is one seed.

### Step 4. Canonical fwd/back/side (FB), then the full phase-1 recipe (about 1.3 GPU-h, then 5.5 GPU-h; about 60–100 more lines)

This is the user's pre-registered tree (CHARTER.md:461-471) applied to the successful config. It runs only if step 3 is FAST.

**Question.** Does adding cam-driven fwd and back travel to a canonical +y side network keep the long stroke? Does forward keep its tripod balance without the full M_y tie? No existing run answers either question.

**Design.**
- **FB_canon.** Pool `stand:0.2,fwd:0.2,side_l:0.2,side_r:0.2,back:0.2`, 1b from scratch, 3000 its.
  - The mirror is a new masked entry point in `crab_hex_omni_mirror.py`. It always applies M_x, and applies M_y and R_z(π) only on rows with |vy_b| < h. That keeps M_y's forward tripod-handedness tie on stand and fwd/back rows, and gives no side row a −y twin.
  - Choose h ≥ 0.08. With facing held, the relay dead band lets fwd rows carry |vy_b| up to about 0.35 × sin(0.133) ≈ 0.046.
- **Gates** (pre-register):
  - both-hands side FAST;
  - `suite__omni` fwd and back ratios and forward tripod no worse than the t3 record at an equal 3000 its (fwd 0.558, back 0.524, side 0.478; `policy/manifest.yaml:113-115`);
  - side survival recorded against W's 0.995. The fast gait falls more.
- **Then P1_canon.** Stages t3 3000 → c4 1500 → c5 1500 → c6 2000 → c6x 4000, 12000 its, with the stage evals from the record. Compare against W with both-hands side, fwd, back, diag, turn and move_turn.
  - move_turn-left drives vy_b negative mid-manoeuvre, so it forces one frame flip. With h = 0.08 that comes after about asin(0.08/v): 13° at 0.35 m/s, 21° at 0.22 m/s. It is in distribution only because training is canonical.

**Outcomes.**
- **Side FAST and fwd/back at t3 level:** run P1_canon. It becomes the candidate head of record if it beats W's side ratio 0.473 without losing the rest.
- **Side SLOW:** co-training with a second axis also forms the carrier. Fall back to the policy switch (W for non-side commands, the step-1 canonical head for side-dominant ones; about 100 lines, about 0.1 GPU-h eval-only) or to distillation (about 150–200 lines).
- **Fwd tripod degrades:** widen h or the M_y mask.

**Risks.**
- Diagonals and move_turn cross the flip boundary.
- Real-robot left-right asymmetry becomes a hand-dependent error that training never sees.

### Step 5. Wrapper-free alternatives and the sub-mechanism split (1.25 GPU-h each; only if the user wants one bare network, or if step 3 fails)

**5a. Curriculum fork (zero code; 1500 its ≈ 0.6–0.7 GPU-h).** Fork the C_fa head (step 2) into the C_lr config for 1500 its. Read both hands at +500 and +1500.
- +y stays fast and −y becomes fast: a formation-order lock-in. The lever is a one-direction-first curriculum.
- −y is slow with a static carrier: 1b.
- +y collapses to the shuffle: 1c.
- **Do not fork a C/C_mo head.** Their slot 8–9 weights never received a gradient (zero input).
- The saturated means (H2) may confound this test.
- The runner here restores `alg_state` (`parkour/scripts/rsl_rl/modules/on_policy_runner_with_extractor.py:743-750`, `resume_alg_state: true`); verify on the launch branch.

**5b. Direction gain (about 40 lines + mirror-table test).** Run C_lr with slots 8–9 × 4, or with an added ±1 sign slot that flips under M_y.
- FAST: 1a (weak input), and a wrapper-free fix exists.
- SLOW: the fix has to be structural (canonicalization or distillation).
- **Prior: open, and worth running alongside step 3.**
  - Walking the correct way needs only a first-order channel (lean or cam bias), which is exactly what H1 says the slow gait uses.
  - A phase-flipped stroke needs direction × clock phase. Slots 8–9 are unnormalized ±0.22–0.35 against ±1 clock slots (`empirical_normalization: false`).
  - This is the cheapest wrapper-free test of 1a. If FAST, it makes the step 3/4 canonicalizer unnecessary.

**5c. No-mirror one-direction arm (about 10 lines + 1.25 GPU-h).**
- **Setup:** +y-only, scalar cmd obs, NO mirror loss. This needs a `none` mode, since `parse_mirror_images` only accepts `fa` (`omni_knobs.py:626-638`).
- **What it separates:** H1's phantom-row story from H6 (the 3× M_x weight under fa), which O→C_mo confounds.
  - FAST: the phantom rows are what hurt.
  - SLOW: the strong M_x tie is needed for the fast gait.

**Budget.** Steps 1–3 cost about 3 GPU-h. Step 4 adds 1.3 + 5.5. Step 5 adds 1.25 per arm.

Two CPU-only side items cost nothing:
- An action-saturation pass over the eval `raw/episode_*.npz` actions, to quantify H2 per arm. No action telemetry exists in tfevents.
- An offline reward replay (the campaign's replay-gate practice) of the penalty terms that price against the long stroke. Per-term split over its 2500–3000, O vs C_mo s3:
  - the fast gait earns +0.12/s more in `track_lin_vel_xy_exp` (0.90 vs 0.78);
  - it pays about −0.09/s more in `reward_action_rate`, and more in `penalty_foot_idle_when_forward`;
  - the clock terms are equal.

  These penalties were never tested as levers; the failed campaigns only added income. Low priority, given that the gait forms early.

## 5. What not to do

- **No more reward INCOME for stroke, centring, cadence or tracking economy.** R is SLOW and the three stroke/posture campaigns failed. The penalty side (action rate, foot idle) is untested; only an offline replay is suggested, not a training arm.
- **No sampling or episode-structure levers.** Band, RSI, dwell, episode length and reversal curricula were all tested as fixes: C_band, C_rsi, LR4_dwell, C_lr and C_lr_10.
- **Don't swap the 4-image mirror for fa in two-direction or full omni as a fix.** LRfa, C_lr and C_lr_10 are slow with fa, and dropping M_y loses the forward tripod-handedness tie (GATE.md:193-195).
- **No additive direction embeddings or direction-conditioned inputs as the fix.** All five two-direction arms had the direction visible. Only the step-5b gain test, as a mechanism probe, is worth running.
- **Don't use std or `KRABBY_ACTION_STD_MAX` as the primary lever.** Saturation is identical in fast and slow arms.
- **No seed-lottery reruns of O_s2 or LR4_s2 now.** The contrast already rests on 9 negative configurations plus W against 5 positive seeds.
- **Don't resume X13 (LR4_dwell_s2).** The run tags collide; rerun it from scratch if it is ever needed.
- **Don't judge the gait on lean alone.** Use feet_ss plus sweep, since C_lr_10's right hand is slow with no lean. Don't compare track_ratio across different episode lengths (C_lr_10).
- **Don't reuse the sideways task's head for omni.** It uses the forward obs term, walks +y only and looks 0.6 m ahead. It would need step 1's canonicalizer anyway.
- **Don't kill arms on the early read.** It separated perfectly here, but it stays a progress signal (CHARTER.md:89-91).
---

## Update 2026-09-30: step 1 run, then the user's harvest + training test

**Step 1, canonical eval: WORKS on 4 of 4 fast heads, with no training** (CHARTER log, `gate/canon_lr.json`).

| Head | lat_l, 0.22 / 0.25 | lat_r via M_y, 0.22 / 0.25 |
|---|---|---|
| C | 0.215 / 0.217 | 0.211 / 0.216 |
| C_mo_s2 | 0.203 / 0.211 | 0.199 / 0.209 |
| C_s2 | 0.194 / 0.208 | 0.190 / 0.203 |
| C_mo_s1 | 0.184 / 0.185 | 0.182 / 0.191 |

- Every head is FAST on both hands, with |L−R| ≤ 0.006 and a long sweep on both hands. cmd mode agrees.
- Reversals (left → right → left, with and without a pause) keep speed, with no stall mode. Falls track each head's
  baseline rate.
- So the M_y maps and the sim are exact at deployment, and **a working two-direction fast side controller exists now:
  any C / C_mo head behind the canonicalizer.**

**The user's follow-up: a canonical harvest → a two-direction RSI bank → C_lr_rsi (C in both directions + RSI 0.2 from
the bank).**
- The bank: 2000 +y and 2000 −y frames from canonical C, median 0.28 m/s. Canonical C degrades at 0.35 m/s (0.15, lean
  0.15), and about a third of the pool came from those holds.
- **C_lr_rsi is SLOW on both hands:** L 0.085 / 0.085, R 0.069 / 0.075, lean 0.21–0.27, sweep 0.035.
- **Reading.** Fast-gait seeding in both directions does not change what two-direction training converges to. This fits
  H1: the network sharing both handednesses forms the shuffle regardless of its start states.

**Where this leaves the lever.** The fix has to be structural. Canonicalization works at deploy time. Canonical TRAINING
(step 3) is the remaining test of whether it also works as a training method, where the network never sees −y states.
After that come canonical FB and then full phase 1 (step 4). Distillation, and a policy switch (W for non-side commands,
the canonical head for side), are the fallbacks.
