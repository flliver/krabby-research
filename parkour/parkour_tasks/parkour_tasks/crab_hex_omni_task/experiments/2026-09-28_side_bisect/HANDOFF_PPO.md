# Side bisect: handoff to the PPO algorithm and hyperparameter session

**Date:** 2026-09-30. This is the state at about 21:10. **From:** the side-bisect session. **To:** the PPO algorithm and hyperparameter evaluation session.

**Purpose.** The user stopped the side bisect's last ladder, the episode-length ladder, at 21:07. The user's request: "stop this ladder. Summarize the tests and results you have so far and pass them of to the PPO algorithm and hyperparameter session".

This file has four parts:
- the 25 training arms and the zero-training evals, with their results;
- the established root cause, and the open question of why PPO does not form the stroke;
- every PPO-side signal recorded;
- the open items, which now pass to you as proposals.

**No one owns the side bisect any more.** Its open proposals pass to you for the user's decision: canonical training (with its budget, kill rules and seed choices) and the mechanism tests. **None of them has user approval (§7).**

**Abbreviations.**
- SB = `parkour/parkour_tasks/parkour_tasks/crab_hex_omni_task/experiments/2026-09-28_side_bisect`, the directory this file belongs in (CH:791).
- Files in SB:
  - CH = CHARTER.md
  - RC = ROOT_CAUSE.md
  - AN = ANALYSIS.md
  - GA = GATE.md
  - CK = CHECKS.md
  - OL = orchestrator.log
  - ST = state.json
- QL = `parkour/outputs/omni_plan/side_bisect/queue.log`.
- ARMS = `parkour/outputs/omni_plan/side_bisect/arms/`.
- CANON = `parkour/outputs/omni_plan/side_bisect/canon/`.
- SSI = `../2026-09-28_side_speed_investigation/REPORT.md`.
- RR = `../2026-09-30_rl_algorithm_review/REPORT.md`; DOS = `RR/DOSSIER.md`.
- RCD = `crab_hex_omni_task/experiments/tools/run_candidates.py`.
- **tb** and **cfg** mean the value was computed for this handoff, on CPU, from the run's tfevents or `params/*.yaml`.
- Cells written "a / b" are at 0.22 / 0.25 m/s. L = +y (lat_l); R = −y (lat_r).
- "4-image arms" = the +y-only runs trained with the 4-image mirror (O, R, C_band, C_rsi). They are read as training phantom −y rows, a reading confounded with H6 (§4.2 (iii)).

**Gate** (`scripts/side22_classify.py:106-118`):
- feet_ss = feet-centroid speed over the hold [4,10) s, measured along each episode's own displacement.
- FAST ≥ 0.155 / 0.175. SLOW ≤ 0.13 / 0.145. DEAD < 0.05 at *either* speed (:112). Anything else is INTERMEDIATE.
- Two-direction arms are classed per hand.
- Since 19:52, a hand counts as FAST only if its along-command class is also FAST (CH:767-771).

---

## 1. Status at handoff

**Nothing is running and the GPU is idle.**
- `systemctl --user list-units --all 'cands-*'` lists 0 units.
- No `isaac_venv/bin/python` process is running.
- QL:343 (21:08:12) reads STOP-MANUAL: "user stopped the episode-length ladder … runner v2 killed, C_lr_30_rsi (X25) stopped gracefully at ~it 1900 via systemctl stop … do not resume X25".

**X25 (C_lr_30_rsi) was stopped and must not be resumed.**
- The user stopped it at 21:07 with `systemctl --user stop`.
- The train log's last line is `Learning iteration 1922/3000`. In the tfevents, the Loss and Policy tags end at step 1922 and the Metrics tags end at 1923.
- The last checkpoint is `model_1900.pt`. There is no eval and no class. Its early read was 0.524 (CH:787-790; QL:342-343).

**Three stop notices say to resume a stopped run. Ignore all three:**
- OL:52 (X13): "relaunch with --reference-chain LR4_dwell_s2 --seed 2 to resume it". CH:419 overrides it.
- OL:72 (X18): "relaunch with --reference-chain C_lr_10_rsi_s2 --seed 2 to resume it". CH:619 overrides it.
- OL:100 (21:07:45, X25), also the last line of notify.log: "relaunch with --reference-chain C_lr_30_rsi --seed 3 to resume it". CH:789-790 overrides it.

**No AWAIT notice in OL has ever been acknowledged.**
- This covers all 22 `await_user:reference_done:*` notices (OL:4 to OL:96) and the 3 stop notices.
- There is no ack path for these notices: `ledger_cli.py ack-flag` acknowledges ledger flags only. ST `user_choice` is {} and `applied_overrides` is [].
- `ledger.json` holds only D01.

**state.json is stale.**
- `phase` is "run", and `current` is still the X25 header (20:23:43).
- X13, X18 and X25 still show `status: "running"` with an `inflight` block. `keys.*` marks the same three runs `interrupted` (wall 0.2 / 0.534 / 0.733 h).
- `ladder.pending` is [].
- This creates a resume hazard, described in §7.J.

**Candidates and tracking.**
- The live `candidates.yaml` is v16: sha256 `8221b99f…`. It equals `scripts/candidates_v16.yaml` and ST `ladder.candidates_sha256`.
- HEAD still has v14 (`45c80f52…`).
- Uncommitted working-tree changes in SB: candidates.yaml (v16), CHANGELOG.md and REPORT.md.
- Untracked: heads.json, ledger.json, state.json and the whole RR directory.
- orchestrator.log and notify.log are gitignored (`*.log`, `crab_hex_omni_task/experiments/.gitignore:18`).

**No code landed during the bisect.**
- Every launch from X01 to X25 used code_sha `07a0037ed5d9`.
- eval_sha has been `3e2898e24cab` since 09:05:32. That change was the default-off `--canon` eval hook, commit 325b2d6a.
- `KRABBY_OMNI_CANON` appears in 0 `.py` files in the repo.

**Commits after RR's 18:44 snapshot.** All of them touch only SB:
- 67b20d8d, 19:00 (controls closed);
- a0b9ce72, 19:10 (ladder pre-registration);
- f533e356, 19:53 (ladder review);
- 47686f1e, 20:25 (C_lr_20_rsi SLOW);
- 675868f0, 21:09 (ladder stopped; CHARTER only).

b48e7ee6 (17:42, C_40 s3 DEAD) came before the snapshot, and RR:7 already cites it.

### 1.1 What changed since RR's 18:44 snapshot (RR:7), with corrections

| # | RR location | RR says | Correct now | Source |
|---|---|---|---|---|
| 1 | RR:7, :250 | C_40_s2 "training since 17:41, early read 0.180 (forming SLOW)"; that read "already excludes FAST" | Finished 18:58, **DEAD**: feet_ss 0.006 / 0.006, lean_abs 0.30, sweep 0.017–0.023. The ratio peaked at 0.23 near it 1500 and was 0.16 by it 2950. Falls end only 0.03–0.09 of episodes. This is a stand attractor, not the shuffle. | CH:705-708 |
| 2 | RR:7 | Live queue; "any edit … blocks that queue" | No queue is live. The terminal state is STOP-MANUAL. The pathspec rules still bind any later launch. | QL:343; RCD:163-183 |
| 3 | RR:3, :7 | Snapshot ends 18:44 | It misses `DONE controls` at 18:58:27 (QL:314), the ladder pre-registration at 19:09 (CH:721-747, v16), the ladder review at 19:52 (CH:748-777), X24 SLOW/SLOW at 20:23, and X25 stopped at 21:07. | QL; CH |
| 4 | RR:21 (§1.3), RR:42 (§2.2) | "+y-only … fa mirror FAST in 4 of 5 (0.184-0.217)" | There are 8 +y-only fa runs: 5 FAST, 1 INTERMEDIATE (C_mo s3), 2 DEAD (C_40 s3 and s2, at a 0.4 moving share). FAST spans 0.184–0.215 on the driver gate; 0.217 is canonical C (AN:322). The fa mirror is not sufficient by itself: every FAST run had a 0.8 moving share. | ARMS |
| 5 | RR:42 | C_fa at CH:686-690 | C_fa is CH:676-683. CH:686-690 is C_40's early read. | CH |
| 6 | RR:129 (§4.5) | "the four RSI-seeded two-direction negatives" | Five, all evaluated SLOW or DEAD: C_lr_rsi, C_lr_10_rsi, C_lr_rsi33, C_lr_10_rsi33 and C_lr_20_rsi. X18 and X25 were stopped before eval. | ARMS; CH:778 |
| 7 | RR:170 (§6.0.1) | Wait for sentinel `DONE controls\|STOP` | The ladder ran 19:09–21:08, after `DONE controls`. queue_eplen2.sh's own terminal line, "DONE eplen" (:160), never appeared. `STOP` also matches the older QL:29, :55 and :241, so tail from the end of the log. | QL |
| 8 | RR:173, :309 | First landing "once the user has judged C_40_s2"; bound loss "only if C_lr_bound goes first" | No explicit user judgement of C_40_s2 is recorded: the next request was the ladder (CH:723-725). C_lr_bound did not trigger, so the canon wrapper need not carry the bound loss. FB_canon's masked mirror is a separate code move (§7.C). | CH:705-725 |
| 9 | RR:177 (§6.0.5) | Side-bisect arms keep the s3 → s2 convention | There is no owner now, so you pick the seeds. Every two-direction and 4-image comparator is seed 3 (for example X14). One-direction comparators also exist at s2 and s1 (X07, X08, X09, X23). The comparator choice is in §7.B. | ST |
| 10 | RR:238, :251, :292, :310, :321 | C_lr_bound runs on the "C_40_s2 SLOW or FAST-late" branch | C_40_s2 is DEAD, so RR's own DEAD branch applies (RR:252): "(ii) stays open but untestable with the designed arms, and side speed stops being a decision gate for rungs 2-3". CH:671-673's SLOW branch named longer training, a higher side share or a slower early std/LR schedule, not a bound loss. | CH:666-673 |
| 11 | RR:21, :240, :242, :302, :310 | Canonical training "owned by the side bisect"; its ≈ 20 GPU-h "not counted" | No one owns the side bisect. The proposals and their budget pass to you for the user's decision, unapproved (§7). | CH:787-791 |
| 12 | RR:245 | C_lr_canon "stop only at ≤ 0.48 … the side bisect's call" | AN:226 says an early read ≥ 0.60 is "a progress note, not a kill rule". `early_read.py:5-6` says its bands are uncalibrated. The kill rule is now yours; calibration is in §6. | AN:226 |
| 13 | RR:242, :246, :254 | RC :467-470, :473-509, :481-482, :486-488 | RC moved at 67b20d8d. Current lines: Order RC:480-482; §6 RC:486-524; C_lr_canon RC:497-500; FB_canon RC:501; P1_canon RC:502-503; risks RC:505-516; fallbacks RC:518-524. The student `--canon` refusal is now `eval_crab_hex_gait.py:721-723`. | `git show cbf8ad06` (the RC that RR read) vs HEAD |
| 14 | RR §2.2, §6 | (omitted) | RR does not mention C_40_s2's DEAD verdict (row 1), the episode-length rungs, T1, T2, the flat-terrain control, or the along-command gate rule. | CH:748-786; RC:427-478 |
| 15 | RR:47 | σ, LR, entropy and clipped-rod share "overlap completely across 17 arms" | There are now 22 classified arms. These signals still overlap between FAST and the full SLOW/DEAD set. Value loss, surrogate and σ (its 400–1000) do split FAST from the ten two-direction arms, but they sort by command pool, not by class (§5.4). | §5.4 |
| 16 | RR:175 | A B\<k\> rerun "per live ladder" after each landing | SB is stale and reference-only. If canonical work moves to a new campaign directory, no SB B\<k\> is needed; the clause applies to whatever ladder is live then. | §7.B |
| 17 | RR:272, :328 | P1_canon is "the side bisect's … lineage"; the fix comes "first, in the side bisect" | No one owns the side bisect; both items pass to you, unapproved. | CH:787-791 |
| 18 | RR:24, :298-302 | The totals exclude canonical training | It adds about 16 GPU-h of training at two seeds per step: 2 × (1.2–1.3 + 1.3 + 5.4–5.5). That is about 20 GPU-h with evals (RR:302), plus about 250 lines + tests for the wrapper (AN:205) and the landing's B\<k\> round. | AN:205, :241; RR:302 |

RR:173 is still correct: `KRABBY_OMNI_CANON` exists in no `.py`; only the eval `--canon` hook exists.

**Stale lines in DOSSIER.md:**
- **DOS:66, :593 and :968:** X20 is described as "live" or "still training". It finished SLOW/SLOW, and X21–X25 came after it. DOS:968 should also say the side bisect was stopped at 21:07.
- **DOS:161:** episode lengths also include 20 s (X24) and 30 s (X25).
- **DOS:568-569:**
  - Add X20 and X24 to the two-direction list. The overall range 0.028–0.103 is unchanged.
  - The SLOW sub-range at DOS:568 becomes 0.052–0.103, because X24 R reads .054 / .052.
- **DOS:571 and :639:** the "4 of 5" claim; see row 4 above.
- **DOS:582:** all 25 runs (not 20) share their PPO settings. Episode length is an env.yaml knob, not an agent.yaml one.
- **DOS:587 and :950:** C_fa and C_40 did run. C_fa is FAST, and C_40 is DEAD at two seeds.
- **DOS:713:** RSI is now 5 of 5 SLOW or DEAD in two-direction arms.
- **DOS:717:** single-command episodes were tested at three lengths, all SLOW or DEAD:
  - 10 s: 3 arms; C_lr_10_rsi L is DEAD.
  - 20 s: X24.
  - 40 s: LR4_dwell, which has a different config (4-image mirror, +y RSI bank) and is DEAD.

  30, 50 and 60 s are untested, and so is the C_lr-family 40-s rung.
- **CH:699** says "Noise std reaches 2.0 as in every arm". In fact 5 of the 22 classified arms stay below 1.99 through it 2999 (§5).

---

## 2. Bottom line

- **The problem.** Every network that serves both side directions walks sideways slowly, and so does every +y-only network trained with the 4-image mirror.
  - The evidence: 10 two-hand arms plus 4 4-image arms, at feet_ss 0.028–0.103 m/s, all SLOW or DEAD.
  - +y-only fa networks at a 0.8 moving share are fast: 5 FAST at 0.184–0.215, plus 1 INTERMEDIATE (C_mo s3) at 0.162 / 0.173.
- **Root cause, established with high confidence** (RC:27-46). Sideways thrust comes from a hip/knee rod stroke locked to the tripod phase. Under the exact mirror M_y, the direction-flipped (odd) part of that stroke does the pushing. Every two-hand network lacks that stroke.
  - Rod odd RMS separates the classes: FAST 1.50–2.16 vs SLOW/DEAD 0.15–0.93.
  - Canonical deploy means a +y head serves −y through the mirror map, with no training. It is FAST on both hands on 4 of 4 heads. That rules out the plant, the sim and the mirror map.
  - Capacity is ruled out for the open-loop fit: a fresh network of the deployed architecture distils the two-hand fast policy, open loop, to about 99 % sign agreement.
- **Not established: why PPO does not form the stroke** in a two-hand network. The candidates are:
  - (i) a direction × phase formation barrier;
  - (ii) per-hand sample rate × early saturation lock-in;
  - (iii) mirror loss against RL (4-image arms only).

  RC rates each at medium confidence, inferred only. No gradient, advantage or action telemetry was ever logged.
- **What the PPO evidence shows.** All 25 runs used the same PPO settings, so the bisect tested no hyperparameter.
  - **Optimizer signals overlap.** Noise std, LR, entropy, surrogate loss and the clipped-rod share all overlap between FAST and the full SLOW/DEAD set.
  - **σ timing.** σ first reaches 1.99 at it 1264 and the 2.0 cap at it 1594, both in C_40, and later in every other arm. The classes split at about it 540–600.
  - **Signals that sort by command pool.** Value loss, surrogate loss and σ (its 400–1000) split the FAST arms from the ten two-direction arms from about it 300–600. But they sort by command pool, not by class: the slow 4-image arms read like the fast ones.
  - **Against all slow arms.** Only the track ratio separates, from it 600. Value loss follows from about it 1000, which fits an outcome effect.
- **The designed test of (ii) did not discriminate.** C_40 (+y at a 0.4 share) is DEAD on both seeds, but it also halves the moving share. RR's DEAD branch applies, and C_lr_bound did not trigger.
- **Episode length: no tested length changed the class, but the question is deprioritised, not closed.**
  - In the C_lr family: 10 s (3 arms) and 20 s (1 arm, confounded with obstacle exposure).
  - 40-s single-command episodes only in LR4_dwell, which has a different config (4-image mirror, +y RSI bank).
  - The user stopped the ladder, so E ≥ 30 in the C_lr_10_rsi config was never evaluated.
- **The proposed fix is canonical training (`KRABBY_OMNI_CANON=vy`).** The network would only ever solve the one-direction problem, where one-direction runs reach the fast class.
  - **Prediction.** It is predicted FAST under every candidate (P3, untested). AN pre-registers a SLOW branch: flip and reversal states re-form the shuffle (AN:232-234).
  - **What it validates.** It validates the lever, not the mechanism (RC:480-482).
  - **Value now in doubt.** AN:212 notes its marginal value shrinks once the reversal manifest passes on existing heads, which it did (§3.4); the policy-switch fallback exists.
  - **Status.** It is not implemented, and it is not approved.
- **The cheapest optimizer-relevant mechanism test is T1:** a checkpoint-series eval of C vs C_lr. It needs no code and costs about 0.5 GPU-h (§7.F).
  - T1 separates formation from collapse. It does not split (i) from (ii).
  - The only designed (i)-vs-(ii) test is T2 with its additive control (§7.G), which can be interpreted now that C_fa is FAST.

---

## 3. Tests and results

### 3.1 Shared setup (cfg, all 25 runs)

**PPO, identical in every `params/agent.yaml`.** Only the seed, the run name and the mirror function differ.
- 256 envs × 24 steps, 5 epochs, 16 minibatches.
- LR 3e-4, adaptive schedule with desired_kl 0.01. The rule divides or multiplies the LR by 1.5, with a floor of 1e-5 and a ceiling of 1e-2 (`ppo_with_extractor.py:393-396`).
- γ 0.99, λ 0.95, clip 0.2, max_grad_norm 1.0, entropy 0.01.
- Initial σ 1.5 (scalar), clamped per dimension at 2.0 (`crab_actor_critic_with_encoder.py:10-11, :50`).
- `clip_actions` 1.0; `empirical_normalization` false.
- `mirror_loss_coeff` 0.5.
- 3000 iterations from scratch.
- Mirror function: the 4-image `crab_hex_omni_symmetry_augmentation` for O, R, C_band, C_rsi, LR4, LR4_dwell and LR4_dwell_s2; `_fa` for all the others.

**Base O = preset 1b** (cfg, X01):
- side_l only (dir band π/2), slots 0.2:0.8:0:0;
- 40-s episodes, with a new command every 10 s;
- lateral speed ellipse 0.25;
- RSI 0.2 from `rsi_bank_omni.npz`, source 1 (+y frames only);
- 4-image mirror;
- clock combine product, tracking std 0.2, L1 weight −2.0, yaw_progress weight 1.0.

| Overlay | Setting |
|---|---|
| fa | `KRABBY_OMNI_MIRROR_IMAGES=fa` |
| scal | `KRABBY_OMNI_CMD_OBS=scalar` (command direction hidden) |
| rsi0 | `KRABBY_RSI_FRAC=0` |
| ell35 | `KRABBY_OMNI_SPEED_ELLIPSE=0.35:0.35:0.25` |
| LR | `KRABBY_OMNI_CMD_POOL=stand:0.2,side_l:0.4,side_r:0.4`, `KRABBY_OMNI_DIR_BAND=''`, `KRABBY_OMNI_SLOTS=0.2:0.8:0:0` |
| Rk | `KRABBY_OMNI_CLOCK_COMBINE=sum`, `KRABBY_TRACK_SIGMA2=0.1`, `KRABBY_TRACK_L1_W=-1.0`, `KRABBY_YAW_PROGRESS_W=0` |
| dwell | `KRABBY_RESAMPLE_S=40:40` |
| E10 / E20 / E30 | `KRABBY_EPISODE_S=10/20/30` with `KRABBY_RESAMPLE_S=20:20 / 30:30 / 40:40` |
| bankA | RSI 0.2, `rsi_bank_side_lr.npz`, `KRABBY_OMNI_RSI_SOURCE=all` |
| bankB33 | RSI 0.33, `rsi_bank_side_lr_b.npz`, source all |
| sh40 | `KRABBY_OMNI_CMD_POOL=stand:0.6,side_l:0.4`, `DIR_BAND=''`, `SLOTS=0.6:0.4:0:0` |

**Pre-arm checks** (CK; CH:248-251):

| Check | Result |
|---|---|
| Knob smokes C / R / RC, 20 its each | overlay_checks 4/4, 4/4, 8/8 (CK:13-30) |
| W harness check, lat_l | 0.096 / 0.103, inside 0.095 ± 0.01: PASS (CK:36-39) |
| Sideways FAST pin, model_3000 | s3 0.196 / 0.193; s2 0.201 / 0.202: PASS, no re-threshold (CK:46-47) |
| W at side commands 0.15 / 0.22 / 0.30 / 0.35 | feet_ss 0.093 / 0.097 / 0.105 / 0.110, while lean04 (lean built over 0–4 s) grows from 0.19 to 0.28 m (CK:57-62) |

### 3.2 One-direction (+y) arms, driver eval `side22_25`

- "Early" is track_ratio_mean averaged over its 800–1000; "final" is averaged over its 2900–3000 (tb).
- lean_abs is the signed median of body minus feet (`side22_classify.py:49`).

| # | Arm | Run | Seed | Change from O | Class | feet_ss | lean_abs | sweep | Early / final | Record |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | O | X01 | 3 | none | SLOW | .084/.094 | .183/.197 | .042/.047 | .564/.568 | O.json; chain_O.log; CH:255 |
| 2 | C | X02 | 3 | fa, scal, rsi0, ell35 | FAST | .215/.214 | .030/.051 | .085/.083 | .671/.798 | C.json; CH:258 |
| 3 | R | X03 | 3 | Rk | SLOW | .075/.078 | .157/.180 | .023/.023 | .473/.462 | QL:3-5; CH:269 |
| 4 | C_mo | X04 | 3 | fa, scal | INTERMEDIATE | .162/.173 | .116/.110 | .056/.057 | .654/.773 | QL:14-16; CH:315-317 |
| 5 | C_band | X05 | 3 | ell35 | SLOW | .093/.100 | .197/.211 | .049/.052 | .561/.579 | QL:24-26; CH:282 |
| 6 | C_rsi | X06 | 3 | rsi0 | SLOW | .069/.077 | .203/.211 | .034/.037 | .498/.536 | QL:37-39; CH:289 |
| 7 | C_s2 | X07 | 2 | as C | FAST | .194/.209 | .049/.056 | .082/.090 | .702/.746 | QL:46-48; CH:296 |
| 8 | C_mo_s2 | X08 | 2 | as C_mo | FAST | .203/.210 | .005/.031 | .086/.087 | .700/.870 | QL:59-61; CH:307 |
| 9 | C_mo_s1 | X09 | 1 | as C_mo | FAST | .184/.185 | .065/.074 | .076/.069 | .765/.853 | QL:72-74; CH:355 |
| 21 | C_fa | X21 | 3 | fa | FAST | .211/.206 | .028/.068 | .084/.066 | .794/.907 | QL:285-292; CH:676-683 |
| 22 | C_40 | X22 | 3 | C + sh40 | DEAD (falls) | .035/.031 | .399/.380 | .043/.050 | .325/.523 | QL:295-302; CH:696 |
| 23 | C_40_s2 | X23 | 2 | as C_40 | DEAD (stands) | .006/.006 | .304/.306 | .023/.017 | .180/.157 | QL:306-313; CH:705-708 |

### 3.3 Two-direction arms, both-hands eval `side22_25lr__omni` (`SB/scenarios_side_lr.yaml`)

| # | Arm | Run | Seed | Change from O | Class L / R | feet_ss L | feet_ss R | lean L; R | sweep L; R | Early / final | Record |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 10 | LRfa | X10 | 3 | LR, fa | SLOW / SLOW | .061/.065 | .087/.088 | .250/.260; .162/.178 | .036/.038; .033/.034 | .536/.523 | QL:88-94; CH:361 |
| 11 | LR4 | X11 | 3 | LR | SLOW / SLOW | .085/.091 | .082/.096 | .117/.134; .114/.127 | .033/.035; .033/.033 | .571/.598 | QL:105-111; CH:368 |
| 12 | LR4_dwell | X12 | 3 | LR, dwell | DEAD / DEAD | .047/.058 | .055/.028 | .184/.208; .206/.164 | .029/.029; .029/.026 | .491/.482 | QL:124-130; CH:405 |
| 13 | LR4_dwell_s2 | X13 | 2 | as X12 | stopped at it 484 (model_400), no eval | — | — | — | — | ratio .514 at it 400 | QL:133-134; CH:416-421 |
| 14 | C_lr | X14 | 3 | LR, fa, rsi0, ell35 | SLOW / SLOW | .072/.077 | .063/.073 | .193/.210; .205/.215 | .034/.035; .033/.039 | .524/.528 | QL:149-157; CH:491 |
| 15 | C_lr_10 | X15 | 3 | C_lr + E10 | SLOW / SLOW | .085/.085 | .067/.073 | .153/.158; −.008/−.006 | .051/.054; .039/.041 | .538/.570 | QL:164-172; CH:495 |
| 16 | C_lr_rsi | X16 | 3 | C_lr with bankA instead of rsi0 | SLOW / SLOW | .085/.085 | .069/.075 | .268/.263; .210/.211 | .035/.035; .034/.037 | .512/.481 | QL:214-221; CH:580 |
| 17 | C_lr_10_rsi | X17 | 3 | C_lr_10 + bankA | DEAD / SLOW | .042/.042 | .078/.081 | .069/.072; .141/.146 | .043/.046; .044/.043 | .545/.578 | QL:230-238; CH:614 |
| 18 | C_lr_10_rsi_s2 | X18 | 2 | as X17 | stopped at it 1326 (model_1300), no eval | — | — | — | — | .535 / — | QL:239-241; CH:618-619 |
| 19 | C_lr_rsi33 | X19 | 3 | C_lr + bankB33 | SLOW / SLOW | .094/.103 | .069/.072 | .209/.222; .175/.181 | .027/.028; .048/.050 | .522/.582 | QL:259-266; CH:628 |
| 20 | C_lr_10_rsi33 | X20 | 3 | C_lr_10 + bankB33 | SLOW / SLOW | .080/.082 | .080/.081 | .154/.160; .166/.170 | .049/.058; .037/.039 | .540/.550 | QL:272-279; CH:643 |
| 24 | C_lr_20_rsi | X24 | 3 | X17 with E20 | SLOW / SLOW | .071/.095 | .054/.052 | .016/.020; .057/.051 | .079/.089; .057/.061 | .504/.591 | QL:326-339; CH:778 |
| 25 | C_lr_30_rsi | X25 | 3 | X17 with E30 | stopped at it 1922 (model_1900), no eval | — | — | — | — | .524 / — | QL:340-343; CH:787-790 |

**Notes on these tables:**
- **The driver eval agrees.** The lat_l driver eval agrees with the both-hands eval within 0.003: LRfa .061/.064, LR4 .085/.092, LR4_dwell .047/.061, C_lr .072/.076.
- **C_rsi is bimodal** (`feet_ss_all`). 35 % / 34 % of episodes are below 0.03, and the walking median is 0.079 / 0.086.
- **LR4_dwell stalls.** 42–51 % of included episodes are below 0.03, and the walkers move at 0.067–0.070. It also had 36–44 wrong-way episodes per hold (CH:767-771).
- **C_lr_rsi33 falls far more than its siblings.** crab_failure over its 2900–3000 is 0.336 (flat 0.290, obstacle tiles 0.328), against 0.03–0.11 in the sibling arms. CH:628-631 does not mention it.

**Pattern:**
- **All 14 slow-family networks are SLOW or DEAD**, at 0.028–0.103. That is the 10 two-direction arms plus the 4 4-image arms (O, R, C_band, C_rsi).
- **The 4-image reading is confounded.** Those arms are read as training phantom −y rows, and that reading is confounded with H6 (§4.2 (iii)).
- **+y-only fa runs at a 0.8 moving share** are 5 FAST and 1 INTERMEDIATE (6 runs).
- **At a 0.4 moving share** (C_40), the same setup is DEAD on both seeds.

### 3.4 Canonical deploy: zero training, 08:14–08:58 (`SB/gate/canon_lr.json`; QL:174-196; CH:547-562)

Each existing +y head serves −y through the mirror map. Each cell is lat_l at 0.22 / 0.25 ; lat_r at 0.22 / 0.25.

| Head | wave mode | cmd mode | Class L / R | sweep (wave) |
|---|---|---|---|---|
| C_mo_s2 (X08) | .203/.211 ; .199/.209 | .203/.207 ; .197/.204 | FAST / FAST | .077–.086 |
| C (X02) | .215/.217 ; .211/.216 | .215/.216 ; .212/.212 | FAST / FAST | .081–.086 |
| C_s2 (X07) | .194/.208 ; .190/.203 | .194/.205 ; .197/.206 | FAST / FAST | .076–.082 |
| C_mo_s1 (X09) | .184/.185 ; .182/.191 | .184/.189 ; .180/.186 | FAST / FAST | .074–.077 |

- In wave mode, |L−R| ≤ 0.006 at every speed.
- **Reversals** (`CANON/*_cmd_rev_*.log`): body speed after the reversal holds is 0.168–0.190, against 0.170–0.194 before the reversal.
- **Falls per 100 episodes**, direct reversal / reversal with a pause:

  | Head | Direct | With pause |
  |---|---|---|
  | C | 25 | 45 |
  | C_s2 | 28 | 41 |
  | C_mo_s2 | 16 | 20 |
  | C_mo_s1 | 5 | 7 |

  The per-hold counts are in §6. Paused reversals raise C's and C_s2's falls.

### 3.5 RSI banks

Both banks were harvested from canonical C (X02, wave mode, checkpoint sha `966e56d5…`). The harvest script is `scripts/harvest_canon_bank.py`.

| Bank | Harvested | Holds | feet_ss L / R per hold | Contents | Used by |
|---|---|---|---|---|---|
| A: `rsi_bank_side_lr.npz` | 09:00–09:05 (QL:197-207) | 0.22, 0.28, 0.35 | .215/.211; .207/.199; **.156/.145** at 0.35 (lean .149/.154, sweep .055/.051) | sha `fd6704ae…`; tags {1: 2000, 2: 2000}; speed p10/p50/p90 0.19/0.28/0.40. **31.7 % (L) and 31.4 % (R) of the candidate pool came from the degraded 0.35 holds.** | X16, X17, X18, X24, X25 |
| B: `rsi_bank_side_lr_b.npz` | 12:22–12:27 (QL:243-253) | 0.22, 0.25, 0.28 | .215/.211; .217/.216; .207/.210 (all FAST) | sha `743f7dd6…`; tags {1: 2000, 2: 2000}; median speed 0.284 / 0.277 | X19, X20 |

Neither bank, as implemented, changed the class. RSI starts underperform in every arm, fast ones included (§4.3).

### 3.6 Controls (X21–X23; CH:657-719)

**C_fa (= O + fa, direction visible) is FAST** at 0.211 / 0.206, early read 0.794.
- Switching O to the fa mirror is enough in one direction.
  - fa both removes the phantom M_y rows and triples the M_x image weight (H6; GA:144-146).
  - Which of the two changes matters is open (P7/5c).
- A visible direction slot does not by itself bring the shuffle back, so the direction-visible two-hand arms can be interpreted (CH:679-681).

**C_40 (one hand at a 0.4 share, direction hidden, fa, RSI off, band 0.35) is DEAD on both seeds.**
- **s3 falls.** 38 and 46 of 100 episodes die in the hold, and crab_failure is 0.72–0.78 from it 2000 on.
- **s2 stands.** The ratio peaks at 0.230 over its 1400–1500.
- Neither phenotype is C_lr's stable shuffle.
- The run halves the total moving share (0.4 vs 0.8). That confound was recorded before the verdict (CH:691-695), so P10 is inconclusive.
- **Early reads** (its 400 / 600 / 800 / 1000):
  - C_40 s3: 0.171 / 0.241 / 0.295 / 0.355.
  - C_40 s2: 0.145 / 0.148 / 0.167 / 0.191.
  - Fast one-hand runs: 0.48–0.56 / 0.56–0.69 / 0.63–0.75 / 0.67–0.78.

**Recorded reading, inferred only from the confounded C_40** (stand share 0.6, RSI off). C_lr's stepping and carrier are supported by pooled two-hand experience; only the direction-flipped stroke is missing (CH:715-716; RC:422-423).

### 3.7 Episode-length ladder, 19:09–21:07 (CH:721-791)

**Design.** C_lr_10_rsi at E = 20, 30, 40, 50 and 60 s. A rung succeeds only if both hands are FAST and pass the along-command check (CH:769-770).
- The rungs were predicted SLOW or DEAD at every length (CH:739-744).
- A FAST rung "would put the episode structure back in play as a lever" (CH:743-744). That stays open for E ≥ 30.
- The review is at CH:748-777. The runner moved from v1 to v2 at 19:52 (QL:320-322).

**C_lr_20_rsi (X24): SLOW / SLOW, FAIL, no seed 2.**
- Along-command check (`arms/C_lr_20_rsi_along.json`): L .069/.091 SLOW; R .049/.051, DEAD by 0.001.
- 0 wrong-way episodes, 0 static, and 3–8 short episodes per hold.

**C_lr_30_rsi (X25):** stopped by the user and never evaluated.

**C_lr_40/50/60_rsi:** not run. They are defined in candidates v16 at :782-850.

**Obstacle telemetry over its 2900–3000** (`scripts/eplen_telemetry.py`; matches CH:784-785):

| Run | r_max (m) | reach_obst | block_steps | falls on flat | falls on obstacle tiles | all falls |
|---|---|---|---|---|---|---|
| X17 (E = 10) | 0.823 | 0.001 | 0.005 | 0.041 | 0.033 | 0.037 |
| X24 (E = 20) | 1.413 | 0.225 | 18.7 | 0.078 | 0.112 | 0.072 |
| X25 (E = 30), its 1800–1923 (last metric step) | 2.554 | 0.423 | 78.0 | 0.132 | 0.196 | 0.154 |

### 3.8 Cost

**As recorded:**
- About 9.9 GPU-h for O through C_mo_s2 plus the checks (CH:311; GA:169 says about 10).
- About 3.5 for C_mo_s1, LRfa and LR4. Total through LR4 about 13.6 (GA:267-268).
- About 3.7 for the controls (CH:709).
- Ladder estimate about 1.5 per arm (CH:746). No actual figure was recorded.

**Training wall time, computed.** These figures exclude all evals and harvests.

| Runs | X01–08 | X09–11 | X12–13 | X14–20 | X21–23 | X24–25 | **All 25** |
|---|---|---|---|---|---|---|---|
| h (tb, first to last event) | 9.12 | 3.34 | 1.30 | 7.34 | 3.47 | 1.81 | **26.38** |

The sum of ST `keys.*.wall_h` gives 26.93 h.

### 3.9 Designed but not run

- **Seed arms whose triggers never fired:**
  - LRfa_s2, C_lr_s2, C_lr_10_s2, C_lr_rsi_s2, C_lr_rsi33_s2, C_lr_10_rsi33_s2, C_fa_s2, C_lr_20_rsi_s2 and R_s2;
  - RC, which needed both C and R to be SLOW (CH:70, 270);
  - O_s2 and LR4_s2: the seed-lottery check, never launched after the user's pause (CH:394-421).
- **Killed or skipped:** C_nb and C_nr were killed at a user stop (CH:305), then skipped by rule once C_mo_s1 was FAST (CH:358).
- **Decision-tree success branches, never triggered:** FB_C_lr, FB_C_lr_10, P1_C_lr, P1_C_lr_10.
- **Ladder:** C_lr_30_rsi_s2 and C_lr_40/50/60_rsi (+ `_s2`). The pure-flat control was the user's call (CH:758-759).
- **Mechanism tests:** RC/AN T1, T2 (+ its additive control), 5c/P7, P9 and 5a.
- **Fix path:** C_lr_canon, FB_canon, P1_canon, and the fallbacks.
- **Not designed at all:** a clean rate test for (ii) (CH:694-695). C_lr_bound appears only in RR; it is not in candidates v16.

---

## 4. Mechanism

### 4.1 Established (RC:27-46, "high confidence")

1. **The thrust split.** Sideways thrust comes from a phase-locked hip/knee rod stroke; fore-aft thrust comes from the cams.
   - C at 0.215 lat_l: rods 0.189, cam yaw 0.014, slip 0.013 m/s.
   - W forward at 0.203: cams 0.175, rods 0.011 (RC:78-81).
   - Radial stroke amplitude: FAST 0.077–0.113 m vs SLOW/DEAD 0.010–0.057 m, r = 0.94 with feet speed (RC:82-84, 108-109).
2. **The odd part does the pushing.** Under M_y, the direction-flipped, phase-locked part pushes at speed.
   - The static odd part (the lean) also pushes, but only at the slow gaits' 0.06–0.10 m/s (RC:30-31).
   - On canonical heads, foot loading stays the same across hands (+0.80 to +0.97) while the stroke reverses (−0.64 to −0.92).
   - So a two-direction policy must output sign(vy) × g(phase) on the rods, while a one-direction policy needs only g(phase) (RC:92-113).
3. **The signature** (RC:127-143, 308-355). It is robust to the choice of 12, 24 or 36 phase bins, and to clock vs data-derived phase.

   | Statistic | FAST | INT | SLOW/DEAD | Separates? |
   |---|---|---|---|---|
   | Rod odd RMS | 1.50–2.16 | 1.46–1.53 | 0.15–0.93 | yes: gap 0.57 (1.6×), Spearman vs feet_ss 0.81 |
   | Joint-space odd RMS (rad) | 0.203–0.269 | | 0.069–0.158 | yes, 1.3× |
   | Even stepping | 1.41–1.83 | | 0.82–2.30 | no (Spearman −0.02) |

4. **The symptom.** Every two-hand network lacks the stroke, and so does every 4-image arm.
   - Two mechanisms produce one symptom: real −y data, or the 4-image mirror's phantom −y rows. The 4-image-arm mechanism is confounded with H6 (RC:145; GA:144-146).
   - Direction rides a static carrier instead, either a knee-bias lean or corner-cam spin.
   - The static odd part is 0.26–2.47 in slow rows but also 0.07–1.05 in fast rows (C_mo_s1 1.03). It is a symptom, not a separator (RC:286).
5. **Fast-class timing.** One-direction fa nets at a 0.8 moving share reach the fast class: 5 of 6, with C_mo s3 INTERMEDIATE.
   - Their track ratio separates from the slow arms by about it 600 (RC:251 puts the split at it 540).
   - The stroke's own onset is unmeasured: odd RMS exists only at final checkpoints (P5/T1, RC:397).
   - C_40 (0.4 moving share, RSI off) forms no lateral gait: DEAD, falling on s3 and standing on s2. That failure is upstream of the stroke, so it is not evidence on (ii) (CH:715-718).
6. **Canonical deploy** is FAST on both hands on 4 of 4 heads (§3.4; AN:318-331; RC:270-279).
7. **The lean is not the invariant.**
   - Slow arms with small leans: LR4 at 0.11–0.13 m, C_lr_10 R at −0.01 m, C_lr_20_rsi at 0.016–0.057 m.
   - Knee bias vs speed: r = −0.28 (GA:239-243; RC:371).
8. **Sweep is no longer a perfect separator.** C_lr_20_rsi L has a fast-range sweep (0.079 / 0.089) but is SLOW. Odd RMS still separates it (0.78 / 0.82, below).
9. **Cadence works only one way.** Above about 0.95 Hz is never FAST, but slow arms can step at FAST cadence. C_lr_20_rsi steps at 0.67–0.78 Hz, slower than the fast heads' 0.81–0.92 Hz (AN:87; GA:110-113).
10. **Capacity, open loop.** A fresh network distils the two-hand fast policy to 98.9–99.0 % held-out sign agreement (open loop, 40 epochs, 2 seeds; RC:184-190, 364).
    - Its stroke reaches 0.585, against the teacher's 0.583.
    - A bare two-hand net has not been shown to walk closed-loop or to survive PPO (RC:540-541).

**New CPU rows.** These come from rerunning `scripts/rootcause/sym/extract.py: one()` + `sym.py: row()` in memory. The rerun reproduces RC's values for C (1.90 / 2.03) and for C_lr (L 0.59 / 0.61, R 0.45 / 0.46).

| Stamp | Hand | Class | Rod odd RMS | Rod even RMS | Static odd | Joint odd RMS (rad) | Note |
|---|---|---|---|---|---|---|---|
| C_lr_20_rsi (X24) | L | SLOW | 0.78 / 0.82 | 2.43 / 2.46 | 0.66–0.67 | 0.127 / 0.135 | sweep 0.079 / 0.089; 0.70 / 0.78 Hz |
| C_lr_20_rsi (X24) | R | SLOW | 0.71 / 0.66 | 2.49 / 2.51 | 0.44–0.47 | 0.130 / 0.113 | 0.67 Hz |
| C_lr_10_rsi33 (X20) | L | SLOW | 0.65 / 0.70 | 2.04 / 2.16 | 0.70–0.71 | | |
| C_lr_10_rsi33 (X20) | R | SLOW | 0.65 / 0.70 | 1.88 / 1.95 | 1.54–1.64 | | |
| C_fa (X21) | L | FAST | 2.06 / 1.99 | 1.57 / 1.58 | 0.33–0.39 | 0.261 / 0.226 | in the FAST band |
| C_40 (X22) | L | DEAD | 0.78 / 0.99 | | 2.0–2.2 | | survivors only (n 43 / 36); not comparable |
| C_40_s2 (X23) | L | DEAD | 0.04–0.05 | 0.04–0.05 | | | standing |

**Reading the new rows.** X24's even stepping (2.43–2.51) is the largest recorded; RC's range was 0.82–2.30. Its long, low-cadence sweeps do not add up to the odd stroke.

**Correction to SSI.** SSI:77-80 says "unidirectional 1b leans". This is superseded: 1b was never one-direction, because the 4-image mirror trains phantom −y rows (GA:139-144).

### 4.2 Not established: why PPO misses the stroke

RC rates (i), (ii) and (iii) each at medium confidence, inferred (RC:48-61, 173-218).

**(i) Direction × phase formation barrier.** The stroke's gradient on direction-independent features cancels across hands. The even stepping and an additive carrier get same-signed gradients and win the race.

For (i):
- Supervised fitting of the interaction succeeds, open loop (§4.1, item 10).
- One-direction nets reach the fast class on schedule.
- Static carriers are present in the slow rows.
- **The open-loop distilled student reads direction from the state** (RC §2.6, "confidence: medium; inference", RC:202-211).
  - Its hand difference is 0.79 of the spread.
  - Its command-flip response is 0.067, inside the slow nets' 0.05–0.17.
  - RC infers from this that a formed stroke sustains itself and that the barrier is at formation.
  - Whether a formed stroke survives two-direction PPO is open (T1, P9, AN 5a).
- H1a (weak direction input) is refuted: the slot-9 first-layer column norm is 2.2–2.9× the median proprio column by it 100 (RC:197-200).

Against or unexplained:
- The two-direction fa arms are not mirror-equivariant: M_y residual 0.25–0.40, against 0.04–0.05 for LR4. Yet they did not learn C's stroke on their +y rows alone (RC:57-59).
- **Fast-gait RSI fails to get around the barrier.** It decays to the shuffle within about 2 s in every RSI arm. RC calls this RSI step its chain's least-supported link.
  - RC reads the generic RSI deficit (in FAST C_mo too: edge reach 0.870 vs 0.980, failure 0.127 vs 0.021) as RSI being inert at init std 1.5 (RC:213-218).
  - RR §4.5 proposes cam braking at reset as an alternative explanation, to be tested by rung 0b (untested).

**(ii) Rate × early saturation lock-in.** A hand-specific stroke gets at most 40 % of samples, while the hand-agnostic shuffle gets all of them (RC:55-57). Saturation statistics, identical in fast and slow runs (RC:248-252):

| Statistic | it 300 | it 600 | it 1500 | it 2999 |
|---|---|---|---|---|
| Plain clipped share of rod means (C's own number; 0.59–0.88 over 22 arms at it 300) | 0.70 | 0.84 | | |
| Share pinned past 1+2σ | 0.01–0.06 | 0.15–0.27 | 0.44–0.57 | 0.62–0.72 |

**How these saturation numbers can be read.**
- **The probes are off-distribution.** They run early checkpoints on final-eval states (RC:536-537).
- **RC reads them two ways.**
  - "A lock is present *at* formation", supporting (ii) as much as (i) (RC:248-252).
  - Saturation at it 300 "cannot select" and only "plausibly locks" by the end (RC:290).
- **At formation, the clip is crossed but not locked.** With σ ≈ 1.65, a mean just past ±1 is not locked: the lock share at it 300 is only 0.00–0.10 (§5.3).
- **Conclusion.** The evidence is compatible with (ii) but does not show lock-in at formation. T1's `--sample-actions` run is the on-distribution read.
- **The designed test failed.** C_40 was confounded (§3.6). No clean rate arm exists, because a 0.8 moving share with +y at 0.4 forces a second command (RC:424-425).

**(iii) Mirror loss vs RL (4-image arms only).**
- The detached-target loss (coefficient 0.5) pulls the +y output toward its −y image (`ppo_with_extractor.py:432-456`). The critic never sees those rows (RC:155-161).
- O's symmetry loss rises from 0.037 to 0.070 (RC:60-61; RC:164 says 0.040 → 0.071).
- The direction-specific share of the +y rod pattern is O 0.42 and C_band 0.44–0.46, against 0.65–0.80 in fast +y runs.
- (iii) is confounded with H6: fa triples the M_x image weight. Only the no-mirror arm (5c/P7) separates them.
- (iii) is irrelevant to the lr runs.

**Other open sub-mechanisms:**
- AN 1c, "formed then collapses once −y is trained" (AN:114), is addressed by T1, 5a and P9.
- SSI's onset-payoff trap (+0.23/s for about 2 s; SSI:27-30) was never tested by intervention. C and C_lr share the reward and split FAST/SLOW, so the reward alone cannot select the class (RC:260).

### 4.3 Ruled out

| Candidate | Ruled out by |
|---|---|
| Plant, sim, −y hand harder; M_y map bug | Canonical deploy 4/4 FAST both hands, \|L−R\| ≤ 0.006; unit tests (RC:363) |
| Capacity or architecture (open-loop fit) | CPU distillation, about 99 %, open loop, 2 seeds (RC:364) |
| Weak direction input (H1a) | Slot-9 weights 2.2–2.9× by it 100 (RC:365) |
| Reward pricing | C vs C_lr on the same reward; R SLOW; replay +0.25 to +0.43 /s for the fast gait (RC:366). Penalty levers (action_rate, foot_idle) were never trained as arms (RC:258-268) |
| Lateral band | C_band SLOW; C_lr SLOW with band 0.35 (RC:367) |
| RSI as implemented (on/off, +y bank, banks A and B) as a class selector | 8 RSI-varied arms SLOW or DEAD; C_mo FAST with RSI on (RC:368). RSI starts underperform in every arm, fast ones included, so effective fast-gait seeding itself is untested (RR §4.5, rung 0b) |
| Within-episode reversals; episode length *as tested* | LR4_dwell DEAD (40-s single command, but 4-image mirror and +y RSI bank). In the C_lr family, E = 10 (3 arms) and E = 20 (1 arm, obstacle-confounded) are SLOW or DEAD (RC:369). E ≥ 30 in the C_lr_10_rsi config was never evaluated: deprioritised, not closed |
| Mirror type in two-direction training | LRfa and LR4 both SLOW (RC:370) |
| The lean as cause | §4.1, item 7 |
| Saturation or σ as the *selector* | Overlapping in both classes at every logged checkpoint; no separation against the full SLOW/DEAD set (RC:372; §5.4). σ at its 400–1000 splits FAST from the two-direction arms only, not from the 4-image arms |
| Critic interference; clock cadence law; ring/relay/heading slots | RC:374-376 |
| Visible direction slot (one direction) | C_fa FAST |

**Not ruled out:**
- **Half the data per direction.** It is not *necessary* for slowness: O is slow with all the +y data, through its 4-image mirror. Whether it is *sufficient* in the two-direction arms is candidate (ii), and C_40 is confounded (CH:691-695). RC:373 lists it as ruled out, but that holds only for necessity.
- **H6** (the M_x image weight) for the 4-image arms (RC:379).
- **Fast-gait seeding that actually takes** (see the RSI row).

---

## 5. PPO-facing evidence (tb, window [c−50, c+50), as in `scripts/early_read.py`)

**How to read these tables.**
- `Policy/mean_noise_std` logs the raw parameter, so it can read 2.0002 (`on_policy_runner_with_extractor.py:545-551`).
- First crossings are per iteration. This handoff re-read them from the tfevents: X22 ≥ 2.0 at 1594, X10 at 2939.
- Entropy is 18 × ½ ln(2πeσ²) of the clamped σ (32.84 at σ 1.5, 38.02 at σ 2). It is not an independent signal.
- crab_failure + total_terminates = 1.000 (checked on C).

### 5.1 Exploration and optimizer signals

"Floor" is the share of iterations with LR ≤ 1.0001e-5. "none" means the threshold was never reached by it 2999.

| Arm | Class | σ 300 | σ 1000 | σ 2000 | σ 2950 | 1st σ≥1.99 | 1st σ≥2.0 | LR 300 | Floor | H 300 |
|---|---|---|---|---|---|---|---|---|---|---|
| O | S | 1.633 | 1.761 | 1.944 | 2.000 | 2499 | 2832 | 4.9e-5 | .34 | 34.37 |
| C | F | 1.637 | 1.806 | 1.977 | 2.000 | 2267 | 2734 | 4.9e-5 | .33 | 34.41 |
| R | S | 1.575 | 1.662 | 1.824 | 1.939 | none | none | 3.5e-5 | .40 | 33.71 |
| C_mo | I | 1.646 | 1.768 | 1.920 | 1.987 | none | none | 4.8e-5 | .36 | 34.50 |
| C_band | S | 1.663 | 1.828 | 1.979 | 2.000 | 2129 | 2426 | 5.3e-5 | .34 | 34.69 |
| C_rsi | S | 1.621 | 1.746 | 1.938 | 1.986 | none | none | 4.0e-5 | .41 | 34.23 |
| C_s2 | F | 1.643 | 1.764 | 1.935 | 1.992 | 2850 | none | 4.9e-5 | .38 | 34.46 |
| C_mo_s2 | F | 1.617 | 1.739 | 1.924 | 1.995 | 2748 | none | 4.2e-5 | .36 | 34.18 |
| C_mo_s1 | F | 1.631 | 1.772 | 1.947 | 2.000 | 2636 | 2944 | 4.2e-5 | .37 | 34.33 |
| LRfa | S/S | 1.651 | 1.793 | 1.965 | 2.000 | 2566 | 2939 | 5.4e-5 | .36 | 34.55 |
| LR4 | S/S | 1.653 | 1.804 | 1.950 | 2.000 | 2491 | 2937 | 6.4e-5 | .33 | 34.57 |
| LR4_dwell | D/D | 1.634 | 1.818 | 1.965 | 1.993 | 2831 | none | 5.3e-5 | .35 | 34.37 |
| C_lr | S/S | 1.691 | 1.863 | 1.979 | 2.000 | 2238 | 2704 | 5.9e-5 | .36 | 34.98 |
| C_lr_10 | S/S | 1.629 | 1.787 | 1.936 | 1.983 | none | none | 4.4e-5 | .37 | 34.32 |
| C_lr_rsi | S/S | 1.696 | 1.880 | 1.999 | 2.000 | 1667 | 2047 | 5.7e-5 | .36 | 35.05 |
| C_lr_10_rsi | D/S | 1.645 | 1.839 | 1.985 | 2.000 | 2157 | 2860 | 5.6e-5 | .31 | 34.49 |
| C_lr_rsi33 | S/S | 1.669 | 1.859 | 1.989 | 2.000 | 2019 | 2410 | 5.3e-5 | .34 | 34.75 |
| C_lr_10_rsi33 | S/S | 1.641 | 1.817 | 1.960 | 1.998 | 2551 | none | 4.5e-5 | .34 | 34.45 |
| C_fa | F | 1.616 | 1.740 | 1.898 | 1.986 | none | none | 4.3e-5 | .37 | 34.17 |
| C_40 | D | 1.663 | 1.944 | 2.000 | 2.000 | 1264 | 1594 | 4.1e-5 | .50 | 34.69 |
| C_40_s2 | D | 1.648 | 1.886 | 2.000 | 2.000 | 1392 | 1603 | 2.7e-5 | .59 | 34.53 |
| C_lr_20_rsi | S/S | 1.670 | 1.866 | 1.994 | 2.000 | 1856 | 2265 | 5.5e-5 | .32 | 34.76 |

The stopped runs read as follows at it 300:
- X13: σ 1.676.
- X18: σ 1.637 (1.783 at it 1000).
- X25: σ 1.704 (1.948 at it 1000, first ≥ 1.99 at it 1389).

### 5.2 Losses and outcomes

`fail` = crab_failure share. TR = track_ratio_mean.

| Arm | sym 300 | sym 2950 | vf 300 | vf 1000 | vf 2950 | fail 1000 | fail 2950 | TR 400 | TR 600 | TR 800 | TR 1000 | TR 2950 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| O | .0488 | .0699 | .0038 | .0048 | .0067 | .030 | .019 | .553 | .557 | .565 | .564 | .568 |
| C | .0260 | .0329 | .0039 | .0080 | .0105 | .129 | .073 | .508 | .606 | .662 | .680 | .798 |
| R | .0516 | .0471 | .0041 | .0042 | .0049 | .041 | .102 | .491 | .509 | .485 | .466 | .462 |
| C_mo | .0282 | .0332 | .0037 | .0068 | .0098 | .041 | .018 | .476 | .564 | .628 | .674 | .773 |
| C_band | .0457 | .0707 | .0033 | .0066 | .0077 | .253 | .175 | .491 | .544 | .555 | .567 | .579 |
| C_rsi | .0496 | .0500 | .0040 | .0055 | .0074 | .038 | .003 | .507 | .504 | .496 | .503 | .536 |
| C_s2 | .0292 | .0336 | .0041 | .0083 | .0077 | .068 | .030 | .510 | .635 | .697 | .706 | .746 |
| C_mo_s2 | .0282 | .0328 | .0038 | .0077 | .0117 | .068 | .153 | .510 | .587 | .665 | .739 | .870 |
| C_mo_s1 | .0297 | .0299 | .0041 | .0078 | .0104 | .037 | .005 | .557 | .685 | .753 | .775 | .853 |
| LRfa | .0261 | .0354 | .0029 | .0044 | .0062 | .133 | .113 | .552 | .521 | .537 | .530 | .523 |
| LR4 | .0414 | .0461 | .0024 | .0042 | .0085 | .159 | .107 | .535 | .554 | .576 | .568 | .598 |
| LR4_dwell | .0408 | .0485 | .0029 | .0036 | .0025 | .178 | .007 | .477 | .498 | .491 | .491 | .482 |
| C_lr | .0272 | .0334 | .0024 | .0039 | .0051 | .197 | .110 | .468 | .523 | .522 | .526 | .528 |
| C_lr_10 | .0247 | .0332 | .0029 | .0045 | .0055 | .038 | .031 | .542 | .517 | .527 | .543 | .570 |
| C_lr_rsi | .0262 | .0304 | .0023 | .0035 | .0081 | .128 | .030 | .443 | .505 | .519 | .506 | .481 |
| C_lr_10_rsi | .0277 | .0369 | .0030 | .0039 | .0041 | .053 | .037 | .537 | .559 | .556 | .548 | .578 |
| C_lr_rsi33 | .0252 | .0340 | .0030 | .0040 | .0056 | .283 | .336 | .481 | .508 | .511 | .533 | .582 |
| C_lr_10_rsi33 | .0269 | .0356 | .0027 | .0038 | .0038 | .062 | .036 | .516 | .529 | .534 | .543 | .550 |
| C_fa | .0288 | .0318 | .0050 | .0080 | .0088 | .149 | .115 | .535 | .673 | .771 | .812 | .907 |
| C_40 | .0196 | .0179 | .0099 | .0340 | .0476 | .489 | .754 | .171 | .241 | .295 | .355 | .523 |
| C_40_s2 | .0160 | .0107 | .0172 | .0532 | .0861 | .088 | .025 | .145 | .148 | .167 | .191 | .157 |
| C_lr_20_rsi | .0259 | .0325 | .0031 | .0052 | .0048 | .114 | .072 | .466 | .482 | .504 | .514 | .591 |

**X25** (stopped): sym .0238, vf .0023 / .0034, fail at 1000 .161, TR .428 / .491 / .528 / .517.

### 5.3 Clipped rod means

**Method.** Deterministic μ on each arm's own final-eval states, following `scripts/rootcause/rep/probe_sat_mirror.py`. It reproduces that file's 8 arms exactly.
- These are off-distribution for early checkpoints (RC:536-537).
- "sat" = share of the 12 rod dimensions with |μ| > 1.
- "lock" = share with |μ| > 1 + 2σ.

| Group (members in order) | sat 300 | sat 600 | sat 1000 | sat 2999 | lock 300 | lock 2999 |
|---|---|---|---|---|---|---|
| FAST: C / C_s2 / C_mo_s2 / C_mo_s1 / C_fa | .70/.66/.59/.66/.68 | .84/.84/.78/.82/.82 | .88/.88/.86/.88/.87 | .94/.94/.94/.95/.94 | .06/.04/.05/.06/.02 | .67/.61/.63/.67/.62 |
| INT: C_mo | .64 | .81 | .89 | .94 | .02 | .65 |
| 4-image: O / R / C_band / C_rsi | .71/.73/.71/.78 | .85/.85/.86/.95 | .89/.94/.92/.97 | .96/.97/.95/.97 | .04/.06/.05/.09 | .71/.77/.72/.70 |
| LRfa / LR4 / LR4_dwell | .73/.62/.60 | .86/.79/.73 | .91/.87/.86 | .96/.94/.98 | .03/.01/.00 | .71/.62/.66 |
| C_lr / C_lr_10 / C_lr_rsi / C_lr_10_rsi | .63/.70/.59/.69 | .78/.83/.79/.83 | .86/.88/.88/.87 | .94/.91/.96/.94 | .01/.07/.00/.03 | .64/.56/.70/.66 |
| C_lr_rsi33 / C_lr_10_rsi33 / C_lr_20_rsi | .66/.69/.73 | .79/.83/.88 | .87/.87/.92 | .93/.93/.96 | .01/.04/.03 | .63/.63/.74 |
| C_40 / C_40_s2 | .75/.88 | .86/.95 | .91/.96 | .95/.97 | .10/.08 | .62/.85 |

At the final checkpoint, policy heads are 91–98 % saturated, with median |μ| 6.1–7.4 against a clip of ±1, fast and slow alike (AN:96).

### 5.4 What separates FAST from SLOW/DEAD

- FAST+INT is 6 arms; SLOW/DEAD is 16 arms: 10 two-direction, 4 4-image and 2 C_40.
- AUC = P(a FAST+INT arm reads higher than a SLOW/DEAD arm). "vs 2-dir" and "vs 4-img" restrict the SLOW/DEAD side to those families.

| Signal | FAST+INT | SLOW/DEAD | AUC | Verdict |
|---|---|---|---|---|
| σ at it 300 / 600 / 1000 / 2950 | 1.616–1.646 / 1.664–1.700 / 1.739–1.806 / 1.986–2.000 | 1.575–1.696 / 1.610–1.789 / 1.662–1.944 / 1.939–2.000 | all .24 / .15 / .18 / .31; vs 2-dir at 400 / 600 / 1000 .07 / .00 / .05 | Overlap vs all. At its 400–1000, lower than every two-direction arm (two-direction 1.703–1.787 at 600), but overlaps the 4-image arms (1.610–1.736); X18 (1.699 at 600) is in the fast range |
| First σ ≥ 1.99 | 2267–2850; never in 2 of 6 | 1264–2831; never in 3 of 16 | – | Overlap, and always after the split |
| LR at it 300 / 1000 | 4.2–4.9e-5 / 2.2–2.9e-5 | 2.7–6.4e-5 / 1.9–3.5e-5 | .31 / .35 | Overlap |
| LR floor share | .33–.38 | .31–.59 (.31–.41 without C_40 ×2) | – | Overlap |
| Rod sat at it 300 / 2999 | .59–.70 / .94–.95 | .59–.88 / .91–.98 | .27 / .31 | Overlap |
| Rod lock at it 300 / 2999 | .02–.06 / .61–.67 | .00–.10 / .56–.85 | .53 / .29 | Overlap |
| Symmetry loss at it 2950 | .030–.034 | fa two-direction .030–.037; 4-image .046–.071 | – | Sorts by mirror type, not by class |
| Value loss at it 300 / 1000 / 2950 | .0037–.0050 / .0068–.0083 / .0077–.0117 | two-direction .0023–.0031 / .0035–.0052 / .0025–.0085; 4-image .0033–.0041 / .0042–.0066 / .0049–.0077 | all .78 / .88 / .84; vs 2-dir 1.00 / 1.00 / .97; vs 4-img .62 / 1.00 / .96 | **Sorts by command pool at it 300**: two-direction arms sit below every one-direction arm, 4-image ones included. Splits FAST from the 4-image arms only from about it 1000, after the track ratio |
| Surrogate at it 300 / 400 / 600 | at 600: −.0129 to −.0112 | at 600: two-direction −.0149 to −.0132; 4-image −.0135 to −.0097; C_40 −.0092 / −.0078 | all .75 / .75 / .75; vs 2-dir .97 / 1.00 / 1.00; vs 4-img .58 / .50 / .50 | Splits FAST from the ten two-direction arms from it 300–400; overlaps the 4-image arms. X18 (−.0129 at 400, −.0128 at 600) is in the fast range |
| crab_failure at it 1000 / 2950 | .037–.149 / .005–.153 | .030–.489 / .003–.754 | .38 / .45 | Overlap |
| Track ratio at it 400 / 600 / 1000 / 2950 | .476–.557 / .564–.685 / .674–.812 / .746–.907 | .145–.553 / ≤.559 / ≤.568 / ≤.598 | .66 / 1 / 1 / 1 | Separates from it 600, thinly (see below) |

**Note on the track-ratio gap.** At it 600 the gap is .005 (C_mo .564 vs C_lr_10_rsi .559). C_lr_10_rsi is an E = 10 arm whose ratio is on a different basis (CH:765-766). Among 40-s arms the gap is .007 (O .557).

### 5.5 What this does and does not show

- **No hyperparameter was tested.** The bisect shows that the shared settings do not *select* the class.
- **The σ cap cannot be the selector.** σ during formation (its 300–600) is 1.57–1.79, against an action clip of ±1.
  - Two-direction arms grow σ marginally faster: 1.703–1.787 vs 1.664–1.700 at it 600.
  - X18 (1.699) and the 4-image arms (1.610–1.736) overlap the fast range.
- **Rod means pass the clip early in every arm.** At least 0.59 of rod means are past the clip at it 300.
  - The lock share is only 0.00–0.10 there.
  - Both are measured off-distribution.
  - RC reads this both as a lock present at formation (RC:248-252) and as unable to select (RC:290).
  - It is compatible with (ii) but does not show lock-in at formation. No bounded-mean or σ-schedule arm has run.
- **Value loss, surrogate and σ (its 400–1000) sort by command pool.** The ten two-direction arms differ from every one-direction arm, slow 4-image arms included.
  - The value loss of the two-direction arms is below every fast arm at its 300 and 1000, and overlaps by it 2950 (LR4 .0085 vs C_s2 .0077).
  - Against the 4-image arms, value loss separates the fast arms only from about it 1000. That is after the track ratio split at it 600, which fits an outcome effect: higher returns give a larger unnormalized value MSE.
  - Treat all three as untested correlates of the two-direction command pool, not as early class separators.
- **The C_40 pair carries the extreme readings:**
  - σ ≥ 1.99 at it 1264 / 1392, and the 2.0 cap at 1594 / 1603;
  - floor share .50 / .59;
  - value loss .034 / .053 at it 1000;
  - sat .75 / .88 at it 300.

  These go with the fall and stand phenotypes; causation is untested.
- **(i) and (ii) cannot be separated from these logs.** No per-sample gradient, advantage or action telemetry exists in X01–X25 (RC:533-535).
- **Formation timing.** The classes split at it 540 (RC:251). Fast runs pass the shuffle ceiling of about 0.53 at its 450–600 (RC:223-226).

---

## 6. Caveats and measurement notes

- **The gate has a direction hole.** feet_ss follows each episode's own displacement, so a wrong-way walker scores fast.
  - `scripts/along_cmd.py` scores along the command, with static episodes scored 0. It reproduces feet_ss within 0.004 on C_fa and C_mo_s2.
  - The hole can only inflate speed, so every SLOW or DEAD verdict stands.
  - Of the FAST verdicts, only C_fa and C_mo_s2 were direction-checked. Apply the check to any canonical arm, where a flag error makes a wrong-way walker (CH:767-771).
- **The DEAD rule's wording differs from the code.** The text says "< 0.05", but the code fires DEAD if *either* speed is below 0.05 (`side22_classify.py:112`). Three DEAD verdicts depend on this:
  - LR4_dwell L (.047/.058);
  - LR4_dwell R (.055/.028);
  - X24 R along-command (.049/.051).
- **Obstacle exposure grows with episode length.** 1b is 80 % flat and 20 % ring, with blocks 2.0–4.5 m from spawn (CH:754-759; §3.7).
  - Exposure did not select the class. E = 10 arms at reach_obst 0.001–0.003 (X15, X17) and 40-s arms at 0.195–0.227 (X14, X16) are all SLOW.
  - The flat-terrain control is in §7.H.
- **RSI changes the stand share.** RSI 0.2 lowers the actual stand share, because staged rows are forced to move: X17 .165, X24 .140, X16 .190, against RSI-off X14 .210 and X15 .203.
  - Staged commands are the frame's own bearing and speed, up to 20° off the lateral axis. In single-command arms they last the whole episode.
  - The 40-s rung was never a single-factor contrast with C_lr_rsi: it has about 3× the staged share (0.20 vs 0.07) and no reversals (CH:760-764).
- **Early reads.** They are not comparable across episode lengths (CH:765-766).
  - E = 20 reads .466 / .482 / .504 / .514; E = 30 reads .428 / .491 / .528 / .517.
  - Calibration on these 22 arms: FAST and INT runs read 0.654–0.794; every SLOW or DEAD run reads 0.180–0.571.
  - A ≤ 0.48 kill line would have caught only R, C_40 and C_40_s2.
  - `early_read.py:5-6` marks the bands as uncalibrated.
- **Resolution and seeds.** Single-seed resolution is 0.03–0.04 m/s hand-to-hand and seed-to-seed, so only FAST vs not-FAST is read (CH:772-773). Every two-direction and 4-image negative is seed 3 only (RC:528-532).
- **The eval is identical across rungs.** The both-hands eval runs its own 18-s episode and never receives `KRABBY_EPISODE_S` or `RESAMPLE_S`. So C_lr_10_rsi L DEAD (0.042) is real slow walking (CH:751-753).
- **The probes are off-distribution.** Early-checkpoint saturation and symmetry probes used final-eval states (RC:536-537). T1 fixes this.
- **Paused reversals raise falls for C and C_s2.**
  - **The record.** CH:560-561 gives falls per 8-s hold of 1–2 % (C_mo_s1), 3–9 % (C_mo_s2) and 7–19 % (C / C_s2), and calls them baseline.
  - **The per-hold counts.** These come from the alive counts in the logs' "facing by hold" lines (`CANON/{head}_cmd_rev_{direct,pause}__omni.log`).

    | Head | First lat_l hold | Post-reversal lat_r_2, with pause | Post-reversal lat_r_2, direct |
    |---|---|---|---|
    | C | 10/98 | 22/87 = 25 % | 6/88 = 7 % |
    | C_s2 | 10/99 | 17/88 = 19 % | 8/89 = 9 % |
    | C_mo_s2 | 2/99 | 9/96 | 7/97 |
    | C_mo_s1 | 2/100 | 2/97 | 1/98 |

  - **What it means.** For C and C_s2, the record's "falls at each head's baseline" (RC:277; AN:328-329) holds for direct reversals only. C_mo_s2's post-reversal falls rise in both modes. This bears on P1_canon's reversal risk (RC:509).
  - **Survival in general.** C's bundle falls 9–14 % in the hold, against 0–1 % for W (AN:41).
- **Stale lines in the SB records:**
  - RC:380-381, RC:545, AN:39 and AN:147 say "C_fa never run".
  - RC:393 says P2 is "training now".
  - RC:534-535 says "C_40 isolates the rate".
  - RC:40-41 says "4 of 5".
  - AN:22-23, AN:78-81 and RC:345 call sweep the perfect separator.
  - The two-direction lists at RC:36-37 and RC:147 predate X20 and X24.
  - CH:275 ("lean gone" for C_mo) is superseded by CH:315-317.
  - CH:742 calls LR4_dwell "RSI off". X12's env.yaml has RSI fraction 0.2 from `rsi_bank_omni.npz`, source 1.
- **Records for O, C and R.**
  - O's and C's classifier lines are not in QL. They are in chain_O.log, ARMS and CH:255-260.
  - R's early read (.473) is recorded at GA:166-167.
  - X13's partial ratio comes from tfevents only.

---

## 7. Open items, now passed to you (proposals, none approved)

**Approval status.** Nothing in this section has user approval.
- **These are proposals.** C_lr_canon, FB_canon, P1_canon, T1, T2, C_lr_bound and the rest come from AN, RC and RR.
- **What the user did approve.** The user's instructions in SB covered:
  - the canonical *eval*, the bank harvest and C_lr_rsi (CH:509-511, :587-588);
  - then each later arm, by name.
- **The last decision was a stop.** The user's last decision was to stop the ladder (CH:787-791).
- **Ask the user before launching any of these.**

**Common gate procedure:**
1. Run the driver `side22_25` eval and the both-hands `scenarios_side_lr.yaml` eval, each with the arm's own `run_meta.env_vars`.
2. Run `scripts/along_cmd.py`.
3. Class with `side22_classify.py`. FAST requires the along-command class to be FAST as well. Apply this rule by hand: queue_eplen2.sh's `decide()` (:29-33) uses only the feet_ss classes and only logs the along-command class (:99-103).
4. Compute odd RMS with `scripts/rootcause/sym/extract.py` + `sym.py`.

**Evaluating a canon-trained head:**
- **Canonicalize exactly once.** Use either the carried env knob or `--canon cmd --canon-hyst 0.08` (training hysteresis matches `cmd`, not `wave`; `eval_crab_hex_gait.py:117-131`). Never use both: M_y applied twice is the identity, so the −y hand would see raw observations.
- **Assert canonicalization.** Check that the stamp's `run_meta.canon` (or the knob) is present, as `canon_evals.sh:36-37` does for its smoke.
- **The driver row is safe.** The driver's lat_l-only `side22_25` row is the identity under canon. Only the both-hands and reversal evals are at risk.

**Common runner:** `SB/scripts/queue_eplen2.sh` is the hardened template (§7.J). It needs edits before reuse for a canon arm:
- Its `lr_eval` never passes `--canon` (:94-96).
- `FALLBACK_ST` is X17's env, which lacks the knob (:22, :116).
- SB paths (:12-21), the step list (:147) and the v16 accept string (:61) are hard-coded.
- Its PYTHONPATH points at the main checkout. Worktree code needs `PYTHONPATH=<worktree>/parkour` (RR:171).

Run unmodified, it would score the R hand without canonicalization and record a false SLOW or DEAD.

**A. Canonical-training wrapper, `KRABBY_OMNI_CANON` (not written)** (AN:214-220; RC:486-500)
- **Wrapper.** Subclass `ParkourRslRlVecEnvWrapper` (`parkour/scripts/rsl_rl/vecenv_wrapper.py`, instantiated at train.py:182).
  - Mirror the policy obs (the critic uses the same group) and inverse-mirror the actions per env.
  - Use a sticky flag with hysteresis h = 0.08 m/s, reset from the command sign on `dones`.
- **Reusable pieces.** Reuse only these:
  - `_canon_hyst_update` (`eval_crab_hex_gait.py:524-535`);
  - `lr.mirror_obs` / `lr.mirror_actions` from `build_mirrors` (`mdp/crab_hex_omni_mirror.py:314`);
  - tests in `tests/unit/test_crab_hex_omni_canon_eval.py`.

  `_canon_policy_step` (:567-577) also calls the policy and recomputes the estimator slice, and a training env wrapper must do neither. PPO already substitutes the estimator output during rollouts (`train_with_estimated_states: true`, X14 agent.yaml:96; `ppo_with_extractor.py:121, :189-198`).
- **Untested by the deploy result.** The wrapper also mirrors the true priv slice that trains the estimator and feeds the critic; the canonical deploy never exercised that. Add a CPU test that the wrapper's M_y round-trip is exact on the full observation, including the priv_explicit and scan slices.
- **Plumbing.** Add the knob in three places, then select the wrapper in train.py and in the eval harness (see "Evaluating a canon-trained head").
  - `crab_hex_omni_task/config/crab_hex/omni_knobs.py`, which moves code_sha;
  - `EVAL_CARRY` at `experiments/tools/run_phases.py:169`, which moves eval_sha;
  - `OVERLAY_DUMP_CHECKS` at `run_candidates.py:1860`.
- **Landing:**
  - The wrapper moves code_sha (CODE_PATHSPECS, RCD:163-170): `--accept-code-change`.
  - The EVAL_CARRY change moves eval_sha (RCD:177-183): `--accept-eval-change WHY`.
  - Records, docs and tests move neither hash.
  - FB_canon's masked mirror (§7.C) is also under CODE_PATHSPECS. Land it default-off in the same landing, or with RR's second landing, to avoid a third code_sha move.

**B. C_lr_canon** (P3; about 1.2–1.3 GPU-h per seed, AN:205)
- **Config.** The C_lr chain (candidates.yaml:353-371) plus `KRABBY_OMNI_CANON=vy`:
  - base 1b from scratch, 3000 its;
  - LR overlay, fa, `KRABBY_RSI_FRAC=0`, ell35.
  - Keep fa: M_y images would restore the phantom rows. Keep RSI off, or mirror the −y bank frames.
- **Evals.** Driver; both hands, canonicalized once (above); and `SB/scenarios_side_reversal.yaml` (`rev_direct__omni`, `rev_pause__omni`). Template: `canon_evals.sh`.
- **Pre-registered** (AN:223-226):
  - success = FAST on both hands, with |L−R| ≤ 0.03;
  - SLOW → seed 2 plus a hand-agreement check;
  - an early read ≥ 0.60 is a progress note, not a kill rule.

  Also apply the along-command rule. It comes from the ladder (CH:769-770), not from AN, and is recommended here because a flag error makes a wrong-way walker.
- **Expected** (RC): FAST, with odd RMS ≥ 1.4, under every candidate (P3, untested). If it is SLOW, the reversal and flip states re-form the shuffle: try C_lr_10 + canon next, then re-plan (AN:232-234).
- **Comparator: choose explicitly.**
  - (a) Seed 3 against the historical reference X14 (C_lr s3, SLOW / SLOW, code_sha 07a0037e, eval_sha 3e2898e2).
    - Justify it with RR §6.0.2's knob-off identity checks: byte-identical config dumps, and identity through it 19 of a 20-it same-seed smoke.
    - It is defensible because the predicted effect is a class flip (14 of 14 slow-family networks are SLOW or DEAD).
  - (b) RR:177's rule: fresh seeds plus a knob-off C_lr control at the new code_sha, about 1.2 GPU-h per seed. Historical heads are references, never controls.

  SB is stale and reference-only. If the work moves to a new campaign directory, no SB B\<k\> rerun is needed; RR:175's B\<k\> clause applies to whatever ladder is live then.

**C. FB_canon** (about 1.3 GPU-h; only if B is FAST; RC:501; AN:241-262)
- **Template.** FB_C_lr (candidates.yaml:393-410), pool `stand:0.2,fwd:0.2,side_l:0.2,side_r:0.2,back:0.2`.
  - Add canon and a masked mirror entry in `crab_hex_omni_mirror.py`: M_x always; M_y and R_z(π) only where |vy_b| < h, with h ≥ 0.08.
  - RR's rung 6 needs the same mask (RR:269). See §7.A for landing it.
- **Gates:**
  - side FAST on both hands;
  - fwd and back ratios no worse than the t3 record at 3000 its: fwd 0.558, back 0.524, side 0.478 (`policy/manifest.yaml:113-115`).
    - That record is a different recipe: C1d t3, with a turn slot in the pool and the clock held (`manifest.yaml:95-118`). FB_canon is 1b-based with no turn, so the record is a reference, not a control.
    - AN:253 also gates on forward tripod, but the manifest stores no tripod value for t3. Set a threshold, with its source, before launch, or drop that gate.
  - side survival recorded against W's 0.995.

**D. P1_canon** (about 5.4–5.5 GPU-h per seed; RC:502-503; AN:255-261)
- **Template.** P1_C_lr (candidates.yaml:433-524): t3 3000 → c4 1500 → c5 1500 → c6 2000 → c6x 4000 (`KRABBY_OMNI_CLOCK_HIDE=1`), all fa, plus canon.
- **Comparison.** Compare against W (`policy/5_c6x_clockfree/model_11995.pt`; its eval needs `CLOCK_HIDE=1`). Adopt if it beats W's side ratio of 0.473 without losing the other categories (AN:259).
- **Risks** (RC:505-516):
  - reversals re-mirror the whole history, and paused reversals raise C-type falls (§6);
  - diagonals are untested;
  - move_turn forces a flip after about asin(h/v): 13° at 0.35 m/s, 21° at 0.22 m/s;
  - forward tripod handedness;
  - a clock-free canonical network is untested;
  - the exporter ships the bare actor, and students refuse `--canon` (`eval_crab_hex_gait.py:721-723`);
  - real-robot left-right asymmetry.
- **Fallbacks** (RC:518-524): a policy switch (about 100 lines, eval only), distillation, or a sign-gated odd head (risk: INTERMEDIATE).
- **Budget.** RR estimates about 20 GPU-h for the whole canonical path (§1.1 row 18). That budget would now be yours, subject to the user's approval.

**E. C_lr_bound (not triggered; not in candidates v16)**
- **Where it is defined.** Only in RR:238 / RR:208, `RR/artifacts/decision/REPORT_draft2.md:314` and `final_check.json:25-27`.
- **The A head it uses.** Bound loss 1e-4 → 1e-3, tanh-bounded log-std with init σ 0.8, and entropy 0.0025, all on the C_lr setup. Its action_rate −0.3 is already the default (X14 env.yaml:1774). It needs your second-landing code.
- **It can be read only one way.**
  - The A head bundles three PPO-side changes.
  - A bounded mean also exposes an action_rate noise tax of +0.27 to +0.99 /s (RR §4.3). That is the same size as the fast gait's margin over the two-direction nets, +0.29 to +0.32 /s.
  - So a FAST result supports lock-in. A SLOW result cannot be read against (ii) without an action_rate-0 control (RR arm B).
- **Status.** It is the only designed optimizer-side probe of (ii). ROOT_CAUSE records no prediction for it. Running it is a fresh decision.

**F. T1 checkpoint series** (P5; no code; about 0.5 GPU-h; RC:427-447)
- **What to run.** Evaluate C (X02) on lat_l `side22_25` and C_lr (X14) on both hands, at model_200 / 400 / 600 / 1000. All 8 checkpoints exist.
  - Add `--sample-actions` runs at the training std (about 1.65) beside the deterministic runs.
  - Then, on CPU, run `sym/extract.py`, `sym.py` and `rep/recon.py`.
- **Expected:**
  - Formation barrier, under (i) or (ii): C_lr's odd RMS stays ≤ 0.9 at every checkpoint, while C's rises through 1.4 by it 600–1000.
  - Formed-then-lost: C_lr reaches ≥ 1.4, then decays. In that case a distilled or warm-started network would not survive PPO.
- **What it can and cannot tell you.** T1 separates formation from collapse; it does not split (i) from (ii). It is the only test that reads saturation during formation on-distribution, which makes it the most relevant to your session.

**G. T2 interaction feature** (P4 + P8; about 60–100 lines + 2.5 GPU-h; RC:449-478)
- **Feature.** C_lr plus two channels, tanh(vy_b/0.05)·sin φ and tanh(vy_b/0.05)·cos φ, with mirror sign +1 under all images. The control is an additive ±1 sign(vy) slot.
- **Readings** (decision table RC:470-476):

  | Feature arm | Control | Reading |
  |---|---|---|
  | FAST | SLOW | (i) |
  | SLOW | SLOW, with C_fa FAST | (ii) |
  | FAST | FAST | Input strength or a first-order carrier suffices |
- **Why it matters now.** It is the only designed (i)-vs-(ii) test. It can be interpreted now that C_fa is FAST (RC:468: "read only after C_fa").
- **Requirements.** A CPU invariance test, knob-off identity, and an `OVERLAY_DUMP_CHECKS` entry.
- **Obs-slicing risk.** The actor slices `obs[:75]`, a 10×75 history and the estimator slice `obs[207:216]`, and the live head is 89 dims.
- **Scope.** It is a mechanism test only. W is clock-free, so this is not the omni fix.

**H. Flat-terrain control** (not run; the user's call; CH:757-759)
- **Setting.** `KRABBY_FLAT_TERRAIN_FLAT_FRAC=1.0` (default 0.8, `crab_hex_env_cfg.py:642`). X24's env.yaml has `parkour_flat.proportion: 0.8`.
- **Before trusting a run,** check the dumped proportion: the knob's meaning depends on `KRABBY_FLAT_TERRAIN_MODE` (`run_phases.py:218-224`).
- Use `scripts/eplen_telemetry.py`.
- It matters only if episode length is revisited. Estimate about 1.5 GPU-h per arm (CH:746).

**I. Other designed items:**
- **AN 5a.** Fork C_fa (X21 model_2999) into the C_lr config for 1500 its and read at +500 / +1500. It separates lock-in from 1b and 1c.
  - Use C_fa, not C or C_mo: their slot 8–9 weights never got a gradient.
  - The resume counter-reset hazard applies.
- **P7 / AN 5c.** A no-mirror +y arm. It needs a `none` mirror mode (about 10 lines) and separates (iii) from H6.
- **P9.** Does a distilled two-hand student stay FAST under about 1000 its of two-direction PPO?
- **Seed lottery.** O_s2 and LR4_s2.
- **Ladder E = 40/50/60.** Defined in v16 at :782-850; predicted SLOW or DEAD; stopped by the user.
- **Housekeeping.** Acknowledge the §1 notices, and fix the stale headers in state.json.

**J. Runners and hazards:**
- **`queue_eplen2.sh`** carries the review fixes F1–F5:
  - it finds the run dir by run name and takes the head from state.json;
  - it detects unit death early;
  - its idle check is the anchored `ps -eo args | grep -qE "^[^ ]*isaac_venv/bin/python …"` (:27);
  - two-hand evals run with `OMNI_KIT_ACCEPT_EULA=yes TERM=xterm PYTHONPATH=…`, `systemd-run --scope MemoryMax=48G` and `timeout 3600` (:90-104);
  - it passes `--accept-candidates-change` on the first launch (:61).

  See the edits it needs before reuse under "Common runner".
- **Other templates.**
  - `queue_controls.sh` for one-hand arms.
  - `canon_evals.sh` for `--canon` evals. Its `drivers_idle` (:14) still uses the unanchored `pgrep -f` pattern, so port the F3 anchored check (queue_eplen2.sh:27) before reusing it.
- **One Isaac process at a time** (`launch_candidates.sh:27-36`).
- **Never SIGKILL Kit:** it corrupts the DerivedDataCache. Stop a run with `systemctl --user stop <unit>` (KillMode=mixed, TimeoutStopSec=360; :14-18), and kill only the bash runners.
- **Resume hazard.** In SB, three launches would *resume* a stopped run through reference_loop:

  | Launch | Resumes |
  |---|---|
  | `--reference-chain LR4_dwell_s2` | X13 |
  | `--reference-chain C_lr_10_rsi_s2` | X18 |
  | `--reference-chain C_lr_30_rsi` | X25 |

  - **Why.** Each works with or without its pinned `--seed`, because `_reference_inflight` matches the header's candidate name and seed (RCD:11951-11961).
  - **Cost.** A resume restarts the PPO counters (CH:419).
  - **Near-misses.** `--reference-chain LR4_dwell --seed 2` and `C_lr_10_rsi --seed 2` exit instead, because those references pin seed 3 (`reference_seed`, RCD:11908-11916).
  - **What to do.** Use a new campaign directory or fresh chain names. SB is reference-only (`candidates: []` at candidates.yaml:918).
- **Watchers.** Tail from the end of the log, and never `pgrep -f` a string that is in the watcher's own command line.
- **Repository.** Never push. The checkout is shared with other sessions, so do not revert changes you did not make.

---

## 8. Where everything is

| What | Path |
|---|---|
| Campaign records | SB/CHARTER.md (dated log), GATE.md, ANALYSIS.md, ROOT_CAUSE.md, CHECKS.md, REPORT.md, CHANGELOG.md |
| Live candidates | SB/candidates.yaml = SB/scripts/candidates_v16.yaml (modified, uncommitted) |
| Orchestrator state | SB/state.json, heads.json, ledger.json (untracked); orchestrator.log, notify.log (gitignored, `*.log`) |
| Gate and analysis scripts | SB/scripts/side22_classify.py, along_cmd.py, early_read.py, eplen_telemetry.py, harvest_canon_bank.py, rootcause/{sym,rep}/ |
| Runners | SB/scripts/queue_eplen2.sh, queue_eplen.sh, queue_controls.sh, canon_evals.sh |
| Scenarios | SB/scenarios_side_lr.yaml, SB/scenarios_side_reversal.yaml |
| RSI banks | SB/rsi_bank_side_lr.npz (A), SB/rsi_bank_side_lr_b.npz (B) |
| Gate results | parkour/outputs/omni_plan/side_bisect/arms/NAME.json (driver), NAME_lr.json (both hands), NAME_along.json |
| Canonical deploy | SB/gate/canon_lr.json; parkour/outputs/omni_plan/side_bisect/canon/*_cmd_rev_*.log |
| Runner log | parkour/outputs/omni_plan/side_bisect/queue.log |
| Training runs | parkour/logs/rsl_rl/crab_hex_omni_flat_walk/2026-09-{29,30}_*_seed{1,2,3}_X{01..25}_s1_A15pB/ (tfevents, params/env.yaml, params/agent.yaml) |
| Earlier diagnosis | ../2026-09-28_side_speed_investigation/REPORT.md |
| Fix building blocks | eval_crab_hex_gait.py:524-535, :567-577 (read only; do not reuse its policy/estimator calls), :721-723; mdp/crab_hex_omni_mirror.py:314; parkour/scripts/rsl_rl/vecenv_wrapper.py |
| Knob plumbing | crab_hex_omni_task/config/crab_hex/omni_knobs.py; experiments/tools/run_phases.py:169 (EVAL_CARRY); experiments/tools/run_candidates.py:1860 (OVERLAY_DUMP_CHECKS) |
| Launcher pathspecs | crab_hex_omni_task/experiments/tools/run_candidates.py:163-190 |
| Head of record (comparator) | crab_hex_omni_task/policy/5_c6x_clockfree/model_11995.pt (W; needs KRABBY_OMNI_CLOCK_HIDE=1) |
| Your documents | ../2026-09-30_rl_algorithm_review/REPORT.md, DOSSIER.md, artifacts/omni_stroke_side/tb_arms.json |

`/dev/shm/sb_tb.pkl` is a tfevents cache in RAM, written while this handoff was prepared. It is transient, it is not a record, and it is safe to delete.

---

*Compiled by a read-only workflow (4 extractors, 1 writer, 3 verifiers: numbers, claims, receiver; 1 reviser; run `wf_bc156787-343`). The verifiers raised 58 issues (13 major, 45 minor). The reviser re-checked each against its source; dispositions are in `handoff/revision_log.md`, and the raw issues in `handoff/verifier_issues.json`.*
