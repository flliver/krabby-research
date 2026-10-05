# PPO construction on the side beds: charter (pre-registered)

**Opened:** 2026-10-01. **Plan (approved and audited, 4 passes):**
`/home/nickmagus/.claude/plans/foamy-wondering-ullman.md`.
**Inputs:**
- the RL review `../2026-09-30_rl_algorithm_review/REPORT.md` (steps 2–4) and its `DOSSIER.md`;
- the side-bisect handoff `../2026-09-28_side_bisect/HANDOFF_PPO.md`.

Everything below was written **before any run of this campaign**. Rules are applied as written. Any
change is logged in §9 with its reason and time, before the result it would affect is read.

## 1. Question

Do PPO-construction changes let a network trained on both lateral directions form the
direction-flipped rod stroke (FAST side walking), or at least improve executed-action quality,
without breaking the one-direction gait? The changes are:
- a bounded mean;
- a gradient-passing σ;
- lower entropy;
- then step size and batch size.

The **user's decisions** shaping the design:
- no canonical training;
- beds C, C_lr and C_lr_rsi;
- C_lr_rsi cam staging is decided by the rung-0b diagnostic;
- P1 gets a C0 lineage control;
- the per-hand gradient-cosine and `_rsi` diagnostics are kept;
- `KRABBY_PPO_MINIBATCHES` is set aside;
- after a C-bed guard failure, A and B continue unchanged, and 4c needs the guard to pass or the
  user's approval.

## 2. Beds and arms

Bed overlays are copied verbatim from side-bisect `candidates_v16.yaml`. All runs are from scratch:
3,000 iterations, 256 envs, base `1b`.

| Bed | Overlay source | Historical reference (not a control) |
|---|---|---|
| C | SB chain C (fa mirror, scalar cmd obs, RSI 0, ellipse 0.35) | X02 s3 FAST .215/.214, X07 s2 FAST |
| C_lr | SB chain C_lr (both side directions, direction visible, fa, RSI 0, ellipse 0.35) | X14 s3 SLOW/SLOW |
| C_lr_rsi | SB chain C_lr_rsi (C_lr + RSI 0.2 bank A, source all); cam staging per §5 | X16 s3 SLOW/SLOW |

| Arm | Knobs |
|---|---|
| C0 | none (incumbent). Two-direction beds carry `KRABBY_PPO_HAND_DIAG=1` (diagnostic only, verified bit-identical). |
| A | `KRABBY_ACTION_BOUND_LOSS=0.001`, `KRABBY_ACTION_BOUND_RAMP=0.0001:0.001:500`, `KRABBY_ACTION_BOUND=1.1`, `KRABBY_STD_PARAM=soft`, `KRABBY_INIT_STD=0.8`, `KRABBY_ENTROPY_COEF=0.0025`; action_rate −0.3 (default); mirror loss 0.5 on raw μ (unchanged) |
| B | A + `KRABBY_ACTION_RATE_W=0` |

Screens:
- C_lr and C_lr_rsi: C0, A and B at seeds 11–13.
- C: C0 and A at seeds 11–13.

Seeds 14–15 are extensions and 16–20 are confirms. s2 and s3 are never used: W and C1d were selected
on s3. Controls are fresh at the campaign's code_sha; historical heads are references only. Arms are
not seed-paired.

## 3. Run order

1. **Pilot:** C__C0, C__A, C_lr__C0 and C_lr__A at seed 11. These are read before anything else
   launches.
   - **Pilot pass:** both A runs pass the iteration-1000 dose read (§4.3), and C__A is FAST if C__C0
     is FAST.
   - **C__A not FAST while C__C0 is FAST:** run C__A and C__C0 at seed 12.
   - **If A fails again and C0 is FAST at both seeds:** take the C-bed guard-fail branch (§4.7) before
     any further two-direction A/B run.
   - **If C0 is not FAST at both seeds:** record "C bed not reliable (provisional)" and continue.
   - **A dose failure:** handled by §4.3.
2. **Blocks:** seed-block-major for the rest (seed 11's remaining configs, then 12, then 13),
   randomised within each block. The order is drawn once and logged in §9 before block 1 starts.

## 4. Pre-registered reads and decisions

### 4.1 Classes

A hand's class comes from `side22_classify.py` on feet_ss:
- FAST ≥ 0.155 / 0.175 m/s;
- SLOW ≤ 0.13 / 0.145;
- DEAD < 0.05 at either speed;
- otherwise INTERMEDIATE.

A hand counts FAST only if `along_cmd.py` also classes it FAST. This fails closed: a hand with no
feet class, or a feet-FAST hand with no along-command class, reads NA (never FAST). Its run is
excluded from every count until the missing read is rerun.

A run is FAST when:
- on two-direction beds, both hands are FAST in the both-hands eval;
- on the C bed, lat_l is FAST in the driver eval.

### 4.2 Kills (A/B arms)

- **Wiring, iteration 200:**
  - consumed-knob lines are present in the train log;
  - `PPO/effective_std` is present;
  - `Loss/action_bound` is present for bound arms;
  - for init-0.8 arms only, `PPO/effective_std` < 1.0.

  If any check fails, kill the run, fix the wiring and rerun; the rerun counts.
- **DEAD, iteration 1000** (window 950–1050): `track_ratio_mean` < 0.40, or `PPO/effective_std` < 0.1.
- **Mechanics** (`scripts/run_arm.sh` + `scripts/verdict.py --check-run`):
  - Each kill has its own exit code: 3 wiring, 4 dose, 5 DEAD, 6 4a divergence (§6).
  - A wiring or dose kill is a true stop. The runner writes `STOP_QUEUE`, so nothing else launches,
    and logs a NOTIFY line.
  - DEAD and divergence kills let the queue continue.
  - Every killed run still gets a record.
  - A DEAD- or divergence-killed run counts as not FAST.
  - A wiring-killed run is excluded, as is a run whose post-hoc wiring read fails because the live
    kill was skipped. Its rerun counts: a fresh reference of the same chain and seed, named in §9.
  - A dose-killed run is excluded and goes to the §4.3 stop.
  - A second valid run of the same chain and seed is a re-roll and is excluded; the first one counts.
  - A read that lacks the telemetry to decide is retried at the next poll. It is latched only once it
    decides. Always-on tags still missing 200 iterations past the window count as a wiring failure.

### 4.3 Dose read (A/B)

- **Read from:** training telemetry at iteration 1000 (window 950–1050) and at the final window
  (2900–2999).
- **Thresholds (amended 2026-10-01 12:30, user decision, see §9):** `PPO/mu_med_{cam,rod}` ≤ 1.5. `PPO/sat15_{cam,rod}` (share |μ| > 1.5) is reported but no longer gates. The original rule also required `PPO/sat15_{cam,rod}` ≤ 0.1, a threshold set before any data.
- **C0 reference:** the same tags in fresh C0.
- **The |μ| > 1 share is an outcome, not the dose.**
- **Ramp timing:** the ramp is keyed on the algorithm's update counter, which `update_dagger` (every
  20 iterations) also advances, so a resumed stage continues the ramp instead of restarting it. The
  500-update ramp therefore ends near iteration 476, not 500. This is immaterial to the dose read at
  iteration 1000.

Failure handling:
- **At iteration 1000:** kill the run and check the wiring.
  - **Wiring wrong:** fix and rerun.
  - **Wiring right:** record "bound did not bind at this coefficient", then stop and ask the user.
    This is a true stop.
- **At the final window only:** record "dose lost late"; the run's class still counts and the flag is
  reported.

### 4.4 Survival tripwire (every A/B run, FAST runs included)

- **Statistic:** hold-fall share = short / (n + short + static), pooled over every lat_* hold of the
  classifying stamp.
- **Reference:** the C0 median on the bed, excluding C0 runs above 0.25. On two-direction beds it is
  the larger of the bed's C0 median and the C-bed C0 median.
- **Trips above:** max(0.15, reference + 0.10).
- **Effect:** a tripped run counts as not FAST.
- **Calibration:** non-collapse 0.000–0.118; FAST driver stamps 0.025–0.085; collapse 0.38–0.46.

### 4.5 Two-direction beds (C_lr, C_lr_rsi; per bed and arm)

**Precedence: VOID > UNLOCK > PARTIAL > NO-FLIP.**

- **VOID:**
  - C0 ≥ 2/3 FAST at three seeds (immediately), or C0 ≥ 2/5 FAST after extension.
  - Recorded "bed not reliably SLOW at fresh seeds".
  - Extend C0 only; A/B seeds 14–15 are the user's call (FYI).
  - A/B counts are descriptive; no UNLOCK, NO-FLIP or 4c.
- **Extension:** any A/B run FAST, or a single C0 run FAST, extends that arm (all arms if C0) and C0
  to seeds 14–15. A single C0 FAST run is a note, not VOID.
- **UNLOCK:** A/B 4/5 FAST with C0 0/5, or A/B 5/5 with C0 ≤ 1/5 (Fisher two-sided p = 0.048 at both
  corners). 4/5 vs 1/5 (p = 0.21) is PARTIAL.
- **PARTIAL** ("inconclusive", never "not the limiter"; no 4c; FYI to the user) if any of:
  - 2–4/5 FAST without meeting UNLOCK;
  - any A/B hand INTERMEDIATE, or a FAST hand in a run not counted FAST;
  - a non-FAST A/B run whose deterministic both-hands feet_ss (hands averaged) rises from model_2500
    to model_2999 by more than both 0.03 m/s and the largest C0 rise on that bed. C0 model_2500 is
    always evaluated.
- **NO-FLIP:** none of the above, and no A/B run FAST at 3 seeds (or ≤ 1/5 at 5). Never issued
  while an extension is pending (e.g. A/B 0/3 with one C0 FAST run); that reads PENDING. Recorded as "no
  reliable flip at this dose", with any FAST runs named and the §4.8 LR numbers printed.
- **Descriptive only:** feet_ss and odd-RMS means with 90 % t-CIs (3 v 3 MDE ≈ 2.75 seed-SD).

### 4.6 Motion quality (every bed)

- **Q-block:** `action_jitter` and `action_fft_freq` on executed actions, hands averaged; the
  both-hands stamp on two-direction beds.
- **"Q better":**
  - all 3 A (or B) seeds below all 3 C0 seeds on jitter (one-sided permutation p = 0.05);
  - and the mean `action_fft_freq` of the arm ≤ C0's.
- **Otherwise descriptive.** Q does not separate FAST from SLOW in existing stamps.

### 4.7 C-bed guard

- **Passes:** A FAST ≥ 2/3.
- **Fails:** A FAST ≤ 1/3 and C0 ≥ 2/3.
  - Run B on C (3 seeds).
  - If B also fails, run the bisection at 2 seeds each, in order, stopping at the first that recovers
    C0's class: C__A_ent01 (entropy 0.01), C__A_std15 (init σ 1.5), C__bound_only.
  - After a guard failure (user decision), A and B continue unchanged on the two-direction beds. Guard
    and bisection results are reported beside any two-direction result, and 4c needs the guard to
    pass or the user's approval.
- **Undefined:**
  - C0 ≤ 1/3: record "C bed not reliable at fresh seeds"; the guard does not decide.
  - A between the bands (2/5 or 3/5) with C0 ≥ 2/3: neither PASS nor FAIL. No B run and no
    bisection; reported to the user.
- **Bisection arms' wiring read:** `PPO/effective_std` present only; no < 1.0 check.

### 4.8 LR-floor confound

Either condition triggers 4a regardless of outcome:
- A/B end-of-iteration `Loss/learning_rate` ≤ 1.0001e-5 in > 90 % of its 500–3000;
- A/B mean `PPO/lr_applied` over its 500–3000 below ½ the bed's C0 median.

### 4.9 Mechanism reads (descriptive)

- `PPO/hand_grad_cos` over its 0–600 (C0 vs A);
- the |μ| > 1 + 2σ lock share over its 300–600;
- `track_ratio_mean_rsi` on C_lr_rsi;
- `Loss/symmetry` per arm (the raw-μ mirror MSE weakens under a bound);
- rod and joint-space odd RMS.

### 4.10 Outcomes

- UNLOCK on any two-direction bed → 4c, if the guard passed (otherwise the user's approval).
- Guard holds and "Q better" with no tripwire → "adopt for motion quality" is a proposal to the user.
  It is not baked and not a 4a trigger.
- PARTIAL → inconclusive.
- NO-FLIP → recorded as above.

The step-3 verdicts get an independent fable·high check before step 4 launches.

## 5. Rung 0b: C_lr_rsi cam staging (decides §2's C_lr_rsi variant)

**Rows:** 8 direct evals of X02, each in its own process, from the landed tree:
- {RSI FRAC 0, FRAC 1.0 source 1} × {staging off, on} × 2 eval seeds;
- a lat_l 0.22 hold from reset, σ 0, ≥ 400 episodes per row.

**Before deciding:**
- Staging-on RSI envs must log a step-0 applied cam target equal to clamp(jv, ±π) rad/s and history
  cam columns equal to clip(jv/π).
- Staging-off rows must log zeros.
- **The check is per env.** With `KRABBY_RSI_CAM_STAGE_DEBUG=1`, every reset with RSI envs prints
  `RSI_CAM_STAGE_DEBUG n_rsi= stage= target_err= hist_err= target_absmax= hist_absmax= bank_absmax=`.
  The errors are maxima over envs, slots and cams of |applied − clamp(jv)| and |history −
  clip(jv/π)|.
- **`scripts/rung0b_read.py`** reads only these lines:
  - on rows: stage=on, n_rsi > 0, both errors ≤ 1e-4;
  - off rows: stage=off, n_rsi > 0, both absmax = 0 and bank_absmax > 0;
  - FRAC-0 rows: no line.
- If the staging-on and staging-off RSI rows are identical, no decision: fix and rerun. FRAC-0 rows
  are identical by construction.

**Rule** (both conditions in both seeds):
- **(a) a deficit exists:** RSI-staging-off fall share in [0, 2) s minus normal-spawn share exceeds
  both 0.04 and the seed-pair spread. The seed-pair spread is the larger, over the conditions, of the
  absolute difference between the two seeds.
- **(b) staging removes at least half of it:** (staging-on − normal) ≤ ½ (staging-off − normal).

If both hold, C_lr_rsi runs as chains `C_lr_rsi_st__*` (staging on in C0, A and B). Otherwise it runs
as `C_lr_rsi__*`, identical to X16's definition.

## 6. Step 4 (conditional)

- **4a step size.**
  - **Trigger:** an UNLOCK, or the §4.8 confound.
  - **Head:** the unlocking arm on its bed. On a floor-only trigger, A on the bed that triggered it (C_lr
    if several did; B if only B triggered).
  - **Arms:** from scratch, seeds 11–13, history-encoder LR pinned at 3e-4:
    - `C_lr__A_fixed`: LR 1e-4, per-epoch KL stop at 0.02, hard stop per minibatch at 0.04;
    - `C_lr__A_epochkl`: per-epoch adaptive, desired KL 0.003, a recalibrated step-KL target.
  - **Control:** the step-3 run of the head at the same seeds. `verdict.py` compares each step-4
    arm with it per seed. Only the head's own chain is declared now: `C_lr__A_fixed`/`_epochkl`.
    A B-head or C_lr_rsi-head 4a chain is added in a new `candidates_v<k>` once the trigger names it.
  - **Mechanism read:** executed change per update ≥ 1.5× control over its 1000–3000.
  - **Wiring at iteration 200** (on top of §4.2), on every logged point through iteration 210:
    - `PPO/lr_applied` = `KRABBY_PPO_LR` (`fixed` only);
    - `PPO/kl_stop` present when the KL stop is set;
    - `PPO/hist_lr`, the history-encoder optimizer's own LR, = 3e-4.
  - **Kill** (polled throughout; exit 6; the queue continues):
    - `PPO/kl_mean` > 0.1 on 50 consecutive iterations;
    - or the 50-iteration mean of `Train/mean_reward` ending at t falls more than 30 % of its
      magnitude below the same mean ending at t − 500.
  - **Outcome:** descriptive. A two-direction FAST where its control run was not FAST triggers the
    extension and UNLOCK rule.
- **4b batch.**
  - **Trigger:** rung 1, i.e. 1,024 envs ≤ 1.15× the 256-env s/iter and ≤ 10 GB.
  - **Arm:** on C_lr, A if any bed UNLOCKed, else C0, at 1,024 envs × 24 steps × 16 minibatches.
  - **Control:** the step-3 256-env run.
  - **Readout:** the §4.5 class rule; odd RMS descriptive.
  - **UNLOCK listing:** a 4a/4b UNLOCK (five seeds, after its extension) is listed apart from the
    step-3 4c triggers. It enters 4c only under step 12's carry rule. INC0 is never an UNLOCK.
- **4c.**
  - **Trigger:** UNLOCK, and the guard passed or the user approves.
  - **Carry rule:** the unlocking run's non-construction settings carry into both the winner and the
    C0 lineage:
    - RSI/staging for C_lr_rsi, env count for 4b;
    - C_lr's settings if both beds unlock;
    - A if A and B both unlock.
  - **4b UNLOCK on the C0 head:** the factor is env count (1,024 vs 256 lineages), and INC0 is skipped.
  - **(1) INC0, run first** if B unlocked, at seeds 11–15:
    - INC0 FAST ≥ B − 1 → stop and ask;
    - otherwise label FB/P1 "the B package vs C0" unless INC0 ≤ 1/5.
  - **(2) FB**, winner vs C0 at seeds 16–18: gate side FAST on both hands in ≥ 2/3 seeds; P1 only if
    it passes.
  - **(3) P1**, winner vs a C0 P1 lineage at seeds 16–18: side (both hands), fwd, back, diag, turn,
    move_turn. W is a reference only.
  - **Stop and ask** before any adoption or bake.

## 7. Launching and hazards

**Launching:**
- Use `scripts/run_arm.sh <reference> …`, which wraps `launch_candidates.sh --campaign-dir <this dir>
  --reference-chain <ref>`. Never the plain launcher.
- One Isaac process at a time. Stop with `systemctl --user stop <unit>`, never SIGKILL.
- Watchers tail `parkour/outputs/omni_plan/ppo_construction/queue.log` and never `pgrep -f` their own
  string.

**Recording:**
- Decisions are written here (§9) and push-notified as FYI.
- Nothing is pushed.

## 8. Landed code

- **Commit** `60532fb9` on `feat/crab-hex-omni-task`, landed 2026-10-01 by fast-forward from worktree
  branch `claude/ppo-construction`. Local only; nothing pushed.
- **code_sha** `f7166d558c5d…` (`run_candidates.code_sha()`).
- **eval_sha** `1d021625844b…` (`run_candidates.eval_sha()`).
- **Dirty-tree check:** clean.
- **Driver dry run:** a scratch copy of `candidates.yaml` (v2) validates.
- **Every run of this campaign uses this code.** A later code change would invalidate the fresh-control
  rule (§2) and is logged here before use.

## 10. x6k: B to 6000 iterations on C_lr (pre-registered 2026-10-03)

(Placed before the log so §9 stays last. The user approved it on 2026-10-03: "Let's run the 6000 iteration
experiment on B", which is RESULTS.md next-experiment 1. Step 3 is closed, and nothing here changes its verdicts:
the forks are not step-3 runs, and `verdict.py` reads only `$OUT/arms`.)

### 10.1 Question

Step 3 left two-direction B one hand short. On C_lr, B's second hand was still rising at iteration 3000 (SLOW →
INT between model_2500 and 2999 in 4 of the 4 one-FAST C_lr B runs). With 3000 more iterations, does that hand:

- **complete:** late formation;
- **plateau:** the stroke is present but does not convert into speed;
- **or trade off against the first hand:** interference?

And does C0's lean-and-shuffle stay frozen at 6000?

### 10.2 Arms

Each arm forks a step-3 head's `model_2999` for 3000 more iterations (`resume_from: {reference}`). It keeps the
forked chain's overlay verbatim and the head's own seed, at code_sha f7166d558c5d (`scripts/candidates_v4.yaml`).

| Fork | Head at model_2999 | Role |
|---|---|---|
| `C_lr__B_x6k_s11`, `_s13`, `_s14` | FAST / INT | one-FAST heads: the question |
| `C_lr__B_x6k_s15` | INT / FAST | one-FAST head: the question |
| `C_lr__B_x6k_s12` | FAST / FAST | stability, and a control on the resume itself |
| `C_lr__C0_x6k_s11` | SLOW / SLOW | C0 control: is the freeze permanent? |

- **Order,** drawn once with `random.Random(20261003).shuffle`:
  1. C_lr__B_x6k_s13
  2. C_lr__B_x6k_s14
  3. C_lr__B_x6k_s12
  4. C_lr__C0_x6k_s11
  5. C_lr__B_x6k_s15
  6. C_lr__B_x6k_s11
- **Numbering:** a fork re-runs index 2999 and ends at `model_5998`, its head (SMOKES.md fork smoke).

### 10.3 Wiring read (kills)

`scripts/resume_read.py` runs once the fork has logged at least 5 iterations. It checks that the fork continues
its head on four points; any failure means it restarted:

- iteration numbering continues (the first step is 2999 or 3000);
- `PPO/effective_std` is within 0.02 of the head's last value (B is about 0.748 against an init of 0.8; C0 is 2.0);
- `Loss/priv_reg_coef` is within 5 % (a restart reads 0);
- `Loss/action_bound_coef` is within 2 % (a restart reads 1e-4 against the head's 1e-3).

A FAIL (rc 3 only) is a wiring kill and stops the queue: fix the cause, then rerun under a fresh name. If it still
fails with correct wiring, stop and ask.

- **A rerun is a new chain,** for example `C_lr__B_x6k_s13_r2` with the same `resume_from`, added in a
  `candidates_v5`. Relaunching the killed name would resume the killed fork. The runner reads each fork's source
  from its chain's `resume_from.reference`.
- **A reader crash** (any other non-zero exit) is logged and retried, never a kill.

The from-scratch reads (iteration 200 wiring, 1000 DEAD / dose, live divergence) are off. Their windows do not
exist in a fork, and their thresholds are calibrated for formation from scratch.

### 10.4 Reads

- **Evals:**
  - the both-hands eval (`side22_25lr__omni`) of `model_3100`, `model_4000`, `model_5000` and the head
    (`model_3100` is the resume-artifact read, 10.5);
  - the driver `side22_25` eval of the head;
  - classes by §4.1 (feet AND along-command);
  - the survival tripwire by §4.4 (fall share above 0.15 counts as not FAST);
  - dose, reported only (median |μ| ≤ 1.5 over the fork's own last 100 logged iterations, 5899–5998);
  - rod and joint odd RMS.
- **Definitions:**
  - The **second hand** of a one-FAST head is the hand that was not FAST in its step-3 record.
  - A fork **completes** if, at its head:
    - the second hand is FAST;
    - the second hand's feet_ss (mean of the 0.22 and 0.25 holds) rose by ≥ 0.03 from model_2999 (the same
      margin as "drops": s15's second hand is 0.0014 m/s from the FAST line, and a flip without a material rise
      is not a completion);
    - the first hand is still FAST;
    - it is not tripped.
  - A hand **drops** if it was FAST at model_2999, is below FAST at the head, and its feet_ss (mean of the 0.22
    and 0.25 holds) fell by at least 0.03.
- **Reader:** `scripts/x6k_read.py` computes every 10.4 / 10.5 statistic from the `arms_x6k` records, the step-3
  records (the model_2999 values) and each fork's own tfevents. It is written while fork 1 trains, with a
  self-test that covers the s15 boundary, and it is committed before the first head is read. verdict.py is not
  used on forks: its windows are absolute iterations from scratch.

### 10.5 Pre-registered reading

The reading covers the four one-FAST B forks; drops count over all five B forks. Precedence: RESUME ARTIFACT >
INTERFERENCE > LATE FORMATION > PLATEAU > MIXED. The first rule that matches decides.

- **RESUME ARTIFACT:** a fork restarts the environment (episodes at the spawn pose, an empty rollout buffer); only
  the algorithm state carries. So a hand that already drops at `model_3100` (below FAST and −0.03 against
  model_2999) is read as a resume artifact. The response is stop and ask. Such a drop never counts toward
  INTERFERENCE, in s12 or anywhere else.

- **INTERFERENCE:** a hand drops in ≥ 2 of the 5 B forks.
- **LATE FORMATION:** ≥ 2 of the 4 one-FAST forks complete.
- **PLATEAU:** none of the 4 completes, and in ≥ 3 of 4 the second hand's feet_ss rises by < 0.03 from
  model_2999 to the head.
- **MIXED:** anything else, reported with the numbers.

Why these thresholds:
- 0.03 is the materiality convention of §4.5's late-rise clause. It is not the eval's resolution: the measured
  same-checkpoint spread between manifests is ≤ 0.005.
- ≥ 2 of 4 is the smallest count that is not a single seed.
- At n = 4 this reading is descriptive; no p-value is claimed.

The other heads:
- **s12 (FAST/FAST):** STABLE if it is still FAST/FAST at the head. Otherwise it is reported, and a drop counts
  toward INTERFERENCE.
- **C0 control:** FROZEN if it is SLOW/SLOW at the head and each hand's feet_ss rose by < 0.03. Otherwise the
  reading is "C0 moved under the 0.04 → 0.1 priv_reg ramp". That ramp runs during the extension (update counter
  3149 → 5000 by about iteration 4760). The step-3 freeze read stands as stated, because it was made at
  coefficient 0. `Loss/priv_reg_coef` at the head is reported in the C0 row.

**Two-direction candidates:** any B fork that is FAST/FAST at its head, not tripped, with dose median ≤ 1.5. These
are listed for the user. There is no bake, and no FB/P1 run without the user.

**Intermediate evals** (model_4000 and model_5000) are descriptive: they time the formation.

### 10.6 Running

- **Command:** `scripts/run_arm.sh` in the drawn order, with these settings:
  - `ARMS_DIR=$OUT/arms_x6k` (fork records stay out of the step-3 arms dir);
  - `RESUME_READ=1`;
  - `EXTRA_LR_EVALS="3100 4000 5000"`.
- **Gate:** fork 1 (`C_lr__B_x6k_s13`) runs alone first. Its resume read and its `model_3100` eval (the RESUME
  ARTIFACT rule) are read before forks 2–6 launch. One eval of about 3 minutes gates about 7 GPU-h.
- **Process rules:** one Isaac process at a time, and no rendering while the queue can launch an arm (§9,
  2026-10-02 08:30).
- **At the end:** records are copied to `x6k/gate/`, the reading goes into `x6k/RESULTS.md`, and an independent
  check follows.
- **Budget:** about 6 × 1.45 ≈ 8.7 GPU-h (1.15–1.25 h of training per fork, plus five evals).

**Pre-launch review (2026-10-03, before any fork ran):** a fable·high audit of this section and an opus·high
code-path review (workflow `wf_01145c14-592`).
- **Applied from the audit:** the resume-artifact rule and the model_3100 gate; the ≥ 0.03 rise in "completes";
  the fork source taken from `resume_from`, plus the rerun sentence; the reader and its windows; the C0 wording.
- **Applied from the code review:** only rc 3 kills; the runner follows an infra `_r<k>` retry; the records-dir
  check.
- **Confirmed by the code review:** every head resolves at code_sha f7166d558c5d; the resume restores Adam,
  alg_state and the adaptive LR; model_3100, model_4000 and model_5000 exist (save_interval 100); the head is
  model_5998.

## 11. FB6k + INC0: confirm B on the FB pool at fresh seeds (pre-registered 2026-10-03)

(Placed before the log. The user approved x6k next-option 1 ("Run 1") on 2026-10-03; the design is
`x6k/next_options.md`, the confirm option of opus·high. Nothing here changes step 3's or x6k's readings.)

### 11.1 Questions

1. **FB6k.** x6k showed B finishing its second side hand by 6000 iterations on C_lr. Does B form both side hands
   at fresh seeds, from scratch, when fwd/back share the pool (each side hand's weight halves from 0.4 to 0.2), and
   do fwd/back hold against a matched C0 lineage?
2. **INC0.** Does `action_rate 0` alone, on the incumbent construction, flip C_lr at 6000? B is the construction
   plus action_rate 0, and INC0 attributes the flip.

### 11.2 Arms (`scripts/candidates_v5.yaml`; overlays are the named chains verbatim)

| Chain | Overlay | Seeds | Iterations | Driver eval |
|---|---|---|---|---|
| `FB6k__B` | `FB_C_lr__B` (+ the RNG-free hand diagnostic) | 16, 17, 18 | 6000 from scratch | `suite` |
| `FB6k__C0` | `FB_C_lr__C0` (+ hand diagnostic): the C0 lineage | 16, 17, 18 | 6000 from scratch | `suite` |
| `C_lr6k__INC0` | `C_lr__INC0` (C_lr C0 + `KRABBY_ACTION_RATE_W=0`) | 11, 12, 13 | 6000 from scratch | `side22_25` |
| `FB6k__{B,C0}_x9k_s{16,17,18}` | the forked chain's, verbatim | the head's | +3000 (fork of model_5999) | `suite` |

- **Pool:** `FB_C_lr` is `stand:0.2,fwd:0.2,side_l:0.2,side_r:0.2,back:0.2`, slots 0.2:0.8:0:0, fa mirror,
  RSI 0, ellipse 0.35, base 1b, clock visible.
- **The x9k forks** are pre-declared and run only when 11.4's extension rule fires.
- **Order:**
  1. The INC0 block: s11, s12, s13.
  2. The INC0 gate (11.5).
  3. The FB6k block in the drawn order (random.Random(20261004), seed-block-major): C0_s16, B_s16, C0_s17,
     B_s17, B_s18, C0_s18.
  4. The triggered forks.

### 11.3 Reads (`scripts/fb6k_read.py`, written and self-tested before launch)

- **Classes:** §4.1 (feet AND along-command; FAST 0.155 / 0.175) on the both-hands eval `side22_25lr__omni` of
  `model_3000`, `model_4500` and the head (`model_5999`). For forks, of `model_6100`, `model_7500` and the head
  (`model_8998`).
- **Suite:** the head's driver stamp gives the `fwd`, `back`, `side` and `diag` ratios
  (`aggregate.suite.by_category.<cat>.ratio.median`) and survival. **diag is descriptive**: it is not trained
  here, and it is a free diagonal read.
- **B dose:** median |μ| ≤ 1.5 on cams and rods over the run's own last 100 logged iterations.
- **Live kills (§4.2):** the from-scratch reads apply as in step 3, for construction arms only (INC0 and C0 are
  not construction arms). These are the iteration-200 wiring, the iteration-1000 dose and live divergence.
  - **The DEAD-at-1000 kill is advisory on FB6k** (`ADVISE_DEAD=1`: logged, never stops a run). Its threshold
    (track 0.40) was calibrated on C_lr, where B's margin was only +0.05, and there is no FB-pool calibration. A
    false kill would remove a seed from the count.
  - **The x9k forks** get the resume read (§10.3) instead.
- **Descriptive:** the Q block (jitter, fft) and rod odd RMS.

### 11.4 FB6k: extension and gate

- **Extension** (revised by the pre-launch review, before any run):
  - **B:** every run that is not FAST/FAST at model_5999 and has no DEAD hand is forked +3000 (x9k). Its final
    read is the fork head. x6k shows flat stretches before the second hand crosses (s15: +0.010 over 2999 → 4000,
    then FAST), so a rise trigger would miss them. The 4500 → 5999 rise is reported, not a trigger.
  - **C0:** the C0 run of the SAME seed is forked whenever B's is (equal budget per seed). It is also forked if a
    C0 hand rose ≥ 0.03 from model_4500 and it is not FAST/FAST.
  - **Resume artifact:** a hand that was FAST at 5999 but is below FAST and −0.03 at model_6100 is a RESUME
    ARTIFACT. Stop and ask.
  - **Rationale:** each side hand's weight halves against C_lr, where formation took ≤ 4000 iterations. So 6000
    is the scaled budget, and the fork covers seeds whose second hand is still rising.
- **Tripwire (§4.4):** threshold max(0.15, ref + 0.10), where ref is the median fall share of the FB6k C0 final
  reads (runs above 0.25 excluded). It applies to B.
- **"B counted FAST/FAST":** at its final read, both hands FAST, untripped, and dose OK. A DEAD or DIVERGED kill
  is a valid not-FAST run, and it is left out of the fwd/back comparison (n reported). A wiring kill means a
  rerun under a fresh name. A dose kill is a §4.3 stop.
- **Missing reads are PENDING, never a decision.** That covers a head; model_4500 for the C0 rise; model_6100
  for the artifact rule; a B dose read on an otherwise counted run; a suite read.
- **fwd/back** are compared at model_5999 for both arms (equal budget; the extension targets side only):
  - **holds:** B's seed-mean ratio ≥ C0's − 0.03;
  - **clearly inferior:** B's seed-mean < C0's − 0.03 AND B's best seed < C0's worst seed.
- **Gate** (the first that matches decides):
  1. **RESUME ARTIFACT:** stop and ask.
  2. **PENDING:** while a read or a triggered extension is missing.
  3. **FAIL:** B counted FAST/FAST ≤ 1/3, or fwd or back clearly inferior. The construction does not survive a
     multi-command pool at this budget. Stop; no P1.
  4. **PASS:** B ≥ 2/3, C0 0/3, no B run tripped, and fwd and back hold. This is the precondition for P1
     (x6k option 2), which is the user's call.
  5. **PARTIAL:** anything else, reported with the numbers. Examples: C0 ≥ 1/3 (the FB pool lets C0 form, so the
     construction effect is unresolved), or fwd/back short but not clearly inferior. No P1 on a PARTIAL.

### 11.5 INC0 gate (before any FB6k GPU is spent)

INC0 is FAST/FAST at its model_5999 head in:
- **≥ 2 of 3:** STOP before FB6k. The flip belongs to action_rate 0, not the construction; stop and ask.
- **1 of 3:** noted. B is labelled "construction + action_rate 0".
- **0 of 3:** action_rate 0 alone does not flip C_lr by 6000 *continuous* iterations. The construction
  attribution is read together with FB6k, whose from-scratch B runs test the premise that a fork equals a
  continuous run. The x6k comparison point (B at seeds 11–13: 3/3 FAST/FAST) was reached as 3000 iterations plus
  an env-restarting fork, and x6k's check rated that premise open.

These also STOP:
- fewer than 3 INC0 reads;
- a dumped `reward_action_rate` weight (`params/env.yaml`) that is not 0, which would mean INC0 is not wired.

### 11.6 Running

- **Command:** `scripts/fb6k_queue.sh`, run detached. It chains the INC0 block, the gate, the FB6k block, the
  extension decision and the forks, each through `run_arm.sh`. Settings:
  - `ARMS_DIR=$OUT/arms_fb6k`;
  - `EXTRA_LR_EVALS="3000 4500"` (forks: `RESUME_READ=1` and `"6100 7500"`).
- **Process rules:** one Isaac process at a time, and no rendering while the queue can launch.
- **Budget:**

  | Block | Calculation | GPU-h |
  |---|---|---|
  | INC0 | 3 × ≈ 2.7 | ≈ 8 |
  | FB6k | 6 × ≈ 2.7 | ≈ 16 |
  | Forks | ≤ 6 × ≈ 1.45 | ≤ 9 |
  | **Total** | | **≈ 24–33** |

- **At the end:** an independent check, then `fb6k/RESULTS.md`. Adoption, a bake or P1 is the user's call. A PASS
  says nothing on smoothness: B's jitter is about 1.6–2× C0's (x6k).

**Pre-launch review (2026-10-03, before any run):** a fable·high audit of this section and an opus·high
code-path review (workflow `wf_5e3c6bef-eb8`).
- **Opus:** no queue break and no wasted GPU. The hashes, validation, chain_info, the live checks, stamp
  discovery and checkpoints all hold.
- **Applied:**
  - a DEAD-killed seed no longer hangs the gate;
  - missing reads are PENDING;
  - every non-FAST/FAST B run is extended, and its seed's C0 with it;
  - the DEAD-at-1000 kill is advisory on FB6k;
  - the INC0 0/3 wording is corrected, and the INC0 action_rate dump is checked;
  - the queue stops on run_arm or reader errors;
  - the Q block and odd RMS appear in the reader rows.
- **Tests:** the reader self-test has 28 cases.

## 12. P1-B6: B lineage vs C0 lineage through the phase-1 chain (pre-registered 2026-10-04)

(Placed before the log. The user approved FB6k next-option 1 ("Run 1") on 2026-10-04. The design is
`fb6k/p1_design.md` by opus·high, written after the FB6k PASS. Nothing here changes steps 3, x6k or FB6k.)

### 12.1 Question

Does the B construction give fast two-hand side walking inside the full phase-1 chain:
- with turning, move_turn and diag **trained**;
- with the clock **cut** (c6x, the shipped recipe);

without costing any other category against a matched C0 lineage? W, the shipped clock-free head, is the
head-to-head reference.

### 12.2 Arms (`scripts/candidates_v6.yaml`)

Every stage overlay is `P1_C_lr__C0` / `P1_C_lr__B` verbatim, plus the RNG-free hand diagnostic.

| Chain | Stages | Iterations | Seeds |
|---|---|---|---|
| `P1B6__{C0,B}_t3` | t3 (base 1a; pool stand .15 / fwd .25 / side_l .16 / side_r .16 / back .08 / turn .2; clock visible), from scratch | **6000** (was 3000; re-sized from FB6k's formation timing) | 16, 17, 18 |
| `P1B6__{C0,B}_tail_s{16,17,18}` | c4 (fork of that seed's t3 head) → c5 → c6 → c6x (`CLOCK_W 0.001`, `APEX_W 0.001`, `KRABBY_OMNI_CLOCK_HIDE=1`, as shipped) | 1500 / 1500 / 2000 / 4000 | its own |

- **Iteration numbering:** t3 runs 0–5999, c4 to 7498, c5 to 8997, c6 to 10996, c6x to 14995 (the head,
  model_14995).
- **B:** the B knobs on every stage. The soft σ and the bound ramp carry across stage forks (SMOKES.md).
- **C0:** the incumbent on every stage.
- **Order** (random.Random(20261005), seed-block-major): t3 C0_s16, B_s16, C0_s17, B_s17, C0_s18, B_s18; the tails
  in the same order.

### 12.3 Reads (`scripts/p1_read.py`, written and self-tested before launch)

- **Classes:** §4.1 (feet AND along; FAST 0.155 / 0.175) on the both-hands eval (`side22_25lr__omni`) at:

  | Checkpoint | When | Role |
  |---|---|---|
  | `model_4500` | t3 | descriptive |
  | `model_5999` | t3 head | the t3 gate |
  | c6 head | pre-cut, clock visible | the clock-cut read |
  | `model_13000` | 2000 iterations into c6x | the cut transient, descriptive |
  | `model_14995` | c6x head, clock hidden | the gate |

- **Suite** (the driver `suite__omni` stamps; t3, c4, c6 and c6x heads), read at the c6x head:
  - fwd, back, diag and side ratio;
  - turn `facing_closed` and `yaw_progress`;
  - move_turn ratio and survival;
  - per-category jitter and fft.
- **Clock-hidden assertion:** every c6x suite and lr stamp's env must carry `KRABBY_OMNI_CLOCK_HIDE=1`.
  Otherwise the read is INVALID EVAL (stop and ask).
- **B dose:** median |μ| ≤ 1.5 over the c6x run's last 100 logged iterations.
- **Tripwire:** threshold max(0.15, ref + 0.10), where ref is the C0-lineage median fall share at the c6x head
  (side holds).
- **Kills:**
  - **t3:** the §4.2 wiring and dose kills (B); DEAD-at-1000 is advisory (`ADVISE_DEAD=1`, as §11.3).
  - **Tails:** the §10.3 resume read on the c4 fork of the t3 head (`RESUME_READ=1`).
  - **Missing reads** are PENDING, never a decision.
- **W:** re-scored at this code_sha with the env of its own suite stamp (clock hidden): suite and both-hands
  side.

### 12.4 t3 stop gate (before the ≈ 25 GPU-h of tails)

**STOP and ask** if any of these hold:
- B has ≥ 1 FAST hand at its t3 head in ≤ 1/3 seeds (formation fails once turning shares the pool);
- B's weaker hand is below INTERMEDIATE (SLOW or DEAD) at the t3 head in ≥ 2/3 seeds. This is the INC0-like
  one-hand outcome, where the construction's second hand fails. At the same per-hand exposure, FB6k's second
  hands were all ≥ INT, so this rule does not false-stop on FB6k's timing;
- C0 is FAST/FAST at t3 in ≥ 2/3 (the construction effect is unresolved);
- a t3 read is missing or NA.

Otherwise run the tails. The t3 FAST/FAST counts are descriptive.

### 12.5 Final gate (c6x heads, clock hidden; the first rule that matches decides)

1. **INVALID EVAL:** a c6x eval without the clock hidden. Stop and ask.
2. **PENDING:** a read is missing.
3. **CLOCK-CUT LOSS:** B is FAST/FAST by hands at the c6 pre-cut read in ≥ 2/3, but FAST/FAST by hands at c6x in
   ≤ 1/3. Stop and ask; the next lever is a clock anneal, not the construction. A dose or tripwire loss with the
   hands still FAST is not a clock-cut loss; it falls through to FAIL.
4. **FAIL:** B counted FAST/FAST (both hands, untripped, dose OK) ≤ 1/3.
5. **COST:** B counted FAST/FAST ≥ 2/3, but a gated category is CLEARLY inferior to the C0 lineage: seed-mean <
   C0's − margin AND B's best seed < C0's worst. The category is named.
6. **PASS:** B ≥ 2/3, C0 FAST/FAST ≤ 1/3, no B trip, and every gated category holds (seed-mean ≥ C0's − margin).
7. **PARTIAL:** anything else, reported with the numbers.

**Gated categories and margins:**

| Category | Margin |
|---|---|
| fwd, back and diag ratio | 0.03 |
| turn `facing_closed` | 0.03 |
| turn `yaw_progress` | 0.01 (≈ 10 % of W's 0.102; 0.03 would allow a 30 % loss of turning rate) |

**move_turn survival and ratio are reported, not gated.** The clock cut alone took W's move_turn survival from
0.865 to 0.365, an open follow-up the user accepted. B's untrained move_turn already collapsed in FB6k, and the
ratio is a median over all episodes, so it tracks survival.

**An NA hand class** (a missing feet or along-command read) is PENDING, never a decided not-FAST.

**Reported, not gated:**
- per-category deltas against the re-scored W;
- **adoption-candidate flag:** a B seed FAST/FAST at c6x whose side ratio beats W's, with no gated category more
  than 0.05 below W's;
- **smoothness flag:** B's seed-mean jitter > 1.5× C0's in any category, or fft > C0's + 0.3. The flag makes a
  smoothness stage (action_rate restored after formation, ≈ 3 GPU-h) mandatory before any adoption.

Any adoption, bake or push is the user's call.

### 12.6 Running

- **Command:** `scripts/p1_queue.sh`, run detached:
  1. the t3 block (`ADVISE_DEAD=1`, `EXTRA_LR_EVALS="4500"`);
  2. the t3 gate;
  3. the tail block (`RESUME_READ=1`, `FORK_SRC_ARMS=$OUT/arms_p1`, `EXTRA_LR_EVALS="13000"`);
  4. the c6 pre-cut reads (`run_arm.sh --lr-eval`, using the c6 stage's own suite-stamp env);
  5. W's re-score (`--lr-eval`, `--row-eval suite__omni`).
- **Records:** `$OUT/arms_p1`.
- **Errors:** any error stops the queue with a NOTIFY line. On the tail block, `STOP_ON_REF_STOP=1` makes a per-run
  STOP inside run_arm.sh end the block.
- **Process rules:** one Isaac process at a time, and no rendering while the queue can launch.
- **Budget:**

  | Stage | Calculation | GPU-h |
  |---|---|---|
  | t3 | 6 × ≈ 2.7 | ≈ 16 |
  | Tails | 6 × ≈ 4.0 | ≈ 24 |
  | Pre-cut reads and W | | ≈ 0.6 |
  | **Total** | | **≈ 41–45** |
  | If the t3 stop fires | | ≈ 17 |

- **At the end:** an independent check, then `p1/RESULTS.md`.

**Pre-launch review (2026-10-04, before any run):** a fable·high audit of this section and an opus·high
code-path review (workflow `wf_0d7bf006-a3d`). Opus found the normal path correct, end to end, against real files.
- **Applied from the audit:**
  - CLOCK-CUT LOSS compares hands with hands;
  - FAIL (no formation) is separated from COST (a category clearly inferior);
  - move_turn is reported, not gated;
  - per-metric margins (yaw_progress 0.01);
  - the t3 second-hand stop rule.
- **Applied from the code review:**
  - NA hand classes are PENDING;
  - `STOP_ON_REF_STOP` on the tail block;
  - continuation dirs (`_c`, `_w`) are mapped to their stamps in any_stamp_of and p1_read;
  - t3 model_4500 is reported;
  - the resume read is pinned to the tail's first stage, and stage changes are logged as such. A test caught and
    fixed a first-poll case.
- **Tests:** tooling tests 53/53.

## 9. Log

- 2026-10-01: charter written. Code is being implemented on worktree `claude/ppo-construction`.
- 2026-10-01, before any run: review fixes (reviewer F1–F11 plus an independent check), applied to the
  tooling.
  - `run_arm.sh`/`verdict.py` (§4.2):
    - distinct kill codes; a wiring or dose kill writes `STOP_QUEUE` and a NOTIFY line;
    - a record for every killed run;
    - DEAD- and divergence-killed runs count as not FAST; wiring- and dose-killed runs are excluded;
    - post-hoc wiring failures are excluded;
    - duplicates count as re-rolls;
    - a read is latched only once it decides;
    - the seed comes from `references:`, so a rerun's fresh name works.
  - Hand class fails closed to NA (§4.1).
  - The `[KNOB] … consumed` checks accept only the exact line. "NOT consumed" never counts, in
    `verdict.py` or in the driver's `construction_log_checks`.
  - Verdict corners:
    - NO-FLIP never while an extension is pending (§4.5);
    - C-guard FAIL needs A ≤ 1/3; the middle band is UNDEFINED (§4.7).
  - Step 4 (§6): the 4a wiring read and kills, the per-seed control comparison and the UNLOCK listing
    were added. `PPO/hist_lr` telemetry was added (logged only when `KRABBY_PPO_HIST_LR` is set).
  - The per-hand diagnostic now runs every 25 training iterations, keyed on the runner's iteration
    instead of the update counter. It stays diagnostic only. The bound ramp stays on the counter
    (§4.3).
  - Rung 0b (§5): the env's debug line is now the tagged per-env line, and `rung0b_read.py` parses
    only that line.
  - `scripts/candidates_v2.yaml` (= `candidates.yaml`): the staging-off C_lr_rsi chains (C0/A/B/INC0)
    carry `KRABBY_RSI_CAM_STAGE: "0"` explicitly, so the dump audit proves their staging state.
    Behaviour is unchanged.
  - GPU knob-off identity smoke (plan step 5): run the pre-change tree twice at the same seed for 20
    iterations first. If those two disagree, judge the new code against that run-to-run spread, not
    against bit-identity.
- 2026-10-01 10:03, rung 0b: **STAGING OFF**; C_lr_rsi runs as `C_lr_rsi__C0/__A/__B` (X16's definition).
  - **Fall share in [0, 2) s, seed 1 / seed 2** (X02, lat_l 0.22 from reset, 400 episodes per row):
    - normal 0.000 / 0.000;
    - RSI with staging off 0.0025 / 0.005;
    - RSI with staging on 0.0025 / 0.000.
  - **Rule:** seed-pair spread 0.0025. (a) fails in both seeds: the deficit is 0.0025 / 0.005, below 0.04.
  - **Staging was verified per env** on all four RSI rows: errors 0 when on; target and history 0 when off, with
    `bank_absmax` 3.27.
  - **Before the decision was read:** `rung0b.sh` and `rung0b_read.py` were fixed to accept plain schedule stamps,
    which carry no suite-only `complete.json`. Tests were added. Details in SMOKES.md §4.
- 2026-10-01 10:27, rung 1: **4b NOT triggered.** Same window, C_lr__C0 config, 200 iterations per size.
  - **Trigger check:** 1,024 envs take 1.583 s/iter, 1.21× the 256-env 1.312 s/iter, above the 1.15× limit; VRAM is
    5.4 GB, within 10 GB.
  - **Other sizes:** 512 envs 1.08×; 2,048 envs 1.63×, past the 1.5× stop, so 4,096 was not run.
  - **No failures:** no NaN or OOM; peak RAM 6.1–6.8 GiB. Details in SMOKES.md §5.
- 2026-10-01: code landed (commit 60532fb9; code_sha f7166d558c5d, eval_sha 1d021625844b). The pilot launches in the CHARTER §3 order: C__C0_s11, C__A_s11, C_lr__C0_s11, C_lr__A_s11, serially via scripts/run_arm.sh. C_lr_rsi will run the `C_lr_rsi__*` chains (rung 0b: staging OFF). 4b is not triggered (rung 1).
- 2026-10-01 11:41, pilot: **C__C0_s11 = SLOW** (feet_ss 0.083 / 0.092, along SLOW, lean_abs 0.19–0.20, sweep 0.048–0.051 — the lean-and-shuffle phenotype; 0 short). This is the first SLOW one-direction fa run at 0.8 moving share; historical: C s3/s2 FAST, C_mo s1/s2 FAST, s3 INT, C_fa s3 FAST. Read: one-direction formation is itself seed-dependent at fresh seeds. Per §3 the pilot continues unchanged (C__A has no 'must be FAST' bar while C__C0 is not FAST). The C-bed guard (§4.7) will decide at 3 seeds; 'C bed not reliable' becomes likely.
- 2026-10-01 12:07, **C__A_s11 (X02) dose-killed at iteration ~1,140** under the original §4.3 rule.
  - **Wiring verified correct:** 6/6 consumed lines, `Loss/action_bound` 8.9, effective σ 0.67.
  - **The share rule failed:** median |μ| cam 0.97 / rod 1.27 (≤ 1.5, pass), but share |μ| > 1.5 was 0.27 / 0.41 (> 0.1, fail). Fresh C__C0_s11 at the same iteration: median 3.69 / 4.57, share > 1.5 0.88 / 0.88.
  - **Training reads, descriptive:** track_ratio 0.744 and rising, vs C__C0_s11 0.51 (SLOW). Applied LR 2.0e-5 vs C0 3.8e-5.
  - **True stop → user decision (12:30):** the dose is median |μ| ≤ 1.5 on cams and rods; sat15 is reported only. verdict.py `dose_at` was updated and its self-test passes (a case for exactly these numbers was added).
  - **The killed run is excluded** (§4.2) and must never be resumed (the orchestrator's AWAIT offers to resume it; ignore that).
  - **Rerun:** a fresh reference `C__A_s11_r2` (chain C__A, seed 11) added in candidates v3; its class counts. The pilot continues with C__A_s11_r2, C_lr__C0_s11, C_lr__A_s11.
  - **Written before any affected result was read.** No other run's dose had been read.
- 2026-10-01 16:18, pilot: **C__A_s11_r2 (X03) = FAST** (feet_ss 0.229 / 0.251, along FAST, 5 / 3 short; lean_abs −0.030 / −0.018; sweep 0.130 / 0.172; rod odd RMS 2.03 / 2.09). Same seed as C__C0_s11 = SLOW (0.083 / 0.092, lean 0.19–0.20, sweep 0.048–0.051, rod odd RMS 0.63 / 0.68). Descriptive (n = 1 per arm, arms not seed-paired). The speed is above every historical side head (max 0.215), and the sweep is about 1.5–2× the historical FAST range (0.076–0.090). Pilot condition met: the dose passed at 1000 under the amended rule (median 0.97 / 1.27), and the 'C__A FAST if C__C0 FAST' clause is vacuous because C0 is SLOW. The pilot continues with C_lr__C0_s11 and C_lr__A_s11.
- 2026-10-01 17:35, pilot: **C_lr__C0_s11 (X04) = SLOW / SLOW** (both-hands L 0.083 / 0.085, R 0.074 / 0.084; along SLOW; lean_abs 0.196–0.203; sweep 0.032–0.038; model_2500 also SLOW / SLOW). This reproduces the historical two-direction shuffle at a fresh seed. C_lr__A_s11 launched 17:35.
- 2026-10-01 18:50, pilot: **C_lr__A_s11 (X05) = L SLOW / R INTERMEDIATE**.
  - **Hands, both-hands eval:** L 0.092 / 0.097 (lean 0.18, rod odd RMS 0.77); R 0.165 / 0.161 (rod odd RMS 1.53 / 1.49).
  - **model_2500:** also L SLOW / R INTERMEDIATE (hands-averaged feet 0.129 → 0.127; no late rise).
  - **Dose:** passed at 1000 and at the final window (median 1.11 / 1.26).
  - **Training track ratio:** still rising at the end (0.536 at 1000 → 0.609 at 2950), against C_lr__C0_s11's flat ~0.48.
  - **First two-direction network to form the odd stroke on one hand:** the historical two-direction maximum rod odd RMS was 0.93.
  - **verdict.py so far:** C_lr A PARTIAL (an INTERMEDIATE hand).
- 2026-10-01 18:55, **pilot read (§3): PASS.** Both A runs passed the iteration-1000 dose read (C__A_s11_r2 and C_lr__A_s11). "C__A FAST if C__C0 FAST" is vacuous (C__C0_s11 SLOW).
  - **Excluded:** the C__A_s11 dose kill (resolved by the user's 12:30 amendment).
  - **Flag:** C__A_s11_r2 carries "dose lost late" (rod median 1.52 at the final window); its class counts.
  - **LR:** A runs sit at the floor 0.59–0.67 vs C0 0.35–0.37; applied LR 1.8–2.1e-5 vs 3.2–3.6e-5, a ratio of about 0.57–0.66, above the ½ trigger. No 4a trigger yet.
- 2026-10-01 18:57, **block order drawn once** (python random.Random(20261001).shuffle within each block):
  - block 0 (seed 11 remainder): C_lr__B_s11, C_lr_rsi__A_s11, C_lr_rsi__B_s11, C_lr_rsi__C0_s11;
  - block 1: C_lr__A_s12, C_lr__C0_s12, C_lr_rsi__C0_s12, C__A_s12, C_lr__B_s12, C__C0_s12, C_lr_rsi__B_s12, C_lr_rsi__A_s12;
  - block 2: C_lr_rsi__B_s13, C_lr_rsi__C0_s13, C_lr_rsi__A_s13, C_lr__B_s13, C_lr__A_s13, C__A_s13, C__C0_s13, C_lr__C0_s13.

  These 20 runs launch serially in this order.
- 2026-10-01 20:09: **C_lr__B_s11 (X06) = L FAST / R INTERMEDIATE**: L 0.227 / 0.243 (sweep 0.110 / 0.145, lean 0.06–0.07); R 0.122 / 0.137 (lean 0.17–0.20). This is the first FAST hand in any two-direction run. Run not FAST (both hands needed).
- 2026-10-01 21:25: **C_lr_rsi__A_s11 = L SLOW / R FAST**: L 0.087 / 0.091 (lean 0.34, sweep 0.03); R 0.220 / 0.235 (lean 0.04–0.05, sweep 0.095). This is the third construction-arm two-direction run with one hand fast or near-fast and the other a lean-shuffle. Descriptive note: a one-hand mode, not the incumbent's both-hands shuffle.
- 2026-10-01 22:40: **C_lr_rsi__B_s11 = L FAST / R INTERMEDIATE** (L 0.216 / 0.218, lean 0.07–0.10, sweep 0.10–0.11; R 0.128 / 0.138, lean 0.15–0.16). Four of the four construction-arm two-direction runs so far have one hand FAST or near-FAST and the other slow (descriptive).
- 2026-10-01 23:57: **C_lr_rsi__C0_s11 = SLOW / SLOW** (L 0.079 / 0.085, R 0.084 / 0.095; lean 0.20–0.22). Block 0 is complete. Seed 11 summary:
  - C: C0 SLOW, A FAST.
  - C_lr: C0 SLOW/SLOW, A SLOW/INT, B FAST/INT.
  - C_lr_rsi: C0 SLOW/SLOW, A SLOW/FAST, B FAST/INT.
  - Every construction run formed the stroke on at least one hand, and none on both.
- 2026-10-02 01:14: **C_lr__A_s12 = L FAST / R SLOW** (L 0.175 / 0.180; R 0.084 / 0.090, lean 0.20). The one-hand mode again.
- 2026-10-02 02:29: **C_lr__C0_s12 = SLOW / SLOW** (L 0.092 / 0.097, R 0.073 / 0.079; lean 0.15).
- 2026-10-02 03:46: **C_lr_rsi__C0_s12 = SLOW / SLOW** (L 0.076 / 0.088, R 0.057 / 0.065; lean 0.17 / 0.28).
- 2026-10-02 04:53: **C__A_s12 = FAST** (0.213 / 0.217, lean ≈ 0, sweep 0.13–0.14). C bed so far: A 2/2 FAST, C0 0/1 (C__C0_s12 pending).
- 2026-10-02 06:08: **C_lr__B_s12 = FAST / FAST**, the first two-direction network ever FAST on both hands: L 0.245 / 0.273 (sweep 0.23–0.26, lean −0.03 to −0.04); R 0.189 / 0.197 (sweep 0.07, lean 0.08); 0 short. **§4.5 extension triggered:** C_lr__B and C_lr__C0 at seeds 14–15. They are queued after block 2 so the drawn order stays undisturbed, together with any extension seed 13 triggers.
- 2026-10-02 07:15: **C__C0_s12 = FAST** (L 0.185 / 0.201, lean 0.07, sweep 0.08–0.09, rod odd RMS 1.31). The incumbent forms the one-direction stroke at s12 but not at s11. C bed so far: A 2/2 FAST (§4.7 guard passes whatever s13 gives), C0 1/2. The C bed has no §4.5 extension; that rule is for the two-direction beds.
- 2026-10-02 07:15, **deviation (one Isaac process):** at the user's request, a viewer-only video of C_lr__B_s12 model_2999 (side holds lat_l/lat_r at 0.22 and 0.25, 20 s each) is recorded with `tools/record_categories.py` while C_lr_rsi__B_s12 trains. It runs under a 22 GB MemoryMax scope, 1 env. The recorder is a viewer (no ledger, no gate) and does not touch the training process; only that run's wall-clock is affected. `record_categories.py` gained a default-preserving `--manifest` option so it can load the side bisect's `scenarios_side_lr.yaml`.
- 2026-10-02 08:30, **incident: C_lr_rsi__B_s12, caused by the 07:15 video deviation.**
  - **What happened:** attempt 1 died at env creation (Warp CUDA out of memory, 07:17:54) while the recorder rendered on the same 16 GB GPU. The driver's one experiment-level retry trained `seed12_X16_s1_A15pB_e2` from scratch (07:20–08:25, same reference, seed and config). Attempt 1 never trained, so the retry counts as the run.
  - **What the runner missed:** it followed only the first training line and looked for the stamp under the `_e2` tag. So it made no live wiring, dose or DEAD reads, and it ended with "STOP no complete driver eval stamp". The driver had filed the stamp under `seed12_X16_s1_A15pB`.
  - **Post-hoc reads (tfevents, the same verdict.py calls):** wiring PASS (σ 0.805 at 200); dose PASS at 1000 (median |μ| cam 0.24 / rod 1.01); not DEAD (track 0.50, σ 0.77).
  - **Driver stamp:** lat_l FAST (0.224 / 0.246, lean ≈ 0, along FAST). The both-hands evals are pending.
  - **Runner fix (run_arm.sh):**
    - the run-dir search takes the latest training line;
    - the watcher starts without a run dir and follows a retry, its reads restarting there;
    - the stamp search strips `_e<k>`;
    - a new `--post <ref> <ckpt>` mode;
    - 4 new tests (tooling 29/29).

    The running runner keeps its old copy (an atomic replace), so the fix applies from the relaunch.
  - **Pause and relaunch:** STOP_QUEUE was set at 08:30. The runner exits after C_lr_rsi__A_s12. A gated relaunch then runs, only if C_lr_rsi__A_s12 logs DONE, so a true stop is never overridden:
    1. `--post C_lr_rsi__B_s12`;
    2. block 2 in the drawn order;
    3. the §4.5 C_lr extension: C_lr__B_s14, C_lr__C0_s14, C_lr__B_s15, C_lr__C0_s15.

    Any further extensions (C_lr_rsi from seed 12, seed-13 triggers) are taken from verdict.py after block 2.
  - **Rule from this:** no rendering or eval process on the GPU while the queue can launch an arm. The video is the only deviation so far.
- 2026-10-02 09:45: **C_lr_rsi__A_s12 = SLOW / SLOW** (L 0.090 / 0.103, R 0.095 / 0.089; lean 0.25–0.27, sweep 0.035–0.044). It is the incumbent's lean-and-shuffle phenotype, and model_2500 is also SLOW/SLOW. Wiring and dose PASS (median |μ| at 1000: cam 0.91, rod 1.34). This is the first construction-arm two-direction run with no fast or near-fast hand (descriptive). Block 1 has finished training. The gated relaunch fired at 09:45 (`--post C_lr_rsi__B_s12`, then block 2, then the C_lr extension).
- 2026-10-02 09:53: **C_lr_rsi__B_s12 (`_e2` retry, `--post`) = L FAST / R INTERMEDIATE.**
  - **Hands:** L 0.224 / 0.248 (lean ≈ 0, sweep 0.09–0.11); R 0.153 / 0.156 (lean 0.15–0.19, sweep 0.05).
  - **Rod odd RMS:** L 1.92 / R 1.88, the highest two-direction values so far.
  - **model_2500:** also FAST / INT.
  - **Record:** complete. verdict.py reads it as a normal run.

  **Block 1 (seed 12) summary:**
  - C: C0 FAST, A FAST.
  - C_lr: C0 SLOW/SLOW, A FAST/SLOW, B FAST/FAST.
  - C_lr_rsi: C0 SLOW/SLOW, A SLOW/SLOW, B FAST/INT.

  Block 2 launched at 09:53 on the fixed runner.
- 2026-10-02 11:10: **C_lr_rsi__B_s13 = L SLOW / R FAST.** L 0.127 / 0.140 (lean 0.18, sweep 0.06), just inside the SLOW band. R 0.238 / 0.267 (lean ≈ 0, sweep 0.16–0.20). Rod odd RMS L 1.40 / R 1.99; model_2500 is also SLOW/FAST. Wiring and dose PASS (σ 0.80 at 200; track ratio 0.48 at 1000). C_lr_rsi__B so far: FAST/INT, FAST/INT, SLOW/FAST. Every run has one fast hand, and it is not always the same side.
- 2026-10-02 12:27: **C_lr_rsi__C0_s13 = SLOW / SLOW** (L 0.067 / 0.070, R 0.065 / 0.070; lean 0.12–0.18; rod odd RMS 0.70 / 0.78; model_2500 is also SLOW/SLOW).
  - **C_lr_rsi C0 at 3 seeds:** 0/3 FAST, every run SLOW/SLOW, so no C0-triggered extension.
  - **C_lr_rsi B at 3 seeds:** 0/3 runs FAST, though every run has one FAST hand. This bed's §4.5 extension now depends only on C_lr_rsi__A_s13.
- 2026-10-02 13:44: **C_lr_rsi__A_s13 = L SLOW / R FAST** (L 0.094 / 0.107, lean 0.20; R 0.228 / 0.226, lean 0.12–0.13, sweep 0.12–0.13; rod odd RMS 0.87 / 1.68; model_2500 is also SLOW/FAST). Wiring and dose PASS.
  - **The C_lr_rsi bed is complete at 3 seeds:**

    | Arm | s11 | s12 | s13 | Runs FAST |
    |---|---|---|---|---|
    | C0 | SLOW/SLOW | SLOW/SLOW | SLOW/SLOW | 0/3 |
    | A | SLOW/FAST | SLOW/SLOW | SLOW/FAST | 0/3 |
    | B | FAST/INT | FAST/INT | SLOW/FAST | 0/3 |

    No A/B or C0 run is FAST, so there is no §4.5 extension on this bed.
  - **Interim verdict.py (`--no-tb`):** C_lr_rsi A and B both PARTIAL (a FAST hand in a run not counted FAST). That reads "inconclusive", not "not the limiter". "Q better" = yes for C_lr_rsi A (descriptive; adopting it is a user proposal, gated by the C guard).
  - **For the final read:** verdict.py still lists the C__A_s11 dose kill under STOPS. That kill was resolved by the user's 12:30 amendment (rerun as C__A_s11_r2), so the final verdict must mark it as resolved, not as an open stop.
- 2026-10-02 15:00: **C_lr__B_s13 = L FAST / R INTERMEDIATE.** L 0.221 / 0.235 (lean 0.03–0.04, sweep 0.12–0.13); R 0.132 / 0.143 (lean 0.12, sweep 0.11). Rod odd RMS 1.95 / 1.65. At model_2500 it was FAST / SLOW, so R rose late. Wiring and dose PASS.
  - **C_lr__B is now 1/3 runs FAST** (s11 FAST/INT, s12 FAST/FAST, s13 FAST/INT).
  - **UNLOCK can no longer be reached on any bed.** C_lr__B can reach at most 3/5 after the s14–15 extension, short of 4/5 with C0 0/5, and the same holds for C_lr__A. So 4c cannot trigger.
  - **The extension still runs as pre-registered:** the §4.5 count fixes PARTIAL vs NO-FLIP, and NO-FLIP is never given while an extension is pending.
  - **Step 4 status:** 4b is not triggered (rung 1). 4a is not triggered on present data: A/B LR floor share 0.51–0.65 (the threshold is > 0.9); A/B applied LR 0.55–0.77× the C0 median (the threshold is < 0.5×). The final verdict.py decides 4a.
- 2026-10-02 16:17: **C_lr__A_s13 = SLOW / SLOW** (L 0.073 / 0.084, R 0.093 / 0.101; lean 0.20–0.24; rod odd RMS 0.65 / 1.14). At model_2500 it was L DEAD / R SLOW (feet below 0.05 on L), so it moved up to the shuffle by 2999. Wiring and dose PASS.
  - **C_lr__A is complete at 3 seeds:** s11 SLOW/INT, s12 FAST/SLOW, s13 SLOW/SLOW. 0/3 runs FAST, so no A extension.
  - **The remaining queue is unchanged:** C__A_s13, C__C0_s13, C_lr__C0_s13, then C_lr__B and C_lr__C0 at s14–15. C_lr__C0_s13 FAST would extend A as well (§4.5).
- 2026-10-02 17:25: **C__A_s13 = FAST** (0.241 / 0.260, lean ≈ 0, sweep 0.21–0.26 — the longest strokes on this bed; rod odd RMS 2.31). Wiring and dose PASS (track ratio 0.76 at 1000).
  - **C bed A:** 3/3 FAST, so the §4.7 guard passes (A ≥ 2/3) whatever C__C0_s13 gives; verdict.py prints it once C0 has 3 seeds.
  - **C bed C0:** 1/2 so far.
- 2026-10-02 18:33: **C__C0_s13 = INTERMEDIATE** (0.147 / 0.154, lean 0.09–0.10, sweep 0.06; rod odd RMS 1.36).
  - **The C bed is complete:**

    | Arm | s11 | s12 | s13 | FAST |
    |---|---|---|---|---|
    | A | FAST | FAST | FAST | 3/3 |
    | C0 | SLOW | FAST | INT | 1/3 |

  - **Guard:** verdict.py reads PASS ("A FAST 3/3 ≥ 2/3; C0 itself not reliably FAST: noted").
  - **Discrepancy for the final check:** §4.7's text also lists "C0 ≤ 1/3 → C bed not reliable at fresh seeds; the guard does not decide", and C0 is at 1/3 here. The two readings disagree on whether the guard PASSes or stays UNDEFINED.
    - verdict.py checks A first. Its reading is that the guard's question is "does A still form the one-direction stroke", and A 3/3 answers it, whatever C0 does.
    - It changes no run. 4c is already unreachable. It only gates whether a "Q better" result becomes an "adopt for motion quality" proposal to the user (§4.8), which is never an adoption.
    - It goes to the independent check and to the user as stated; it is not decided here.
- 2026-10-02 19:50: **C_lr__C0_s13 = SLOW / SLOW** (L 0.063 / 0.077, R 0.074 / 0.072; lean 0.20–0.24; rod odd RMS 0.39 / 0.26; model_2500 is also SLOW/SLOW).
  - **Block 2 (seed 13) is complete.** C_lr C0 is 0/3 FAST, so there is no A extension.
  - **Queue:** C_lr__B_s14, C_lr__C0_s14, C_lr__B_s15, C_lr__C0_s15, then the final read.
- 2026-10-02 20:10, **verdict.py: superseded dose kills.** A dose kill whose own iteration-1000 read passes the amended median rule is now "superseded": the run stays excluded, its rerun counts, and it is neither a 4.3 stop nor a pilot DOSE.
  - **Only case:** C__A_s11, median |μ| cam 0.97 / rod 1.27 at 1000.
  - **Fail-closed:** with `--no-tb` the dose read is unread and the kill stays a stop.
  - **Tests:** self-test PASS (the pilot DOSE case now uses a real dose failure, plus a new superseded case); tooling tests 29/29.
  - **Full verdict now:** STOPS none; pilot PASS.
- 2026-10-02 20:10, **finding (descriptive): same-seed training is deterministic here.** The killed C__A_s11 (X02) and its rerun C__A_s11_r2 (X03), same seed and config, match exactly through iteration 1140 on `Loss/surrogate`, `PPO/mu_med_rod` and `Train/mean_reward`.
  - So a same-seed rerun is a replay, not a new sample (consistent with the D18 re-roll exclusion).
  - C_lr_rsi__B_s12's `_e2` retry is what the uninterrupted run would have been.
  - "Same-seed reruns diverge" (D18) does not hold for this training path on this machine.
- 2026-10-02 21:04: **C_lr__B_s14 = L FAST / R INTERMEDIATE.** L 0.218 / 0.244 (lean ≈ 0, sweep 0.16–0.18); R 0.138 / 0.151 (lean 0.10, sweep 0.04). Rod odd RMS 1.95 / 1.12. At model_2500 it was FAST / SLOW, so R rose late again. Wiring and dose PASS.
  - **C_lr__B so far:** FAST/INT, FAST/FAST, FAST/INT, FAST/INT, i.e. 1/4 runs FAST.
  - **Pattern:** in all four runs the left hand is FAST.
- 2026-10-02 22:21: **C_lr__C0_s14 = SLOW / SLOW** (L 0.079 / 0.085, R 0.075 / 0.078; lean 0.21–0.23; rod odd RMS 0.40 / 0.30; n 89–97 per hold, so a few short episodes; model_2500 is also SLOW/SLOW). C_lr C0 so far: 0/4.
- 2026-10-02 23:39: **C_lr__B_s15 = L INTERMEDIATE / R FAST.**
  - **L:** 0.156 / 0.174 (lean 0.12–0.13). It sits right on the FAST bounds (0.155 / 0.175), FAST on the first and just short on the second.
  - **R:** 0.237 / 0.258 (lean 0.02–0.03).
  - **Rod odd RMS:** 1.74 / 1.96.
  - **Late rise:** at model_2500 it was SLOW / FAST, so L rose late.
  - **Checks:** wiring and dose PASS.
  - **C_lr__B is final at 5 seeds: 1/5 runs FAST.** Every run has one FAST hand (L in 4, R in 1), and in 4 of 5 the other hand is INTERMEDIATE.
  - The last run, C_lr__C0_s15, is training.
- 2026-10-03 00:55: **C_lr__C0_s15 = SLOW / SLOW** (L 0.064 / 0.079, R 0.090 / 0.092; lean 0.19–0.31; rod odd RMS 0.50 / 0.70).
  - **The queue has ended:** 31 runs in total (C__A_s11 excluded as superseded).
  - **Final `verdict.py`, with tfevents** (`verdict/verdict.{json,md}`; gate JSONs copied to `gate/`):
    - C_lr A PARTIAL (0/3 vs C0 0/5);
    - C_lr B PARTIAL (1/5 vs 0/5);
    - C_lr_rsi A PARTIAL (0/3 vs 0/3);
    - C_lr_rsi B PARTIAL (0/3 vs 0/3);
    - C-bed guard PASS (A 3/3; C0 1/3);
    - pilot PASS; STOPS none;
    - UNLOCK none, so 4c is not triggered;
    - LR-floor 4a trigger none; 4b not triggered (rung 1);
    - "Q better" + "adopt for motion quality" = yes for C_lr_rsi A only, a proposal to the user (§4.10).
  - **Step 4 runs nothing.** The final read workflow `wf_594e4e5d-d98` is running:
    - mechanism telemetry tables;
    - the independent fable·high re-derivation of the verdicts;
    - an opus review of verdict.py;
    - a fable reading of the mechanism data.

    The write-up goes to `RESULTS.md`; the driver regenerates `REPORT.md`.
- 2026-10-03 02:30, **step 3 closed: final read.**
  - **Verdicts** (`verdict/verdict.{json,md}`, re-run after the post-result edits below, unchanged):
    - C_lr A and B PARTIAL;
    - C_lr_rsi A and B PARTIAL;
    - C guard PASS, with the 4.7 record "C bed not reliable at fresh seeds (C0 1/3)";
    - no UNLOCK, no 4a, no 4b, so step 4 runs nothing;
    - "Q better" and the "adopt for motion quality" proposal for C_lr_rsi A only.
  - **Independent fable·high check** (`wf_594e4e5d-d98`): it reproduces every line from `gate/` and the tfevents, and no finding changes a verdict. Rulings: guard PASS (UNDEFINED is admissible; the user's call); the C__A_s11 supersede is consistent with the amendment; C_lr_rsi__B_s12 is valid; MQ is a proposal for C_lr_rsi A only.
  - **Opus·high implementation review:** two confirmed mismatches, the guard precedence (the user's call) and the Q seed set. The Q fix changes no line.
  - **Post-result verdict.py edits:** the guard record string; Q restricted to the 3 v 3 screen seeds; sat15 reported per run. None changes a verdict.
  - **Mechanism reads:** `verdict/mechanism_reads.{json,md}` (`scripts/mechanism_reads.py`).
  - **Write-up and decisions for the user:** `RESULTS.md` (guard reading; MQ proposal, recommended decline; ranked next experiments, outside the approved plan).
  - **Corrections to this log:**
    - the 00:55 "31 runs" is 29 records (28 counted + 1 excluded);
    - C__A_s13 and C_lr__A_s13 also carry "dose lost late" (final rod median 1.625 / 1.647; class counts);
    - some seed-11 entries (C_lr__A_s11, C_lr_rsi__C0_s11, C_lr_rsi__B_s11) quoted model_2500 speeds; the `RESULTS.md` table, generated from `gate/`, is the record;
    - "the `_e2` retry is what the uninterrupted run would have been" is an inference: the recorder shared the GPU during its first hour.
- 2026-10-03 (pre-launch), **x6k registered (§10):** the user approved "the 6000 iteration experiment on B". Pre-launch it got: candidates_v4 (six forks; dry run and strict validation clean, every head resolves at code_sha f7166d558c5d), the `resume_read.py` wiring read (self-test, plus real tfevents: a from-scratch start FAILs every check), runner changes (41 tooling tests), and a fable/opus pre-launch review whose fixes are applied (§10.6). The live `candidates.yaml` is now v4; the step-3 verdict is byte-identical under v3 and v4. Fork 1 (`C_lr__B_x6k_s13`) launches alone as the §10.6 gate.
- 2026-10-03 07:37, **x6k fork 1 (`C_lr__B_x6k_s13`) resume read PASS:** first step 2999 (= the source's last); σ 0.74852 vs 0.74860; priv_reg 0.03833 vs 0.0383; bound coefficient 1e-3 vs 1e-3. The fork continues its head.
  - **Reader:** `scripts/x6k_read.py` (§10.4) is committed before any head is read. Its self-test covers 15 cases, including the s15 boundary and the precedence; tooling tests are 42/42.
- 2026-10-03 08:57, **x6k fork 1 `C_lr__B_x6k_s13`: COMPLETES; the §10.6 gate passes.**
  - **Per-hand feet** (mean of the 0.22 and 0.25 holds):

    | Checkpoint | L | R |
    |---|---|---|
    | model_2999 | FAST 0.228 | INT 0.137 |
    | model_3100 | FAST 0.233 | INT 0.148 |
    | model_4000 | FAST 0.232 | FAST 0.196 |
    | model_5000 | FAST 0.231 | FAST 0.236 |
    | head model_5998 | FAST 0.240 | FAST 0.239 |

    The second hand rose +0.101.
  - **Other reads:** fall 0.003; dose median |μ| 0.30 / 1.00 over the last 100 iterations; rod odd RMS 2.09 / 2.00; priv_reg at the head 0.1000.
  - **Gate:** no resume artifact (the first hand is still FAST at model_3100), and the fork is a two-direction candidate by §10.5.
  - **Queue:** forks 2–6 launched at 08:57 in the drawn order (s14, s12, C0_s11, s15, s11).
- 2026-10-03 10:20, **x6k fork 2 `C_lr__B_x6k_s14`: COMPLETES** (a candidate).
  - **Per-hand feet:**

    | Checkpoint | L | R |
    |---|---|---|
    | model_2999 | FAST 0.231 | INT 0.144 |
    | model_3100 | FAST 0.249 | INT 0.161 |
    | model_4000 | FAST 0.248 | FAST 0.250 |
    | model_5000 | FAST 0.261 | FAST 0.255 |
    | head | FAST 0.266 | FAST 0.251 |

    The rises are +0.035 / +0.107.
  - **Other reads:** fall 0.000; dose 0.28 / 1.03; no resume artifact.
  - **Count:** 2 of the 4 one-FAST forks complete, which is the LATE FORMATION count. The reading stays PENDING until all five B forks are read, because a drop in 2 of 5 or an artifact outranks it.
  - **Fork 3 (s12):** launched at 10:20; resume read PASS (σ 0.75545 vs 0.75540).
- 2026-10-03 11:43, **x6k fork 3 `C_lr__B_x6k_s12` (the FAST/FAST head): STABLE** (a candidate).
  - **Per-hand feet:**

    | Checkpoint | L | R |
    |---|---|---|
    | model_2999 | FAST 0.259 | FAST 0.193 |
    | model_3100 | FAST 0.250 | FAST 0.224 |
    | model_4000 | FAST 0.238 | FAST 0.210 |
    | model_5000 | FAST 0.258 | FAST 0.250 |
    | head | FAST 0.260 | FAST 0.262 |

  - **Other reads:** fall 0.000; dose 0.28 / 1.00; no drop and no artifact.
  - **Candidates so far:** s12, s13, s14.
  - **Next:** fork 4 (the C0 control) launched at 11:42.
- 2026-10-03 13:10, **x6k fork 4 `C_lr__C0_x6k_s11` (the C0 control): FROZEN.**
  - **Resume read:** PASS (σ 1.99969 vs 1.99976; priv_reg continues).
  - **Per-hand feet:**

    | Checkpoint | L | R |
    |---|---|---|
    | model_2999 | SLOW 0.084 | SLOW 0.081 |
    | model_3100 | SLOW 0.085 | SLOW 0.080 |
    | model_4000 | SLOW 0.086 | SLOW 0.082 |
    | model_5000 | SLOW 0.086 | SLOW 0.080 |
    | head | SLOW 0.086 | SLOW 0.082 |

    The rises are +0.001 / +0.001.
  - **Other reads:** fall 0.025; median |μ| in the last 100 iterations is 6.71 (cam) / 9.77 (rod), more saturated than at 2999. priv_reg at the head is 0.1000.
  - **Reading:** over 3000 more iterations, including the 0.04 → 0.1 priv_reg ramp, the incumbent did not move. The step-3 freeze reading holds, and its two-direction SLOW is not slowness.
  - **Next:** fork 5 (s15) launched at 13:10.
- 2026-10-03 14:35, **x6k fork 5 `C_lr__B_x6k_s15`: COMPLETES** (a candidate).
  - **Resume read:** PASS.
  - **Per-hand feet:**

    | Checkpoint | L | R |
    |---|---|---|
    | model_2999 | INT 0.165 | FAST 0.248 |
    | model_3100 | INT 0.163 | FAST 0.242 |
    | model_4000 | FAST 0.174 | FAST 0.262 |
    | model_5000 | FAST 0.188 | FAST 0.235 |
    | head | FAST 0.219 | FAST 0.255 |

    The rises are +0.055 / +0.007, so the second hand is a material completion, not a boundary flip.
  - **Other reads:** fall 0.000; dose 0.31 / 1.07.
  - **Count:** 3 of the 4 one-FAST forks complete, and no B fork has a drop. With one fork left, INTERFERENCE (drops in ≥ 2 of 5) is unreachable. Only a resume artifact in s11 can outrank LATE FORMATION.
  - **Next:** fork 6 (s11) launched at 14:34.
- 2026-10-03 15:59, **x6k complete: the pre-registered reading is LATE FORMATION** (`x6k/x6k_read.{json,md}`; records copied to `x6k/gate/`).
  - **Last fork:** fork 6 `C_lr__B_x6k_s11` COMPLETES. R went 0.149 → 0.144 (3100) → 0.222 (4000) → 0.244 (5000) → 0.224 at the head; L stayed FAST throughout (0.237 at the head). Resume read PASS; no artifact.
  - **One-FAST forks:** all four complete (s11, s13, s14, s15; second-hand rises +0.055 to +0.107), with no drops in any B fork and no resume artifact.
  - **s12:** STABLE.
  - **C0 control:** FROZEN.
  - **Candidates:** all five B forks are two-direction candidates (FAST/FAST at the head, falls 0.000–0.003, dose median |μ| 0.27–0.31 / 1.00–1.07).
  - **Pending:** the §10.6 independent check (workflow `wf_698962eb-d6d`).
- 2026-10-03 16:40, **x6k closed.** The §10.6 independent check (fable·high, `x6k/CHECK.md`) PASSES. It reproduces every number and the reading LATE FORMATION; no head is marginal (the smallest margin to a FAST line is +0.050).
  - **Confound ratings:**
    - priv_reg ramp: open (not the sole cause; C0 did not move under it);
    - env restart: weakens;
    - eval noise: ruled out;
    - selection: weakens;
    - from-scratch reproduction: open;
    - beyond the C_lr bed: serious (scope).
  - **Correction to §10.5's C0 sentence:** the step-3 freeze onset (iterations 500–1000) was at priv_reg 0, but the step-3 heads at 2999 were at 0.0383. The FROZEN outcome is unaffected.
  - **Q block** (CPU): B's jitter at 6000 is 0.117–0.155, against C0 0.065–0.104 and A 0.043–0.066. That is the action_rate 0 cost, and it rose from 3000 in 4/5 seeds.
  - **Write-up and next options:** `x6k/RESULTS.md`, with opus's confirm design in `x6k/next_options.md`. The ranked options are FB6k + INC0 (≈ 26–30 GPU-h), P1 behind it (≈ 42), a head-fork probe (≈ 6), and completing the C0 control (≈ 6).
  - **No adoption and no bake.** 4c stays formally untriggered: this is a result with an amended budget, so it is the user's call.
- 2026-10-03 (pre-launch), **FB6k + INC0 registered (§11):** the user approved x6k next-option 1 ("Run 1"). Pre-launch it got:
  - candidates_v5 (dry run clean; the step-3 verdict is unchanged under v5);
  - `fb6k_read.py` (reader, INC0 gate and extension rule; self-test 28 cases);
  - `fb6k_queue.sh` (INC0 → gate → FB6k → forks; stops on errors);
  - run_arm `ADVISE_DEAD`;
  - 46 tooling tests;
  - a fable/opus pre-launch review, whose fixes are applied (§11.6).

  The queue launches now; INC0 runs first.
- 2026-10-03 18:57, **INC0 s11 (`C_lr6k__INC0_s11`, C_lr C0 + action_rate 0, 6000 its from scratch) = L FAST / R SLOW**, so not FAST/FAST.
  - **Per-hand feet:**

    | Checkpoint | L | R |
    |---|---|---|
    | model_3000 | FAST 0.183 | SLOW 0.094 |
    | model_4500 | FAST 0.229 | SLOW 0.106 |
    | head | FAST 0.230 | SLOW 0.104 |

    The second hand is flat from 3000 to the head: no late formation, unlike B.
  - **Other reads:** fall 0.003; the dumped action_rate weight is 0 (wiring OK).
  - **Descriptive:**
    - This is the first incumbent-construction run ever to form a FAST side hand on C_lr; C0 was 0/16 hands. So the action_rate penalty is part of what holds C0's stroke back.
    - INC0 pays a large smoothness cost: jitter 0.365 / fft 2.89, against B 0.12–0.15 / 1.07–1.37 and C0 0.065–0.104.
    - Its rods stay saturated (median |μ| 4.31; cam 0.51).
- 2026-10-03 21:25, **INC0 s12 = L INT / R FAST**, so not FAST/FAST.
  - **Per-hand feet:**

    | Checkpoint | L | R |
    |---|---|---|
    | model_3000 | SLOW 0.122 | INT 0.160 |
    | model_4500 | INT 0.137 | FAST 0.180 |
    | head | INT 0.135 | FAST 0.208 |

    The second hand is flat from 4500 to the head.
  - **Other reads:** jitter 0.359 / fft 3.19; rod median |μ| 5.14; dump OK.
  - **Gate:** INC0 FAST/FAST is 0/2, so the §11.5 stop (≥ 2/3) is now unreachable. FB6k will launch after INC0 s13.
  - **Pattern (descriptive):** action_rate 0 on the incumbent forms one hand, and the second stays flat, unlike B's late second hand.
- 2026-10-03 23:50, **INC0 s13 = L FAST / R INT**, so not FAST/FAST.
  - **Per-hand feet:**

    | Checkpoint | L | R |
    |---|---|---|
    | model_3000 | FAST 0.211 | SLOW 0.094 |
    | model_4500 | FAST 0.230 | SLOW 0.124 |
    | head | FAST 0.216 | INT 0.150 |

    Unlike s11 and s12, R is still rising (+0.026 from 4500).
  - **Other reads:** jitter 0.594 / fft 4.12, the jerkiest run of the campaign; rod |μ| 4.23.
  - **§11.5 INC0 gate: 0/3 FAST/FAST** → "ACTION_RATE ALONE INSUFFICIENT". action_rate 0 alone does not flip C_lr by 6000 continuous iterations.
    - It does form one FAST hand in 3/3 runs (C0 at 3000 had 0/16 hands), at 2.4–4× B's jitter.
    - The construction attribution is read together with FB6k.
  - **Next:** the FB6k block launched at 23:50 (C0_s16 first).
- 2026-10-04 02:25, **FB6k__C0_s16 = SLOW / SLOW.**
  - **Per-hand feet:**

    | Checkpoint | L | R |
    |---|---|---|
    | model_3000 | 0.096 | 0.101 |
    | model_4500 | 0.104 | 0.121 |
    | head | 0.096 | 0.111 |

  - **Suite at model_5999** (ratio, survival):

    | Category | Ratio | Survival |
    |---|---|---|
    | fwd | 0.691 | 0.97 |
    | back | 0.715 | 0.96 |
    | side | 0.437 | 0.98 |
    | diag (descriptive) | 0.546 | 0.99 |

  - **Other reads:** median |μ| 7.59 / 9.15; jitter 0.096; fall 0.02. The C0 lineage freezes on the FB pool too.
  - **Next:** FB6k__B_s16 launched at 02:24.
- 2026-10-04 04:57, **FB6k__B_s16 = FAST / FAST at model_5999** (no extension).
  - **Per-hand feet:**

    | Checkpoint | L | R |
    |---|---|---|
    | model_3000 | FAST 0.197 | INT 0.131 |
    | model_4500 | FAST 0.210 | FAST 0.195 |
    | head | FAST 0.203 | FAST 0.202 |

  - **Suite at model_5999** (ratio, survival), against the same seed's C0:

    | Category | B ratio | B survival | C0_s16 ratio |
    |---|---|---|---|
    | fwd | 0.996 | 1.00 | 0.691 |
    | back | 0.956 | 1.00 | 0.715 |
    | side | 0.856 | 1.00 | 0.437 |
    | diag (descriptive, untrained) | 0.917 | 1.00 | 0.546 |

  - **Other reads:** dose median |μ| 0.54 / 0.88 (OK); fall 0.000; jitter 0.152 / fft 1.42; rod odd RMS 1.60 / 1.85.
  - **Live checks:** wiring PASS. track_1000 was 0.451, close to the advisory DEAD line of 0.40, which the §11.3 advisory covers.
- 2026-10-04 07:29, **FB6k__C0_s17 = SLOW / SLOW.**
  - **Per-hand feet:**

    | Checkpoint | L | R |
    |---|---|---|
    | model_3000 | 0.107 | 0.085 |
    | model_4500 | 0.108 | 0.080 |
    | head | 0.106 | 0.085 |

  - **Suite at model_5999** (ratio, survival):

    | Category | Ratio | Survival |
    |---|---|---|
    | fwd | 0.579 | 0.80 |
    | back | 0.605 | 0.84 |
    | side | 0.443 | 0.93 |
    | diag | 0.483 | 0.83 |

  - **Other reads:** median |μ| 6.84 / 9.22; jitter 0.106.
  - **Extension:** no C0 hand rose ≥ 0.03, so this run is extended only if B_s17 is.
  - **Next:** FB6k__B_s17 is running.
- 2026-10-04 10:02, **FB6k__B_s17 = FAST / FAST at model_5999** (no extension).
  - **Per-hand feet:**

    | Checkpoint | L | R |
    |---|---|---|
    | model_3000 | INT 0.161 | INT 0.149 |
    | model_4500 | FAST 0.178 | FAST 0.232 |
    | head | FAST 0.203 | FAST 0.198 |

  - **Suite at model_5999** (ratio, survival), against C0_s17:

    | Category | B ratio | B survival | C0_s17 ratio |
    |---|---|---|---|
    | fwd | 0.933 | 1.00 | 0.579 |
    | back | 0.917 | 1.00 | 0.605 |
    | side | 0.859 | 1.00 | 0.443 |
    | diag | 0.822 | 0.99 | 0.483 |

  - **Other reads:** dose 0.52 / 0.88 (OK); fall 0.000; jitter 0.152.
  - **Count:** B is 2/2 counted FAST/FAST, so the gate's B criterion (≥ 2/3) is met. PASS still needs C0_s18 not FAST/FAST, no B trip, and fwd/back to hold over three seeds.
- 2026-10-04 12:34, **FB6k__B_s18 = FAST / FAST at model_5999** (no extension).
  - **Per-hand feet:**

    | Checkpoint | L | R |
    |---|---|---|
    | model_3000 | FAST 0.181 | SLOW 0.117 |
    | model_4500 | FAST 0.211 | INT 0.172 |
    | head | FAST 0.226 | FAST 0.187 |

  - **Suite at model_5999** (ratio, survival 1.00 everywhere):

    | Category | Ratio |
    |---|---|
    | fwd | 0.968 |
    | back | 0.933 |
    | side | 0.874 |
    | diag | 0.964 |

  - **Other reads:** dose 0.51 / 0.84 (OK); fall 0.000; jitter 0.157.
  - **Count:** B is 3/3 counted FAST/FAST at 6000 from scratch, so no B extension.
  - **Gate:** the outcome now depends only on FB6k__C0_s18 (running). PASS if it is not FAST/FAST, with no B trip and fwd/back holding.
- 2026-10-04 15:08, **FB6k complete; the pre-registered §11.4 gate reads PASS** (`fb6k/fb6k_read.{json,md}`; records in `fb6k/gate/`).
  - **FB6k__C0_s18:** SLOW / SLOW (0.101 / 0.098); suite fwd 0.645, back 0.687, side 0.450, diag 0.582; |μ| 8.34 / 9.42.
  - **No extensions triggered.** The three B runs ended FAST/FAST at model_5999, and no C0 hand rose ≥ 0.03.
  - **Counts:** B counted FAST/FAST 3/3 (0.203 / 0.202, 0.203 / 0.198, 0.226 / 0.187); C0 0/3.
  - **Tripwire:** reference 0.020, threshold 0.15; no trip.
  - **fwd / back:** B mean 0.966 / 0.935 vs C0 0.638 / 0.669, so both hold.
  - **diag (descriptive):** B 0.82–0.96 vs C0 0.48–0.58.
  - **INC0 gate:** 0/3, "ACTION_RATE ALONE INSUFFICIENT".
  - **Pending:** the §11.6 independent check (workflow `wf_a570110e-2aa`). P1 is the user's call.
- 2026-10-04 16:30, **FB6k + INC0 closed.** The §11.6 independent check (fable·high, `fb6k/CHECK.md`) PASSES. It reproduces every number and both gates (FB6k PASS; INC0 0/3, dump 0.0).
  - **Caveats carried:**
    - 3 v 3 is a two-sided Fisher p of 0.10;
    - B's jitter is 1.4–1.6× C0's (fft equal, so stroke rather than chatter);
    - clock-visible here, so the clock-free target is untested;
    - the priv_reg ramp coincides with the second-hand crossings;
    - B's survival on the untrained move_turn holds collapses (0.00–0.48 vs C0 0.00–0.82).
  - **diag** is genuine diagonal travel (91–100 % of path windows within 20°).
  - **INC0:** action_rate 0 drives the first hand (3/3 runs with one FAST hand, against C0 0/16); the construction adds the second hand and cuts jitter from 0.36–0.60 to 0.15.
  - **Write-up and design:** `fb6k/RESULTS.md`; the P1-B6 design by opus·high is in `fb6k/p1_design.md` (≈ 45 GPU-h, with a t3 stop).
  - **P1 is the user's call.**
- 2026-10-04 (pre-launch), **P1-B6 registered (§12):** the user approved FB6k next-option 1 ("Run 1"). Pre-launch it got:
  - candidates_v6 (dry run clean; the step-3 verdict is unchanged under v6);
  - `p1_read.py` (reader, t3 gate, `--c6`; self-test);
  - `p1_queue.sh`;
  - run_arm `--lr-eval`, `--row-eval`, `STOP_ON_REF_STOP` and the pinned resume read;
  - 53 tooling tests;
  - a fable/opus pre-launch review, whose fixes are applied (§12.6).

  The queue launches now: the t3 block first.
- 2026-10-04 19:03, **P1B6__C0_t3_s16 = SLOW / SLOW at the t3 head.**
  - **Feet:** model_4500 0.114 / 0.100; head 0.119 / 0.100; fall 0.033.
  - **t3 suite** (clock visible; descriptive):

    | Category | Read |
    |---|---|
    | fwd | 0.699 |
    | back | 0.746 |
    | diag | 0.589 |
    | side | 0.512 |
    | turn facing_closed / yaw_progress | 0.388 / 0.050 |
    | move_turn survival / ratio | 0.465 / 0.453 |

  - **Next:** P1B6__B_t3_s16 is running.
- 2026-10-04 21:32, **P1B6__B_t3_s16 = L INT / R SLOW at the t3 head**, so no FAST hand.
  - **Feet:** model_4500 SLOW / SLOW 0.124 / 0.108; head 0.147 / 0.127 (rising); fall 0.000; dose 0.52 / 0.83 (OK); wiring PASS.
  - **t3 suite against C0_s16:**

    | Category | B | C0_s16 |
    |---|---|---|
    | fwd | 0.889 | 0.699 |
    | back | 0.796 | 0.746 |
    | diag | 0.676 | 0.589 |
    | side | 0.632 | 0.512 |
    | turn facing_closed | **0.038** | 0.388 |
    | turn yaw_progress | **0.003** | 0.050 |
    | move_turn survival | 0.09 | 0.465 |

  - **Descriptive reading:** with turn at 0.2 in the t3 pool, B's side stroke had not formed by 6000 at this seed, and B barely turns at all.
  - **Gate:** the §12.4 gate is read after all six t3 runs. This seed counts toward "weaker hand below INT".
