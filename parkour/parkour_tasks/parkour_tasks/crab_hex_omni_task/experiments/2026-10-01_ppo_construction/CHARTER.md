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
- **Thresholds:**
  - `PPO/mu_med_{cam,rod}` ≤ 1.5;
  - `PPO/sat15_{cam,rod}` ≤ 0.1.
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

To be filled in at landing: code_sha, eval_sha and the commit.

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
