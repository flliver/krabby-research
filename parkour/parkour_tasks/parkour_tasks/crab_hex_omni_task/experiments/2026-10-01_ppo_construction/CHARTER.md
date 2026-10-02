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
