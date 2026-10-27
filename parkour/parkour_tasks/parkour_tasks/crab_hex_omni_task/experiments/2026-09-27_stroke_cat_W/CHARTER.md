# Per-category stroke weights (S_cat): the W fine-tune

Phase B of the per-category plan: one `--reference-chain S_cat` arm, scored offline against the
stroke_posture W-screen's own B pair. No fresh baseline, no candidate queue.

- **Plan of record:** `/home/nickmagus/.claude/plans/per-category-stroke-weights.md` (Context,
  Design, Phase A steps 1-4, Phase B steps 5-8; approved 2026-09-27, "run every step without
  asking").
- **The machine-readable spec:** [`candidates.yaml`](candidates.yaml) -- chain `c6y` (verbatim, never
  run), chain `c6y_scat` (the one arm, run only via `--reference-chain S_cat`).
- **The offline replay gate:** [`replay_gate/REPLAY_GATE_CAT.md`](replay_gate/REPLAY_GATE_CAT.md)
  (Phase A step 4).
- **The dose-smoke script:** [`stroke_cat_dose_smoke.py`](stroke_cat_dose_smoke.py).

## The question

On W (the clock-free omni head of record), does a per-CATEGORY loaded stance-stroke income --
each category dosed to 10 % of its own tracking income, fwd = back tied, move_turn held at the old
single-weight dose -- land its dose, lengthen side/diag steps beyond band, and clear every block and
guard against the reused single-weight-screen baseline B, for a further 4000 iterations forked from
W?

## Decisions 1-7 (settled, plan Context)

1. **Separate the stroke weights per command type.** `KRABBY_OMNI_STROKE_CAT_W` holds one multiplier
   per category on top of `KRABBY_OMNI_STROKE_W`.
2. **Test in BOTH settings.** The W fine-tune (clock-free, this campaign) first; t3 formation
   (clock-held) second, in a sibling campaign, only after this one closes.
3. **Equal share.** Each category's delivered stroke income is set to 10 % of its own tracking
   income (the replay solver's R4 bind, per category).
4. **Reuse the earlier baseline pair, no fresh B.** W's controls are the stroke_posture W-screen's
   own B s3/s2 (paths below). t3's controls (Phase C, not this campaign) are the formation B_f pair.
   Both pairs ran at eval_sha `36e5a8af7000` / code_sha `709161927627`.
5. **W s3 pilot first.** W s2 and Phase C run automatically only if S_cat s3's side `stride_along`
   rises against B s3 (point delta > 0). Otherwise STOP-and-ask.
6. **A speed-driven stride gain counts as success.** The gate is side `stride_along`.
   `stride_per_speed`, cadence and taps label the mechanism only (faster walking vs longer steps at
   the same speed); they do not gate.
7. **move_turn is held at the old single-weight dose** (added after the audit). `m_move_turn = 1`, so
   move_turn's effective weight is `STROKE_W` -- the same dose the single-weight S screen already ran
   at (≈ 0.117 on W). Equal share applies to every other category. The first screen's S arm read
   `F_mt` WORSE at that dose; ≈ 5x would be a known risk the audit flagged and this decision avoids.

## The reused controls (decision 4)

| | xid | stamp | code_sha | eval_sha |
|---|---|---|---|---|
| B s3 | X01 | `parkour/logs/rsl_rl/gait_eval/omni_phases/2026-09-26_stroke_posture/seed3_X01_c6y_A15pB/suite__omni/seed001/2026-09-26_08-39-30` | `709161927627…` | `36e5a8af7000…` |
| B s2 | X02 | `parkour/logs/rsl_rl/gait_eval/omni_phases/2026-09-26_stroke_posture/seed2_X02_c6y_A15pB/suite__omni/seed001/2026-09-26_10-25-08` | `709161927627…` | `36e5a8af7000…` |

Both are the stroke_posture campaign's own baseline (`chain: c6y`, both knobs at eps) -- the SAME
chain this campaign's `candidates.yaml` keeps verbatim but never runs. `S_cat` is judged against them
directly; this campaign records no new baseline.

## The dose rule and the step-6 final arithmetic

**Replay weights (10 % delivered share, W economy, hand-min, fwd = back tied)** --
`replay_gate/REPLAY_GATE_CAT.md` section 1:

| Category | fwd | back | side | diag | turn | move_turn |
|---|---|---|---|---|---|---|
| solved w_c (replay scale) | 0.08647 | 0.08647 (tied) | 0.6147 | 0.1593 | 0.1533 | not solved (m = 1) |

**Knob values at replay scale** (PROVISIONAL, already written into `candidates.yaml`'s `c6y_scat`
overlay):
- `KRABBY_OMNI_STROKE_W = 0.08646779396708197`
- `KRABBY_OMNI_STROKE_CAT_W = fwd:1.0,back:1.0,side:7.109225977086123,diag:1.8426117426622708,turn:1.7727771075114038,move_turn:1.0`

**The knob stores MULTIPLIERS on `STROKE_W`, so the dose smoke itself runs at these replay-scale
values** (plan Phase B step 6). After the smoke measures r_c per category, the FINAL (training-scale)
table is:

- `STROKE_W = w_fwd_rep / r_fwd`
- `m_c = (w_c_rep / r_c) / STROKE_W` for **side** and **diag**
- `m_back = 1` (the fwd = back tie)
- `m_move_turn = 1` (decision 7)
- `m_turn = w_turn_rep / STROKE_W` (turn is replay-set, not r-corrected: training turn holds reach
  their target and ungate the term, so r_turn is unreliable -- Design, plan Context)

Before landing, **assert every effective weight `STROKE_W x m_c` against the target table to within
1 %.** Fill the final table into `candidates.yaml`'s `c6y_scat` overlay before the first
`--reference-chain S_cat` launch.

**The smoke rule.** r_c = delivered (`Episode_RewardCat`, post-clip, last 200 its) / replay-predicted
(the term priced at the smoke doses on B's own raw npz). Rescale fwd (and back via the tie) and side,
diag by `1/r_c` **only when r_c is in [0.5, 2]**. turn stays replay-set; move_turn follows fwd
(decision 7). **Any r_c outside [0.5, 2] on a row that sets a dose: STOP-and-ask** with the numbers
and the stance-length histogram (mirrors stroke_posture's SA3).

## The knob smoke (driver overlay audit)

Before landing the final table, run the driver's own overlay audit on the composed stage, not just
the by-hand dose smoke:

```
run_candidates.py --campaign-dir <this dir> --smoke-stage 1a:20 \
    --smoke-overlay KRABBY_OMNI_STROKE_W=<final> \
    --smoke-overlay KRABBY_OMNI_STROKE_CAT_W=<final table>
```

- `--smoke-stage 1a:20` composes ONLY the smoke-overlay keys on bare preset `1a` (20 iterations,
  minutes of GPU) -- it does not compose the full `c6y_scat` chain overlay.
- **Require every `overlay_checks` entry True** against the real `params/env.yaml` it dumps. A False
  halts the reference run as `config_mismatch` only AFTER training, and that halt is not resumable --
  the smoke exists so this is caught before the 1.8 GPU-h reference run starts.
- **It records `candidates_sha256`** (the yaml's hash at smoke time). Any later yaml edit therefore
  needs `--accept-candidates-change` on the next launch that reads it (plan item 4).

## The reuse control

**S_cat's composed stage env and its knob-ON config dump must differ from old B's recorded stage env
and `params/env.yaml` ONLY in:**
- `reward_stride_length`'s weight (the func-swap weight, `STROKE_W` in place of B's eps `0.001`);
- its `cat_scale` params (present on S_cat, absent -- no `cat_scale` key at all -- on B, which sets no
  `KRABBY_OMNI_STROKE_CAT_W`).

Any other difference between the two composed envs or the two dumps is a **STOP-and-ask**: it would
mean S_cat is not "B plus the per-category stroke term," and the reused B pair would no longer be a
valid control. `stroke_cat_dose_smoke.py --print-env`'s composed env is the first place to check this
(diff it against B's recorded stage env by hand before the GPU smoke).

## The pilot rule (decision 5)

Compare **S_cat s3's side `stride_along` episode MEAN** against **B s3's 0.0964** (the stroke_posture
W-screen's own B s3 value).

- **Point delta > 0** -> launch `--reference-chain S_cat --seed 2`, then close (Phase C, t3, is a
  separate sibling campaign and starts only after this one's step-8 gate, whatever its verdict, per
  the plan's Phase C header -- "runs after step 8 whatever its verdict, unless a STOP fired").
- **Otherwise: STOP-and-ask**, showing the per-seed readout for BOTH B seeds (B s3's 0.0964 and B
  s2's own value) beside S_cat s3's, so the user can read the pilot against the seed spread, not just
  the single B s3 point.

## The iteration-~500 overdose stop (plan Phase B step 7)

At S_cat s3 iteration ~500, compute **each translation row's (fwd, side, back, diag) logged stroke
share of its tracking income**:

- share = `Episode_RewardCat/<cat>/reward_stride_length` / tracking income on that row, **as a ratio
  of MEANS over >= 50 iterations** (never a mean of per-iteration ratios -- the plan's dose-arithmetic
  hardening, audit pass 1).
- **Tracking income is the replay's own definition**: `track_lin_vel_xy_exp` (translation categories
  only; turn and move_turn are excluded from this check -- they are replay-set / held at m = 1, not
  dosed to a fresh 10 % target here).
- **STOP-and-ask if any row exceeds 15 %** (1.5x the 10 % target) -- **without killing the run.** The
  training process keeps going while the user is asked; this is a share check on the running log, not
  a live abort.

## The pre-registered step-8 gate (verbatim from the plan)

Pooled over s3 and s2, S_cat minus the reused B at the same seed. Band = max(pooled bootstrap MDE,
|B s3 - B s2|) per key.

- **Dose landed (lower bound only, as in the stroke_posture charter):** fwd and side logged income
  >= 0.5 x the replay prediction at the final table on B's raws, at both seeds.
  - Income ABOVE the prediction is stroke change, never a failure.
  - Overdose is the iteration-~500 share stop above, not a step-8 criterion.
  - **A landed FAIL means "no reading": STOP-and-ask, never a stride FAIL.** The dose-landed check and
    the stride check are independent; a failed landed check does not get scored as a failed stride
    check, because a dose that never arrived tells you nothing about whether longer steps would have
    followed one that did.
- **Stride (decision 6):** pooled side `stride_along` delta > band.
- **Mechanism label, reported and not gating:** pooled delta of side achieved speed,
  `stride_per_speed` (a MEAN key), `cadence_ratio`, touchdown rate and taps, each against its own
  band. The label reads "faster walking" or "longer steps at the same speed."
- **Health:**
  - moving ratios >= 0.30 on fwd, side and back at both seeds;
  - no floor WATCH trip;
  - unison clear under B's seed band;
  - **the pooled F and F_mt blocks not WORSE**, scored OFFLINE by `score_reference.py` (below) -- a
    CPU script that calls the driver's own `decide_pooled` on the reference runs' screen records
    against the old B screens, with the old campaign's D00s/D00bs. It also reports T, LEN and the 23
    guards (the full `decision_rule`, not just F/F_mt -- those two are what step 8 calls out as the
    binding health criterion; the others are read the same way).
- **Reported, not gating:**
  - side `stance_stroke`, `stroke_share` and `body_offset`;
  - LEN;
  - each category's delivered share.
- **Confound, stated:** the controls ran a day earlier under code_sha `709161927627`. Step 6's
  identity checks (the reuse control above) are the control for this confound -- they prove the ONLY
  composed-env difference between S_cat and B is the two per-category knobs, so a day of unrelated
  code drift on the same pathspecs is what the confound statement is warning about, not a design
  difference.
- **A PASS goes to the user for adoption; nothing is baked here.**

## `score_reference.py`

**Inputs** (module docstring, verified against the CLI below):
- `--old-campaign` = `experiments/2026-09-26_stroke_posture` -- supplies the reused controls'
  headers, the decision rule, `resolution.36e5a8af7000.json` (D00s), and (via
  `run_candidates.seed_terms_for` on a minimal ctx stub built from the OLD ledger/candidates at the
  HISTORICAL eval_sha) the seed terms D00bs / D00c.
- `--base-xids X01 X02` -- the reused B s3/s2 headers, read from the old campaign.
- `--new-campaign` = `experiments/2026-09-27_stroke_cat_W` (this dir) -- holds the `S_cat` reference
  runs, read from `ladder.references["S_cat"]` (`--reference S_cat`, not `--cand-xids`: the CLI takes
  a **reference name**, not a `--cand-campaign` flag -- checked directly against `score_reference.py
  --help` / its argparse definition, not assumed).
- `--cand-spec S` -- the OLD campaign's candidate id `S` (only its `absolute_units` and
  `speculative_default` fields are read; neither S_cat nor this campaign needs its own candidate
  entry for this).
- `--seeds 3 2` (default; the two `S_cat` reference seeds).

**The exact command the gate runs** (also given verbatim in the script's own module docstring, "The
per-category plan's W reading"):

```
score_reference.py --old-campaign experiments/2026-09-26_stroke_posture --base-xids X01 X02 \
    --new-campaign experiments/2026-09-27_stroke_cat_W --reference S_cat --cand-spec S \
    --json <out>.json
```

Every header's `eval_sha` must equal the old campaign's (`36e5a8af7000…`); `code_sha` is REPORTED,
never asserted -- the reference runs train under the per-category knob's code, so theirs differs from
the controls' by construction (the confound statement above; the reuse control is what stands in for
that check).

## The 4 replay-gate FAILs and the user's decision

Phase A step 4's offline replay gate (`replay_gate/REPLAY_GATE_CAT.md`) found **4 restated-criterion
FAILs** across both economies (W and t3); the stride_per_speed band passed in both. Summary (full
detail and every number: the linked file):

1. **t3 R5 in-plant no-flip** (Bf_s3 vs the E7 creeper, move_turn): depends on the table. t3-only.
2. **t3 R8 hand gap:** E3 turn hand gap CI excludes 0. Scale-invariant; heads are new.
3. **t3 R3 tap test (side):** negative on all three t3 heads. Scale-invariant; heads are new.
4. **W R3 tap test on E1** (X08:c6, a healthy stamp): CI excludes 0; W itself PASSES. Scale-invariant,
   identical to the earlier p15 gate. Whether it counts depends on the clause's scope (W only, or
   every healthy head).

**User decision, 2026-09-28: "Proceed as planned."** Both t3-only findings (1, 3) have no training
effect on THIS campaign (move_turn is not in t3's pool; W is a different economy). Findings 2 and 4
are weight-table-invariant properties of heads that already exist and already trained at these
settings under the single-weight screens. Full record: [`replay_gate/REPLAY_GATE_CAT.md`](replay_gate/REPLAY_GATE_CAT.md).

## Budget

| Item | GPU-h |
|---|---|
| Window 0 (config identity + dose smoke) | ~0.3 |
| S_cat s3 | ~1.8 |
| S_cat s2 (if the pilot passes) | ~1.8 |
| **This campaign's total** | **~3.9** |

Phase C (t3, a sibling campaign) is out of this budget and out of this file's scope.

## Log

- 2026-09-28: `candidates.yaml` written (Phase B step 5); `--dry-run` clean (validator all ok; the
  only notes are the two D00s resolution inputs, absent because they live under the main checkout's
  `parkour/logs`, not this worktree). This CHARTER and `stroke_cat_dose_smoke.py` written alongside it
  (Phase B steps 5-6 tooling). Never launched from this worktree (CHARTER's own rule, inherited from
  stroke_posture: D00s's inputs and the launch itself belong to the main checkout).
