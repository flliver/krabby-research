# Per-category stroke weights (S_cat): the t3 formation (clock-held) leg of the plan.

Phase C of the per-category plan: one `--reference-chain S_cat_f` arm, scored offline against the
stroke_formation t3 gate's own B_f pair. No fresh baseline, no candidate queue -- the same shape as
Phase B's W campaign (`../2026-09-27_stroke_cat_W/CHARTER.md`), applied to t3's clock-held,
from-scratch setting.

- **Plan of record:** `/home/nickmagus/.claude/plans/per-category-stroke-weights.md` (Context, Design,
  Phase A steps 1-4, Phase C steps 9-10; approved 2026-09-27, "run every step without asking"; the
  2026-09-28 reboot STATUS block records the user's override -- "Run W s2 and t3 as planned" -- after
  the W s3 pilot read FAIL).
- **The machine-readable spec:** [`candidates.yaml`](candidates.yaml) -- chain `t3f` (verbatim from
  `../2026-09-26_stroke_formation/candidates.yaml`, never run), chain `t3f_scat` (the one arm, run only
  via `--reference-chain S_cat_f`).
- **The offline replay gate (shared with W):**
  [`../2026-09-27_stroke_cat_W/replay_gate/REPLAY_GATE_CAT.md`](../2026-09-27_stroke_cat_W/replay_gate/REPLAY_GATE_CAT.md)
  (Phase A step 4 ran once, both economies; its t3-economy table is this campaign's provisional dose).
- **The dose-smoke script:** [`stroke_cat_t3_dose_smoke.py`](stroke_cat_t3_dose_smoke.py).

## The question

At t3 (C1d's turn-first curriculum, clock HELD, trained from scratch), does a per-CATEGORY loaded
stance-stroke income -- fwd/back/side dosed and rescaled to 10 % of their own tracking income (turn and
diag replay-set; move_turn held at the old single-weight dose) -- land its dose, lengthen side steps
beyond band, and clear every gated block and guard against the reused single-weight-screen baseline
B_f, over a full 3000-iteration formation run from scratch?

## Decisions 1-7 (settled, plan Context) and the 2026-09-28 override

1. **Separate the stroke weights per command type.** `KRABBY_OMNI_STROKE_CAT_W` holds one multiplier
   per category on top of `KRABBY_OMNI_STROKE_W`.
2. **Test in BOTH settings.** The W fine-tune (clock-free) first; t3 formation (clock-held, this
   campaign) second, only after Phase B's step 8 gate closes -- "runs after step 8 whatever its
   verdict, unless a STOP fired" (plan Phase C header).
3. **Equal share.** Each category's delivered stroke income is set to 10 % of its own tracking income
   (the replay solver's R4 bind, per category).
4. **Reuse the earlier baseline pair, no fresh B_f.** t3's controls are the stroke_formation t3 gate's
   own B_f s3/s2 (paths below). Both ran at eval_sha `36e5a8af7000` / code_sha `709161927627`.
5. *(W-only.)* The W s3 pilot gate (side `stride_along` rising against B s3) has no t3 analogue: the
   plan does not gate t3's s2 launch behind a pilot read -- both seeds always run.
6. **A speed-driven stride gain counts as success.** The gate is side `stride_along`.
   `stride_per_speed`, cadence and taps label the mechanism only; they do not gate.
7. **move_turn is held at the old single-weight dose** (added after the audit). `m_move_turn = 1`, so
   move_turn's effective weight is `STROKE_W` -- ≈ 0.136 at t3's replay scale, ≈ 0.119 training scale
   (the formation screen's own single-weight w_S). move_turn is not in t3's training pool regardless
   (t3f's `categories` list has no `move_turn`), so this weight is inert in training here; it is kept
   only for arithmetic parity with W's table.
8. **2026-09-28 user override (reboot STATUS block).** The W s3 pilot (decision 5) read FAIL: shorter
   side stride (`experiments/2026-09-27_stroke_cat_W/pilot/PILOT.md`). The plan's default on a pilot
   FAIL is to close the line and never launch W s2 or Phase C. **The user overruled that default**:
   "Run W s2 and t3 as planned" -- W s2 and this campaign (Phase C) both run regardless of the W s3
   pilot's own verdict. Phase C's own gate (step 10, below) is unaffected: it still reads t3's own
   dose-landed and stride numbers on their own terms, not on W's pilot result.

## The reused controls (decision 4)

| | xid | stamp | code_sha | eval_sha |
|---|---|---|---|---|
| B_f s3 | X01 | `parkour/logs/rsl_rl/gait_eval/omni_phases/2026-09-26_stroke_formation/seed3_X01_t3_A15pB/suite__omni/seed001/2026-09-26_22-56-51` | `709161927627…` | `36e5a8af7000…` |
| B_f s2 | X02 | `parkour/logs/rsl_rl/gait_eval/omni_phases/2026-09-26_stroke_formation/seed2_X02_t3_A15pB/suite__omni/seed001/2026-09-27_04-08-09` | `709161927627…` | `36e5a8af7000…` |

Both are the stroke_formation campaign's own baseline (`chain: t3f`, both knobs at eps) -- the SAME
chain this campaign's `candidates.yaml` keeps verbatim but never runs. `S_cat_f` is judged against them
directly; this campaign records no new baseline. (`heads.json` in that campaign confirms X01 = B_f s3,
X02 = B_f s2.)

**Because the reference is a from-scratch run, it is exempt from the dead_walker/dead_plant checks the
six B_f/S_f/P_f candidate screens in the old campaign were each subject to.** `run_candidates.py`'s
from-scratch early abort and its `dead_plant` soundness check both arm only for `item["kind"] ==
"candidate"`, never `"reference"` (the code comment there: "a REFERENCE chain is exempt from the
dead_plant soundness check as it is from the dead_walker abort -- its formation readout must never be
[cut]"). `t3f_scat` trains from scratch exactly as `t3f` did, but as a **reference** run, so no
mid-training abort can cut it short the way it could have cut a candidate's s3 line. This is a known
asymmetry between how B_f/S_f/P_f were screened and how S_cat_f is screened -- noted, not remedied (the
plan's own choice: a reference chain records its head and parks, with no DECISION and no baseline
queue).

## The dose rule and the step-9 final arithmetic

**Replay weights (10 % delivered share, t3 economy, hand-min, fwd = back tied)** --
`../2026-09-27_stroke_cat_W/replay_gate/REPLAY_GATE_CAT.md` section 1, t3 economy table:

| Category | fwd | back | side | diag | turn | move_turn |
|---|---|---|---|---|---|---|
| solved w_c (replay scale) | 0.1362 | 0.1362 (tied) | 0.5653 | 0.1796 | 0.2377 | not solved (m = 1) |

**Knob values at replay scale** (PROVISIONAL, already written into `candidates.yaml`'s `t3f_scat`
overlay):
- `KRABBY_OMNI_STROKE_W = 0.1362210975498265`
- `KRABBY_OMNI_STROKE_CAT_W = fwd:1.0,back:1.0,side:4.150041141570944,diag:1.318716993366581,turn:1.7449369222400448,move_turn:1.0`

**The knob stores MULTIPLIERS on `STROKE_W`, so the dose smoke itself runs at these replay-scale
values** (plan Phase C step 9, mirroring Phase B step 6). After the smoke measures r_c per category,
the FINAL (training-scale) table is:

- `STROKE_W = w_fwd_rep / r_fwd`
- `m_side = (w_side_rep / r_side) / STROKE_W`
- `m_back = 1` (the fwd = back tie)
- `m_move_turn = 1` (decision 7)
- `m_turn = w_turn_rep / STROKE_W` (turn is replay-set, not r-corrected: even though turn IS in t3's
  training pool -- `t3f`'s `categories` includes `turn` -- training turn holds reach their target and
  ungate the term the same way they did on W, so r_turn is unreliable; the formation campaign's own
  t3 dose smoke already measured r_turn ≈ 0.42 by this mechanism -- Design, plan Context)
- `m_diag = w_diag_rep / STROKE_W` (diag is replay-set for a different reason: diag is **not** in t3's
  training pool at all -- `t3f`'s `categories` list has no `diag` -- so the term is simply inert on
  diag in training; there is no r_diag to measure)

**Only fwd and side are rescaled by a measured r_c.** back and move_turn are held at multiplier 1; turn
and diag are set directly from the replay table divided by `STROKE_W`, with no r-correction, for the
two distinct reasons above.

Before landing, **assert every effective weight `STROKE_W x m_c` against the target table to within
1 %.** Fill the final table into `candidates.yaml`'s `t3f_scat` overlay before the first
`--reference-chain S_cat_f` launch.

**The smoke rule.** r_c = delivered (`Episode_RewardCat`, post-clip, last 200 its) / replay-predicted
(the term priced at the smoke doses on B_f's own raw npz, in B_f's own clock-held `params/env.yaml`
economy -- the dose smoke's method, as the formation campaign's own t3 dose smoke used). Rescale fwd
(and back via the tie) and side by `1/r_c` **only when r_c is in [0.5, 2]**. turn and diag stay
replay-set; move_turn follows fwd (decision 7). **Any r_c outside [0.5, 2] on a row that sets a dose
(fwd or side): STOP-and-ask** with the numbers and the stance-length histogram (mirrors stroke_posture's
SA3 and W's step 6).

## The knob smoke (driver overlay audit)

Before landing the final table, run the driver's own overlay audit on the composed stage, not just the
by-hand dose smoke:

```
run_candidates.py --campaign-dir <this dir> --smoke-stage 1a:20 \
    --smoke-overlay KRABBY_OMNI_STROKE_W=<final> \
    --smoke-overlay KRABBY_OMNI_STROKE_CAT_W=<final table>
```

- `--smoke-stage 1a:20` composes ONLY the smoke-overlay keys on bare preset `1a` (20 iterations,
  minutes of GPU) -- it does not compose the full `t3f_scat` chain overlay. Run it on a SCRATCH COPY of
  this campaign's yaml (mirrors W's step-6 practice), so its drill records stay out of this campaign.
- **Require every `overlay_checks` entry True** against the real `params/env.yaml` it dumps. A False
  halts the reference run as `config_mismatch` only AFTER training, and that halt is not resumable --
  the smoke exists so this is caught before the ≈1.37 GPU-h reference run starts.
- **It records `candidates_sha256`** (the yaml's hash at smoke time). Any later yaml edit therefore
  needs `--accept-candidates-change` on the next launch that reads it (plan item 4).

## The reuse control

**S_cat_f's composed stage env and its knob-ON config dump must differ from old B_f's recorded stage
env and `params/env.yaml` ONLY in:**
- `reward_stride_length`'s weight (the func-swap weight, `STROKE_W` in place of B_f's eps `0.001`);
- its `cat_scale` params (present on S_cat_f, absent -- no `cat_scale` key at all -- on B_f, which sets
  no `KRABBY_OMNI_STROKE_CAT_W`).

Any other difference between the two composed envs or the two dumps is a **STOP-and-ask**: it would
mean S_cat_f is not "B_f plus the per-category stroke term," and the reused B_f pair would no longer be
a valid control. `stroke_cat_t3_dose_smoke.py --print-env`'s composed env is the first place to check
this (diff it against B_f's recorded stage env by hand before the GPU smoke).

## The pre-registered step-10 gate

Step 8's rule (the W leg's gate), adapted to t3 with the band floor below -- pooled over s3 and s2,
S_cat_f minus the reused B_f at the same seed. Band = max(pooled bootstrap MDE, |B_f s3 − B_f s2|) per
key.

- **Dose landed (lower bound only, as in the stroke_posture/W charters):** fwd and side logged income
  >= 0.5 x the replay prediction at the final table on B_f's raws, at both seeds.
  - Income ABOVE the prediction is stroke change, never a failure.
  - **A landed FAIL means "no reading": STOP-and-ask, never a stride FAIL.** The dose-landed check and
    the stride check are independent; a failed landed check does not get scored as a failed stride
    check, because a dose that never arrived tells you nothing about whether longer steps would have
    followed one that did.
- **Stride (decision 6):** pooled side `stride_along` delta > band.
  - **Band floor (audit pass 3).** The B_f pair's side `stride_along` seed gap is degenerate (0.0800 vs
    0.0810, |Δ| 0.001), so its bootstrap-MDE band alone would fall to about 0.0023 -- roughly 1σ of the
    run-to-run spread, an unreasonably tight bar. **The t3 side `stride_along` band is therefore
    max(pooled MDE, 0.0049)** -- the largest seed gap on record for this key, W's own B pair's gap.
    B_f s3's weakness (per `GATE_t3.md`'s confound note) shows on the fwd and back rows (seed gaps
    0.048–0.068), not on this decisive key, so the floor is not compensating for a known-bad seed on
    the gating row itself.
- **Mechanism label, reported and not gating:** pooled Δ of side achieved speed, `stride_per_speed` (a
  MEAN key), `cadence_ratio`, touchdown rate and taps, each against its own band. The label reads
  "faster walking" or "longer steps at the same speed."
- **Health:**
  - moving ratios >= 0.30 on fwd, side and back at both seeds;
  - the t3 floor at both seeds (`category_survival`, `min_new 0.10`, `min_prev 0.30`), with no floor or
    dead_walker REJECT -- **though as noted above, S_cat_f as a reference run is structurally exempt
    from the dead_walker early abort and the dead_plant check that could have cut a candidate's s3
    line short; only the stage-eval floor read at the end of the full 3000 its applies here**;
  - unison clear under B_f's seed band (tripod coherence, roll_rms);
  - **the pooled F block not WORSE**, scored OFFLINE by `score_reference.py` (below) -- a CPU script
    that calls the driver's own `decide_pooled` on the reference runs' screen records against the old
    B_f screens, with the old campaign's D00s/D00bs. It also reports T, LEN and the 23 guards (the
    full `decision_rule`, not just F -- that is what step 10 calls out as the binding health
    criterion; the others are read the same way). **NOTE: unlike W's rule, the t3 rule has NO F_mt
    block** -- t3f trains no move_turn (it is not in `t3f`'s `categories`), so `decision_rule` here
    drops F_mt entirely (as it drops block RIG, for the reason given in `candidates.yaml`'s header
    comment). There is nothing for "F_mt not WORSE" to mean at t3.
- **Reported, not gating:**
  - side `stance_stroke`, `stroke_share` and `body_offset`;
  - LEN (at t3 the cadence is clock-locked, so LEN is reported only, as in the formation campaign's own
    gate);
  - each category's delivered share.
- **Confound, stated:** the controls ran a day earlier under code_sha `709161927627`. Step 9's identity
  checks (the reuse control above) are the control for this confound.
- **A PASS goes to the user for adoption; nothing is baked here.**

## The 15 % share report (no iteration-500 stop for t3)

W's step 7 has an iteration-~500 overdose STOP-and-ask (a live check on the running log, translation
rows only). **t3 has no equivalent mid-run stop** (plan "Left for implementation" item 17: "W s2 and t3
have no iteration-500 overdose check"). Instead, the step-10 gate's share report itself must **flag any
category whose delivered stroke share of its own tracking income exceeds 15 %** (1.5x the 10 % target),
computed the same way as W's stop would have been -- a ratio of means over the run's own logged window,
never a mean of per-iteration ratios -- but read once, at the end of the full run, not as a live
mid-training check. A flag here is reported alongside the gate, not a kill switch: the run has already
finished by the time it is read.

## `score_reference.py`

**Inputs** (module docstring, verified against the CLI directly -- `score_reference.py --help` /
its `argparse` definition, not assumed):
- `--old-campaign` = `experiments/2026-09-26_stroke_formation` -- supplies the reused controls'
  headers, the decision rule, `resolution.36e5a8af7000.json` (D00s), and (via
  `run_candidates.seed_terms_for` on a minimal ctx stub built from the OLD ledger/candidates at the
  HISTORICAL eval_sha) the seed terms D00bs / D00c.
- `--base-xids X01 X02` -- the reused B_f s3/s2 headers, read from the old campaign.
- `--new-campaign` = `experiments/2026-09-27_stroke_cat_t3` (this dir) -- holds the `S_cat_f`
  reference runs, read from `ladder.references["S_cat_f"]` (`--reference S_cat_f`, not
  `--cand-campaign`: the CLI takes a **reference name**, exactly as for W's `S_cat`).
- `--cand-spec S_f` -- the OLD campaign's candidate id `S_f` (the formation stroke arm; only its
  `absolute_units` and `speculative_default` fields are read; neither S_cat_f nor this campaign needs
  its own candidate entry for this).
- `--seeds 3 2` (default; the two `S_cat_f` reference seeds).

**The exact command the gate runs:**

```
score_reference.py --old-campaign experiments/2026-09-26_stroke_formation --base-xids X01 X02 \
    --new-campaign experiments/2026-09-27_stroke_cat_t3 --reference S_cat_f --cand-spec S_f \
    --json <out>.json
```

Every header's `eval_sha` must equal the old campaign's (`36e5a8af7000…`); `code_sha` is REPORTED,
never asserted -- the reference runs train under the per-category knob's code, so theirs differs from
the controls' by construction (the confound statement above; the reuse control is what stands in for
that check).

## Budget

| Item | GPU-h |
|---|---|
| t3 dose smoke | ~0.2 |
| S_cat_f s3 | ~1.37 |
| S_cat_f s2 | ~1.37 |
| **This campaign's total** | **~2.9** |

Matches the plan's Phase C budget row (t3 smoke ≈ 0.2; t3: 2 × 1.37 ≈ 2.7; ≈ 2.9 total).

## Log

- 2026-09-28: `candidates.yaml` written (Phase C step 9). `--dry-run` clean (validator all ok; the only
  notes are the two D00s resolution inputs, absent because they live under the main checkout's
  `parkour/logs`, not this worktree). This CHARTER and `stroke_cat_t3_dose_smoke.py` written alongside
  it. Never launched from this worktree (CHARTER's own rule, inherited from stroke_posture/W: D00s's
  inputs and the launch itself belong to the main checkout). The reference runs launch only after
  Phase B's S_cat s3/s2 finish (plan Phase C header).
