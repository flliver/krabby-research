# The facing/tracking candidate ladder — CHARTER and verification record

Approved plan: `/home/nickmagus/.claude/plans/the-facing-and-tracking-fancy-breeze.md` (user,
2026-09-18), "the facing/tracking candidate ladder — an autonomous, speculative campaign for
`crab_hex_omni_task`". Successor to the omni-formation campaign
(`experiments/2026-09-16_1805_omni_formation`), which is CLOSED at its 2b decision point; this
campaign never writes into that directory and never resumes one of its heads as a control (every
screen is trained from scratch at equal compute). The formation campaign's recorded 1c-004, 2a-reeval
and 2b eval rows are this campaign's RESOLUTION inputs and its M1 WATCH references — measurements,
not baselines.

Machine-readable specification: `candidates.yaml` (sha256
`7303819ea5cf6895d82f9018c99058fa4bd70819aca31e5f443764116c839027`, pinned in `state.json` at the
first launch; a changed hash is refused without `--accept-candidates-change <why>`). Driver:
`experiments/tools/run_candidates.py`; resolution: `experiments/tools/resolution.py`; user CLI:
`experiments/tools/ledger_cli.py`; launcher: `experiments/tools/launch_candidates.sh`. Nothing in
this CHARTER is read by a tool — `candidates.yaml` is, and the two must agree.

Abbreviations used below: `$R` = `/home/nickmagus/krabby/krabby-research`; `$OMNI` =
`$R/parkour/parkour_tasks/parkour_tasks/crab_hex_omni_task`; `$CAMP` =
`$OMNI/experiments/2026-09-16_1805_omni_formation`; `$LAD` = this directory
(`$OMNI/experiments/2026-09-19_facing_tracking_ladder`); `$TOOLS` = `$OMNI/experiments/tools`;
`$PY` = `/home/nickmagus/krabby/isaac_venv/bin/python`. Every path a tool reads is a CLI or yaml
input — no driver contains a dated path.

## Question

Every omni head since 1c-004 walks in every commanded world direction but **sacrifices facing for
translation**. On the seed-3 2b head of the formation campaign
(`crab_hex_omni_teacher/2026-09-18_12-58-58/model_14997.pt`): the `move_turn` hold ends with a
**1.218-rad** heading error and never settles; `turn_l90` ends at **0.68** rad; the turning rate sits
at **0.07–0.09 rad/s** under every head since 1c-004 (half the kinematic ceiling estimate
0.15–0.18 rad/s); per-direction speed ratios are **0.60 / 0.47 / 0.51 / 0.60** (fwd/lat_l/diag/back);
completion **0.50**; moving-hold walking tripod **0.38**, down from **0.49** at 1c-004.

Does a ranked ladder of candidate recipes — each screened once at equal compute against the recipe
the ladder has already adopted, adopted or rejected by a measured decision rule, speculatively when
the outcome is inside the eval resolution — improve facing AND translation together, and which of
the five established mechanisms is the binding one?

## The five mechanisms (with their numbers)

Established by `$CAMP/ANALYSIS_1c_failure_2026-09-18.md`, `$CAMP/replay_reward_economy.md`,
`experiments/lit-review-omnidirectional-gait-formation.md`, the three exploration reports of
2026-09-19, the design panel and its critics, and three audit passes.

| # | mechanism | the numbers |
|---|---|---|
| 1 | **Price** | The whole facing sub-economy spans **0.423/s**, and the *reachable* gain from turning at the plant's **0.08 rad/s** is **0.206/s** — the price of a 20 % speed loss on a **1.638/s** translation stake. The facing-POSITION term (`reward_facing_exp`, `KRABBY_YAW_W` 0.2) is **0.121/s** of that; **0.302/s** is the even `track_ang_vel_z_exp` kernel, which pays a robot that never turns. (**Provenance:** the five income rates 0.423 / 0.206 / 1.638 / 0.121 / 0.302 per second are **from the 2026-09-19 exploration, not reproducible from a committed record** — the exploration reports live in the session scratchpad, and a grep of `$CAMP/replay_reward_economy.md`, `$CAMP/ANALYSIS_1c_failure_2026-09-18.md` and the lit review finds none of them. They are the design notes' sizing of the sub-economy, not a measurement this repo can re-derive; the committed per-term income tables they were read off are in `$CAMP/replay_reward_economy.md`. The plant's **0.08 rad/s** IS recorded — `$CAMP/CHARTER.md` and the 2b decision table, 0.07–0.09 rad/s under every head since 1c-004.) Ignoring facing costs translation **nothing**: the command term re-projects a held world travel direction into the body frame every step (clamp-only re-cap), so world-frame tracking is heading-independent by construction. |
| 2 | **Kinematics** | `ang_vel_max` **0.3** is ≈2× the plant's estimated **0.15–0.18 rad/s** turn ceiling (lit review §3), and the `move_turn` slot (0.22 m/s + 0.3 rad/s ⇒ ≈**0.58 m/s** outer-foot speed at `r_foot` ≈ 1.2 m) is infeasible at ANY speed the sampler can command (walk floor **0.21 m/s** ⇒ ω ≤ **0.01 rad/s** while walking) — yet it is **20–25 %** of every window's rows. A policy asked for the impossible learns to drop the half it cannot deliver. |
| 3 | **Observation** | Steering slots 6/7 clip the facing error at **0.6 rad** and slot 5 (yaw-rate command) saturates at **0.2 rad** of error, so the policy is magnitude-blind above 0.6 rad and the **0.2-s** observation history cannot integrate a **12-s** turn. |
| 4 | **Curriculum** | The 1a → 1b → 1c hard switches lost translation at every handoff (fwd speed ratio **0.76 → 0.16**), the soft gates were blind to it, and move+turn was never taught incrementally; the RSI bank holds **no turn and no backward frame**. |
| 5 | **Economy** | The angular kernel pays ≈**1.0/s** for standing still on the ≈**88 %** of rows that command no turn; the staged `KRABBY_OMNI_ANG_TURN_ONLY` lever moves the replay's walker-vs-creeper margin **+16.6 % → +37.5 %** — but it also erases the turn-row settle bonus (finish vs dawdle becomes near-indifferent), so it is **demoted** here (C11, conditional). |

## Vocabulary (one word, one meaning)

- **incumbent / control** — the effective recipe's own seed-3 screen (same `recipe_id`, same
  `code_sha`, 8000 it); `B0` at the start. **ok screen** = `status ok`, seed as required,
  `recipe_id == effective`, `code_sha == current`; when several exist it means the NEWEST.
- **ok run** = `status ok, code_sha == current` (any recipe). The `D00c` lookup needs only an ok run
  of the same `recipe_id`.
- **base recipe** of a candidate — the recipe composed of the EFFECTIVE choices of the candidates
  ranked BEFORE it (`base_recipe_id`; identical to the effective recipe while the ladder runs in rank
  order). An out-of-rank re-run (`decide --choose run`) uses that rank-composed recipe, and its
  control is that recipe's ok s3 screen (the last adoption before it, else `B0`/`B<k>`; if none
  exists, `next_item` inserts the recipe-baseline rerun first). Its seed-2 screen is the **twin** a
  pooled decision pairs with. **Effective entry** of a candidate = its newest non-squashed ledger
  entry; TERMINAL, `after` and `compose_recipe` read effective entries only.
- **anchor** — the screen of the last ADOPT that was clear at its final level (s3 clear, or pooled
  clear), user-confirmed, or user-overruled to ADOPT (once its recipe has an ok s3 screen), else
  `B0`; **recomputed by `compose_recipe` under the current `code_sha`** and composed at dispatch, not
  frozen on the item (the header records it for the record only — implementation decision 1).
- **recipe head** — a pure lookup: the final-stage checkpoint of the ok s3 screen of the effective
  recipe; used only by milestones. `recipe_head_s2` is the same lookup at seed 2.
- **FLOOR gate** — the non-final-stage survival floor (`FLOOR_SURVIVAL_MIN` 0.30) or the from-scratch
  `dead_walker` floor (`D03`), meaning "nothing usable was produced" → candidate = clear REJECT
  (`rejected_by_floor`); on a seed-2/twin/batched run it is WATCH (`disputed`) and the run continues
  to its screen. **RETENTION** — a per-stage relative check (`RETENTION_SURV_DROP` 0.10,
  `RETENTION_RATIO_DROP` 0.20), always WATCH + notify, never a verdict.
- **INTERRUPT** — the mechanical stop of a running child (train or eval) after an override, a STOP
  or a SIGTERM (`interrupt_check`, status `interrupted`); never a decision.
- **halt** — the driver exits 1 with `phase = halt:<xid> <why>` (infra, nan, dead_plant, eval_infra,
  config/eval mismatch); never a decision. `RESUMABLE_HALTS = ("halt:infra", "halt:eval_infra")`.
- **Ledger statuses** — `speculative` (uncertain, machine-chosen, pair pending) · `decided` (machine,
  no pair pending — a clear screen, or a MISSING-guard REJECT recorded `confidence: uncertain`) ·
  `final` (pair-resolved, or kind resolution/measurement/baseline, or milestone PASS/WATCH) ·
  `confirmed` (user, frozen — by `confirm` or by a milestone `--accept`) · milestone-only `open`
  (between (i) and the close; `chosen: null`, exempt from the options pin, not TERMINAL) ·
  `overruled` (`override.by ∈ {user — frozen —, driver, milestone}`; its effective choice stands) ·
  `squashed` · `skipped` · `await` (milestone FAIL; sub-status `accepted_i` after a user `--accept` of
  a (i) FAIL, until the promotion lands). **TERMINAL** = `{decided, final, confirmed, overruled,
  skipped}` (`ledger_cli.TERMINAL_STATUSES`). Options are pinned: `CANDIDATE_OPTIONS = (ADOPT,
  REJECT, SKIP)`, `MILESTONE_OPTIONS = (PASS, WATCH, FAIL)`.
- **Undecided** candidate = no effective entry. `decide <Cid> --choose run` squashes the effective
  entry whatever its status and runs the SQUASH algorithm with it as `D_k`, so the candidate is
  undecided by that one rule and no milestone's `after` holds until the re-run lands.
- `confidence` = the seed-3 screen's classification, immutable (`clear|uncertain`);
  `pooled_confidence` = the pair's (`clear|uncertain|disputed`; `none` in a predicate when no pair
  ran). `decision(X).chosen` in a predicate = the EFFECTIVE choice (override applied).
  `screen.<key>` = the effective recipe's ok s3 screen.
- **Experiment kinds** — `baseline`, `candidate`, `twin`, `seed2`, `batched_s2` (a milestone's seed-2
  chain), `milestone` (the 2a→2b promotion), `milestone_s2` (M3's seed-2 promotion). Only `candidate`
  arms `dead_walker`; every other kind runs its full window. `SCREEN_KINDS = (baseline, candidate,
  twin, seed2, batched_s2)` — a promotion is not a screen of the recipe it promotes.
- **Ledger ids** in write order: `D00` resolution, `D01` turn-rate measurement, `D02` B0 s3 baseline,
  `D03` the `dead_walker` floor measurement (from B0's log), `D04` B0 s2 baseline (→ `D00b`),
  candidates from `D05`; per-chain seed resolutions are `D00c-<chain>`.

## Decision rule — plan decision 5, VERBATIM

> **Decision rule (pure function of `summarize()` + the resolution record):** F block = facing on
> `turn_l90`, `turn_r90`, `move_turn`: the per-episode *closed fraction* `clip(1 −
> final/max(initial, 0.15), −1, 1)` (`None` = uninformative when `initial < 0.15`) and the signed
> `yaw_progress_rate`; T block = `speed_ratio` on `fwd`, `lat_l`, `diag`, `back`; guards (four) =
> completion, `tripod_walking_dir` (min over the four direction holds), `turn_survival_l90` =
> `surv[turn_l90]/surv[back]` and `turn_survival_r90` = `surv[turn_r90]/surv[turn_l90]` (falls DURING
> the turns, not the cumulative survivals nested inside completion; a conditional survival is MISSING
> when its denominator is 0 or unavailable, and the bootstrap drops zero-denominator resamples).
> WATCH, never scored: `track_ratio_tail` (the training-log ratio includes the ~20 % move+turn rows,
> so a successful facing fix lowers it by design), the slot-gated `track_ratio_move_tail`,
> `dir_ratio[move_turn]`, `tripod_by_hold[move_turn]`, `settle`, right-handed holds, the cumulative
> survivals. **One standardiser everywhere:** `R_i = R_eval,i`, the metric's D00 paired-bootstrap
> half-width, never re-fitted. **Block statistic** `z_F = mean_i(Δ_i / R_i)` (a closed fraction and a
> rad/s rate weigh equally); **block resolution** `R_F = max(R_F^eval, R_F^seed)` where `R_F^eval` is
> the paired-bootstrap 95 % half-width of `z_F` itself and `R_F^seed` is the block-level seed term of
> decision 6 — likewise `R_T` and the two absolute-units blocks; guards one-sided at 99 % on their
> own per-metric `R` (`max(R_eval,i, R_seed,i)` single-seed). Null candidate all-TIE ≈ 0.95² × 0.99⁴
> ≈ 0.87 under eval noise (pinned), and `B0_s2` submitted against `B0_s3` is TIE on both blocks and
> every guard by construction (pinned). Per block: BETTER if `z > R`, WORSE if `z < −R`, else TIE.
> **Clear ADOPT** = ≥ 1 block BETTER, no WORSE (blocks or guards), and anchored drift OK; **clear
> REJECT** = any WORSE, or a FLOOR gate (a MISSING metric never blocks a REJECT); **UNCERTAIN** = all
> TIE → speculative pick = the candidate's `speculative_default` when declared (C1c: `candidate`),
> else the sign of `z_F + z_T` (exact tie → incumbent), notify, pair queued. **Anchored drift** (`z`
> of candidate vs anchor ≥ −R on both blocks and every guard) is a precondition of ANY ADOPT, clear
> or speculative, s3 or pooled — a clear-ADOPT screen that fails it is recorded `confidence:
> uncertain` (`evidence.anchored_drift: FAIL`), status `speculative`, pick REJECT, and queues the
> pair exactly like an all-TIE UNCERTAIN; at the pooled level drift = the candidate's two-seed mean
> deltas vs the anchor's s3 (and its s2, paired seed-wise, when the anchor recipe has an ok s2 screen)
> at `R_F^pooled`/`R_pooled,i`, and a pooled drift pass with a pooled clear ADOPT is an ADOPT (a
> different pick → self-overrule as usual). MISSING in a guard → a `decided` REJECT (TERMINAL) with
> `confidence: uncertain`, `evidence.missing: [<guards>]` + WATCH and NO pair (a pair cannot supply
> the metric; `decide --choose run` re-runs the candidate later); MISSING in F/T → that metric leaves
> the block mean and the best outcome is a speculative ADOPT. **Absolute-units** candidates (C3a, C3b
> — any candidate that changes the command cap/clip or, if ever added, the speed ellipse): F =
> `facing_closed` + whole-hold `net_yaw_rate_toward`, T = `achieved_vx` m/s.

`candidates.yaml`'s `decision_rule.text` carries this paragraph ASCII-transliterated (modulo the
markdown emphasis markers, and without the paragraph's leading heading phrase "Decision rule (pure
function of `summarize()` + the resolution record):" — a mechanical whitespace-normalised diff shows
that one difference and no other), frozen verbatim into the `D00` entry, so a later edit of the rule is
visible in the ledger rather than only in a plan file. The metric ids are DOTTED PATHS into
`run_phases.summarize()`'s output, listed per block in `decision_rule.blocks` /
`decision_rule.guards` / `decision_rule.watch`.

A recorded entry's `evidence.*.blocks` is keyed by the SLOT LABEL `F`/`T` with the real block name in
`block` (so an absolute-units candidate's notify line reads `F=F_abs`). Guards are ONE-SIDED: a guard
above `+R` is reported TIE with `above_R: true`, never BETTER.

## Resolution policy — plan decision 6, VERBATIM

> **Resolution is measured, cited, and grows with evidence.** `D00` = paired episode bootstrap
> (B = 2000, paired on `env_index`; standardised block means bootstrapped directly) on explicitly
> listed run dirs (`resolution_inputs:` in `candidates.yaml`, full paths — e.g.
> `outputs/omni_plan/reeval_2a/slow__omni/slow__omni/seed001/2026-09-18_16-42-00`): the corrected
> re-evals `outputs/omni_plan/reeval_corrected/{1a,1b,1c}` (listed `provenance_ok`, informational —
> degenerate n), the 1c-004 slow/light/ring, 2b slow/light/ring and the three `reeval_2a` rows (the
> half-widths; n ≥ 50 per hold). **Provenance rule:** a run dir is refused unless its
> `run_meta.env_vars` carries every `EVAL_CARRY` key that has NO entry in `EVAL_CARRY_DEFAULTS` (the
> five 2026-09-17 keys); a defaulted key absent from `env_vars` is taken at its default and recorded
> as `env_vars_defaulted`. **Pairing:** the candidate-vs-control `R_eval` per metric comes from
> CROSS-HEAD pairs on `env_index` (1c-004 vs 2b, 1c-004 vs 2a-reeval, per row); the same-head pair
> (one `slow__omni` re-eval of the 2b head, 4 min GPU, vs the recorded 2b eval) is reported
> separately as the process-noise floor, never as the candidate R. Floors from the sideways CHARTER
> (`2026-09-11_0414_sideways_formation/CHARTER.md:227`: completion ±0.10, walking tripod ±0.056; ring
> completion WATCH at milestones unless the bootstrap resolves it). `facing_closed` needs `initial`,
> which the recorded JSON lacks: `resolution.py` re-scores the saved `raw/episode_*.npz` through the
> new `facing_metrics` for slow/light rows; **every ring row** uses the documented proxy `1 −
> final/max(heading_error_max_abs, 0.15)` on both sides regardless of raw presence
> (`facing_closed_source` recorded). **Seed terms.** Per metric `R_seed,i = |s3 − s2|` of a control
> pair on the same chain; per block `R_F^seed = max(|z_F(control s2 as candidate vs control s3)|,
> rms_i(Δ_i^seed / R_eval,i) / √k)` (the second term is the independence-model spread of a k-metric
> block mean and cannot cancel across metrics, whereas the signed pair statistic alone can cancel to
> near zero when the seed deltas alternate in sign). Sources: **`D00b`** = B0's pair (window 2), cited
> by every decision on the `scratch_1c` chain AND by every decision on any other chain until that
> chain's own **`D00c-<chain>`** exists (the chain's first ok same-chain pair — formed by a twin, a
> seed-2 run or a batched milestone run, written as a lookup step BEFORE any pooled decision that
> would cite it, never re-fitted later); a pooled decision whose candidate and base are on different
> chains uses the `max` of the two. Single-seed decisions: blocks at `R_F = max(R_F^eval, R_F^seed)`,
> guards at `max(R_eval,i, R_seed,i)`. **Pooled decisions** keep `R_i = R_eval,i` in `z`, compute `z`
> on the two-seed mean deltas, and use `R_F^pooled = max(R_F^eval/√2, R_F^seed)` for blocks and
> `R_pooled,i = sqrt((R_eval,i/√2)² + R_seed,i²)` for guards. `lat_r`/`move_turn_r` borrow
> `lat_l`/`move_turn`'s R until `resolution.py --curriculum-runs` re-derives them at the first
> milestone. Levels are recorded and cited in every DECISION line. **Tests:** N = 1000 pairs of
> independent episode resamples of the 1c-004 run as candidate/control with R in the eval-only
> construction (seed terms 0) → all-TIE frequency in `[0.85, 0.98]` (the independence bound 0.87 is a
> floor — blocks and guards share fall events — not a target; fixed RNG seed) with the per-block
> marginal TIE frequency 0.95 ± 0.03 when `R_F^eval` is re-derived by the same paired bootstrap on the
> same-run resample distribution (a calibration check of the bootstrap; under D00's cross-head R the
> marginal is asserted only ≥ 0.92), and ≥ 0.85 with the full construction; `B0_s2` vs `B0_s3` →
> all-TIE on blocks and guards; `R_F(D00b) ≥ R_F(D00)`; six equal-magnitude alternating-sign seed
> deltas → `R_F^seed > 0`; `R_F^seed ≥ |z_F(pair)|`; a curriculum-chain decision before its `D00c`
> cites `D00b`'s block term.

**Amended by the code (PLAN_AMENDMENTS 11, ratified):** the ring-row proxy sentence is superseded.
`resolution.py` re-scores EVERY input row from its raws, ring included, and labels
`facing_closed_source ∈ {metric, proxy, mixed, None}`; proxy rows are never pooled against metric
rows. The 2026-09-18 review measured the `heading_error_max_abs` proxy to **sign-flip** against the
per-episode metric on the recorded 1a slow row (+0.30/+0.28/+0.29 on lat_l/diag/back vs the metric's
−0.51/−0.39/−0.40) and to overstate `turn_r90` threefold, because `max > initial` whenever the body
first turns the wrong way. The plan's proxy is available behind `--ring-proxy` and is not used.
`facing_closed_by_hold` has NO `heading_error_max_abs` fallback at all: an aggregate without
`heading_error_initial_abs` yields `{hold: None}` for every hold and `facing_closed_source is None`,
so every row recorded before 2026-09-19 must be re-scored from `--save-raw` raws to carry the
quantity. `None` is recoverable; a wrong-signed number is not.

### The D00 record as delivered

`resolution.json` (schema 1, created 2026-09-19T09:05:39-04:00, md5 `6675368cf2f0490b1a1d41fc8d3c6a20`
— re-run after the C9 `question` correction below; every `R_eval`, block and guard number is
bit-identical to the 06:21 pass, which differed only in `created` and `candidates_sha256`),
B = 2000, `rng_seed` 20260919, paired on `env_index`, n = 100 paired episodes per pair, screen row
`slow__omni`, `control_head: 1c-004` on every row (pinned in `candidates.yaml`
`decision_rule.control_head` rather than inferred from the order of `resolution_inputs`: the
per-metric R is the MAX over the pairs a control forms, so a reorder would silently re-fit D00 after
decisions had been taken against it). **13 inputs → 9 scored, 1 process-floor, 3 informational, 0
refused**; 6 cross-head pairs + 1 process-floor pair (`2b vs 2b_floor @slow__omni`); `seed_terms {}`
(D00b lands at B0 s2, `D00c-<chain>` at each chain's first ok pair).

Per-metric half-widths on the SCREEN row (`slow__omni`) — the numbers every DECISION line quotes:

`n` is the paired episode count (100 on every metric, by construction); `n_valid` is how many of
those pairs carried the metric on BOTH sides — the quantity decision 6's "n ≥ 50 per hold" is about,
and the one `resolution.json` records per metric (`below_min_n` is false on every row below).

| metric | R_eval | level | block(s) | n | n_valid | source |
|---|---|---|---|---|---|---|
| `facing_closed_by_hold.turn_l90` | 0.06019 | 0.95 | F, F_abs | 100 | 58 | raw |
| `facing_closed_by_hold.turn_r90` | 0.03860 | 0.95 | F, F_abs | 100 | 57 | raw |
| `facing_closed_by_hold.move_turn` | 0.04844 | 0.95 | F, F_abs | 100 | 57 | raw |
| `yaw_progress_by_hold.turn_l90` | 0.00718 | 0.95 | F | 100 | 58 | json |
| `yaw_progress_by_hold.turn_r90` | 0.00762 | 0.95 | F | 100 | 57 | json |
| `yaw_progress_by_hold.move_turn` | 0.00805 | 0.95 | F | 100 | 57 | json |
| `net_yaw_by_hold.turn_l90` | 0.00731 | 0.95 | F_abs | 100 | 58 | json |
| `net_yaw_by_hold.turn_r90` | 0.00715 | 0.95 | F_abs | 100 | 57 | json |
| `net_yaw_by_hold.move_turn` | 0.00820 | 0.95 | F_abs | 100 | 57 | json |
| `ratio_by_hold.fwd` | 0.04844 | 0.95 | T | 100 | 98 | json |
| `ratio_by_hold.lat_l` | 0.03981 | 0.95 | T | 100 | 82 | json |
| `ratio_by_hold.diag` | 0.03452 | 0.95 | T | 100 | 77 | json |
| `ratio_by_hold.back` | 0.05053 | 0.95 | T | 100 | 65 | json |
| `achieved_speed_by_hold.fwd` | 0.01211 | 0.95 | T_abs | 100 | 98 | json |
| `achieved_speed_by_hold.lat_l` | 0.00876 | 0.95 | T_abs | 100 | 82 | json |
| `achieved_speed_by_hold.diag` | 0.00863 | 0.95 | T_abs | 100 | 77 | json |
| `achieved_speed_by_hold.back` | 0.01112 | 0.95 | T_abs | 100 | 65 | json |
| `completion` | 0.15500 | 0.99 one-sided | guard | 100 | 100 | json |
| `tripod_walking_dir` | 0.04173 | 0.99 one-sided | guard | 100 | 64 | json |
| `turn_survival_l90` | 0.07920 | 0.99 one-sided | guard | 100 | 100 | json |
| `turn_survival_r90` | 0.02818 | 0.99 one-sided | guard | 100 | 100 | json |

Block half-widths (standardised by the merged `R_eval,i`) and per-row guards — the milestone
comparison reads a row's OWN row:

| row | F | T | F_abs | T_abs | completion | tripod_dir | turn_surv_l90 | turn_surv_r90 |
|---|---|---|---|---|---|---|---|---|
| **slow__omni** (screen) | **0.54951** | **0.51767** | **0.57766** | **0.51767** | 0.1550 | 0.0417 | 0.0792 | 0.0282 |
| light__omni | 0.53404 | 0.46084 | 0.53657 | 0.46084 | 0.1750 | 0.0449 | 0.0863 | 0.0390 |
| ring__omni | 0.47710 | 0.52611 | 0.50891 | 0.52611 | **0.1650** | 0.0445 | 0.0808 | 0.0323 |

The RESOLUTION line, as `resolution.json` carries it and the driver emits it:

```
RESOLUTION D00 | paired episode bootstrap B=2000 on env_index | blocks F R=0.550@0.95 (MDE per member 0.004..0.033); T R=0.518@0.95 (MDE per member 0.018..0.026); F_abs R=0.578@0.95 (MDE per member 0.004..0.035); T_abs R=0.518@0.95 (MDE per member 0.004..0.006) | guards @0.99 one-sided completion+-0.155, tripod_walking_dir+-0.042, turn_survival_l90+-0.079, turn_survival_r90+-0.028
```

**Null construction** (N = 1000 trials, two independent episode resamples of the 1c-004 `slow__omni`
run as candidate/control, fixed RNG seed): all-TIE **0.888** with `R` re-derived by the same paired
bootstrap on the same resample distribution (per-block marginals F 0.953, F_abs 0.956, T 0.962,
T_abs 0.962; guards 0.988–0.994), and all-TIE **0.851** under D00's own cross-head `R` (blocks F
0.949, F_abs 0.943, T 0.960, T_abs 0.960; guards 0.990–1.000 **except `turn_survival_r90` 0.949**).
The plan's independence bound 0.87 is a floor, not a target; the shipped band is `[0.85, 0.98]` and
0.851 sits **within 0.1 σ of the 0.85 floor** (the Monte-Carlo σ at N = 1000 is
`sqrt(0.851 × 0.149 / 1000) ≈ 0.011`, so the 0.001 margin is ≈ 0.09 σ — the estimate is essentially
ON the floor, not above it). A re-measure that drops below it is a signal to LOOK, never to widen
the band.

**Process floor = 0.0 on all 26 metrics, BY CONSTRUCTION** (pair `2b vs 2b_floor @slow__omni`, run
`.../campaign/seed3_2b_006_A15pB_floor/slow__omni/seed001/2026-09-19_06-17-18`, 4 min GPU
2026-09-19). The harness is bitwise deterministic at a fixed `env_seed`: `raw/episode_00.npz` is
md5-identical between the recorded 2b eval and the same-head re-eval, and `run_meta` differs only by
`started_utc` and the two defaulted carry keys. Recorded as a FACT ABOUT THE HARNESS, not as a
resolution — any future floor measurement must vary `env_seed`; episode-sampling noise is what the
independent-resample construction already measures.

Charter floors are CITED, not applied: the sideways campaign's completion ±0.10 (the bootstrap
resolves it at ±0.155, wider, and supersedes it) and walking tripod ±0.056 (resolved at ±0.042,
tighter, and likewise superseded) —
`2026-09-11_0414_sideways_formation/CHARTER.md:227`.

`lat_r` / `move_turn_r` **borrow** `lat_l` / `move_turn`'s R (no curriculum row among the D00
inputs); `resolution.py --curriculum-runs <two or more curriculum__omni dirs>` re-derives them at the
first milestone. Two WATCH quantities are thin and `resolution.json`'s `notes` say so per row:
`facing_by_hold.turn_l90.settle` produced NO half-width on any row (n_valid 1 slow / 2 light / 0
ring — decision 5's MISSING rules apply), and `facing_by_hold.turn_r90.settle` has only 35 / 39 / 33
usable episodes (< 50), so its half-width is reported rather than dropped. Both are WATCH anyway.

## Chains

A chain is a list of stages; `base` names the PRESET the stage runs on (`KRABBY_OMNI_PHASE`), never a
new preset. Every difference from that preset is an env OVERLAY, recorded in `params/env.yaml` and
audited per window by `overlay_checks` — which is what makes "the recipe" a fact rather than a claim.

- **`scratch_1c`** — one stage: base `1c`, from scratch, 8000 it, eval `[slow]`, floor `dead_walker`.
  The incumbent recipe at equal compute; `B0`'s chain.
- **`scratch_c6`** — one stage: base `1a`, from scratch, 8000 it, overlay = c6's pool + `DIR_BAND=""`
  + `SLOTS 0.15:0.55:0.1:0.2` + `RSI_SOURCE all` + `FACING 0.2:0.6` + `YAW_W 0.2`, eval `[slow]`,
  floor `dead_walker`. C1c's attribution arm.
- **`curriculum`** — the user's additive capability curriculum, six stages on the `1a` preset. The
  economy is UNIFORM along the chain (`FACING 0.2:0.6`, `YAW_W 0.2`, `DIR_BAND=""`, RSI from the
  merged bank in every stage): the 1c failure was a drift under a fixed economy, and a stage whose
  economy differs from its neighbour's is exactly the handoff that lost forward travel at 1a→1b.
  Consecutive stages differ only in the pool (+ the RSI source at c1→c2).

| stage | it | RSI | pool (`name:weight`) | slot projection | eval | floor |
|---|---|---|---|---|---|---|
| c1 | 1000 | `0` | `stand:0.2,fwd:0.8` | 0.20:0.80:0:0 | curriculum | survival through `fwd` < 0.30 |
| c2 | 1000 | `all` | `stand:0.2,fwd:0.3,side_l:0.2,side_r:0.2,back:0.1` | 0.20:0.80:0:0 | curriculum | survival through `back` < 0.30 |
| c3 | 1000 | `all` | `stand:0.15,fwd:0.25,side_l:0.16,side_r:0.16,back:0.08,turn:0.2` | 0.15:0.65:0.20:0 | — | none |
| c4 | 1500 | `all` | `stand:0.15,fwd:0.22,side_l:0.14,side_r:0.14,back:0.1,turn:0.15,fwd_turn:0.05,side_l_turn:0.025,side_r_turn:0.025` | 0.15:0.60:0.15:0.10 | curriculum | survival through `move_turn_r` < 0.30 |
| c5 | 1500 | `all` | `stand:0.15,fwd:0.15,side_l:0.1,side_r:0.1,back:0.1,omni:0.15,turn:0.1,fwd_turn:0.07,side_l_turn:0.04,side_r_turn:0.04` | 0.15:0.60:0.10:0.15 | — | none |
| c6 | 2000 | `all` | `stand:0.15,fwd:0.14,side_l:0.09,side_r:0.09,back:0.09,omni:0.14,turn:0.1,fwd_turn:0.06,side_l_turn:0.035,side_r_turn:0.035,omni_turn:0.07` | **0.15:0.55:0.10:0.20 = SLOTS_OMNI** | slow + curriculum | none (final stage) |

Total **8000 iterations = B0**. Backward is in the pool from c2 (user amendment 2026-09-19): it has
no RSI seed and relies on the fore-aft mirror image of forward, and its eval hold `back` sits right
after `lat_r` in the prefix-ordered `curriculum__omni` row so c2's floor and RETENTION read it. Every
enabled type keeps ≥ `POOL_FLOOR_SHARE` **0.025** once enabled; move+turn ramps **0.10 → 0.15 →
0.20** and never above the incumbent's 0.20; c6's regime shares equal the incumbent's; every row sums
to 1.00 and projects to the stated slot mix (re-derived, pinned). Teacher pool for milestones:
`rescale_to(SLOTS_TEACHER)` of c6 (round-trip pinned). **Confound, recorded in C1's `question`:**
c4–c6 draw world-held move+turn at the 0.3 rad/s cap, which the plant cannot deliver at any
commandable speed (caveat 1) — C1b is the reduced-demand re-screen and C1c the attribution arm.

## Ranked candidates

Rank order IS `candidates.yaml`'s list order. Each is screened on the EFFECTIVE recipe at seed 3 for
8000 iterations; its control is that recipe's own ok s3 screen.

| # | id | change | mechanism | rationale / confound | replay | conditional |
|---|---|---|---|---|---|---|
| B0 | baseline | `scratch_1c`, 8000 it, seed 3 then seed 2 | — | the incumbent recipe at equal compute; the recorded 1c-004 (5000 it) is a zero-GPU prior row, never a control. Never stops on a gate. `D03` records B0's its-800–1000 `track_ratio_mean` mean → the candidates' `dead_walker` floor `max(0.35, 0.75 × that)` | — | — |
| C1 | curriculum | chain `curriculum`; `teacher_overlay: {}` | curriculum (4) | the user's #1: one command type at a time, floor shares kept, move+turn ramped. CONFOUND: c4–c6 command the infeasible world-held move+turn at 0.3 rad/s | none | — |
| C1b | curriculum, reduced-demand turns | chain `curriculum` + `pool_edit {ang_scale: {"*_turn": 0.3}}` (0.09 rad/s cap on `*_turn` rows) | curriculum (4) + kinematics (2) | was the curriculum rejected because of the infeasible demand? A 70 % smaller demand on the rows that cannot be delivered; training distribution only | none | `decision(C1).chosen == REJECT and (decision(C1).floor_stage is None or decision(C1).floor_stage >= c4)` |
| C1c | pool at 8000, single stage | chain `scratch_c6`; `teacher_overlay: {}`; `speculative_default: candidate` | attribution | separates the POOL from the STAGING. On all-TIE the simpler recipe is the speculative pick (Occam) and the pair still runs; its chain's seed resolution is `D00c-scratch_c6` | none | `decision(C1).chosen == ADOPT` |
| C2 | yaw-progress 3.0 | `KRABBY_YAW_PROGRESS_W=3.0` | price (1) | raises the reachable prize **0.206 → 0.366/s** without paying a robot that never turns (the even angular kernel does that and is untouched) | `--yaw-progress-w 3` | — |
| C3a | cap 0.21 | `KRABBY_OMNI_ANG_VEL_MAX=0.21`, `KRABBY_OMNI_CLOCK_TURN_R=1.72`; `absolute_units`; `eval_carry` both | kinematics (2) | cadence preserved (1.72 × 0.21 = 0.361 vs the shipped 1.2 × 0.3 = 0.36); also cuts the angular kernel's marginal turn incentive ≈36 %, so C2 belongs first. The clip stays 0.2, leaving the P-control's linear region 0.133–0.14 rad (near bang-bang) | none (the replay cannot price a cap change) | — |
| C3b | cap 0.18 paired | `ANG_VEL_MAX=0.18`, `ANG_VEL_CLIP=0.12`, `CLOCK_TURN_R=2.0`; `absolute_units`; `eval_carry` all three | kinematics (2) | the paired variant at the kinematic bound, dead band 0.08 rad, cadence preserved (2.0 × 0.18 = 0.36) | none | `decision(C3a).chosen != REJECT or decision(C3a).pooled_confidence in (uncertain, disputed)` |
| C4 | turn-slot facing bonus | `KRABBY_OMNI_FACING_TURN_W=0.8` | price (1) | settle bonus on the turn slots **0.12 → 0.60/s** across the settled-0.1-rad vs 1.2-rad span; exactly **0** on every translation row, so the translation stake is untouched | `--facing-turn-w 0.8` | — |
| C5 | facing obs sincos | `KRABBY_OMNI_FACING_OBS_ENCODING=sincos`; `eval_carry` | observation (3) | (sin, cos) of the UNCLIPPED error is bounded by construction and keeps the magnitude the 0.6-rad clip destroys. Screened from scratch, which is what an observation change needs | none | — |
| C6 | coupled tracking | `KRABBY_OMNI_TRACK_FACING_SIGMA=1.0` | coupling (1) | makes ignoring facing COST translation, on every row (stock kernel × `exp(−(herr/1.0)²)`). After the cheaper fixes, because it lowers translation income a candidate before it may have bought | `--coupled-sigma 1.0` | — |
| C7 | reduced-demand move+turn | `pool_edit {ang_scale: {"*_turn": 0.3}}` on whatever pool the recipe carries | kinematics (2) | C1b's lever inside an ADOPTED curriculum; training distribution only, eval holds unchanged | none | `recipe.has_pool and decision(C1b).chosen != ADOPT` |
| C8 | stand share 0.05 | `overlay_by_chain`: curriculum / scratch_c6 → `pool_edit {set_share: {stand: 0.05}, renormalise: true}`; scratch_1c → `KRABBY_OMNI_SLOTS=0.05:0.65:0.1:0.2`; `teacher_overlay {SLOTS 0.05:0.55:0.15:0.25}` | economy (5) | the stand share is 0.15 of every window and the shipped terms pay a stander ≈1.0/s. 0.05 with the freed share over the WALKING types only (turn and move+turn shares unchanged) triples walking exposure without moving the regime mix the facing work is measured on | none | — |
| C9 | progress weight 1.5 | `KRABBY_OMNI_PROGRESS_W=1.5` | economy (lit rec 3) | raises the translation stake — the opposite side of every facing trade above, so it runs LATE. NOTE the replay's `--progress-w` alias **DOES** arm its turn-hold income table when supplied — it is one argparse dest with two spellings (`replay_reward_economy.py:640-650`; PLAN_AMENDMENTS 15, caveat 25) | `--progress-w 1.5` | — |
| C10 | RSI turn frames | `KRABBY_OMNI_RSI_TURN_SOURCE=2`; **deferred** | seeding (lit rec 6) | the bank holds no turn frame, so a turn always starts from a standing pose. Harvester (`--mode turn`), the staged-turn tuple and the staged-slot rule are OUTSIDE the code freeze: reached with its predicate true → `await_user:C10 code` | none | `screen.yaw_progress_turn_min < 0.10` |
| C11 | ang-turn-only | `KRABBY_OMNI_ANG_TURN_ONLY=1` | economy (5) | DEMOTED: +16.6 % → +37.5 % replay margin, but it erases the turn-row settle bonus. Runs only if the creeper signature returns on the MOVE-ONLY rows (a creeper reads ≈0.25 there, a walker ≈0.6) | `--new-ang-turn-only` | `screen.track_ratio_move_tail < 0.45` |
| C12 | body-locked travel | `KRABBY_OMNI_TRAVEL_FRAME=body`; **deferred**, `user_decision` | task definition | the only change that makes move+turn kinematically honourable — and it redefines the task of record, so it is the USER's call (`await_user:C12`, answered by `ledger_cli decide C12 --choose run\|skip`). Knob outside the code freeze | none | — |

## Milestones

`M1` after C3b, `M2` after C7, `M3` (final) after C12; `after` = every candidate up to and including
that one has a TERMINAL entry and is not undecided (a deferred candidate ends `skipped` or decided
like any other). A milestone is up to three ordinary experiments, each with its own
xid/header/resume/interrupt/squash:

1. **(i) batched seed-2 confirmation** — a zero-GPU pooled decision, preceded by one experiment
   `M<k>.s2` (kind `batched_s2`) whenever `R_k` has no ok s2 screen. Pair = candidate `R_k` (s3 = its
   ok s3 screen; s2 = `M<k>.s2` or an existing twin — never a duplicate) vs base `R_{k−1}` (`B0` at
   M1). It runs BEFORE any promotion. Agree or pooled-uncertain-same-pick → every `decided` adoption
   in the block becomes `final`. Contradiction → driver overrule of the block's LAST unfrozen
   `decided` adoption (`override.by: milestone`) + SQUASH; (i) re-runs ONCE on the reduced recipe; a
   second contradiction → `chosen: FAIL`, status `await`.
2. **(ii) promotion** `M<k>.promo` (kind `milestone`) — the recipe head through **2a then 2b** with
   the recipe's 2a/2b overlays, `EVAL_FULL` at both heads, every row re-summarised from its aggregate
   JSON. **`M1.promo` is never omitted** (there is no previous milestone, and M1's heads are M2's
   rolling reference); from M2 it is omitted when `recipe_id` equals the previous milestone's (the
   entry carries forward as `chosen: PASS`, `evidence.carried_from: M<k−1>`). At M1 the verdict is
   **WATCH-only** vs the recorded seed-3 2b rows and, for the 2a head, vs
   `outputs/omni_plan/reeval_2a` (`milestone_reference` in `candidates.yaml`). From M2: **FAIL = ≥ 2
   of the three 2b rows WORSE** vs the previous milestone's 2b head (a row is WORSE when either block
   or any guard is WORSE; blocks per row at `R_F = max(R_F^eval(row), R_F^seed(chain))`, guards at
   the row's own `R_eval,i`); **the three 2a rows are WATCH at every milestone**. FAIL → the same
   driver overrule + one automatic rerun per eligibility (SHARED across (i) and (ii)); a second FAIL,
   or a FAIL with nothing unfrozen to drop → `await_user:M<k>` (sticky across relaunches).
3. **(iii) M3 only** `M3.final_s2` (kind `milestone_s2`) — the final recipe's seed-2 head through
   2a → 2b from `recipe_head_s2`; `phase=done` follows it, and the user names the head of record from
   both seeds' 2b rows.

**Ring completion IS SCORED as a guard from M2 on this campaign** — decision 6's normative sentence
is "ring completion WATCH at milestones UNLESS the bootstrap resolves it", and D00 resolves it
(`by_row["ring__omni"].metrics.completion.R_eval = 0.165`), so a ring-row completion drop beyond
0.165 makes that row WORSE. The MILESTONE line names the branch it took
(`ring_completion: scored (...)`), and both branches are pinned by tests
(PLAN_AMENDMENTS 16, ratified by the orchestrator 2026-09-19).

## Seed policy

Screens on **seed 3**. Uncertain → the base recipe's **seed-2 twin** (only if no ok s2 screen exists
for that base — `B0` s2 covers everything screened on `B0`) and then the candidate's **seed 2**, as
the next GPU windows; the pooled decision is TERMINAL. Clear ADOPT → batched at the milestone. Clear
REJECT → no pair. A contradicting batched pair on a frozen decision is WATCH, never a flip. A user
`confirm`/`overrule`/`decide run` of a `speculative` decision drops that candidate's own pending
seed-2 run (and interrupts it in flight, `squashed_by: user_choice`); the base recipe's twin is NEVER
interrupted or squashed — it is the base's s2 screen, needed by later pairs and by `M<k>.s2`.

## Budget

0.43 h/1000 it; 2a 2.4 h, 2b 1.9 h; 3.3–3.9 min per eval row; a screen/twin/s2 run is **3.7 h** on
`scratch_1c` and **3.9 h** on `curriculum`. B0 s3 + s2 **7.4 h**; C1 3.9 h; the ten knob screens
(C2, C3a, C3b, C4–C9, C11 — C3b/C7/C11 conditional) ≈ **37–39 h**; pairs (≈2–4 twins + ≈4 candidate
s2 runs) ≈ **22–31 h**; M1/M2/M3 ≈ **8.6 h each** plus `M3.final_s2` **4.7 h**; **core ≈ 101–112 h**.
Zero-incident extras: C1b 3.9 h, C1c 3.7 h, C10/C12 screens 3.7–3.9 h each plus their control reruns
→ ≈ **101–135 h ≈ 4.2–5.6 GPU-days** before incident reruns. Incident reruns: a milestone (i)
contradiction adds ≤ 3.9 h (only when the reduced recipe lacks an s2) plus ≤ 3.9 h per block member
screened after the dropped adoption; a (ii) FAIL adds 4.3 h (+ 3.9 h when the reduced recipe has no
s2) plus the same per re-screened member — at most **one automatic rerun per eligibility**. Each user
overrule discards and reruns ≤ 3.9 h per squashed run. An out-of-rank re-run whose rank-composed base
has no ok s3 screen adds a 3.7–3.9 h `B<k>`; each uncertain candidate screened after a code landing
adds a 3.7–3.9 h twin. Replay wall time ≈ 10 min each (CPU). **There is no wall-clock cap (user
decision): one long-lived unit, `--max-wall-h` parks at `await_user:relaunch` if one is ever wanted.**

## Speculation and SQUASH

**Speculation = proceed on the chosen branch, notify, resolve by evidence or by the user.** An
uncertain screen takes the speculative pick and the campaign keeps moving; the confirming pair runs
as the very next GPU windows. Pooled outcomes are ALL TERMINAL, evaluated in this order: pooled clear
→ its pick; else the two seeds' own signs (`sign(z_F + z_T)` of candidate s_k vs base s_k) disagree →
pick = incumbent, `pooled_confidence: disputed`; else the candidate's `speculative_default` when
declared, else the pooled sign (exact tie → incumbent). Same pick as the s3 decision → `final`; a
DIFFERENT pick → driver **self-overrule** (`overruled`, `override.by: driver`, a `SELF-OVERRULE`
line, SQUASH of dependents).

**SQUASH** (trigger = an override changes decision `D_k`'s effective choice — user, driver pair, or
milestone): recompute the recipe; the squash set is every experiment and SKIP entry whose
`base_recipe_id` differs from the recomputed base, plus milestones built on a changed recipe, plus
every SKIP entry whose predicate now holds, plus — closed transitively — every SKIP whose
`conditional_refs` include a candidate squashed in this pass, plus every running experiment whose
candidate's newest `user_choice` is `skip`. Never `D_k`'s own screen. `D_k`'s own seed-2 run only on
a USER `confirm`/`overrule`/`decide run` of a still-`speculative` `D_k` (never on a driver or
milestone override). Headers move to `state.ladder.squashed[xid]` with `squashed_by`/`reason`; stage
records stay; run dirs and eval dirs stay on disk (the driver never deletes); dependent decisions
become `status: squashed` and their candidates undecided again. The squash view is **DERIVED and
recomputed after every ledger write**, so it survives a crash; reverting an overrule resurrects an
experiment squashed by derived-view causes (a)–(c) whose files still exist — user squashes, causes
(d)/(e), are **sticky and never resurrected**. A SQUASH or MILESTONE line that squashes a `confirmed`
or `overruled(user)` entry names it with `frozen:` so the user can re-confirm after the re-screen.
A squashed LIVE window (running, or a resumable infra halt with `inflight`) is archived `interrupted`
with `pre_squash_status` and restored to `running` on resurrection, so the record never claims a dead
window trains (implementation decision 6).

**A user `decide <Cid> --choose skip` never squashes or interrupts the BASE recipe's twin** booked for
`Cid`'s pair (kind `twin` is exempt from derived-view cause (d) and from `interrupt_reason`'s skip
clause) — PLAN_AMENDMENTS 20.

## How to confirm, overrule, decide, resume

`$PY`, `$LAD` and `$TOOLS` as defined at the top. Every command below is exactly what the
notification line quotes, with ONE spelling difference: a notify line is written on one physical
line and its sanitiser maps `|` to `/`, so an AWAIT or DECISION line renders the alternation
`--choose run|skip` as **`--choose run/skip`** (likewise `ADOPT/REJECT/SKIP`). That slash is "either
word", not a literal argument — type `--choose run` OR `--choose skip`. Everything else in a quoted
command is runnable as printed.

```bash
OMNI=/home/nickmagus/krabby/krabby-research/parkour/parkour_tasks/parkour_tasks/crab_hex_omni_task
LAD=$OMNI/experiments/2026-09-19_facing_tracking_ladder
TOOLS=$OMNI/experiments/tools
PY=/home/nickmagus/krabby/isaac_venv/bin/python

# launch (prints the unit name on stdout; no timeout wrapper, MemoryMax=48G, TimeoutStopSec=360)
"$TOOLS/launch_candidates.sh" --campaign-dir "$LAD"
# watch it (unchanged from the phase driver; a warning during a CPU-only window is expected)
"$TOOLS/heartbeat_phases.sh" <unit> "$LAD" 1800

# read
$PY $TOOLS/ledger_cli.py --campaign-dir $LAD pending
$PY $TOOLS/ledger_cli.py --campaign-dir $LAD show [<id>] [--json]

# answer a DECISION line
$PY $TOOLS/ledger_cli.py --campaign-dir $LAD confirm <Did> [--why "..."]
$PY $TOOLS/ledger_cli.py --campaign-dir $LAD overrule <Did> --choose ADOPT|REJECT|SKIP --why "..."

# answer a SKIP line, an await_user:<Cid>, or re-run a decided candidate out of rank
$PY $TOOLS/ledger_cli.py --campaign-dir $LAD decide <Cid> --choose run|skip [--why "..."]

# answer a milestone await
$PY $TOOLS/ledger_cli.py --campaign-dir $LAD resume-milestone <Mid> (--accept | --drop <Cid>) [--why "..."]

# the NORMAL stop (and its undo)
$PY $TOOLS/ledger_cli.py --campaign-dir $LAD stop [--why "..."]
$PY $TOOLS/ledger_cli.py --campaign-dir $LAD unstop

# the EMERGENCY stop: SIGTERMs the driver only (KillMode=mixed), so its own handler stops the
# training/eval child with SIGTERM first -- a SIGKILL of a live Kit corrupts the DerivedDataCache.
systemctl --user stop <unit>
```

The CLI writes ONLY `ledger_overrides.jsonl` (append-only, fsync, one uuid per line) — never
`ledger.json`, never `state.json`. A running driver ingests it at its next `ingest_overrides()`
(within one poll tick) and answers `OVERRIDE-APPLIED <uuid> ...`; a stopped driver ingests it at its
next launch, and the CLI prints "driver not running — applied at next launch" when
`state.ladder.unit` is inactive. `overrule` of a `resolution`, `measurement` or `baseline` entry is
REFUSED (those are measurements, not choices). `confirm` of an `overruled(driver)` /
`overruled(milestone)` entry **reverts** the driver's override and freezes the entry's own `chosen`;
to freeze the driver's pick instead, use `overrule --choose <what stands>` (implementation
decision 3). `resume-milestone --drop <Cid>` is refused unless `Cid` is an ADOPT in that milestone's
block, so it always changes the recipe and re-queues the milestone.

**Records, all in `$LAD`:** `candidates.yaml` (the spec) · `resolution.json` (D00) · `ledger.json`
(driver-owned, atomic) · `ledger_overrides.jsonl` (CLI-owned) · `state.json` / `heads.json` (the
`run_phases` contract, so `heartbeat_phases.sh` works unchanged) · **`REPORT.md`** (every block ends
`>>> ENTRY <id>`) · **`CHANGELOG.md`** (one row per experiment) · **`notify.log`** (the thirteen tags;
untracked by the keep-set rule — every line it carries is also in the ledger, REPORT and CHANGELOG) ·
`orchestrator.log` · `STOP` (a live control signal, never a record).

**Notification contract** — `notify.log` line = `%Y-%m-%d %H:%M TAG <id> | field | ...`, one physical
line, absolute paths, produced by `ledger_cli.format_notify` and parsed by
`ledger_cli.parse_notify` (round-trip tested). The fourteen tags: `RESOLUTION, DECISION, CONFIRM,
SELF-OVERRULE, OVERRIDE-APPLIED, SQUASH, MILESTONE, RETENTION, INFRA, HALT, DONE, RELAUNCH, AWAIT,
MIGRATION` (`MIGRATION`, added by the per-category evaluation plan of 2026-09-21, carries the
eval-only rescreen/reeval/rescore items that move a paused campaign onto a new `eval_sha`; its
lines say "eval-only" — nothing trains).
Each DECISION line carries the blocks with their `z` and `R`, the guards, the anchored-drift verdict,
the resolution id and levels, the WATCH row, the next item, and the exact `confirm:` / `overrule:`
commands. The assistant's persistent Monitor tails the file and relays every new line as a push
notification plus a chat card.

**Dry run** (writes nothing but the campaign dir's own `logs/`, no state, no ledger, no GPU):

```bash
$PY $TOOLS/run_candidates.py --campaign-dir /tmp/ladder_dry --candidates $LAD/candidates.yaml --dry-run
```

**A launch is REFUSED while the code tree is dirty**, and a changed `code_sha` at relaunch is refused
without `--accept-code-change <why>` (which then queues a recipe-baseline rerun `B<k>`; old screens
stay as records and nothing is squashed). `code_sha` = sha256 over the blob ids of
`git ls-tree -r HEAD` under `parkour/parkour_tasks/.../crab_hex_omni_task/{mdp,config}`,
`.../crab_hex_forward_task/{config,mdp,scripts/gait_eval}`, `parkour/scripts`,
`parkour/parkour_isaaclab`, `$TOOLS/run_phases.py` and `experiments/eval/scenarios_omni.yaml`, after
an `fnmatch` filter drops `*.md`, `*/experiments/20*`, `*/tests/*` and `tests/*`; the dirty check is
`git status --porcelain` over the same pathspecs, filtered identically (untracked lines count). The
ladder's own three tools are OUTSIDE it and are hashed informationally per entry (`tools_sha`), never
a refusal. **A record or docs commit — `$LAD`, this CHARTER, the README, `docs/FOLDER_LAYOUT.md` —
leaves `code_sha` unchanged**, which is what lets the campaign commit its own records while it runs.

## Held fixed

Plant **A15+B** (`assets/crab.usda`); the Revision-4 reward economy (`KRABBY_TRACK_SIGMA2` 0.04,
`KRABBY_TRACK_L1_W` −2.0, `KRABBY_OMNI_ANG_VEL_STD` 0.5, `KRABBY_OMNI_CLOCK_COMBINE` product) unless
a candidate says otherwise; the facing-error observation clip `KRABBY_OMNI_FACING_OBS_CLIP=0.6` in
every window (C5 changes the ENCODING, not the clip); seeds **3** (screen) and **2** (confirm);
**8000-iteration** screens scored on the flat canary `slow__omni` alone; the eval manifest's three
LEGACY rows are shape-frozen (`slow__omni`, `light__omni`, `ring__omni` — a new row may carry its own
schedule and `episode_length_s`, which `curriculum__omni` does); the 18-channel action layer, the
actuator model, PPO hyper-parameters, 256 envs, the scan grid (108 rays, obs 1149); the 37 reward
NAMES (the forward 35 + `reward_yaw_progress` + `reward_facing_turn` at weight 0.0 by default) and
every weight/ramp not named by a candidate; `KRABBY_FLAT_TERRAIN_GEOM=ring` in every omni process;
one Isaac process at a time; `code_sha` frozen for C1–C9 and C11 (C10 and C12 are deferred, outside
the freeze); presets are NEVER edited during the campaign — stages and candidates are env OVERLAYS on
the real presets `1a` / `1c` / `2a` / `2b`, and a closing **Revision 5** bakes whatever was adopted.
Nothing is pushed (the user publishes).

## Caveats

### From the plan (carried verbatim in intent)

1. `move_turn` / `move_turn_r` are **infeasible by construction** on this plant (walking at the
   0.21 m/s floor leaves ω ≤ 0.01 rad/s at `r_foot` 1.2 m; and the manifest rule
   `1.0 s + 1.5708/cap ≤ 8 s` fails for any cap < 0.2244 rad/s). They are scored RELATIVELY (in F)
   and their translation, tripod and the training-log tracking ratios are watched. A candidate that
   "fixes" them will turn more and walk less there.
2. **Single-seed screens:** seed variance is the dominant uncertainty; the per-block `R^seed` from
   `D00b`/`D00c` and the confirming pairs are the mitigation, not a cure.
3. Milestone rows at **M1 compare unequal compute and a different lineage** (WATCH only); from M2 the
   comparison is rolling and equal.
4. **RETENTION is never a verdict;** the curriculum's first run seeds the stage trace that later runs
   are compared against.
5. World-held move+turn during c4–c6 is the SAME infeasible demand as the eval hold; C1b/C7 reduce
   the demand (0.09 rad/s on `*_turn` rows) without making it feasible; C3a/C3b lower the global cap;
   C12 is the redefinition.
6. `KRABBY_OMNI_ANG_TURN_ONLY` is **demoted:** it erases the turn-row settle bonus; it runs only if
   the creeper signature returns on move-only rows (`track_ratio_move_tail < 0.45`).
7. The **replay screen is advisory** and one-hold decidable until B0's raws exist; it never rejects,
   and it cannot price a cap change.
8. A user overrule late in a block **discards finished work** (hours) — the accepted branch-mispredict
   cost; records are kept.
9. **C10 and C12 are outside the code freeze;** landing them changes `code_sha`, re-runs the control
   and the base twins.
10. Isaac under a **48 G cap:** a cgroup kill is itself a cache-corruption risk, so every stop path is
    SIGTERM first; the infra detector stops retrying on the first OOM/cache signature and hands the
    cache to the assistant (`mv DerivedDataCache DerivedDataCache.broken_<date>`).

### Found during implementation (PLAN_AMENDMENTS 1–26; the code is ground truth)

11. **(1) Twin ruling.** A pool twin of a legacy window is bitwise on
    `slot_id / speed_cmd / heading_target / vel_command_b` for every row and on `travel_dir_w` for
    MOVING rows only (a non-moving row holds `yaw + 0` where the window held its own band —
    unobservable: `speed_cmd == 0`). Bitwise twins ALSO require the pool to be written in
    `POOL_TYPES` order; a pool written out of order still validates, projects and trains the same
    distribution but is **not** a bitwise twin.
12. **(2) `pool_<name>_draws` is a LIFETIME running fraction** (overwritten every step, never reset),
    not "per resample"; `pool_<name>_frac` is the step-weighted one. Read `_draws` ≈ the weights only
    on a fresh run.
13. **(3) `speed_scale` bound.** A construction-time refusal keeps every scaled draw above the
    gait-clock stop threshold 0.2 m/s: `speed_scale > pool_speed_scale_floor(...)` = **0.88** for
    world/lateral/backward types and **0.6286** for `fwd` at the shipped ellipse (0.35, 0.25, 0.25).
    No candidate uses `speed_scale`; the ladder validator applies the same bound at compose time. The
    way to buy room is a wider `speed_ellipse` or a smaller `lin_vel_clip` in the same overlay.
14. **(4) The clock rule refuses `KRABBY_OMNI_CLOCK_TURN_R = 0`** as well as `radius × cap < 0.25`
    (`CLOCK_TURN_MIN_SPEED`). Inert on every preset (1.2 × 0.3 = 0.36). A missing radius or a missing
    `ang_vel_max` is its own NAMED `RuntimeError`.
15. **(5) `KRABBY_OMNI_FACING_OBS_ENCODING` applies through `omni_knobs._omni_obs_terms(cfg)`**, not
    the plan's literal cfg path; `omni_knobs.parse_cmd_pool("")` → `None` (the math module's parser is
    strict and raises on blank).
16. **(6) The cfg-identity `dumps` fixture does not create `KRABBY_CFG_DUMP_DIR`** — pre-create it
    (step 0's lesson).
17. **(7) The unit suite needs `--continue-on-collection-errors`:** 32 PRE-EXISTING collection errors
    in unrelated trees (`zmq`/`serial`/`aiortc`/`tkinter` under `tests/unit/{hal,teleop,controller}`,
    `test_zero_copy`, `test_observation_types`). Without the flag the run aborts at collection.
18. **(8) The README documents the six new knobs, `cmd_pool`, and `CLOCK_TURN_R` as "> 0"** (it said
    "≥ 0"; the rule refuses 0).
19. **(9) `_stop_child` (SIGTERM → `TERM_GRACE_S` 120 s → SIGKILL) is used on EVERY path that stops a
    live child** — train's stall / nan / dead_plant / dead_walker / stop / exception exits too, not
    only the eval timeout (HEAD did `proc.kill(); proc.wait(60)`). Cost: up to +120 s per such death.
    Reasons: those four exits kill a running Kit exactly like the timeout does (the DerivedDataCache
    hazard), and HEAD's bare `wait(timeout=60)` raising `TimeoutExpired` fell into the generic handler
    and reclassified a `nan`/`dead_plant` death as `infra`, firing a spurious `_r2` relaunch.
    KNOCK-ON (unverifiable without Isaac): on a `dead_walker` exit the head is resolved by
    `latest_run_ckpt(...)` AFTER `_stop_child`, i.e. after up to one grace of SIGTERM, so if Kit lets
    the trainer finish one more checkpoint save in that window the head carried forward is a LATER
    checkpoint than HEAD would have filed. Bounded (one save), probably unreachable — **watch the
    first real `dead_walker` abort.** Ctrl-C/SystemExit on an ATTACHED driver now stops the child
    explicitly (`except (KeyboardInterrupt, SystemExit): _stop_child(proc); raise`, pinned by two
    tests); under `systemd-run` a SIGTERM raises no Python exception, so the stop arrives via
    `interrupt_check`.
20. **(10) Three further `_run_eval` timing deltas, all deliberate:** the 3600/5400-s timeout is
    enforced at TICK granularity (worst case `timeout + EVAL_POLL_S + TERM_GRACE_S` per attempt);
    EVERY attempt now costs at least one `EVAL_POLL_S` tick (measured: a success path that took 2 s
    takes ≈30 s — ≈30 s per eval row per campaign, accepted); and the `RETRY_WAIT_S` pause is ticked
    (a stop inside it ends the call `interrupted`) and is SKIPPED after the last attempt, so a
    `max_attempts=1` call (the ladder's shape) no longer burns 120 s of dead time before the infra
    detector runs. `_wait_isaac_clear(interrupt_check)` still runs after EVERY failed attempt, the
    last included — what waits on it is the NEXT ROW.
21. **(11) `facing_closed` for an already-recorded row is available ONLY by re-scoring `raw/*.npz`.**
    See the amendment under "Resolution policy" above: no `heading_error_max_abs` fallback, the proxy
    sign-flips, `facing_closed_source ∈ {metric, proxy, mixed, None}`, and a proxy row is never pooled
    against a metric row. A milestone that needs the quantity re-scores the `run_dir` from its raws
    (`--save-raw` rows only).
22. **(12) `PYTHONUNBUFFERED=1` on EVERY training child**, set AFTER the `extra_env` merge so no
    overlay can unbuffer it (and deliberately NOT in `TRAIN_ENV_RESERVED`, which is about records that
    lie, not about the watchdog). The poll loop reads the child's stdout FILE, so a block-buffered
    child looks stalled until its buffer fills — which `STALL_S` calls an infra death and which
    `dead_walker` reads as "no `track_ratio` prints yet". Verdicts measured unchanged (ok / nan /
    no-checkpoint), but it DOES change the interleaving of the training log that the stall watchdog,
    the `Traceback` scan and `dead_walker` all read, and the `env {...}` line in the campaign log and
    CHANGELOG — so a later reader can tell a line-ordering difference caused by this from one caused
    by a code change.
23. **(13) Additive record keys on the default path:** `state["current"]` gains `"extra_env": {}`, and
    `eval_morph`/`eval_obst`'s dicts gain `eval_status`/`eval_rc` (the phase driver's keys, not the
    ladder's). `expo_lines`' terrain line also gained `(move rows only <x>)`, which re-renders an OLD
    campaign's line as `n/a`; `REPORT.md` files already on disk are untouched.
24. **(14) Reward/observation details.** Under `sincos` the `facing_obs_clip` is IGNORED (sin/cos are
    bounded by construction; clipping first would cap slot 7 at `cos(0.6) = 0.825` and destroy the
    magnitude mechanism 3 says is missing, i.e. make C5 silently inert on every preset).
    `facing_sigma` is KEYWORD-ONLY with no default (Isaac's static argument check reorders defaulted
    args). `reward_facing_turn` carries `parkour_name` (accepted and ignored) so its dump block is
    params-identical to `reward_tracking_yaw`. The turn-slot gate is `slot_id >= SLOT_TURN`, NOT
    `cmd_wz != 0` (the dead-band gate closes exactly when a turn succeeds, which would reward
    overshoot). The obs param is declared in `crab_hex_omni_env_cfg.py` and that declaration is
    **load-bearing for every eval row** now that `EVAL_CARRY_DEFAULTS` injects the encoding: a rename
    or removal kills all three rows of every phase at config build, after the training window is
    spent (pinned at unit level by `test_the_shipped_obs_term_declares_the_encoding_every_eval_now_carries`).
25. **(15) `move_turn_r` (0.22, dir 0, face 0) is a right-handed COUNTERPART, not a reflection** of
    `move_turn`: its ellipse cap is the fore-aft **0.35** vs `move_turn`'s lateral **0.25**, and 90°
    inside 8 s never closes — WATCH only, and never read as a handedness asymmetry. The replay's
    `--progress-w` alias arms the turn-hold table when supplied; a `segment_id` guard refuses raws
    that are not `slow__omni`'s; `--co` is now an ambiguous abbreviation (no committed invocation
    abbreviates).
26. **(16) Ring completion is SCORED at milestones on this campaign** (R ±0.165) — see the
    Milestones section. The safer of two readings of two conflicting plan sentences, deliberately
    chosen and pinned in both directions; reverting it is three lines plus two test rewrites, and
    costs a milestone FAIL → a driver overrule of an adoption → a 4.3 h promotion rerun plus
    per-member re-screens.
27. **(17) `turn_survival_r90`'s cross-head `R_eval` (0.028) is DEGENERATE.** The 2b and 2a-reeval
    heads sit at exactly 1.0 on it, so the paired difference carries only 1c-004's variation and the
    guard does not hold its nominal 99 % (measured null TIE ≈ 0.949 under this R vs 0.994 under the
    calibrated one; all-TIE 0.851 vs the 0.85 floor). No code change: `D00b`'s seed term (B0's pair,
    window 2) is expected to widen it, and the RESOLUTION line lists "guards still carried by the
    EVAL half-width". **A lone WORSE on that guard while it is so listed is to be read as UNCERTAIN
    and confirmed by the user, not as a clear REJECT.**
28. **(18) For C3a/C3b the absolute-units T block is mathematically IDENTICAL to the relative one**
    (`ratio = achieved / a manifest-constant command`): identical `z_T` and `R` (relative difference
    1e-8). Only `F_abs` differs (`net_yaw_rate_toward` vs `yaw_progress_rate`: `R_F` 0.550 vs
    `R_F_abs` 0.578). Faithful to the plan, not a bug — `T` and `T_abs` are the same statistic
    whenever `cmd_mean` is fixed.
29. **(19) The D00 process floor is 0.0 on every metric BY CONSTRUCTION** — see the D00 record above.
    Any future floor must vary `env_seed`.
30. **(20) A user `decide <Cid> --choose skip` never squashes or interrupts the BASE recipe's twin**
    booked for `Cid`'s pair.
31. **(21) `rp.gate_1c_verdict` / `rp.gate_facing_verdict` are printed as WATCH lines** in the
    ladder's stage report and in DECISION/MILESTONE fields (the chain's final stage; the 2a/2b
    promotion stages) — CONTEXT ONLY, never a verdict. This campaign's verdicts come from the
    decision rule and the milestone comparison, nothing else.
    **Expect `gate_1c: FAIL` on essentially every screen, and do not read it as a bad screen.**
    The gate's floors are `GATE_1C_COMPLETION` **0.70** and `GATE_1C_TRIPOD` **0.40** on each
    moving hold, and EVERY head this campaign can cite sits below them: completion **0.50–0.60**
    (1c-004, the 2a re-eval, 2b) and walking tripod **0.377–0.489** (2b 0.377, 2a 0.39,
    1c-004 0.489; 1a 0.267 and 1b 0.405 lower still). The gate was written for the formation
    campaign's pass/fail question; here the SCREEN is the decision rule against the incumbent's own
    ok s3 screen, so a FAIL line is WATCH context — the thing to look at is whether it MOVED.

32. **(22) The V3 legacy twin smokes, and what "bitwise" means in them.** 20-iteration smokes with
    the new code default-off, normalised through `smokes_base/normalize_log.py` and compared against
    the step-0 HEAD capture (`parkour/outputs/omni_plan/smokes_ladder_base`): **1c is identical**
    (2080 normalised lines = 2040 base + the two new labels × 20); **1a is identical except ONE
    print digit** — `Mean priv_reg: 0.3693` (base) vs `0.3694` at normalised line 1330, every reward
    and metric line identical. A second 1a run flipped that same digit back, so it is run-to-run GPU
    nondeterminism at a rounding boundary of the `priv_reg` loss, not a code change. The `sincos` +
    `FACING_TURN_W` + `TRACK_FACING_SIGMA` + pool smoke counts **37 distinct `Episode_Reward/`
    labels** (expected 37), and its `params/env.yaml` records all four intended facts: the swapped
    `func` (`track_lin_vel_xy_exp` → `track_lin_vel_xy_exp_facing_coupled`), `facing_sigma: 1.0`
    among its params, `facing_obs_encoding: sincos`, and the `cmd_pool` string as requested. NOTE
    the evidentiary limit: the normaliser keeps four-decimal print precision, so a smoke PASS proves
    agreement to **1e-4 across ~102 labels × 20 iterations on 1a and 1c only** — the BITWISE claim
    is carried by the int-view tensor differential (FAILS: 0, identical RNG counts) and by the
    character-identical legacy code block, not by the smoke, and 1b/2a/2b/2c rest on the cfg-dump
    delta plus the preset-independence of the path. Record: `parkour/outputs/omni_plan/
    smokes_ladder/RESULT.md` (its V3(a) table and root-cause summary are STALE — the runbook's
    normaliser path was wrong; the appended orchestrator section at the bottom is the correct
    reading, and the stale verdicts are marked "superseded" in place).
33. **(23) RATIFIED 2026-09-19 — `decide_pooled` evaluates the two-seed sign DISAGREEMENT before the
    MISSING-member `uncertain` cap**, in the plan's stated order: pooled clear → disagree (→
    `disputed`, pick = incumbent) → `speculative_default` → pooled sign. The step-7 review found the
    shipped order reversed, so two seeds disagreeing in sign could land ADOPT/`uncertain` instead of
    `disputed → incumbent`; the ratified order is the plan's and the code follows it.
34. **(24) RATIFIED 2026-09-19 — `overruled → confirmed` by the user is legal only for
    `overruled(driver)` and `overruled(milestone)` entries**, per the plan's transition table. An
    `overruled(user)` entry is already frozen by the user's own choice: it takes a NEW `overrule`,
    never a `confirm`. (`confirm` of a driver/milestone override still REVERTS it and freezes the
    entry's own `chosen` — see "How to confirm, overrule, decide, resume".)
35. **(25) Recorded — Four narrowings the step-7 review found and the orchestrator carried as
    caveats rather than code changes.** (a) The driver traps **SIGINT like SIGTERM**
    (`install_sigterm`), so an attached Ctrl-C stops the child cleanly and CANNOT abort the driver
    mid-write — deliberate (an orphaned 256-env Kit holding the GPU is the worse failure), and the
    only way to kill it hard is `systemctl --user kill -s SIGKILL <unit>`, which risks the
    DerivedDataCache; see caveat 38. (b) `CODE_DROP_GLOBS` carries a **fourth pattern `tests/*`**
    beyond the plan's three — inert on this tree (nothing under a top-level `tests/` is inside the
    `code_sha` pathspecs), kept because it is the same filter `code_sha` applies. (c) A single-seed
    decision whose candidate and base sit on **different chains** widens with `max(D00c, D00b)`
    rather than the candidate's own chain term — conservative (it can only widen R, never narrow
    it), and it is what `seed_terms_for` does. (d) **RETENTION is silent on a chain stage's FIRST
    run**: there is nothing to compare against, and the trace that run seeds is compared from the
    second run on (caveat 4 states the principle; this is the mechanical consequence).
36. **(26) The step-7 review's driver fixes**, each pinned by a named test. Milestone entries are
    **not confirm/overrulable** — their verb is `resume-milestone --accept|--drop`; `--accept` is
    **refused unless the milestone's newest non-squashed entry is `await`** (both the CLI and the
    driver-side ingest refuse, naming the actual status); **resurrection after a revert is
    TRANSITIVE** — the squash view is recomputed to a fixpoint and reconciled with `state.json` at
    every launch, so reverting one overrule resurrects the whole dependency chain in ONE ingest
    instead of one level per pass; the **drift anchor requires a CLEAR pooled level** when a pooled
    level exists (a `final` ADOPT whose pooled level was `uncertain`/`disputed` is not an anchor);
    the validator applies `ang_vel_max > clip` with the **default clip 0.2** when no clip is
    overlaid (a cap in (0.15, 0.2] used to pass load and die at config build); the 2a/2b teacher
    pool is rescaled from the **COMPOSED** final-stage pool, not the raw YAML one; `_wait_isaac_clear`
    precedes **every** train/eval launch, first attempts included; the **replay child has a timeout
    and an exception guard** (`_stop_child` on any exception; advisory `status: timeout`, never a
    halt); and pooled DECISION lines **cite their levels** (`@0.95/0.99`) and read `s3 + s2` instead
    of `@None/None`.

### Further sentences the implementation requires

37. **`remaining <= 1` means "finished"** in the ladder's resume path while `run_phases` uses
    `<= 0` — the two resume paths differ by one iteration on purpose.
38. **SIGINT is handled like SIGTERM** (`install_sigterm`), so Ctrl-C on an attached driver no longer
    force-aborts it; the handler only sets the flag and SIGTERMs `rp.CURRENT_CHILD`.
39. **The turn-cap floor is BOTH** the plan's explicit `TURN_CAP_MIN` 0.15 AND the manifest
    recomputation — the feasibility rule is `TURN_SETUP_S 1.0 + (π/2)/cap ≤ hold_s`, so the 12-s turn
    hold accepts any cap ≥ `1.5708/11 = 0.1428`; both are checked and the stricter (0.15) bites. A cap candidate must also clear `CAP_CADENCE_MIN` 0.35 unless its `question`
    records the cadence change (C3a and C3b both do).
40. **A crash strictly between `run_experiment` returning `ok` and `_close_milestone_promo` writing
    the entry re-books the 4.3 h promotion under a new xid.** Not guarded (a wrong guard would skip a
    legitimate re-promotion after a `--drop`); the stage records are persisted, so manual recovery is
    possible. Flagged, not guessed.
41. **`--curriculum-runs` is exercised on synthetic rows only** so far; `lat_r` / `move_turn_r` keep
    borrowed R until the first milestone re-derives them.
42. **Verification 8 (the stop drill) still needs a real Isaac child:**
    `launch_candidates.sh --campaign-dir <copy> --smoke-stage 1a:20`, then
    `systemctl --user stop <unit>` mid-child. Expect `phase=await_user:stop`, no `_r2`, the unit
    inactive in < 360 s, and the smoke's `inflight` header resumed by the next launch.

### From the re-verification of the fix batch (PLAN_AMENDMENTS 27–28, 2026-09-19)

43. **Pool strings the driver renders are canonical** (amendment 27): every weight and scale is rounded
    to 12 significant digits before `repr()`, so a pool that went through parse → normalise → rescale
    renders exactly like its yaml literal (`fwd:0.14`, never `fwd:0.14000000000000004`), and `recipe_id`
    — sha256 over the rendered strings — is stable under float noise. Still repr precision, never fixed
    decimals; the projection/sum round-trip pins (1e-6 / 1e-9) hold. Before this, the re-rendered 2a/2b
    teacher pool had moved the C8-and-later ADOPT-path recipe ids while leaving the distribution
    unchanged (max |Δweight| 2.8e-17); nothing had been recorded under either id.
44. **Reconciliation after every recipe-moving ledger write** (amendment 28): `_dispatch_candidate` and
    `_dispatch_floor_reject` reconcile the record through `squash_and_report` (headers archived or
    resurrected, SQUASH line / REPORT block / CHANGELOG row, `supersedes` set), not `recompute()` alone
    — the plan's "recomputed after EVERY ledger write (decision or override)". The milestone verb
    recognises an already-applied answer by uuid (accept and drop) and relays it as such, like the other
    two verbs; a second `--accept` on an `accepted_i` milestone is already applied. The drift anchor's
    seed-3 screen is resolved through `ok_screen` under the CURRENT `code_sha` (falling back to the
    baseline anchor when none exists), never the recorded old-code header — so after
    `--accept-code-change` the anchor re-resolves as decision 9 requires. Recorded, not changed:
    `decide_pooled`'s guard-MISSING branch precedes the disagreement check (the pick is identical, the
    label reads `uncertain` rather than `disputed`); the SELF-OVERRULE line's second field shows the
    seed-3 pick with the pooled level and the effective pick in its own field; a `decide <Cid> run`
    re-screen's header carries `supersedes: []` because the replaced window is kept as a record.

## Log

- **2026-09-18 — plan approved** (`the-facing-and-tracking-fancy-breeze.md`, 9 audit passes, pass 9
  CONVERGED on all three lenses). Campaign directory created; the formation campaign
  (`$CAMP`) closed at its 2b decision point and never written to again.
- **2026-09-18 — step 0 reference captures at HEAD `5030872`:** the `1a` 20-iteration smoke (64 envs,
  seed 3) → `parkour/outputs/omni_plan/smokes_ladder_base`, and the three cfg-identity suites
  (`KRABBY_CFG_DUMP_DIR`) → `parkour/outputs/omni_plan/dumps_ladder_base`. Verification 2/3 compare
  against these two directories. Lesson recorded (PLAN_AMENDMENTS 6): the dumps fixture does not
  create its output directory — pre-create it.
- **2026-09-18/19 — Part A landed (steps 1–4), all default-off.** Step 1 (pool sampler, per-type
  scales with the −1 sentinel, slot-gated tracking telemetry, the six knob rows + the clock rule):
  unit suite **1007 passed / 5 skipped / 303 deselected / 32 pre-existing collection errors**,
  reproduced 3× identically; the independent HEAD-vs-working-tree differential (5 cfg scenarios × 6
  rounds × 18 tensors, int-view bitwise) reports **FAILS: 0** with identical RNG counts
  (`rand` 30, `rand_elems` 4090) on both builds, the cfg field delta exactly `{cmd_pool}`, the legacy
  nine-name metric set unchanged and `type_id == −1` on every legacy row. Step 2 (`reward_facing_turn`
  as the 37th name at weight 0.0, the facing-coupled tracking twin, the `sincos` encoding and the
  encoding-aware mirror signs): **+31 tests** (rewards 45→55, observations 17→23, mirror 23→33,
  phases 128→133), full suite **1081 passed / 5 skipped / 32 errors**, nine mutations of the new code
  all killed. Step 3 (`facing_closed` in the gait harness, the `run_phases` overlay/interrupt seams):
  narrow suite **251 passed / 4 skipped / 32 errors**, full **1107 passed**, a 13/13 mutation sweep,
  and a real-data backward-compat smoke on a COPY of the formation campaign's state — `state.json`
  55214 B and `heads.json` 665 B re-saved **byte-identical**, `summarize()` reproducing
  `tripod_walking_dir` 1a 0.267 / 1b 0.405 / 1c 0.489 / 2b 0.377 and `yaw_progress_turn_min` 1c 0.092
  on all eight recorded aggregates. Step 4 (`curriculum__omni` manifest row, the replay's per-term
  weight flags): `scenarios_omni.yaml` **+50 / −0, still strictly append-only**; the no-new-flags and
  pre-existing-flags replay dry runs are `diff`-EMPTY against HEAD, the flagged run prints 18
  turn-hold data rows with no `nan`/`inf`, failure injection degrades to one `error` line with the
  report still written, and the new `segment_id` guard exits 1 on non-`slow__omni` raws.
- **2026-09-19 — step 5 landed the ladder** (`run_candidates.py`, `resolution.py`, `ledger_cli.py`,
  `launch_candidates.sh`, `$LAD/candidates.yaml` / sha256 `7303819e…`, plus
  `test_crab_hex_omni_candidates.py` and `test_crab_hex_omni_resolution.py`). **No line counts are
  recorded here** — the earlier ones (`run_candidates.py` 6214, the candidate suite 4716,
  `candidates.yaml` 520) were copied from the step-5 report and went stale within the day as the
  review fixes landed; what a reader wants instead is `state.ladder.tools_sha`, stamped at the
  first launch over the four tools plus the replay, and `wc -l` on the tree in hand.
  `candidates.yaml` re-validated strict with
  `check_replay=True`: **0 reasons on all 20 subjects**. `decision_rule.text` is decision 5 VERBATIM
  (ASCII transliteration only) and `decision_rule.control_head: 1c-004` is pinned rather than
  inferred.
- **2026-09-19 06:21 — `D00` computed** (`resolution.json`, schema 1): 13 inputs → **9 scored, 1
  process-floor, 3 informational, 0 refused**; every measured number bit-identical to the previous
  pass. The facing re-score window was verified against the recorded per-episode keys with **max abs
  deviation 0.0e+00 on all 13 rows**. Numbers and the RESOLUTION line: see "The D00 record as
  delivered".
- **2026-09-19 06:17 — the process-floor eval** (4 min GPU): `seed3_2b_006_A15pB_floor/slow__omni/
  seed001/2026-09-19_06-17-18` against the recorded 2b `slow__omni` row — completion **0.50 both**,
  `tripod_dir` **0.377 vs 0.38**, `yaw_progress_turn_min` **0.0715 vs 0.071**, `raw/episode_00.npz`
  md5-identical. Process floor **0.0 on all 26 metrics, by construction** (amendment 19).
- **2026-09-19 — ONE orchestrator ratification and three step-5 findings** (PLAN_AMENDMENTS 16–19),
  all recorded in Caveats 26–29. The **ratification** is 16: ring completion is SCORED as a guard
  from M2 (R ±0.165 — decision 6's escape clause fires, the MILESTONE line names the branch, both
  branches pinned). The other three are step-5 **findings carried as caveats**, not decisions the
  orchestrator took: 17, `turn_survival_r90`'s cross-head R is degenerate, so a lone WORSE on it is
  to be read as uncertain until `D00b` widens it (no code change); 18, `T_abs ≡ T` for every non-cap
  candidate, only `F_abs` differs; 19, the D00 process floor is 0 by construction and any future
  floor must vary `env_seed`.
- **2026-09-19 — two further orchestrator ratifications** (PLAN_AMENDMENTS 23–24, Caveats 33–34),
  both from the step-7 review: `decide_pooled` checks the two-seed sign DISAGREEMENT before the
  MISSING-member `uncertain` cap (the plan's order), and `overruled → confirmed` by the user is
  legal only for `overruled(driver|milestone)` entries.
- **2026-09-19 — Verifications 1–3 (Part F).** **V1** the unit suite as instructed
  (`--continue-on-collection-errors`): **1438 passed**, the 32 pre-existing collection errors in
  unrelated trees, no failures. **V2** the cfg-identity dumps: **215 passed**, and all **12/12
  forward + sideways dumps byte-IDENTICAL** to the step-0 capture, with the omni delta exactly the
  three default-off additions (`cmd_pool`, the `facing_obs_encoding` declaration, the
  `reward_facing_turn` block at weight 0.0). **V3** the legacy twin smokes: **1c bitwise**, **1a
  bitwise modulo one run-to-run nondeterministic `priv_reg` print digit** (0.3693 / 0.3694, flipped
  back by a second run); **37 distinct reward labels** under the C4/C5/C6 overlay; and `env.yaml`
  recording the swapped tracking `func`, `facing_sigma`, `sincos` and the `cmd_pool` string. Caveat
  32 carries the evidentiary limit (the smoke is 4-decimal on 1a/1c only; the bitwise claim is the
  tensor differential's). Record: `parkour/outputs/omni_plan/smokes_ladder/RESULT.md`, whose stale
  V3(a) verdicts are marked "superseded by the addendum" in place.
- **2026-09-19 — the step-7 review** (eight dimensions, read against the code and the records rather
  than the step reports): verdict **GO-WITH-FIXES** — one blocking defect (the squash view was not a
  pure function of the ledger, so a revert resurrected only one dependency level), ten substantial
  code findings, this records batch, a coverage batch and a minor list; nothing found touched B0's
  s3/s2 windows or the D00 record. The **fix batch of 2026-09-19** landed them before the Part G
  commit and the B0 launch: the driver fixes are Caveat 36 (amendment 26), the two ratifications
  Caveats 33–34 (23–24), the four recorded narrowings Caveat 35 (25), and the records corrections
  are in this file — the C9 row and yaml `question` (the replay alias DOES arm the turn-hold table),
  this Log's ratification count, the `n_valid` column, the σ wording, the mechanism-1 provenance
  note, and the closing entry of `$CAMP/CHARTER.md`.
- **2026-09-19 — `code_sha` at HEAD `5030872` is `c4c8abd522b3db36…`** with **13 dirty code paths**
  (the step 1–4 edits), so a real launch is REFUSED until Part G commits 1–4 land. With the working
  tree exactly as it stands, the post-commit hash is **`4c3f99c9a2c5f279…`** — `code_sha` reads
  `git ls-tree -r HEAD`, so committing the code moves it; step 5's report named `c4c8abd5…` as the
  post-commit value, which is HEAD's value instead. The authoritative number is whatever
  `state.ladder.code_sha` records at the first launch, and the record commits of this CHARTER, the
  README and `docs/FOLDER_LAYOUT.md` do not move it. **No `tools_sha` is recorded here:** the three
  ladder tools sit OUTSIDE `code_sha` and kept changing through the review, so any value written in
  this file is stale the moment a fix lands. `tools_sha` is informational, never a refusal, and the
  authoritative value is the one `state.ladder.tools_sha` stamps at the first launch (the dry run
  prints the same hash over tools + replay).
- **2026-09-19 — step 6 wrote the records** (this CHARTER, the omni README's pool/knob/ladder
  sections, the `docs/FOLDER_LAYOUT.md` tree lines, and the closing Log entry in `$CAMP/CHARTER.md`).
  `parkour/scripts/README.md` was checked and left alone: it documents the USD/asset scripts and has
  never mentioned the omni drivers. The step-7 review then corrected them (see the review entry
  above); `candidates.yaml`'s only edit was C9's `question`, after which `resolution.py` was re-run
  and every `R_eval`, block and guard number came back bit-identical — the new `resolution.json`
  differs from the 06:21 one only in `created` and `candidates_sha256`.
- **2026-09-19 — the two post-review fix rounds and their re-verification.** Round 1 closed the
  synthesis list F1–F11 and the queue (transitive resurrection with a launch reconcile, `--accept`
  only from `await`, milestone kinds unoverrulable, the anchor's pooled level, the default-clip cap
  rule, the composed final-stage pool, `_wait_isaac_clear` before every launch, the replay timeout,
  pooled levels on the DECISION line, already-applied override relay, seven functions given real
  tests, the transition guard on resurrection, `s3 + s2`, metric-family labels, the torn override
  line): suite 1445 → 1478. Three fresh lenses then found four substantial gaps, closed by round 2
  (caveats 43–44: canonical pool rendering, reconcile after every recipe-moving decision write, the
  milestone verb's already-applied guard, the anchor under the current `code_sha`; plus the five
  small items in the queue's R5): suite → **1491 passed, 5 skipped, 32 pre-existing collection
  errors**, `--dry-run` rc 0 with B0 and every `scratch_1c` recipe id unchanged and every pooled
  recipe id moved once to its canonical value (nothing had been recorded under the old ids). A final
  single-lens review of round 2 precedes the Part G commits.
- **2026-09-19 11:08–11:11 — Verification 8, the stop drill, PASSED (stop half).** On a copy of this
  directory (`/tmp/ladder_v8`, `candidates.yaml` + `resolution.json` only) the launcher started unit
  `cands-ladder_v8-1789830527` with `--smoke-stage 1a:20` (`code_sha 4c3f99c9a2c5` — the committed
  tree); with the 20-iteration training child alive, `systemctl --user stop <unit>` returned in **46 s**
  (budget 360 s): the driver logged "stop requested — terminating the training child", classified the
  window `running (sigterm) — no decision from this experiment`, wrote `phase = await_user:stop` and the
  `AWAIT await_user:stop | stopped (sigterm) | records are complete; relaunch with …` line; zero
  `_r2`/retry lines; no child left behind. The resume half (a relaunch picks the `inflight` header up)
  is recorded below.
- **2026-09-19 11:11 — Verification 8, the resume half, PASSED.** Relaunching the drill directory
  (`cands-ladder_v8-1789830676`) found the `inflight` header, logged "X01:1a — resuming at model_19.pt
  (1 iterations left of 20)" and "the window had already reached its final iteration before the stop
  -- continuing at its eval", and closed the smoke hook at `await_user:smoke_done` ("relaunch without
  --smoke-stage to run the campaign"); the unit exited on its own in 5 s, no child left. Verification
  1–8 is therefore complete before the first real window.
- **2026-09-19 11:12 — CAMPAIGN LAUNCHED.** Unit **`cands-2026-09-19_facing_tracking_ladder-1789830730`**
  (`launch_candidates.sh --campaign-dir $LAD`), `code_sha 4c3f99c9a2c5`, `tools_sha 28eb7860f4fa`, on
  the committed tree (81a1aa3). The driver ingested `resolution.json` and emitted `RESOLUTION D00`
  (blocks F R=0.550@0.95, T R=0.518@0.95, F_abs R=0.578@0.95, T_abs R=0.518@0.95; guards @0.99
  one-sided completion ±0.155, tripod_walking_dir ±0.042, turn_survival_l90 ±0.079, turn_survival_r90
  ±0.028) and started **`X01` = B0 seed 3** (`1c` preset from scratch, 8000 iterations,
  `seed3_X01_1c_A15pB`). Heartbeat every 1800 s (`/tmp/heartbeat_ladder.log`); the assistant's
  persistent monitor tails `notify.log` and relays every line as a push notification. Expected next
  lines: `DECISION D02` (B0 s3 baseline, ≈ 3.7 h + evals), then `X02` = B0 seed 2, then `RESOLUTION
  D00b`. The `DerivedDataCache.corrupt-2026-08-13` quarantine (101 GB) was deleted on the user's instruction
  at 11:45 the same day (disk 477 → 377 GB used; the live 4.0 GB cache and the running unit untouched).
- **2026-09-19 14:15 — B0 seed 3 (X01) landed; `MemoryPeak` measured.** `systemctl --user show <unit> -p MemoryPeak` after the
  8000-iteration window and its `slow__omni` eval: **9.27 GB** (`MemoryPeak=9271455744`) — the 48 G cap holds a 5× margin and is
  not lowered yet (the ring/light evals of the milestones and the 100-episode evals have not run under this unit).
- **2026-09-19 14:15 — Verification 6 (B0 pre-flight) PASSED on X01.** `overlay_checks` empty (the `1c`
  preset carries no overlay — nothing to audit), the `slow__omni` eval's `run_meta.env_vars` carries exactly
  the seven `EVAL_CARRY` keys (the two defaulted ones included: `FACING_OBS_ENCODING=raw`,
  `ANG_VEL_CLIP=0.2`), `KRABBY_RSI_FRAC=0`, `KRABBY_PLAY_DEBUG_VIS=0`, ring geometry; `eval_status ok`;
  `decision_table.py --phases X01:1c` renders the row; `D02` (baseline, final) and `D03` (dead_walker floor
  0.422 = max(0.35, 0.75 × 0.563)) written. **B0 seed 3 screen:** completion 0.56, `tripod_walking_dir`
  0.384, speed ratios fwd/lat_l/diag/back 0.602/0.542/0.553/0.611 (move_turn 0.441, WATCH), yaw progress
  0.071/0.070 rad/s, `facing_closed` turn_l90 0.544 / turn_r90 0.845 / move_turn 0.132, finals turn_l90
  0.724 rad, move_turn 1.295 rad, turn survivals 0.952/0.983, training `track_ratio_tail` 0.637
  (`track_ratio_move_tail` 0.636), 44/100 falls. Within D00 of the 1c-004 prior row and the 2b lineage on
  every scored quantity — the incumbent is what it was. X02 = B0 seed 2 started 14:15.
- **2026-09-19 17:19 — B0 seed 2 (X02) landed; `RESOLUTION D00b` written; C1 started.** X02 screen:
  completion 0.68, `tripod_walking_dir` 0.472, speed ratios fwd/lat_l/diag/back 0.628/0.462/0.542/0.583
  (move_turn 0.353), yaw progress 0.081/0.084 rad/s, `facing_closed` turn_l90 0.629 / turn_r90 0.860 /
  move_turn 0.310 — a better walker than seed 3 on completion, tripod and facing, which is what the seed
  terms now carry. **D00b:** the control check (B0 s2 submitted against B0 s3) decides all-TIE on every
  block and guard, in both unit systems; `R_F(D00b) ≥ R_F(D00)` holds on every block. Block levels from
  here on: **F R = 2.043** (R_eval 0.550, R_seed 2.043), **T R = 0.597** (0.518 / 0.597), F_abs 1.993,
  T_abs 0.597 — the facing block's seed term is ~4× its eval half-width, so a single-seed screen can
  clear F only by moving the standardised block mean by more than two eval half-widths (≈ +0.12 on the
  turn-hold closed fractions and +0.015 rad/s on the progress rates at once); most facing candidates
  will therefore land UNCERTAIN and run their pairs — the designed behaviour, not a defect. Guards:
  completion ±0.155 (eval; seed 0.120), tripod_walking_dir ±0.088 (seed), turn_survival_l90 ±0.079 and
  turn_survival_r90 ±0.028 both still carried by the degenerate eval half-width (caveat 27: a lone
  WORSE on either is read as uncertain and confirmed by the user). `MemoryPeak` after two windows
  10.67 GB. **X03 = C1 (curriculum) stage c1 started 17:19** (1000 it, `1a` base + the c1 pool).
- **2026-09-19 19:19 — `DECISION D05`: C1 (the user's additive curriculum) REJECTED by the stage-c4 FLOOR**
  — survival through `move_turn_r` on the curriculum row 0.230 (floor 0.30) at 4500 of 8000 iterations;
  clear, no pair; the c4 head kept (`…_seed3_X03_c4_A15pB/model_4496.pt`). Profile: through the ninth hold
  0.40 (c1, forward-only) → 0.00 (c2, lateral+back added) → 0.23 (c4, turning added 2500 it earlier;
  yaw progress 0.034/0.031 rad/s, speed ratios fwd/lat/back 0.65/0.45/0.60) — a chain still forming, not
  a dead one; the 0.30 floor was set without a B0 number on this row (B0 has no curriculum-row eval) and
  is uncalibrated. Driver proceeded to **C1b** (X04, the same chain with `ang_scale 0.3` on the `*_turn`
  types; its c1/c2 reproduce C1's). **User decision (19:40): option 1 — wait for C1b's own c4 result**
  before relaxing the floor; if C1b also lands near 0.2, the floor (not the demand) is the problem and C1
  is re-run under a relaxed c4 floor (`candidates.yaml` c4 `min`, relaunch with
  `--accept-candidates-change`, `decide C1 --choose run`); if C1b clears 0.30, the turn demand was the
  problem and the ladder's own answer stands.
- **2026-09-19 20:14 — `DECISION D06`: C1b REJECTED by the stage-c2 FLOOR (survival through `back` 0.250 vs
  0.30); `D07`: C1c SKIP (predicate `decision(C1).chosen == ADOPT` false); X05 = C2 (yaw-progress 3.0,
  `scratch_1c`) started 20:15.** The decisive evidence on the floors: C1b's c1 reproduced C1's c1 to the
  digit (survival 0.81/0.78/0.63/0.59/0.40, ratios 0.62/0.19/0.60, track tail 0.664 — deterministic), and
  its c2 ran the IDENTICAL overlay (the `*_turn` scale touches no c1/c2 type) yet landed at 0.25 through
  `back` where C1's c2 landed at 0.36 (fwd survival 0.59 vs 0.71; lateral ratio 0.41 vs 0.36) — the
  run-to-run divergence begins at c2 (RSI resets enter there). Two identical configurations straddle the
  0.30 floor, so the c2/c4 floors gate on run noise (band ≈ ±0.1 at 2000 iterations from scratch), not on
  whether the chain is alive. Recommended amendment (awaiting the user): relax the non-final floors to
  `min: 0.10` (a dead chain, not a forming one), keep the final screen as the judge, relaunch with
  `--accept-candidates-change`, `decide C1 --choose run` (queued after X05 resumes); C1b re-queued by hand
  only if C1's re-run is rejected at its screen.
- **2026-09-19 20:25 — CAMPAIGN PAUSED by the user** ("pause for now; resume later with a more narrowly scoped
  plan"). Normal stop: `ledger_cli stop` → the driver terminated X05's training child (C2, `scratch_1c`,
  interrupted at `model_400.pt` of 8000, header left `inflight` and resumable), parked at
  `phase = await_user:stop` in 3 min, and the unit went inactive; the `STOP` file stays in place so a
  relaunch is refused until `ledger_cli unstop`. Ledger at the pause: D00, D01, D02, D03, D04, D00b
  (records), **D05 C1 REJECT (c4 floor)**, **D06 C1b REJECT (c2 floor)**, **D07 C1c SKIP**; two B0
  screens (X01 s3, X02 s2) are the incumbent; no candidate has been adopted. **Open question for the
  next plan:** the non-final curriculum floors (c2/c4 at 0.30) gate on run-to-run noise (an identical c2
  configuration scored 0.36 and 0.25); the prepared amendment (floors → 0.10, re-run C1) was NOT applied.
  The approved plan is saved verbatim beside this file as `PLAN_2026-09-18_facing_tracking_ladder.md`
  with the deviation log `PLAN_AMENDMENTS.md`. **To resume as-is:** `ledger_cli unstop`, then
  `launch_candidates.sh --campaign-dir $LAD` (X05 resumes from its last checkpoint; the queue continues
  C2 → C3a → …). To resume under a narrowed plan: edit `candidates.yaml`, relaunch with
  `--accept-candidates-change "<why>"`; a code change needs `--accept-code-change` and a `B<k>` rerun.
- **2026-09-21 — PER-CATEGORY EVALUATION PLAN APPROVED; the campaign stays paused until it lands.** Saved verbatim beside
  this file as `PLAN_2026-09-21_per_category_eval.md` (610 lines). It supersedes this plan's decisions 4 (floors), 5
  (blocks/guards by hold) and A6 (the `curriculum__omni` row) and leaves the ranked candidate list, the speculation/squash
  protocol and the milestone structure intact. What it changes: one manifest row `suite__omni` runs ten waves in a single Kit
  launch, each env taking one single-category episode (2 s settle at the pinned yaw, then 12 s of that category's command), so
  survival is the hold's OWN completion and no category's number depends on another's; seven categories (`stand, fwd, side,
  back, diag, turn, move_turn`) each score metrics matched to their motion (stillness and drift; speed ratio and a new
  direction error; facing closure, turn rate and settle); a new `eval_categories.py` derives which categories a recipe's
  command distribution actually trains, and blocks, guards, stage floors, RETENTION and milestones are restricted to those
  while every untrained category is still run and printed as a WATCH row. `move_turn` is decisive in its own block `F_mt`
  (user amendment) with `cat.move_turn.ratio` as the anti-sacrifice guard. The stage floors become per-category, absolute
  (0.10 newly enabled / 0.30 previously enabled) and BAND-tested at 99 %, which is what un-inverts the C1 and C1b rejections:
  neither would have fired. An `eval_sha` split from `code_sha` (with a proof that the training tree did not move - verified
  identical at `81a1aa3`, `75dd943`, `9f61eb2`, `9c2c1fba`) lets the migration RE-EVALUATE the recorded heads instead of
  retraining B0: rescreen X01/X02, reeval the three D00 heads, write `D00s` then `D00bs`, rescore X03:c4 and X04:c2 and resume
  those chains from their next stage, then the ordinary queue. Audit: one pass, three lenses, 9 blocking + 19 substantial + 44
  minor findings all applied; further passes declined by the user, so execution steps 1, 3 and 5 carry the review lenses.
- **`MemoryPeak` policy.** The launcher runs the driver under `MemoryMax=48G`,
  `MemorySwapMax=0`, `OOMPolicy=continue`, `KillMode=mixed`, `TimeoutStopSec=360`; the measured
  `MemoryPeak` of the unit is recorded HERE after `B0` completes (`systemctl --user show <unit> -p
  MemoryPeak`), and the cap is lowered only with that evidence.

## Per-category evaluation (2026-09-21)

Plan of record: `PLAN_2026-09-21_per_category_eval.md` (saved verbatim beside this file; audited in
five further passes after its pass-1 review — passes 2–6, `PLAN_AMENDMENTS.md` item 30). It answers
the paused campaign's floor rejections (D05 C1, D06 C1b): their `curriculum__omni` floors read
*prefix survival* on a ten-hold episode, which counts every earlier fall regardless of whether that
hold was trained, and is noisy at the threshold (two identical c2 configurations scored 0.36 and
0.25 against a 0.30 floor). It replaces this CHARTER's Decision rule (decision 5) and Resolution
policy (decision 6) sections above wherever they conflict — those sections stay as the record of the
2026-09-19 rule; the ladder's ranked candidate list, speculation/squash protocol and milestone
structure are otherwise unchanged. **Supersedes:** the 2026-09-18 plan's decision 4 (floors),
decision 5's blocks/guards by hold, and decision 6's curriculum-row borrow (`BORROW` /
`--curriculum-runs`, retired behind `rule_version == 2`); decision 6's resolution / seed-term /
pooled-R machinery stays.

**Vocabulary.** **category** — one of `stand, fwd, side, back, diag, turn, move_turn` (mirror pairs
share a category; **hand** l|r reports handedness inside it). **hold** — one command row of the
suite. **suite** — the manifest row `suite__omni`. **decisive / present / watch** — the categories a
stage's command distribution trains at/above, at all, and below the pool floor share
(`eval_categories.trained_categories`); the decisive set of a comparison is the CANDIDATE's decisive
set. **eval_sha** — a sha256 over the eval-side files, split from `code_sha`. **rescreen / reeval /
rescore** — the three eval-only migration items (D9); none trains. **N/A block** — a block with no
decisive member; never TIE, never MISSING.

### D1–D3: the suite and its metrics

`suite__omni` (`kind: category_suite`) is ONE Kit launch, TEN waves in wave order; wave k = every
env runs `[settle 2 s, hold_k 12 s]` at 100 envs/wave (1000 episodes total,
`env_index = wave*100 + env`), so a fall in one hold never contaminates another hold's number — the
fix for the prefix-survival contamination above. The harness re-seeds before every wave
(`env.seed(env_seed*1_000_003 + wave)`) and resets between waves ON THIS ROW ONLY; every legacy row
(`slow/light/ring/curriculum__omni`) runs its single-pass loop unchanged, because that reseed writes
process state and any env write after Kit startup can flip a same-seed eval (see "Isaac headless
launch quirks" / process-state sensitivity). Holds, in wave order:

| wave | label | category/hand | speed | dir (world) | face | note |
|---|---|---|---|---|---|---|
| 0 | `stand` | stand | 0.00 | 0 | 0 | |
| 1 | `fwd` | fwd | 0.25 | 0 | 0 | = legacy |
| 2 | `lat_l` | side/l | 0.22 | +π/2 | 0 | = legacy |
| 3 | `lat_r` | side/r | 0.22 | −π/2 | 0 | = curriculum row |
| 4 | `back` | back | 0.22 | π | 0 | = legacy |
| 5 | `diag` | diag | 0.25 | +π/4 | 0 | = legacy |
| 6 | `turn_l90` | turn/l | 0.00 | 0 | +π/2 | = legacy |
| 7 | `turn_r90` | turn/r | 0.00 | 0 | −π/2 | NEW meaning: 90° right from face 0 (legacy: the return turn +π/2 → 0) |
| 8 | `move_turn` | move_turn/l | 0.22 | 0 | +π/2 | = legacy |
| 9 | `move_turn_r` | move_turn/r | 0.22 | 0 | −π/2 | mirror of `move_turn` |

`n` = 100 per hand, 100 for a singleton category, 200 for a paired one; the floor band and RETENTION
read the distinct-env count `n_env` (100 for every category — a paired category's two hands share
the same startup-randomised robots). New per-hold metrics, world rows only, gated like `facing`:
`motion` (`speed_abs_mean_m_s, pos_drift_m, path_len_m, yaw_drift_rad, foot_lift_count,
contact_churn_per_s, n_steps`) and `direction` (`direction_error_rad, cross_track_ratio,
travel_speed_m_s`, commanded-speed-> 0 holds only) — `gait_eval.metrics.hold_motion_metrics` /
`travel_direction_metrics`. The suite's `aggregate.suite` block carries `hold_labels` (the 11-label
union), `hold_of_wave`, `category_of_hold`, `hand_of_hold`, `n_by_hold`, per-hold OWN-hold
`survival_by_hold`, `trained_categories`, and `by_category` — `report.aggregate`'s pooled-both-hands
reduction (never a mean of hand medians) that `summarize()`'s `cat` dict (D5) reads. Category metric
roles:

| category | decisive members | guards | WATCH |
|---|---|---|---|
| stand | — | `survival` | `drift_speed`; per-hold `motion_by_hold` |
| fwd, side, back, diag | T: `ratio`; T_abs: `achieved_speed` | `survival`; `cat_tripod_min` | `direction_error`, `cross_track_ratio`, `heading_final` |
| turn | F: `facing_closed`, `yaw_progress`; F_abs: `facing_closed`, `net_yaw` | `survival` | `settle_s`, `heading_final` |
| move_turn | F_mt: `facing_closed`, `yaw_progress`; F_mt_abs: `facing_closed`, `net_yaw` | `survival`; **`ratio`** (`achieved_speed` under `absolute_units`) | `tripod`, `achieved_speed`, `direction_error`, `cross_track_ratio`, `settle_s` |

`move_turn` is decisive in its OWN block `F_mt`/`F_mt_abs` — user amendment 2026-09-21, applied after
the pass-1 review — because the hold's demand is infeasible by construction and it is where the
sacrifice the campaign exists to fix is largest (the 2b head ends at 1.218 rad there versus 0.68 on
the in-place turn); the separate block keeps it from drowning, or being drowned by, the in-place
turn's F, and `cat.move_turn.ratio` (or `achieved_speed` under `absolute_units`) is a GUARD so a
candidate cannot win F_mt by walking less. `drift_speed` stays WATCH this window (lower-is-better; a
standing collapse is still caught by `cat.stand.survival`).

### D4: trained categories (`eval_categories.py`)

New Isaac-free module at the omni task root (`$OMNI/eval_categories.py`, outside `mdp/`/`config/`,
hence in `eval_sha` not `code_sha`), loading `crab_hex_omni_command_math.py` and `omni_knobs.py` BY
PATH (never a dotted import, which would run Isaac's `import_packages`).
`trained_categories(env) -> TrainedCategories(decisive, present, watch, hands, demand, source,
facing_band_hi, encoding, notes)` reads the stage's COMPOSED training env (never the overlay alone):
`present` = share > 0, `decisive` = share ≥ `POOL_FLOOR_SHARE` at the suite's full demand, `watch` =
present − decisive with a note (`watch (reduced demand)` / `watch (one hand)` / `watch (below floor
share)`). `CATEGORY_HOLDS`, `HAND_OF_HOLD`, `TYPE_TO_CATEGORIES` map pool types to categories
(`omni` → all four bearings; `fwd_turn|side_*_turn|back_turn|omni_turn` → `move_turn` alone); a scale
edit demotes ONLY the categories its own type maps to (`{"*_turn": 0.3}` touches `move_turn`, never
`turn` — this is what un-inverts the C1b/C7 floor reading). `turn` is decisive for every recipe
regardless of the FACING band or `sincos` encoding (`turn_annotation` prints the band/encoding as
information on the RESOLUTION/DECISION lines, never a demotion). The two pure helpers both tools
share, never re-implemented: `expand_rule(rule, decisive, present, *, absolute_units)` and
`composite_guards(screen, decisive)`.

### D5: decision rule v2

`summarize()` gains `cat[c] = {n, n_env, survival, ratio, achieved_speed, tripod, facing_closed,
yaw_progress, net_yaw, settle_s, heading_final, drift_speed, direction_error, cross_track_ratio,
by_hand}` per category, read from `aggregate.suite.by_category`. `decision_rule` becomes
`rule_version: 2` with `per_category` templates:

> F `{turn: [facing_closed, yaw_progress]}`; **F_mt `{move_turn: [facing_closed, yaw_progress]}`**;
> T `{fwd|side|back|diag: [ratio]}`; F_abs `{turn: [facing_closed, net_yaw]}`; **F_mt_abs
> `{move_turn: [facing_closed, net_yaw]}`**; T_abs `{...: [achieved_speed]}`; guards
> `{every category: [survival]}` + **`{move_turn: [ratio]}`** + `composite: [cat_tripod_min]`, and
> **`guards_abs: {move_turn: [achieved_speed]}`** in place of the ratio guard for an
> `absolute_units` candidate.

`eval_categories.expand_rule` turns this into today's flat `{blocks, guards, watch}` shape
restricted to the decisive set (ids `cat.<c>.<key>`, `cat_tripod_min`), keeps every PRESENT
category's `survival` guard, and moves every other member/guard of a non-decisive category to WATCH
with its note. **The block set is derived from the expanded rule everywhere it is enumerated, never
hard-wired** — a candidate scores `{F, F_mt, T}` or, under `absolute_units`, `{F_abs, F_mt_abs,
T_abs}`. A block with no member is **N/A** (never TIE); an all-N/A comparison is a terminal
`status: decided, chosen: SKIP, confidence: none` with no pair (squash cause (b) exempts it from ever
being re-screened). Resolution (D00s/D00bs) uses `expand_rule_union` — the UNION of both unit
systems — so an `absolute_units` candidate is decidable. Guards are one-sided at 99 % on
`R_i = max(R_eval,i, R_seed,i, R_binom,i)`; `R_binom` (survival guards only) uses the clipped
`p̄ = clip((surv_a+surv_b)/2, 0.05, 0.95)`, so a survival saturated on both heads floors at 0.072
(n=100) / 0.051 (n=200) instead of reading MISSING. **Speculative sign:** the all-TIE pick and
`decide_pooled`'s per-seed/pooled-fallback signs use `sign(Σ_b z_b / R_b)` over the non-N/A blocks —
each in units of its own resolution, because F_mt's seed spread is ≈ 3× F's and a raw `Σ z_b` would
be dominated by it.

### D6: floors and RETENTION

`floor: {kind: category_survival, row: suite, min_new: 0.10, min_prev: 0.30}` on every non-final
chain stage. For each decisive category: `min_prev` if it was decisive at any earlier stage
(`eval: []` stages included), else `min_new`. The test is a BAND test — FAIL only when the one-sided
99 % upper binomial bound on `n_env` (the category's distinct-env count, 100 for every category) is
below the threshold (effective point thresholds at n=100: passes down to p̂ ≈ 0.05 new / ≈ 0.21
previous) — which is what un-inverts C1's and C1b's 2026-09-19 rejections: on the recorded per-hold
survivals (C1 c2 `back` 0.58, C1b c2 0.52, C1 c4 `move_turn` 0.66–0.76) neither floor would have
fired. RETENTION compares `cat.c.survival` and a per-category retention metric (`ratio` for
translation, `facing_closed` for turn/move_turn plus `ratio` for move_turn, `drift_speed` for stand
— lower is better) against the incumbent trace, per metric-kind threshold (survival 0.10, `ratio` a
20 % relative drop, `facing_closed` an ABSOLUTE drop of `max(0.10, R_eval)`, `drift_speed` a rise of
≥ 0.02 m/s absolute AND 20 % relative); WATCH + notify, never a gate.

### D7: milestones

The 2026-09-18 three-row verdict structure is KEPT with the suite in place of `slow`: the 2b head
FAILS iff ≥ 2 of the three rows {suite, light, ring} are WORSE. The **suite row** is WORSE iff ≥ 2
decisive UNITS are WORSE (a unit = a block, or a counting guard; `cat_tripod_min` does not count);
nominal suite-row WORSE rate under the null ≈ 1 % at B0's set (3 blocks + 8 counting guards).
**light and ring keep the frozen v1 per-row rule byte-for-byte**, scored from three named sources:
the rule (D00s copies D00's frozen `decision_rule` verbatim as `by_row_rule`), the R (D00's copied
`by_row` tables) and the seed terms (D00b's v1 terms) — so their scoring never changes under this
plan.

### D8: `eval_sha`

`CODE_PATHSPECS` loses `scenarios_omni.yaml`, `run_phases.py` and `gait_eval`;
`EVAL_PATHSPECS = (gait_eval/, eval_crab_hex_gait.py, scenarios_omni.yaml, run_phases.py,
eval_categories.py)`. Every header/entry carries `eval_sha`; `ok_screen` additionally requires it to
match the current one, so a screen measured on the old eval is never a control or an anchor until
rescreened. A changed `eval_sha` is refused without `--accept-eval-change WHY`; **a campaign with a
recorded `code_sha` and no `eval_sha` counts as changed** (every pre-suite campaign, this one
included) — `main()` runs the acceptance path and its ALIAS PROOF before the `code_sha` comparison,
so a real launch is refused naming `--accept-eval-change`, never `--accept-code-change` (which would
queue an 8000-iteration B0 rerun). The alias proof: `code_sha_at(commit, pathspecs)` finds the newest
commit whose training-side tree (`CODE_PATHSPECS_V1`, the pre-split pathspec) equals the recorded
`lad.code_sha`, then proves that SAME commit's narrowed-pathspec tree equals the current `code_sha()`
— measured identical at `81a1aa3`, `75dd943`, `9f61eb2`, `9c2c1fba`, `464be672` (`lad.code_sha` =
`4c3f99c9a2c5…`, the narrowed value `a617f5ee59d1…`) — else refused ("the training tree also moved:
use `--accept-code-change`"). A rescreen's old screen is kept under `header.screens_superseded`,
never overwritten.

### D9: migration and the resume sequence

`migration_item(ctx)` runs BEFORE `find_inflight`/`_next_item_raw` (whose baseline-rerun rule would
otherwise book an 8000-iteration B0 rerun the moment `ok_screen` goes stale). Order, each item
eval-only and notified `MIGRATION` (a rescore PASS under `MIGRATION` too; only a rescore FAIL, which writes a REJECT entry, under `DECISION`):

1. **rescreen X01, then X02** — every stale-`eval_sha` `ok` header on the suite, filed beside the
   kept `slow` row (old screen kept under `screens_superseded`).
2. **reeval** the three D00 inputs (`1c-004`, `2b`, `2a-reeval`) on the suite.
3. **`D00s`** — resolved IN-PROCESS from the cross-head suite pairs (env-clustered paired bootstrap,
   B = 2000), with D00's `light`/`ring` `by_row` tables and rule copied across; written to
   `resolution.<eval_sha[:12]>.json`.
4. **`D00bs`** — the seed terms per block/guard from the two rescreened B0 screens against `D00s`.
5. **rescore** the floor-rejected chains: X03 (C1) at its c4 checkpoint, X04 (C1b) at its c2
   checkpoint, each re-decided under D6 on that stage's trained categories; **PASS** re-opens the
   chain at its next stage (C1 → c5, C1b → c3, the latter `rescored_awaiting_predicate` since C1b
   `depends_on: [C1]`); **FAIL** keeps the REJECT effective, now under the current `eval_sha`.
6. **the ordinary queue, in rank order** — a `candidate` header resumes only when `_next_item_raw`
   would dispatch it NOW, so **X05 (C2) parks until C1, C1b and C1c are terminal** (logged "not
   resumed (waiting on C<id> at rank k)").

Both D00s and D00bs compute their acceptance (a null band / a construction check) BEFORE any write;
a failure writes only the diagnostic `resolution.<eval_sha[:12]>.<kind>.rejected.json` (kind =
`D00s`|`D00bs`) and halts `halt:resolution <kind>` (stop-and-wait) until `--accept-resolution <kind>
WHY` is given at relaunch. A failed migration item halts the CAMPAIGN (`halt:migration <item>
<why>`), never a recorded header.

### D10: resolution on the suite

`bootstrap_pair` draws clustered on the WITHIN-WAVE env index (both heads share the draw; a paired
category's 100/100 hand split and the within-env correlation across categories, including
`cat_tripod_min`, are preserved). A stamp is read only after `resolve_eval_dir`'s FRESHNESS check
(`complete.json`, written LAST by `write_run`, plus a checkpoint-sha match) — never a partially
written raw set.

### Relaunch (execution step 8)

```bash
$PY $TOOLS/ledger_cli.py --campaign-dir $LAD unstop
"$OMNI/experiments/tools/launch_candidates.sh" --campaign-dir "$LAD" \
  --accept-eval-change "per-category suite (user 2026-09-21)" \
  --accept-candidates-change "categories, floors, rule v2"
```

Budget before the first training window resumes: 7 suite evals (2 rescreens, 3 reevals, 2 rescores)
≈ 50–60 GPU-min plus the D00s bootstrap (CPU) and ≈ 6 GB of raws/JSON, then C1 c5–c6 (≈ 1.7 h
including its stage evals) before C1's screen; every later screen and promotion carries one suite
(≈ 0.75–0.95 GB). Read, in order: the MIGRATION lines, the `D00s` then `D00bs` RESOLUTION lines (the
F and F_mt seed-dominance sentences first — a `halt:resolution` stops here), then the C1 and C1b
rescore verdicts, then the queue.

### Caveats carried from the plan

1. `move_turn` is scored RELATIVELY — its facing block compares two heads under the same infeasible
   demand, and `cat.move_turn.ratio` guards against winning it by walking less; its tripod and
   direction numbers stay WATCH.
2. `turn` is decisive for every recipe although each trains a narrower FACING band than the suite's
   90° demand; the comparison is symmetric and the band/encoding are printed, never a demotion.
3. The facing blocks are expected to stay seed-dominated, F_mt far more than F: on the recorded B0
   pair split along the v2 blocks, `R_F^seed` ≈ 1.2 and `R_F_mt^seed` ≈ 3.7; `D00bs` measures every
   block before any decision and states per block whether a single-seed screen can clear it at all.
4. `run_phases.py` sits on the eval side of the hash; its training seam is pinned by a source hash
   (`test_run_phases_training_seam_is_unchanged`), not by `code_sha`.
5. The stage floors are absolute and band-tested at 99 % on the category's distinct-env count (100,
   paired categories included); the translation/move_turn proxies are legacy 8-s holds while the
   suite's are 12-s, so the proxies OVERSTATE the suite's survival (≈ p^1.5 on 12 s; still above the floors) — calibrated on
   conditional per-hold survivals, a proxy the C1/C1b rescores validate with the first real
   per-category numbers.
6. A suite eval costs ≈ 2× a legacy row in time and ≈ 3.2–3.5× in disk; raws are kept (re-scoring
   needs them). The migration adds ≈ 1 GPU-h and ≈ 6 GB before the first training window resumes.
7. `turn_r90` and `move_turn_r` change MEANING relative to the legacy rows (both now mirror their
   left-handed twin from face 0); a predicate reading a screen's legacy per-hand key
   (`predicate_context`, e.g. C10's `screen.yaw_progress_turn_min`) now reads the suite screen — C10
   is deferred and its predicate is re-expressed on `cat.turn.yaw_progress` when undeferred.
8. `by_category` pools both hands; a handedness asymmetry is visible only in the WATCH `by_hand`
   rows.
9. Three blocks and nine guards cost sensitivity: a null candidate reads some WORSE about 15 % of
   the time, so a clear REJECT is worth reading before it is trusted; every DECISION line quotes the
   overrule command. The survival guards' binomial floor is unpaired, so a category's survival must
   drop by ≈ 0.10–0.16 (n=100, p̄ 0.9–0.5; ≈ 0.07–0.12 for side/turn/move_turn at n=200) before it
   reads WORSE — fewer spurious REJECTs at the price of sensitivity to falls.
10. A present-but-watch category (e.g. `move_turn` after a C1b/C7-style `ang_scale` adoption) keeps
    only its `survival` guard from then on (`rescale_to` keeps `ang_scale`) — its facing and speed
    are WATCH lines for every later screen and milestone, and only a survival regression can REJECT
    or FAIL.
