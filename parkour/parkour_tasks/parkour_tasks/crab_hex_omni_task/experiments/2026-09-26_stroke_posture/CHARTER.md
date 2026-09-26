# Stroke and posture: replace the dead stride term and test what lengthens omni side steps

A screen of one extra omni phase-1 stage after the head of record W.

- **Plan of record:** `/home/nickmagus/.claude/plans/let-s-examine-the-stride-gleaming-dream.md` (steps 9–15;
  approved by the user 2026-09-25, "run the whole plan without asking").
- **SA0 decisions (binding):** [`SA0_DECISIONS.md`](SA0_DECISIONS.md), with the evidence beside it:
  [`REPLAY_GATE.md`](REPLAY_GATE.md), `gate_p15/`, [`REPLAY_GATE_CHECK.md`](REPLAY_GATE_CHECK.md),
  [`POWER_PREVIEW.md`](POWER_PREVIEW.md).
- **The machine-readable spec:** [`candidates.yaml`](candidates.yaml). The window-0 smoke:
  [`window0_dose_smoke.py`](window0_dose_smoke.py).
- **Investigation and design records:** `/home/nickmagus/.claude/plans/let-s-examine-the-stride-gleaming-dream.evidence/`
  (`wya9ai1ml.json` investigation, `wrup2f3cq.json` design panel, `wkt6sd8vn`/`wddov7duf`/`wy7chufw4.json` audit
  passes 1–3, `plan_reviewed_v1.md`).

## The question

Two questions, one screen:

1. **Is the replaced stride term alive and command-general?** The old `reward_stride_length` is dead in training
   (float32 `compute_first_*` edge bug, ~2 s after each reset) and mis-specified. Its replacement, a per-foot loaded
   stance-stroke income, must pay in real Isaac past 2 s, on every moving command type, at the dose the replay set.
2. **Does stroke income or posture lengthen omni side steps?** The omni side gait takes short steps because the body
   leads its six-foot centroid by ~0.23 m on side holds (a posture outcome, not a stride-reward effect). Arm S pays
   stroke; arm P pays re-centring. LEN (side and diag `steps_per_m`) reads whether either lengthens the step.

## Evidence (short)

- **Stride is dead, negligible and mis-specified** (73-agent investigation, 30 of 32 findings upheld). It logs 4e-5/s
  in training against clock 0.30/s; an event paid through `weight·dt` is ~0.1 % of tracking even when timed right; it
  credits body progress to every planted leg (on side ~40 % real stroke, 35 % slip), pays sway 1.24× on side, pays 0
  on turn-in-place, and projects move_turn on the body command. Air-time has the same edge bug.
- **Symmetry works for every command type.** The 4-image mirror loss is correct for A15+B; no fix. New terms are
  unit-tested for equivariance under {I, M_y, M_x, R_z(π)}.
- **The short side steps are posture.** The sideways head on the same plant, with the same dead stride term, strokes
  long; back at the same speed keeps a full stride. On W (re-measured, n = 200 side episodes): side `body_offset`
  +0.231 m, side `stride_along` 0.083 vs fwd 0.240 (0.35×), side tracking 0.473; side `steps_per_m` 32.1 vs fwd 14.7.
- **Replay gate (step 7) and power preview (step 8):** see SA0_DECISIONS.md. Premise holds (+0.231 m ≥ 0.15;
  0.35 ≤ 0.6). Provisional LEN block R 1.194 (seed-dominated); per-member MDE side 1.644, diag 1.938 steps/m; S's
  minimum effect (+0.02 m side stride) ≈ −6.2 steps/m, above the MDE. Readout-key MDEs: side `body_offset` 0.017 m,
  side `stride_along` 4 %, side `stance_stroke` 0.003 m, side `stroke_share` 0.033, fwd `stance_stroke` 0.027 m.

## Settled decisions

**The user (2026-09-25; plan Context):**

1. **Scope.** Fix the reward mechanics for all command types; add a lead/trail posture metric and one
   posture-targeted control arm.
2. **Stride.** Replace it with a per-foot loaded stance-stroke income. This supersedes reward-simplification
   candidate #2 for stride; air-time is out of scope.
3. **Symmetry.** The mirror loss stays; new terms are only unit-tested for equivariance.
4. **Sequencing.** Built behind default-off knobs; replay gates at 0 GPU-h first. W was adopted, so the recipe is
   clock-free: every stage forked from W carries K (`require_env`), every eval carries `KRABBY_OMNI_CLOCK_HIDE=1`
   (`EVAL_CARRY`).
5. **Adoption** (audit pass 1). S is adopted when it is alive, its extra stride is stroke-borne (not slip), and
   nothing gets WORSE. Longer side steps (LEN BETTER) are not required.
6. **Dose.** One command-general weight, bound by fwd (R4). No per-category scaling. The sideways-head side share is
   reported, not capped (audit pass 2).
7. **Queue.** No hold after S; the order is B s3 → B s2 → S → P.
8. **Scope.** Arm C and the P re-dose stay options, each run only on the user's yes at its stop.
9. **S manipulation check** (audit pass 2). Posture cannot cap it: delivered dose plus any stroke change. Side
   `stance_stroke` is the outcome.

Also the user's: R4 binds on omni rows only (audit pass 2); move_turn and diag floors park and flag (audit pass 3).

**SA0 (the orchestrator, 2026-09-26, under the standing instruction; the user was push-notified):**

1. **p = 1.5**, not 2 (≈1.85× the delivered side pull at the same 10 % fwd cap; lunge 3.0× instead of 5.1×).
2. **Keep the rotational credit (ω×r)** on turn and move_turn; watch the move_turn stroke share at SA4.
3. **Eval keys: offline readout** (`replay_stroke_posture.py readout` at each park); `eval_sha` unchanged.
4. **Stand excluded, command-only normaliser:** yes.
5. **Floors:** W's suite survival (`survival_by_hold`) − 0.20; move_turn and diag park and flag.
6. **Restated R1, R2, R3, R6, R8, P1** (each literal form measured something other than what it protects).
7. **R5 as written FAILS** on the cross-plant lunge (a per-foot stroke income cannot see unison or phase).
   Mitigation adopted: the **unison/phase flag** on S (and C) at every readout park.
8. **Turn progress guard accepted as is** (watch turn stroke income at SA4).
9. **Reward-simplification #2:** see the note below.

**Adopted parameters** (the math's defaults): stroke RHO0 0.5, DRHO 0.3, POWER 1.5, C_MAX 1.5; progress guard τ 4 s,
LO 0.15, WIDTH 0.15; contact open 50 N, hold 20 N, N_OFF 2, min 5 loaded steps; centring σ 0.20, τ 2 s, min cmd 0.12,
φ 0.10–0.25. **Doses:** w_S = **0.0865** (R4 binds on W fwd at 10 % of tracking income, delivered), w_P = **0.155**
(P2 prize parity with S; band [0.124, 0.194]) — both subject to the window-0 dose smoke.

## Design

| | |
|---|---|
| Chain `c6y` | one stage, preset `1a`, C1d's c6 overlay verbatim (as w2's c6x), **4000 its** |
| Fork | W's **baked** copy `crab_hex_omni_task/policy/5_c6x_clockfree/model_11995.pt` (sha256 `3a527683eccd…`, byte-identical to the run-dir checkpoint; `resume_from: {checkpoint: …}`) |
| Phase-1 total | 16000 its (C1d 8000 → c6x 4000 → c6y 4000) |
| Chain overlay | c6 overlay + **K** (`KRABBY_CLOCK_W` 0.001, `KRABBY_APEX_W` 0.001, `KRABBY_OMNI_CLOCK_HIDE` 1) + `KRABBY_OMNI_STROKE_W` 0.001 + `KRABBY_OMNI_CENTRING_W` 0.001 + `KRABBY_OMNI_REWARD_SPLIT` 1. In the CHAIN, so B inherits them (w2 put K in W's candidate overlay; a verbatim copy would give B the clock back). |
| `require_env` | K + the split flag, pinned on every composed stage env (baseline, twins, every candidate) |
| Baseline **B** | c6y with both knobs at ε; s3 and s2 (both fork the same baked W); its pair is D00bs. No exact twin of "W + 4000 its" exists, so B is the in-campaign control. |
| Arm **S** | `KRABBY_OMNI_STROKE_W` = w_S; s3 (s2 when paired) |
| Arm **P** | `KRABBY_OMNI_CENTRING_W` = w_P; s3 (s2 when paired) |
| Arm **C** | both doses; **conditional (SA5)**, not queued |
| Arms vs B | exactly one weight each (declared `overrides_chain`; `--dry-run` shows no other key differs) |
| Eval | `suite__omni` at stage end; HIDE carried (`EVAL_CARRY`; `eval_carry` on S and P too) |
| D00s | from the unchanged E1/E2 stamps (eval_sha `36e5a8af7000`), exactly as w2: control X08-c6, pair X08-c4, B = 2000 paired on env_index, rng 20260923; computed by the driver at the first launch |
| Budget | 1.8 GPU-h per arm-seed; typical ≈ 9.4 (window 0 0.35 + B pair 3.6 + S pair and P 5.4); cap 17 |

**Queue** (the driver's own order): B s3 → B s2 → S s3 → [S s2] → P s3 → [P s2]. A seed-2 pair is queued by the
driver when the s3 line is an ADOPT (`pair_on: [adopt]`) or all-TIE (`_decide_s3`'s own branch, so decision 5's "an
all-TIE single-seed S gets its s2 pair" needs no extra entry). `pair_on` `neutral`/`lone_guard` are refused without an
`adopt_requires` block, and this rule names none. A clear s3 REJECT (a block or guard WORSE) is decided with no pair.
**The same all-TIE branch pairs P** (+1.8 GPU-h, inside the cap; the plan's budget line "+ P pair").

## The rule

- **Rule v2, `control: baseline`.** Every arm is judged against B's own screens, never another arm.
- **Blocks** (95 %, two-sided): F, F_mt, T and RIG **verbatim from w2** (RIG is not adopt-required here; it carries
  the flag/park machinery), plus **LEN** = side and diag `steps_per_m`, lower is better. LEN is decisive for the
  mechanism reading only; no `adopt_requires` (decision 5).
- **Guards** (99 %, one-sided): w2's 23. R_i = max(R_eval, R_seed), survival guards floored at R_binom.
- **Floors:** W's suite survival − 0.20: stand 0.80, fwd 0.75, side 0.795, back 0.78, diag 0.67, turn 0.79,
  move_turn 0.165. **move_turn and diag PARK AND FLAG** (`floor.park_categories`): the run continues to its screen,
  its line carries an immediate `floor_park` flag, and the survival guard against B decides. A floor on any other
  category is a candidate REJECT (class BREAK) and parks a baseline, as in w2.
- **Flags** (w2's, against B): turn `facing_closed` −0.05, move_turn `facing_closed` −0.10, any survival −0.05, any
  walking ratio −0.05; taps +25 %; `swing_apex` −30 %; `support_feet` −10 %; speed-driven; removal verification
  (every arm is clock-free); `floor_park`. Expect margin parks on move_turn: W's move_turn is its weakest category.
- **Orchestrator flags at SA4** (offline readout): term income > 25 % of tracking in any category; side
  `knee_cap_frac` up > 20 %; the **unison/phase flag** on S and C (tripod coherence below its seed band, or
  `roll_rms` collapsing, vs B; SA0 decision 7).
- **WATCH:** move_turn survival and pitch (W's forward nose-dive); the w2 WATCH set.
- **The driver's ADOPT/REJECT is mechanical.** S's adoption is decision 5, read at SA4:

  **Adoptable S** (pooled over s3 + s2): the S manipulation check passed; side `stroke_share` not lower beyond its
  MDE; no block or guard WORSE. LEN BETTER is reported as mechanism evidence, not required.

- **Manipulation checks**, each against its MDE:
  - **S** (decision 9), two halves: *the dose landed* — delivered income (`Episode_RewardCat`, post-clip) on fwd and on
    side ≥ 0.5 × the term priced at w_S on B's raw npz (a lower bound only: income above the prediction is stroke
    change, never failure); *any stroke change* — at least one of side `stroke_share`, fwd `stance_stroke`, side
    `stance_stroke`, or a per-category delivered income moved beyond its MDE. Side `stance_stroke` (+0.010 m) stays
    the outcome for the LEN/POS rows.
  - **P:** Δ`body_offset` ≤ −0.05 m.
  - Any reading that stroke income is insufficient, and the C offer, needs S's check to pass.
- **Minimum effects:** side `stride_along` +0.02 m; side `stance_stroke` +0.010 m; `body_offset` −0.05 m; side ratio
  +0.05.

### Outcome table (plan step 14)

| Outcome | Reading |
|---|---|
| S LEN+, POS TIE | stroke income is enough without re-centring |
| S LEN+, POS+ | stroke income re-centres the body |
| S manipulation TIE | the dose did not bite; no mechanism reading → stop and ask: stop, or exceed R4 (the user's call; R4 already takes the largest weight) |
| S manipulation passed; side `stance_stroke`/LEN TIE; P POS+, LEN+ | posture-capped: posture is the bottleneck → offer C (SA5, +3.6 GPU-h with its pair) |
| P POS+, LEN TIE | room is not sufficient; the limiter is cadence, the tracking economy, or the ~0.22 m/s actuator ceiling |
| P POS TIE | manipulation failed → re-dose once at 2× (+1.8 GPU-h) or drop P |
| any block or guard WORSE | not adoptable; the mechanism reading still stands |
| the unison/phase flag on S | recorded with the line; a lunge/unison gait is not adoptable whatever the blocks say |

P is also a side-row emphasis: a POS+ reads "posture or emphasis". Recorded, not controlled.

## Stops

| Stop | What | Who |
|---|---|---|
| **SA1** fork | Confirmed: the fork is W's baked copy (sha256 `3a527683eccd…`, `bundle_policy.py --check` passes); `window0_dose_smoke.py` re-checks the hash | done |
| **SA2** landing | Rebase, run every test, merge `feat/omni-stroke-posture` into `feat/crab-hex-omni-task`. The merge moves `code_sha` for MQ's paused ladder (and for the forward/sideways drivers: step 4 touches `parkour_isaaclab`), so MQ's next relaunch needs `--accept-code-change` and reruns its B<k>. The user said to run without asking: the orchestrator lands it and push-notifies. Expect a conflict on the reward-simplification `CANDIDATES*.md` (another session's uncommitted edits; this branch does not touch them). | orchestrator, notify |
| **SA3** dose smoke out of band | r outside [0.5, 2] on the row that sets a dose, side r_S outside the band, or diag's P share > 10 % after rescale → stop with the numbers and the stance-length histogram; the user decides whether to re-base R4/P2 on training telemetry (replay-vs-training gaps of 2–13× are on record for contact terms) | user |
| **SA4** every park | Every DECISION and every flag: run `replay_stroke_posture.py readout` on the arm's and B's raw npz (`--vs`, MDEs), the S/P manipulation checks, the orchestrator flags, the outcome table row | orchestrator reads; user decides each DECISION |
| **SA5** arm C | Offered only on the "posture-capped" row: +3.6 GPU-h with its pair | user |
| **SA6** close | Fill the outcome table; the user decides: adopt/bake S (the 2a+ phase-out ramp of the swapped term is a follow-up), follow up on the posture's origin, hand `loaded_stance_edges()` to the air-time task. Finalise REPORT/CHANGELOG/memory; never push. | user |

## Caveats

- **Tibia relief amplifies P.** `penalty_tibia_deviation_in_stance` (−0.28, 1.7× on side) is priced in the replay;
  re-centring relieves it, so P's side income is larger in the total economy than its own term shows.
- **P is also a side-row emphasis.** λ = |v_y|/‖v‖ pays only lateral rows; a POS+ may be emphasis, not posture.
- **The turn progress guard sits on the heads' turn ratio.** Every omni head turns at 0.24–0.43 of the commanded yaw
  rate, inside the guard's 0.15–0.30 band, so turn credit carries a tracking tax (SA0 decision 8).
- **Wrapper parity uses training rollouts.** The eval harness assigns the slot from the hold's category; the command
  term's own slot differs (prologue, settle, resample edges), so the eval-hook replay is not a wrapper replica. The
  wrapper→replay parity check (N steps to 1e-5) runs on the dose smoke's training dump.
- **R5 fails cross-plant** (the forward-era lunge earns 3.0× healthy at p = 1.5): hence the unison/phase flag.
- **The creeper earns 1.91× healthy on move_turn** through ω×r (≈1 % of move_turn tracking; no-flip holds).
- **Single-seed S/P deltas must clear the seed term** (the posture keys are seed-dominated).
- **Clock-campaign flags carried by RIG** (removal verification, margins) will park; each is read at SA4, and only
  removal verification is a defect by construction (every arm is clock-free).

## Reward-simplification candidate #2 (recorded here, not in `CANDIDATES*.md`)

- **The stride half is superseded** by `KRABBY_OMNI_STROKE_W` (a func swap under the name `reward_stride_length`, so
  its telemetry and every `KRABBY_STRIDE_W`/`KRABBY_PHASEOUT` reference keep resolving).
- **The air-time half is untouched**: it stays with its own task (#28). `loaded_stance_edges()`
  (`mdp/crab_hex_omni_stroke_math.py`: open > 50 N, close after 2 steps < 20 N, never reads `compute_first_*` or the
  contact timers) is offered to it.
- v4 (the per-foot touchdown stride, 2026-08) was dead in training too (the same float32 bug).
- Why here: `CANDIDATES.md`/`CANDIDATES_DETAIL.md` carry another session's uncommitted edits in the main checkout.
  Move this note there once those land.

## Window 0 (≈ 0.35 GPU-h; after SA2; one Isaac process at a time)

Shared shell (every command: no `KRABBY_*` exported in the shell; Isaac under a memory cap; `setsid nohup` for
anything long):

```bash
R=/home/nickmagus/krabby/krabby-research
PY=/home/nickmagus/krabby/isaac_venv/bin/python
CAMP=parkour/parkour_tasks/parkour_tasks/crab_hex_omni_task/experiments/2026-09-26_stroke_posture
TOOLS=parkour/parkour_tasks/parkour_tasks/crab_hex_omni_task/experiments/tools
D=$R/parkour/outputs/omni_plan/stroke_posture/window0
MB=$(git -C $R merge-base feat/crab-hex-omni-task feat/omni-stroke-posture)   # fb203f7d at the time of writing
CAP="systemd-run --user --scope -q -p MemoryMax=48G -p MemorySwapMax=0"
test -z "$(env | grep '^KRABBY_')" || { echo "unset every KRABBY_* first"; exit 1; }
mkdir -p $D
```

### (i) Knob-off identity dumps and the knob-on dump

How the existing dumps were made (`parkour/outputs/omni_plan/dumps_clock_knobs_off/`, `dumps_motion_knobs_off/`):
the three config-identity modules, reusing one dump directory, one headless Kit boot per fixture; the dumps land as
`cfg_<fwd>.json`, `cfg_side_<phase>.json`, `cfg_omni_<phase>.json` (forward 1a/2a/2b/2c/3a/2c_noplant/2c_legacy,
sideways 1a–3a, omni 1a/1b/1c/2a/2b/2c). Each child strips the parent's `KRABBY_*` and inherits `PYTHONPATH`, and
`crab_hex_phase_cfg_dump.py` puts only its own `parkour/parkour_tasks` first, so `PYTHONPATH` must name the tree's
`parkour_tasks` AND `parkour` (for `parkour_isaaclab`, which the branch touches).

```bash
# the merge-base reference tree (detached; removed at the end)
git -C $R worktree add --detach $R/.claude/worktrees/stroke-posture-mb $MB
for T in $R/.claude/worktrees/stroke-posture-mb $R; do
  tag=$([ "$T" = "$R" ] && echo br || echo mb)
  mkdir -p $D/dumps_$tag
  ( cd $T && env RUN_CRAB_HEX_CFG_IDENTITY=1 KRABBY_CFG_DUMP_DIR=$D/dumps_$tag OMNI_KIT_ACCEPT_EULA=yes \
      PYTHONPATH=$T/parkour/parkour_tasks:$T/parkour \
      $CAP timeout 5400 $PY -m pytest -p no:cacheprovider -v \
        tests/integration/test_crab_hex_phase_configs.py tests/integration/test_crab_hex_sideways_configs.py \
        tests/integration/test_crab_hex_omni_configs.py ) > $D/dumps_$tag.log 2>&1
done
# the screen's chain overlay (c6y), knob-OFF (the three stroke/posture knobs UNSET), both trees
C6Y="KRABBY_OMNI_CMD_POOL=stand:0.15,fwd:0.14,side_l:0.09,side_r:0.09,back:0.09,omni:0.14,turn:0.1,fwd_turn:0.06,side_l_turn:0.035,side_r_turn:0.035,omni_turn:0.07 \
KRABBY_OMNI_SLOTS=0.15:0.55:0.1:0.2 KRABBY_OMNI_DIR_BAND= KRABBY_OMNI_RSI_SOURCE=all KRABBY_OMNI_FACING=0.2:0.6 \
KRABBY_YAW_W=0.2 KRABBY_CLOCK_W=0.001 KRABBY_APEX_W=0.001 KRABBY_OMNI_CLOCK_HIDE=1"
for T in $R/.claude/worktrees/stroke-posture-mb $R; do
  tag=$([ "$T" = "$R" ] && echo br || echo mb)
  ( cd $T/parkour && env $C6Y KRABBY_OMNI_PHASE=1a KRABBY_PLANT=A15+B OMNI_KIT_ACCEPT_EULA=yes TERM=xterm \
      PYTHONPATH=$T/parkour/parkour_tasks:$T/parkour \
      $CAP timeout 900 $PY $T/tests/integration/crab_hex_phase_cfg_dump.py --headless \
        --task Isaac-Crab-Hex-Omni-Walk-v0 --task Isaac-Crab-Hex-Omni-Walk-Play-v0 \
        --out $D/dumps_$tag/cfg_omni_c6y_off.json ) > $D/dumps_$tag/cfg_omni_c6y_off.log 2>&1
done
# byte-identity, every fixture (forward, sideways, omni, c6y): no output = PASS
for f in $D/dumps_mb/cfg_*.json; do cmp -s $f $D/dumps_br/$(basename $f) || echo "DIFF $(basename $f)"; done
ls $D/dumps_mb/cfg_*.json | wc -l; ls $D/dumps_br/cfg_*.json | wc -l      # equal counts
# the knob-ON dump: arm C's doses + the split flag, branch tree only
( cd $R/parkour && env $C6Y KRABBY_OMNI_STROKE_W=0.0865 KRABBY_OMNI_CENTRING_W=0.155 KRABBY_OMNI_REWARD_SPLIT=1 \
    KRABBY_OMNI_PHASE=1a KRABBY_PLANT=A15+B OMNI_KIT_ACCEPT_EULA=yes TERM=xterm \
    PYTHONPATH=$R/parkour/parkour_tasks:$R/parkour \
    $CAP timeout 900 $PY $R/tests/integration/crab_hex_phase_cfg_dump.py --headless \
      --task Isaac-Crab-Hex-Omni-Walk-v0 --task Isaac-Crab-Hex-Omni-Walk-Play-v0 \
      --out $D/dumps_br/cfg_omni_c6y_on.json ) > $D/dumps_br/cfg_omni_c6y_on.log 2>&1
cd $R && $PY - <<'EOF'
import json, sys
sys.path.insert(0, ".")
from tests.integration.test_crab_hex_phase_configs import diff_paths, fmt_diffs
D = "parkour/outputs/omni_plan/stroke_posture/window0/dumps_br"
off, on = (json.load(open(f"{D}/cfg_omni_c6y_{k}.json")) for k in ("off", "on"))
for task in ("Isaac-Crab-Hex-Omni-Walk-v0", "Isaac-Crab-Hex-Omni-Walk-Play-v0"):
    d = diff_paths(off[task]["env"], on[task]["env"], limit=500)
    print(task, sorted({p[:3] for p, _, _ in d}))
    print(fmt_diffs(d))
    assert not diff_paths(off[task]["agent"], on[task]["agent"]), "the runner cfg moved"
EOF
git -C $R worktree remove $R/.claude/worktrees/stroke-posture-mb
```

**Pass:** every `cmp` silent with equal counts; the knob-on diff touches only `rewards.reward_stride_length`
(`func` `RewardStrideLength` → `RewardFootStroke`, its `params` → the six-foot asset/sensor cfg, `weight` 0.0865),
`rewards.reward_body_centring` (added, weight 0.155) and `reward_category_split` (true). Anything else, or any
knob-off DIFF: stop, with the diff.

### (ii) The 300-iteration dose smoke (arm C, outside the ladder)

`window0_dose_smoke.py` composes arm C's stage env from `candidates.yaml` with the driver's own `compose_recipe` (B's
`c6y` env + both doses read from S's and P's overlays), adds the README **play** block's policy-input keys
(`KRABBY_OMNI_CLOCK_HIDE=1`, `KRABBY_FLAT_TERRAIN_GEOM=ring`, `KRABBY_OMNI_FACING_OBS_CLIP=0.6`; NOT the eval block's
`KRABBY_CLOCK_W`/`APEX_W` 1.0 and `KRABBY_RSI_FRAC` 0, refused by name) and the parity recorder's
`KRABBY_OMNI_STROKE_PARITY_{DUMP,ENVS,STEPS}` (in `extra_env`: `run_phases._subenv` strips a parent shell's
`KRABBY_*`), checks W's sha256, and calls `run_phases.train(…, 300 its, resume W)`.

```bash
cd $R
$PY $CAMP/window0_dose_smoke.py --print-env          # CPU: read the env before committing the GPU
setsid nohup $CAP $PY $CAMP/window0_dose_smoke.py > $D/window0_doseC.out 2>&1 < /dev/null &
# done when $D/window0_doseC.out ends "WINDOW0 DOSE SMOKE DONE"; $D/window0_doseC.json names run_dir
RUN_DIR=$(python3 -c "import json;print(json.load(open('$D/window0_doseC.json'))['run_dir'])")
# r per category = delivered (Episode_RewardCat/<cat>/<term>, post-clip, per s of dwell, last 200 its)
#                  / replay-predicted (the term priced at the same doses on W's own suite raws)
$PY $TOOLS/replay_stroke_posture.py income \
  parkour/logs/rsl_rl/gait_eval/omni_phases/2026-09-23_clock_deprecation_w2/seed3_X03_c6x_A15pB/suite__omni/seed001/2026-09-25_20-31-29 \
  --w-stroke 0.0865 --w-centring 0.155 --train-log-dir $RUN_DIR --json $D/window0_doseC_r.json
# wrapper -> replay parity on the smoke's own training rollouts (16 envs x 300 steps)
$PY $TOOLS/replay_stroke_posture.py parity $D/parity_window0_doseC.npz --tol 1e-5
```

**Reading r** (the "training telemetry vs replay" table; `r_S`, `r_P` per category; `abs_log`/`clip_log` against
the replayed absorbed share and clip-bind fraction):

- **r_S is the fwd row's**; side's r_S must also lie in [0.5, 2]. **r_P is the side row's**; diag's r_P is reported
  and w_P/r_P must keep diag ≤ 10 % of its tracking income.
- In band: w ← w/r for each term (never inflate beyond 2×), edit S's and P's overlay in `candidates.yaml` BEFORE the
  first launch. Anything else: **SA3**.
- A non-zero `Episode_RewardCat/<cat>/reward_stride_length` on the moving categories over the last 200 iterations is
  the proof the term pays past 2 s in real Isaac. The parity check must PASS (max |diff| < 1e-5 on per-step income
  and `b`).

### (iii) Fill and check

```bash
cd $R && $PY $TOOLS/run_candidates.py --campaign-dir $CAMP --dry-run    # must be clean (validator all ok, no CONFLICTS)
bash $TOOLS/launch_candidates.sh --campaign-dir $CAMP                    # D00s is the first launch's migration item
```

## Log

- 2026-09-26: directory created (plan steps 7–8, SA0). Step 9: this charter, `candidates.yaml` and
  `window0_dose_smoke.py`. Driver options added for this screen (byte-identical when absent; the clock campaign's two
  yamls dry-run identically before and after): `floor.park_categories` (park-and-flag floors, an immediate
  `floor_park` flag) and a candidate's `overrides_chain` (a declared override of a chain-overlay value is not a
  compose conflict). `--dry-run` clean from the worktree (the E1/E2 stamps are notes there only because a worktree
  has no `parkour/logs`).
