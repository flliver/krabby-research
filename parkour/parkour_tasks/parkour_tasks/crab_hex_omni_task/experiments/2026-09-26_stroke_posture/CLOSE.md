# Close — stride/posture screen (2026-09-26, SA6)

**Status: DONE.** The driver's queue emptied at 10:11:
- B s3 (X01) and B s2 (X02) ran;
- S s3 (X03) → D05 clear REJECT;
- P s3 (X04) → D06 clear REJECT;
- no seed-2 pairs (clear rejects are not paired);
- arm C and the P re-dose did not run, since both are user-gated (decision 8).

GPU used: about 7.5 h (window 0 ≈ 0.25 plus 4 × ≈ 1.8).

**Driver records:** REPORT.md, CHANGELOG.md, ledger.json, state.json, heads.json, notify.log.
**Readouts:** `sa4/D05.md`, `sa4/D06.md` (JSON beside them).
**Window 0:** WINDOW0.md. **SA0:** SA0_DECISIONS.md.

## The two questions

### 1. Is the replaced stride term alive and command-general? — **Yes.**

- **Alive in real Isaac.** The old `reward_stride_length` logs 4e-5/s: it is dead about 2 s after every reset
  (float32 `compute_first_*`). The per-foot stroke term pays past 2 s on every moving category in training
  (`Episode_RewardCat`).
- **Wrapper = replay** to ≤ 3e-7.
- **It delivers what the replay predicted** (S's training, r = log ÷ replay):

  | fwd | side | back | diag | move_turn | turn |
  |---|---|---|---|---|---|
  | 0.72 | 1.50 | 0.77 | 0.76 | 1.19 | 0.21 |

  Turn's 0.21 is a command-distribution effect: training turn holds reach their target and the relay drops to 0.
- **Mirror-equivariant**, and pinned by 116+ unit tests. Six exploits found by two adversarial reviews are closed:
  - a dip-hidden shuffle;
  - backward paddling;
  - a stance carried into stand;
  - tripod-locked rocking;
  - yaw credit;
  - a cold-start guard.
- **Knob-off configs are byte-identical** to the pre-branch code: forward, sideways and omni.

### 2. Does stroke income or posture lengthen omni side steps? — **Neither lever did, at the planned doses, in 4000 iterations from W.**

| Arm | Manipulation check | Side outcome (vs B s3; band = max(MDE, B's seed Δ)) | Driver |
|---|---|---|---|
| **S** (stroke, w_S 0.117 = R4, fwd delivered 10 % of tracking) | dose landed (r ≥ 0.72 on every translation row) but **no stroke change beyond the seed band** on any category → **TIE** | body_offset +0.004, stride −0.003, stance_stroke +0.000, LEN TIE (side steps/m 27.2 → 27.7) | REJECT (F_mt WORSE) |
| **P** (centring, w_P 0.160) | side body_offset **−0.005 ± 0.004** vs the −0.05 m bar → **FAIL** | stride +0.004, stance_stroke +0.005, LEN TIE (27.2 → 26.6) | REJECT (F_mt WORSE) |

**Outcome-table rows:**
- **S:** "manipulation TIE — the dose did not bite". The orchestrator stopped S at R4 and did not exceed it.
- **P:** "manipulation failed". The 2× re-dose was not run; it is user-gated.

**No mechanism reading is licensed.** Neither arm moved its own target. The screen therefore cannot say whether
posture or stroke income limits the side step. It says only that these two rewards, at these doses, over one
4000-iteration fine-tune of a converged head, do not move the side gait.

**Unison/phase flag:** clear on both arms and every category.

**move_turn "WORSE" is weak evidence of harm.**
- Both arms drop move_turn facing and survival against **B s3**, the better of B's two seeds. move_turn tracking
  ratio: B s3 0.582, B s2 0.536, S 0.549, P 0.521.
- W's own move_turn survival swung 0.62 → 0.255 → 0.28 → 0.365 inside its clock-free stage.

The REJECTs stand under the rule, but they are not evidence that the terms damage move_turn.

## What is (and is not) adopted

- **Not adopted:** S and P, both REJECT.
- **Stays in the tree, default-off:**
  - `KRABBY_OMNI_STROKE_W`, `KRABBY_OMNI_CENTRING_W`, `KRABBY_OMNI_REWARD_SPLIT`;
  - the math, wrappers, replay tool, parity recorder and telemetry split.

  Every knob-off config is unchanged.
- **The shipped presets still carry the old, dead `reward_stride_length`** (weight 0.5 in window 0). Removing it is
  reward-simplification candidate #2's stride half, now unblocked: its replacement was built and tested, and was
  not adopted.

## For the user (SA6 decisions; nothing below has run)

1. **Follow-ups, if the side-step question should be pushed further.** Each is user-gated.
   - **P re-dose at 2×** (+1.8 GPU-h). Tests whether ANY reward moves the side posture in a 4000-it fine-tune.
     P's own term barely changed body_offset, and the posture is the diagnosed cause.
   - **Exposure during formation instead of a fine-tune.** Arm S or P through the C1d stages from t3, not from the
     converged W. The posture is already present at X08:t3 and in B0 (formation), so a fine-tune may be too late
     to move it.
   - **S beyond R4.** A side-weighted dose, or ~3×, would exceed the single fwd-bound weight of plan decision 6.
   - **The posture's origin.** Turn-first curriculum, the narrow side band, clock shaping: out of this plan's scope.
2. **The dead stride term:** hand its removal back to reward-simplification #2 (recommended), or keep it.
3. **`loaded_stance_edges()`:** offer it to the air-time task (#28). It uses the same float32 edge fix.
4. **Housekeeping:**
   - the branch is fast-forwarded into `feat/crab-hex-omni-task`, local only and not pushed;
   - MQ's next relaunch needs `--accept-code-change`, because `code_sha` moved;
   - the reward-simplification #2 note is in CHARTER.md; move it into `CANDIDATES*.md` once that session's edits
     land.
