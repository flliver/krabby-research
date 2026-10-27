# Close — per-category stroke weights (S_cat), both settings (2026-09-28, plan step 11)

**Status: CLOSED. Both gates FAIL. Nothing is adopted.** The knob `KRABBY_OMNI_STROKE_CAT_W` stays in the tree,
default-off; with it off, every config is byte-identical.

Plan: `~/.claude/plans/per-category-stroke-weights.md` (audited to cap 3, approved 2026-09-27).

## Records

| Setting | Records |
|---|---|
| W, clock-free fine-tune from the head of record | this dir: [GATE_W.md](GATE_W.md), [WINDOW0.md](WINDOW0.md), [pilot/PILOT.md](pilot/PILOT.md), [replay_gate/REPLAY_GATE_CAT.md](replay_gate/REPLAY_GATE_CAT.md) |
| t3, clock-held formation from scratch | [../2026-09-27_stroke_cat_t3/GATE_t3cat.md](../2026-09-27_stroke_cat_t3/GATE_t3cat.md), [WINDOW0_t3.md](../2026-09-27_stroke_cat_t3/WINDOW0_t3.md) |

Both gates were checked independently by fable·high.

## The question and the answer

**Question.** With the stroke term's weight separated per command type (each category's delivered stroke income set
to 10 % of its own tracking income; fwd = back forced by the mirror loss; move_turn held at the old dose), do omni side
strides get longer?

**Answer: no, in either setting.**

| | W (clock-free) | t3 (clock-held, from scratch) |
|---|---|---|
| Final table | `STROKE_W` 0.1262, side ×2.93 (0.370) | `STROKE_W` 0.1123, side ×3.46 (0.389) |
| Side delivered share | ≈ 9 % (single weight: 2–3.5 %) | ≈ 9 % (single weight: 2–3 %) |
| Dose landed (fwd / side r) | 0.67 / 1.34 (s3), 0.77 / 1.06 (s2) | 1.47 / 1.27 (s3), 0.98 / 1.14 (s2) |
| Side stride_along Δ, s3 / s2 | −0.0099 / +0.0015 | +0.0047 / −0.0078 |
| Pooled Δ against band | **−0.0042 vs 0.0049: FAIL** | **−0.0015 vs 0.0049: FAIL** |
| stride_per_speed (pooled) | −0.020, beyond band (shorter per unit speed) | −0.070, beyond band |
| Taps | up, beyond band | up at both seeds |
| Side lean (body_offset) | +0.010, more | +0.036, more |
| Health | FAIL: fwd unison flag at s3; 4 guards WORSE; F and F_mt TIE | FAIL: diag unison flag at s2; F BETTER (carried by s3 against the weak B_f s3) |
| Seed agreement | s3 damaged, s2 flat | s3 flattered by the weak control, s2 damaged (survival and completion down) |

**Mechanism.** At about 3 × the single weight's side income, the policy bought neither faster side walking nor longer
steps at the same speed. The consistent signature, in both settings: stride per unit speed down, taps up, lean up. That
is more, shorter contacts. The audit's pre-run probe had flagged the reason:
- at matched speed, the stroke term pays LESS for longer side strides (partial Spearman −0.09 to −0.44);
- about half of that comes from the slip gate, because side stances slip (rho about 1.5).

## What the line has now tested

| Arm | Dose | Setting | Record | Side stride |
|---|---|---|---|---|
| Single-weight stroke (S) | fwd-bound | W fine-tune | stroke_posture CLOSE.md | no change |
| Single-weight stroke (S_f) | fwd-bound | t3 formation | stroke_formation GATE_t3.md | no change |
| Centring (P, P_f) | — | W fine-tune and t3 formation | same two records | no change; turning hurt |
| Per-category equal-share stroke (S_cat, S_cat_f) | side about 3 × | W fine-tune and t3 formation | this close | no lengthening; shorter per unit speed, more taps |

## Process notes

- **The user overrode the W s3 pilot STOP** to run W s2 and t3 as planned.
- **The t3 dose smoke's raw side r (4.61) was a telemetry spike.** It was one 0.008-s-dwell iteration. The user chose the
  dwell-weighted r (1.45).
- **Episode_RewardCat per-category rates carry rare heavy-tailed spikes.** They come from small dwells and single large
  stances, and at a side ×3 multiplier they reach 6–33/s. They are not learned: medians stayed flat in every run.
  Future dose smokes should use a dwell-weighted or spike-robust r.
- **Reference-run tags collide with the reused baselines' tags** (`seed3_X01_*`). `prepare_resume` finds run dirs by tag,
  so a mid-training stop could resume from the old baseline's checkpoint. No stop happened.
- **GPU used:** about 7.3 h in total.
  - W: window 0 ≈ 0.3, two runs ≈ 3.6.
  - t3: smoke and checks ≈ 0.3, two runs ≈ 3.0.

## For the user (nothing below has run)

1. **Close the stroke-reward line for side stride** (recommended). Three dose regimes and two settings all failed to
   lengthen side strides. Keep the knobs default-off.
2. **If side stride is still the goal, test the premise directly:** why omni side walking is slow and short. The zero-GPU
   step is a speed-matched comparison with the sideways head. Candidate levers that are not reward income:
   - the side speed command range and tracking economy;
   - the slip gate / stance definition on side (rho about 1.5);
   - the curriculum.
3. **Housekeeping.**
   - All commits are local on `feat/crab-hex-omni-task`; nothing is pushed.
   - `code_sha` moved to `fb17d2ea`, so other campaigns' next relaunch needs `--accept-code-change`.
   - The worktree `.claude/worktrees/stroke-posture` and the detached `.claude/worktrees/stroke-cat-mb` can be removed.
