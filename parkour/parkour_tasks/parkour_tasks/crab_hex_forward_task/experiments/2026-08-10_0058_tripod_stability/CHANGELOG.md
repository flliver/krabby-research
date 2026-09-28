# tripod_stability: tripod-first reward tuning (Milestone 18 Task 1 follow-on)

**Predecessor**: `2026-08-09_1526_gait_tuned/` — the teacher-stack carry-up study, which baked
`reward_action_rate=-0.3`/`reward_delta_torques=-1e-6` and produced the from-scratch flat-walk
checkpoint this campaign builds on (tripod 0.401, tippy 7.97%, stride 0.163m, slip 2.31%,
completion 100%).

**Motivation**: the user observed the current gait leans forward and isn't always stably
supported by its planted legs — a real concern for future off-center payload carrying. Per
TASK-1-REWARD-SHAPING.md, priorities in order: (1) improve tripod_score, (2) don't degrade the
other gait metrics, (3) improve upright stability. Plan:
`~/.claude/plans/the-newest-version-of-cheerful-codd.md`.

**Method**: Step 0 quantified the problem from existing eval artifacts before spending any GPU —
signed mean pitch +0.209/+0.210 rad (~12°) across two independent gait-evals, confirming a real,
repeatable structural lean (not noise). Phase A screens config-only reward knobs (stance-count
bracket, air-time weight/threshold, signed pitch penalty, angular-velocity damping) one lever per
1000-iter resume, judged only by gait-eval (training metrics have previously diverged from
held-out gait quality). A combined-winners validation follows. If config knobs can't move tripod,
an explicit tripod contact-schedule reward is the pre-authorized fallback. The winning config gets
a from-scratch confirmation run before being proposed as the new baked default.

## Results

(updated per run; see `RESULTS.md` for the full table)

## Verdict

(pending)
