<!-- paths-note -->
> **Paths note (2026-09-09):** this campaign moved from `sim_fine_tuning/2026-08-13_1230_perpendicular_mounts/` to `parkour/parkour_tasks/parkour_tasks/crab_hex_forward_task/experiments/2026-08-13_1230_perpendicular_mounts/`. Absolute paths below (`/home/.../sim_fine_tuning/...`, `parkour/logs/rsl_rl/...`) name raw artifacts that stay on disk untracked; the tracked checkpoint of record is `head/` (see `bundle.yaml`) and the eval summaries are in place / under `evals/`.

# Perpendicular-mounts correction (Arm A: centered yaw defaults)

User hardware clarification 2026-08-13: every leg is mounted perpendicular to the frame with
its yaw range centered on that mount. The config's splay defaults (F/R ±0.342, M ±0.143 —
Go2-inherited proportions) had no hardware basis and parked front/rear legs 8.9° from the
±28.54° mechanism limit at 82% cam gear, with their ±0.24 rad command window off-center —
the root cause of mid-legs-only thrust (see the thrust/DOF analyses in chat + mirror_symmetry
campaign artifacts).

**Change (this commit)**: all 6 hip + 6 camshaft defaults → 0.0 (mount neutral). Standing
stance remains the policy's choice within the command window. Arm B (widening the camshaft
action scale toward the full ±2.07 rad shaft travel = full ±28.54° hip range; today's window
commands only ~±4.4° of yaw) is deferred pending Arm A results.

## Verification ladder

1. **Statics at 0 defaults: PASS** — pitch +0.38°, roll +0.05° (leveler than old pose), A share
   52.6%, CoM +5.7 mm. Load shifted mids-heavy statically (ML/MR ~284 N vs F/R 73–145 N; fore-
   aft spread now from mount spacing only). No self-collision over 300 zero-action steps.
2. Cam operating point (all legs identical now): gear 0.323 (100% of peak), stroke ±28.54°
   symmetric, command window centered — the "after" of the diagnosis table.
3. Fine-tune canary (2000 iters from symmetric reference; expect obs/action re-centering shock;
   catastrophe check only) → then the real validation: 20k from-scratch + mirror loss 0.5.
4. Primary gate: thrust distribution (per-group shaft |v| + reversal share — success = front/
   rear shafts working); plus stride, tripod/duty, lean, standard guardrails.

## Canary: PASS — campaign PAUSED for full geometry audit (user directive)

2000-iter fine-tune from the symmetric reference across the re-centering shock: reward 0.47 →
27.5 (full recovery, above pre-shift level), no NaN. Eval: completion 90% (1 fall), tripod
0.420, duty 0.302/0.308 (balance intact), slip 2.7%, stride 0.143. Front/rear shafts still
idle (0.11 rad/s vs mids 3.72) — habit persistence, as expected for a fine-tune; the
thrust-redistribution verdict requires the 20k from-scratch run.

**20k validation and Arm B are BLOCKED pending the user's full robot-geometry audit.** Known
discrepancies queued for that audit: sim mass 106 kg vs URDF ~23 kg; source of the −0.14°
zero-action roll the knee tune compensates; the FR+RL floating-feet settle pattern; systematic
joint frame/sign verification vs CAD; cam K re-derivation from current CAD; collision shapes
vs CAD.

---
2026-08-13 19:xx UPDATE: Arm B (camshaft action-window widening) is SUPERSEDED by the
geometry_velocity_actions campaign (sim_fine_tuning/2026-08-13_1900_geometry_velocity_actions/):
cam channels are now VELOCITY targets (true continuous one-direction spin, matching the
hardware quick-return motor), which subsumes and obsoletes the position-window widening.
