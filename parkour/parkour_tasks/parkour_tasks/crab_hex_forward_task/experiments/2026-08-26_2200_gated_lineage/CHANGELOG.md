# PLAN F gated lineage — CHANGELOG

| element | dose | halvings | reset | verdict detail | ckpt |
|---|---|---|---|---|---|
| P0_null | None | 0 | False | fail 0.13 canary ok | 2026-08-26_21-41-53/model_26899.pt |
| P0_null | RSI refresh FAILED rc=124 — keeping previous bank |
| FIX | — | — | — | User question exposed blind-turning gap: heading command was reward-visible but not obs-visible on flat tiles (delta_yaw zeroed there). observations.py now fills the steering channels with true heading error when KRABBY_HEADING is set (matches turn_walk_v1 injection semantics; env-var-gated so old policies unaffected). Orchestrator restarted at P1 rung 0 with the fix + harvest-marker fix + adopted rsi_bank_P0_null (2051 states) | — |
| REVISION (user, 2026-08-27) | PLAN F-r: front-loaded search | training starts at ITERATION 0 with every element active; elements move later IFF proven formation-breaking (5k gate 0.40/0.60 + trend extension; group bisection; deferred elements activate at the formation head with PLAN F ladders). Prior lineage (P0 head model_26899) retained as reference | — | — | — |
| REVISION (user-approved plan, 2026-08-27) | round-based sequential accumulation | 5k rounds 0/5k/10k/15k cap 20k; per-round no-change baseline controls; RELATIVE verdicts; bakes per round; FAILED recording after round 3; seed-3 schedule replay; REPORT.md reporting contract | — | — | — |
| HALT | round-1 control crashed | — | — | — | — |
| HALT | round-1 control crashed | — | — | — | — |
| HALT | r1_025_terrain_curriculum training crashed | — | — | — | — |
> NOTIFY: round-search complete — schedule [yaw_income@5k, terrain50@5k, terrain_recal@5k, terrain_curriculum@5k, dr_push@5k, dr_masscom@5k, safety_pack@5k, clearance_pack@10k, turning@10k, goalvel_income@10k], failed [speed_band, episode40]
