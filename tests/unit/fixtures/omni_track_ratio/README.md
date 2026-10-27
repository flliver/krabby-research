# `track_ratio_mean` extracts from the seed-3 omni campaign training logs

Source: `crab_hex_omni_task/experiments/2026-09-16_1805_omni_formation/logs/seed3_{1a,1b}_*_train.log`
(the campaign's own run logs; `*.log` is gitignored under `experiments/`, so the decision-bearing
lines are extracted here instead of depending on a 5 MB untracked file).

Each file is `grep "Metrics/base_parkour/track_ratio_mean"` over the whole log, in order, one line
per training iteration, VERBATIM including the trainer's spacing — so
`run_phases.py::series()` / `track_ratio_series()` parse them exactly as they parse a real log.

They exist as the regression for the 2026-09-18 translation floor
(`GATE_TRACK_RATIO_MIN` 0.6 over the last 500 iterations, and the from-scratch `dead_walker` abort
at iteration 1000 on the its-800-1000 mean):

| stage | iterations | its 800-1000 mean | last-500 mean | gate floor 0.6 | abort floor 0.5 |
|---|---|---|---|---|---|
| 1a (from scratch) | 1000 | 0.747 | 0.693 | PASS | PASS (no abort) |
| 1b (resumed 1a)   | 1000 | 0.433 | 0.406 | FAIL | (n/a — not from scratch) |

1c (5000 its, resumed 1b) is not extracted — 230 KB — but its last-500 mean is 0.246, also FAIL.
The chain's real history is 0.747 -> 0.433 -> 0.245 while every gate of the day passed it.
