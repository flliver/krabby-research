# Step-11 provenance copies

These files are verbatim copies of what ran at PLAN step 11 (2026-10-24). They were run from the session scratchpad,
so the absolute `/tmp/claude-1000/.../scratchpad/...` paths inside them (the `LOG=` lines and the clip script call in
`s11_queue.sh`, `s11_L180.sh` and `clip_F19.sh`, plus a few reader notes in `fill_plan.json`) point to locations
that no longer exist.

- `fill_plan.json`: the verified cell-by-cell source map. Workflow `wf_0dba6758-b7f`: survey → fable critic →
  revise.
- `s11_queue.sh`, `s11_L180.sh`: the serial GPU queues (every row runs through `TP/gate_row.sh`). Their sentinel log
  is `s11_queue.log`, which `report.py --rows-log` reads.
- `clip_F19.sh`: the F19 student veto-clip render. The mp4s are local-only in `F19/clip/` (git-ignored).
- `envs/`: the forward-3a added-mass env files (`fwd3a.env` + `KRABBY_DR_MASS=L:L`).

The stamps themselves live under `parkour/outputs/omni_plan/task4/step11/lr/`, and the keep-set summaries of them
are under `evals/step11/`.
- `s11_queue_sentinels.txt`: a committed copy of `s11_queue.log`. `*.log` is git-ignored here; this is the file
  `report.py --rows-log` read.

- Correction (2026-10-25): the step-11 report read `s11_queue.log`, not the committed copy (REPORT_baseline_vs_final.md,
  "Step-11 rows" provenance line). `s11_queue_sentinels.txt` is its byte-identical committed copy (same numbers); on a
  clean checkout, where the git-ignored `.log` is absent, pass `s11_queue_sentinels.txt` to `report.py --rows-log`.
