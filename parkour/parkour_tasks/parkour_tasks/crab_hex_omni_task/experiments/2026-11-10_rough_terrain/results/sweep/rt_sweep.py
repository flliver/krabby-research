"""Report-only checkpoint sweep after the step-9 HALT (CHARTER 2026-11-11 HALT entry, option 'checkpoint sweep').

Runs the campaign's own queue (rt_queue.run: lock, Isaac exclusivity, DDC/disk guards, pinned-build stamps) on a
plan built here, NOT in the repo, so no pinned/hashed campaign file changes:
  W.check (build == pin, clean tree), then for each intermediate RT_s19 checkpoint:
  R_suite es1, rt_slopes_b2 es1-2, rt_ledge_down_b2 es1-2, named RTSW_<sha8>_<rid> under the campaign OUT/lr.
Nothing here reads or writes results/step9/gates.json; the verdict is frozen.

    python -I -B rt_sweep.py --block sweep [--dry | --list]
"""
from __future__ import annotations

import sys

sys.dont_write_bytecode = True
from pathlib import Path  # noqa: E402

SCR = Path("/home/nickmagus/krabby/krabby-research/.claude/worktrees/rough-terrain/parkour/parkour_tasks/"
           "parkour_tasks/crab_hex_omni_task/experiments/2026-11-10_rough_terrain/scripts")
sys.path.insert(0, str(SCR))
import rt_machinery as M  # noqa: E402
import rt_queue as Q  # noqa: E402

RUN = Path("/home/nickmagus/krabby/krabby-research/.claude/worktrees/rough-terrain/parkour/logs/rsl_rl/"
           "crab_hex_omni_teacher/2026-11-10_23-43-01_RT_s19")
ITERS = (29500, 31000, 33000)
ROWS = (("R_suite", (1,)), ("rt_slopes_b2__omni", (1, 2)), ("rt_ledge_down_b2__omni", (1, 2)))


def plan_sweep() -> list:
    rows = M.rt_spec().load_rows()
    by_id = {r["id"]: r for r in rows["kind_rows"] + rows["retention_rows"]}
    steps = M.check_steps("W")
    for it in ITERS:
        ck = RUN / f"model_{it}.pt"
        if not ck.is_file():
            raise ValueError(f"missing checkpoint {ck}")
        sha = M.sha256_file(ck)
        for rid, ess in ROWS:
            for es in ess:
                steps.append(M.gate_step(f"W.{it}.{rid}.es{es}", name=M.row_name("RTSW", sha, rid), ck=ck,
                                         row=by_id[rid], es=es, tp_out=M.OUT))
    return steps


_orig_plan = M.plan
M.BLOCKS = (*M.BLOCKS, "sweep")
M.plan = lambda block, *a, **k: plan_sweep() if block == "sweep" else _orig_plan(block, *a, **k)

if __name__ == "__main__":
    sys.exit(Q.main())
