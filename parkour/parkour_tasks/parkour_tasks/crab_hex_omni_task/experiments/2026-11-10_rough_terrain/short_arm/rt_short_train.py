#!/usr/bin/env python3
"""ARM step 2 / step 4 (short_arm/ARM.md, Amendment A): the launcher of the arm's one training run, RT_s20s.

    /home/nickmagus/krabby/isaac_venv/bin/python -I -B rt_short_train.py --name RT_s20s [--wait-idle]
        [--dry-run | --preflight-only]

RT_s20s resumes TDR (``policy/rsd_9_p3b_rodspeed_dr/model_28991.pt``, sha 03c3e472, counter 30450 restored from the
checkpoint's alg_state) on the UNCHANGED ``envs/RT_train.env`` with seed 20 and 1010 iterations (28991 -> 30000; saves
model_29500.pt and model_30000.pt on the runner's absolute save_interval 100), 256 envs. It is ``scripts/rt_train.py``
unchanged: this launcher imports ``rt_machinery`` under exactly that module name (the same sys.modules entry rt_train
binds as ``M``), sets ``M.TRAIN_NAME = "RT_s20s"``, ``M.TRAIN_ITERS = 1010``, ``M.TRAIN_SEED = 20`` (``TRAIN_ENVS``
stays 256) BEFORE importing rt_train, and returns ``rt_train.main(argv)``. rt_train.parse_args builds its defaults at
call time, so its pre-registration guard (``--iterations`` / ``--num-envs`` / ``--seed`` must equal M.TRAIN_*) and PF14
(params/agent.yaml seed, max_iterations, run_name) then enforce the ARM's values; the argv the trainer gets is
``--seed 20 --max_iterations 1010``, the expected final file ``model_30000.pt`` (final_iteration(28991, 1010)).

Refuses (exit 2, before anything is imported or launched): any ``--name`` other than RT_s20s, ``--smoke``, and any
``--checkpoint`` / ``--env-file`` other than the defaults (TDR's checkpoint, envs/RT_train.env). Not changed by
this file: RT_train.env, TDR's checkpoint, rt_train.py, rt_machinery.py (defaults bound at import time elsewhere stay
seed 19; the arm's own gate path is rt_short.py, which does not reuse plan_gate / gate_ident). The filename keeps
``train`` so distill.sh's ISAAC_PAT counts this launcher as an Isaac process for other waiters, as rt_train.py is.

Normally run through the queue (``rt_short_launch.sh strain``: lock, build check, env check, then this with
``--wait-idle``); the dry pass is ``rt_short.py --block strain --dry``.
"""
from __future__ import annotations

import sys

sys.dont_write_bytecode = True

import argparse  # noqa: E402
from pathlib import Path  # noqa: E402

SCR = Path("/home/nickmagus/krabby/krabby-research/.claude/worktrees/rough-terrain/parkour/parkour_tasks/"
           "parkour_tasks/crab_hex_omni_task/experiments/2026-11-10_rough_terrain/scripts")
sys.path.insert(0, str(SCR))
import rt_machinery as M  # noqa: E402  (exactly this module name: rt_train's ``import rt_machinery as M`` is this object)

NAME, ITERS, SEED = "RT_s20s", 1010, 20


def refuse(why: str) -> int:
    print(f"RT SHORT TRAIN refused: {why}", flush=True)
    return 2


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    pre = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    pre.add_argument("--name")
    pre.add_argument("--checkpoint")
    pre.add_argument("--env-file")
    pre.add_argument("--smoke", action="store_true")
    ns, _rest = pre.parse_known_args(argv)
    if ns.name != NAME:
        return refuse(f"--name must be {NAME} (got {ns.name!r}); this launcher runs the arm's one pre-registered run")
    if ns.smoke:
        return refuse("--smoke is not part of the arm")
    if ns.checkpoint is not None and ns.checkpoint != str(M.TDR_CK):
        return refuse(f"--checkpoint must stay TDR's {M.TDR_CK} (got {ns.checkpoint!r})")
    if ns.env_file is not None and ns.env_file != str(M.RT_TRAIN_ENV):
        return refuse(f"--env-file must stay {M.RT_TRAIN_ENV} (got {ns.env_file!r})")
    if (M.TRAIN_NAME, M.TRAIN_ITERS, M.TRAIN_SEED, M.TRAIN_ENVS) != ("RT_s19", 6000, 19, 256):
        return refuse("rt_machinery's TRAIN_* are not the step-8 values this launcher overrides: the machinery moved")
    M.TRAIN_NAME, M.TRAIN_ITERS, M.TRAIN_SEED = NAME, ITERS, SEED
    import rt_train  # noqa: PLC0415  (after the override; module level only reads M.TRAIN_OUT)

    assert rt_train.M is M, "rt_train bound a different rt_machinery"
    print(f"RT SHORT TRAIN name={NAME} iterations={ITERS} seed={SEED} envs={M.TRAIN_ENVS} "
          f"final=model_{M.final_iteration(M.TDR_ITER, ITERS)}.pt", flush=True)
    return rt_train.main(argv)


if __name__ == "__main__":
    sys.exit(main())
