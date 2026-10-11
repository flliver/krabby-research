#!/usr/bin/env python3
"""Omni phase 2, stage p2r step 1 -- the controller-level stand-mode launcher (P2R_PLAN.md step 1; CHARTER 12).

An EVAL-ONLY wrapper around the eval-locked harness ``crab_hex_forward_task/scripts/eval_crab_hex_gait.py``: it
never edits the harness (eval_sha does not move) and lives in the campaign's scripts dir (code_sha does not move).
The name must end in ``eval_crab_hex_gait.py`` so ``run_arm.sh``'s ``drivers_idle`` sees it as an eval.

WHAT IT DOES. One mode state per env, keyed on the command term's LATCHED stand slot (``cmd_term.stand_mask()``
with ``_slot_latch`` armed at runtime): settle and stand holds are stand; a turn hold stays turn for its whole
length, even once it settles. Entry on the first stand step, exit on the step the slot changes. On an episode
reset (``episode_length_buf == 0``) the turning bit is RE-DERIVED from the raw relay output (the fix for the
suite's cross-wave latch carry-over: the move_turn wave's settle inherited the turn wave's latch on ~62 % of envs).

* ``--p2r-mode hold``  -- applied action = 0 on stand rows (the statue), S1's action elsewhere.
* ``--p2r-mode st2``   -- applied action = the specialist's (``--p2r-stand-ckpt``, ST2 ``model_24993``) on stand rows,
  S1's elsewhere. The specialist sees a CLONE of the obs carrying its OWN estimator output in the priv_explicit
  columns, ``hist_encoding=True`` (the harness's path). Both policies run every step.
* ``--p2r-mode probe`` -- S1 acts on every row (the run must reproduce REFS_S1_s18 bitwise). Per step and probe
  ``--p2r-probe NAME=PATH``: ``p2r_dmu_<NAME>`` = 0.5 * sum_d ((mu_probe - mu_S1) / sigma_S1,d)^2, mu on the ANCHOR
  path (``act_inference(..., hist_encoding=False)`` on the obs carrying the TRUE priv_explicit columns, each
  network on its own clone), sigma_S1 = S1's ``effective_std()``. Plus ``<stamp>/p2r_bank.npz``: a reservoir sample
  (<= ``--p2r-bank-max`` rows, a private generator, seed 0) of the recorded NON-STAND steps' obs with the true priv
  columns (hold label not ``stand`` / the suite settle; settled turns included), their hold label and telemetry
  category, and ``s1_sigma``.
* ``--p2r-mode restance`` -- NOTE(p2r Z2, 2026-10-10; P2R_Z2_PLAN.md step 1 "Probe"): applied action = the scripted
  re-stance controller ``R(obs)`` (``mdp/crab_hex_omni_restance.py``, loaded BY PATH -- the very function the PPO
  statue prior distils) on stand rows, S1's action elsewhere: the ``hold`` branch with ``R(obs)`` in place of zeros.
  ``obs`` is exactly the tensor the harness hands the policy (R reads only the current step's cam columns 18..28 and
  hip-pitch position columns 29..34, both in the proprio head, which the harness's priv_explicit overwrite never
  touches). NOTE(p2r Z2 cams-only, 2026-10-10): R reads no load signal any more (the kinematic re-step), so the
  variant no longer needs ``KRABBY_OMNI_STAND_PROXY=hold`` (Z2's chain runs it off; with it on, R simply ignores the
  held columns). R's constants come from the environment's ``KRABBY_RESTANCE`` (refused unset, or without
  ``KRABBY_OMNI_STAND_OBS=clock``).
  ``--p2r-residual JSON`` (``{"mu_bias": [18 floats]}``; the plan's residual pass) adds a fixed per-dim bias to R on
  stand rows, BEFORE the env's own action clip. ``--p2r-stand-bank`` reservoir-samples the stand rows (alive, stand
  mode; <= ``--p2r-bank-max``, a private generator, seed 0) into ``<stamp>/p2r_stand_bank.npz``: the TRUE obs (true
  priv columns, as the probe bank captures them), R's target and branch, the policy-visible proxy columns, the hold
  label and telemetry category, the step / episode index, ``s1_sigma``, R's params and the bias -- the CPU stand
  fidelity check's input (``p2r_z2_fidelity.py``).
  ``--p2r-observe`` (NOTE(p2r Z2c diagnosis, 2026-10-10)): the CLOSED-LOOP view -- the harness's own policy (the
  checkpoint under ``--checkpoint``) acts on EVERY row, stand rows included; R(obs) and its branch are still computed
  and recorded on stand rows, and the stand bank (with ``--p2r-stand-bank``) also stores the applied action ``mu``.
  The swap-live proof does not apply to an observe stamp (``run_meta`` ``restance.observe`` true); refused with a
  residual.

RECORD (the shared contract). Raw npz per episode: ``p2r_stand_mode`` (T,) bool, and (probe) ``p2r_dmu_<NAME>`` (T,)
float32; (restance) ``p2r_branch`` (T, 6) int8 (R's per-leg ``BRANCH_*`` code on stand rows, -1 off-stand),
``p2r_target`` (T, 18) float32 (R's target WITHOUT the residual bias on stand rows, 0 off-stand) and
``p2r_obs_proxy`` (T, 6) float32 (obs columns 83..88 exactly as the policy saw them, every row). ``run_meta.json``
key ``p2r_mode``: variant, enter_rule, launcher/harness/S1 shas, specialist, probes (probe variant only),
rng_restored, stand_mode_share (recorded env-steps), switches_per_episode_mean, settle_mode_share_by_wave (suite
rows, else null), bank (relative path or null) -- plus the reported extras ``stand_mode_share_by_hold``,
``priv_cols``, ``raw_series``, ``bank_rows``; (restance) ``restance`` (params, knob spelling, module sha, residual,
stand-row branch shares) and ``stand_bank`` / ``stand_bank_rows``. The restance swap-live proof (applied action ==
``p2r_target`` + bias on every stand-mode step, and == REFS before each episode's first stand entry) is
:func:`restance_swap_live`, read from the raws alone.

PATCH POINTS (verified against the harness at sha256 342339cc..., commit 60532fb9):
* ``:140`` ``args_cli = parser.parse_args()`` at IMPORT time -> this launcher strips its own args from ``sys.argv``
  BEFORE ``runpy.run_path(HARNESS, run_name="p2r_harness")`` (which runs argparse / AppLauncher / imports; ``main()``
  only runs under ``__main__``).
* ``runpy.run_path`` returns a COPY of the module globals, so every patch goes into ``g["main"].__globals__``
  (patching ``g`` would be a silent no-op):
  - ``make_on_policy_runner`` (imported ``:179``, called ``:708``) -> the patched factory: deep-copy ``train_cfg``,
    S1's runner via the original factory, the extra runners on a stand-in env with the CPU / CUDA / numpy / python
    RNG states saved and restored around them, the slot latch armed, ``runner.load`` (``:710``) wrapped to hash S1's
    checkpoint, ``runner.get_inference_policy`` (``:728``) replaced so the harness's ``policy`` is the mode wrapper,
    and ``env.step`` (``:1071``) / ``env.get_observations`` (``:900`` / ``:907``) wrapped to snapshot the TRUE
    priv_explicit columns before the harness overwrites them in place (``:1027-1029``) and to track ``alive``.
  - ``_install_schedule_override_world`` (``:393``; called ``:770`` for the single pass / wave 0 and ``:897`` for every
    later wave) -> also starts a new record pass (per-wave buffers, the hold-label map for the bank).
    ``_install_schedule_override`` (``:370``, body-frame rows) is refused: the stand slot is an omni-term concept.
  - ``_resolve_scenario`` (``:225``, called ``:591``) -> captured (the suite's settle label).
  - ``R`` (``gait_eval.report``; ``R.write_run`` at ``:1327``) -> a proxy whose ``write_run`` adds the raw series and
    ``run_meta["p2r_mode"]``, and writes the bank BEFORE the real ``write_run`` (whose ``complete.json`` stays last).
* The wrapper's return value is what the harness applies and records (``:1030`` -> ``:1048`` -> ``:1071``).
* ``simulation_app.close()`` is called only under the harness's ``__main__`` (``:1365``), so this launcher calls it
  in its own ``finally``.

ENV FALLBACK (``run_arm.sh``'s ``lr_eval`` passes no extra CLI args): ``P2R_MODE``, ``P2R_STAND_CKPT``,
``P2R_PROBES`` (``NAME=PATH`` separated by commas), ``P2R_BANK_MAX``, ``P2R_RESIDUAL`` (the JSON path),
``P2R_STAND_BANK`` (``1`` = on), used only when the CLI flag is absent -- set them through ``P2_ENV_EXTRA``. None of
them is a ``KRABBY_*`` variable, so ``run_meta.env_vars`` is unchanged. (The restance probe's own ``KRABBY_RESTANCE``
-- and ``KRABBY_OMNI_STAND_PROXY`` when a stamp sets it -- ARE ``KRABBY_*`` variables: they travel in ``P2_ENV_EXTRA``
too and are recorded.)

Launch exactly as the harness (``run_arm.sh`` with ``P2_EVAL=<this file>`` and the mode args in ``EVAL_EXTRA_ARGS``).
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import os
import random
import re
import runpy
import sys
import traceback
from pathlib import Path

_THIS = Path(__file__).resolve()
# scripts -> 2026-10-07_omni_phase2 -> experiments -> crab_hex_omni_task -> parkour_tasks (inner)
_TASKS = _THIS.parents[4]
HARNESS = _TASKS / "crab_hex_forward_task" / "scripts" / "eval_crab_hex_gait.py"
#: NOTE(p2r Z2, 2026-10-10): the scripted re-stance controller R (pure torch; loaded by path, see :func:`load_restance`)
RESTANCE_MODULE = _TASKS / "crab_hex_omni_task" / "mdp" / "crab_hex_omni_restance.py"

VARIANTS = ("hold", "st2", "probe", "restance")
ENTER_RULE = "latched_stand_slot"
DEFAULT_BANK_MAX = 20000
BANK_NAME = "p2r_bank.npz"
BANK_SEED = 0
STAND_BANK_NAME = "p2r_stand_bank.npz"
STAND_LABEL = "stand"
PROLOGUE_LABEL = "prologue"
RAW_STAND = "p2r_stand_mode"
RAW_DMU_PREFIX = "p2r_dmu_"
RAW_BRANCH = "p2r_branch"
RAW_TARGET = "p2r_target"
RAW_OBS_PROXY = "p2r_obs_proxy"
#: the env knobs the restance variant reads: it needs STAND_OBS=clock (refused otherwise); STAND_PROXY is only
#: recorded (NOTE(p2r Z2 cams-only, 2026-10-10): R reads no proxy, so the variant no longer requires it)
STAND_OBS_VAR, STAND_PROXY_VAR = "KRABBY_OMNI_STAND_OBS", "KRABBY_OMNI_STAND_PROXY"
_PROBE_NAME = re.compile(r"^[A-Za-z0-9_]+$")
_ENV_KEYS = {"mode": "P2R_MODE", "stand_ckpt": "P2R_STAND_CKPT", "probes": "P2R_PROBES", "bank_max": "P2R_BANK_MAX",
             "residual": "P2R_RESIDUAL", "stand_bank": "P2R_STAND_BANK", "observe": "P2R_OBSERVE"}


def sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ===================================================================================== argv
def _parse_probe(spec: str) -> tuple[str, str]:
    name, sep, path = spec.partition("=")
    if not sep or not name or not path:
        raise SystemExit(f"p2r: --p2r-probe expects NAME=PATH, got {spec!r}")
    if not _PROBE_NAME.match(name):
        raise SystemExit(f"p2r: probe name {name!r} must match {_PROBE_NAME.pattern} (it becomes an npz key)")
    return name, path


def strip_p2r_args(argv: list[str], environ=None) -> tuple[argparse.Namespace, list[str]]:
    """``argv`` WITHOUT the program name -> (the p2r options, the harness's argv in its original order).

    ``allow_abbrev=False``: a harness flag can never be swallowed as an abbreviation of a p2r flag. A p2r flag absent
    from the CLI falls back to its ``P2R_*`` environment variable (``run_arm.sh lr_eval`` passes no extra args).
    """
    environ = os.environ if environ is None else environ
    ap = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    ap.add_argument("--p2r-mode", dest="mode", choices=VARIANTS, default=None)
    ap.add_argument("--p2r-stand-ckpt", dest="stand_ckpt", default=None)
    ap.add_argument("--p2r-probe", dest="probes", action="append", default=None)
    ap.add_argument("--p2r-bank-max", dest="bank_max", type=int, default=None)
    ap.add_argument("--p2r-residual", dest="residual", default=None)
    ap.add_argument("--p2r-stand-bank", dest="stand_bank", action="store_const", const=True, default=None)
    ap.add_argument("--p2r-observe", dest="observe", action="store_const", const=True, default=None)
    opts, rest = ap.parse_known_args(list(argv))
    if opts.mode is None and environ.get(_ENV_KEYS["mode"]):
        opts.mode = environ[_ENV_KEYS["mode"]]
    if opts.stand_ckpt is None and environ.get(_ENV_KEYS["stand_ckpt"]):
        opts.stand_ckpt = environ[_ENV_KEYS["stand_ckpt"]]
    if opts.probes is None and environ.get(_ENV_KEYS["probes"]):
        opts.probes = [p for p in environ[_ENV_KEYS["probes"]].split(",") if p.strip()]
    if opts.bank_max is None and environ.get(_ENV_KEYS["bank_max"]):
        opts.bank_max = int(environ[_ENV_KEYS["bank_max"]])
    if opts.bank_max is None:
        opts.bank_max = DEFAULT_BANK_MAX
    if opts.residual is None and environ.get(_ENV_KEYS["residual"]):
        opts.residual = environ[_ENV_KEYS["residual"]]
    if opts.stand_bank is None:
        raw_sb = str(environ.get(_ENV_KEYS["stand_bank"], "")).strip()
        if raw_sb not in ("", "0", "1"):
            raise SystemExit(f"p2r: {_ENV_KEYS['stand_bank']} must be blank / 0 / 1, got {raw_sb!r}")
        opts.stand_bank = raw_sb == "1"
    if opts.observe is None:
        raw_ob = str(environ.get(_ENV_KEYS["observe"], "")).strip()
        if raw_ob not in ("", "0", "1"):
            raise SystemExit(f"p2r: {_ENV_KEYS['observe']} must be blank / 0 / 1, got {raw_ob!r}")
        opts.observe = raw_ob == "1"
    stray = [a for a in rest if a.startswith("--p2r")]
    if stray:
        raise SystemExit(f"p2r: unrecognised p2r argument(s) {stray}")
    return opts, rest


def load_restance(path: Path | str = RESTANCE_MODULE):
    """The re-stance controller module, loaded BY PATH (pure torch; never through the task package, whose
    ``__init__`` pulls in Isaac). The SAME file the PPO statue prior imports, so the probe proves the distilled
    target bit for bit."""
    import importlib.util  # noqa: PLC0415

    spec = importlib.util.spec_from_file_location("_p2r_restance", str(path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def load_residual(path, action_dim: int) -> dict:
    """``--p2r-residual`` JSON -> ``{"path", "sha256", "mu_bias", "source"}``: exactly ``{"mu_bias": [action_dim
    finite floats]}`` plus an optional ``"source"`` provenance string; anything else is refused (a typo'd key must
    never run as a zero bias)."""
    import json  # noqa: PLC0415
    import math  # noqa: PLC0415

    p = Path(path)
    if not p.is_file():
        raise SystemExit(f"p2r: --p2r-residual file not found: {p}")
    try:
        d = json.loads(p.read_text())
    except ValueError as exc:
        raise SystemExit(f"p2r: --p2r-residual {p} is not JSON ({exc})") from exc
    if not isinstance(d, dict) or "mu_bias" not in d or set(d) - {"mu_bias", "source"}:
        raise SystemExit(f"p2r: --p2r-residual {p} must be {{\"mu_bias\": [{action_dim} floats]}} (+ optional "
                         f"\"source\"), got keys {sorted(d) if isinstance(d, dict) else type(d).__name__}")
    bias = d["mu_bias"]
    if (not isinstance(bias, list) or len(bias) != action_dim
            or not all(isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) for v in bias)):
        raise SystemExit(f"p2r: --p2r-residual mu_bias must be {action_dim} finite numbers, got {bias!r}")
    return {"path": str(p), "sha256": sha256_file(p), "mu_bias": [float(v) for v in bias],
            "source": d.get("source")}


def _validate_restance(opts: argparse.Namespace, environ) -> None:
    """NOTE(p2r Z2, 2026-10-10): the restance variant's own refusals (environment knobs, R's constants, the bias)."""
    if opts.mode != "restance":
        if opts.residual:
            raise SystemExit(f"p2r: --p2r-residual is the restance variant's bias; refused with --p2r-mode {opts.mode}")
        if opts.stand_bank:
            raise SystemExit(f"p2r: --p2r-stand-bank is the restance variant's bank; refused with --p2r-mode {opts.mode}")
        if opts.observe:
            raise SystemExit(f"p2r: --p2r-observe is a restance-variant view; refused with --p2r-mode {opts.mode}")
        opts.restance = None
        return
    stand_obs = str(environ.get(STAND_OBS_VAR, "")).strip()
    # NOTE(p2r Z2 cams-only, 2026-10-10): only the stand bit is needed (R reads cams + hip-pitch positions; no proxy)
    if stand_obs != "clock":
        raise SystemExit(f"p2r: --p2r-mode restance needs {STAND_OBS_VAR}=clock in the environment (got "
                         f"{stand_obs!r}): the stand rows are the clock-column stand-bit rows")
    proxy = str(environ.get("KRABBY_OMNI_STAND_PROXY", "")).strip()
    if proxy not in ("", "0"):
        raise SystemExit(f"p2r: --p2r-mode restance (cams-only) needs KRABBY_OMNI_STAND_PROXY off (got {proxy!r}): the "
                         "Z2 reads mark a stamp with the held proxy on INVALID, so the row would be wasted")
    rst = load_restance()
    try:
        params = rst.parse_restance(environ.get(rst.RESTANCE_VAR))
    except ValueError as exc:
        raise SystemExit(f"p2r: {exc}") from exc
    if params is None:
        raise SystemExit(f"p2r: --p2r-mode restance needs {rst.RESTANCE_VAR} (R's constants; 'default' = the seeds)")
    opts.restance = params
    if opts.observe and opts.residual:
        raise SystemExit("p2r: --p2r-observe runs the checkpoint's own action; a --p2r-residual bias has nothing to bias")
    if opts.residual:
        opts.residual = load_residual(opts.residual, rst.ACTION_DIM)


def validate_opts(opts: argparse.Namespace, *, check_files: bool = True, environ=None) -> argparse.Namespace:
    """Normalise ``opts.probes`` to an ordered ``{name: path}`` and refuse every inconsistent combination.

    ``environ`` (default ``os.environ``) is read only by the restance variant (its ``KRABBY_*`` knobs)."""
    environ = os.environ if environ is None else environ
    opts.residual = getattr(opts, "residual", None)
    opts.stand_bank = bool(getattr(opts, "stand_bank", False))
    opts.observe = bool(getattr(opts, "observe", False))
    if opts.mode not in VARIANTS:
        raise SystemExit(f"p2r: --p2r-mode (or P2R_MODE) must be one of {VARIANTS}, got {opts.mode!r}")
    probes: dict[str, str] = {}
    for spec in opts.probes or []:
        name, path = _parse_probe(spec.strip())
        if name in probes:
            raise SystemExit(f"p2r: duplicate probe name {name!r}")
        probes[name] = path
    opts.probes = probes
    if opts.mode == "st2" and not opts.stand_ckpt:
        raise SystemExit("p2r: --p2r-mode st2 needs --p2r-stand-ckpt (the specialist checkpoint)")
    if opts.mode != "st2" and opts.stand_ckpt:
        raise SystemExit(f"p2r: --p2r-stand-ckpt is the st2 specialist; refused with --p2r-mode {opts.mode}")
    if opts.mode == "probe" and not probes:
        raise SystemExit("p2r: --p2r-mode probe needs at least one --p2r-probe NAME=PATH")
    if opts.mode != "probe" and probes:
        raise SystemExit(f"p2r: --p2r-probe is for --p2r-mode probe only (got {opts.mode})")
    if opts.bank_max is None or int(opts.bank_max) <= 0:
        raise SystemExit(f"p2r: --p2r-bank-max must be a positive integer, got {opts.bank_max!r}")
    opts.bank_max = int(opts.bank_max)
    _validate_restance(opts, environ)
    if check_files:
        for p in ([opts.stand_ckpt] if opts.stand_ckpt else []) + list(probes.values()):
            if not Path(p).is_file():
                raise SystemExit(f"p2r: checkpoint not found: {p}")
    return opts


# ===================================================================================== RNG
def save_rng() -> dict:
    import numpy as np  # noqa: PLC0415
    import torch  # noqa: PLC0415

    st = {"torch": torch.get_rng_state(), "numpy": np.random.get_state(), "python": random.getstate(), "cuda": None}
    if torch.cuda.is_available() and torch.cuda.is_initialized():
        st["cuda"] = torch.cuda.get_rng_state_all()
    return st


def restore_rng(st: dict) -> None:
    import numpy as np  # noqa: PLC0415
    import torch  # noqa: PLC0415

    torch.set_rng_state(st["torch"])
    np.random.set_state(st["numpy"])
    random.setstate(st["python"])
    if st["cuda"] is not None:
        torch.cuda.set_rng_state_all(st["cuda"])


def rng_equal(a: dict, b: dict) -> bool:
    import numpy as np  # noqa: PLC0415
    import torch  # noqa: PLC0415

    if not torch.equal(a["torch"], b["torch"]) or a["python"] != b["python"]:
        return False
    na, nb = a["numpy"], b["numpy"]
    if na[0] != nb[0] or not np.array_equal(na[1], nb[1]) or tuple(na[2:]) != tuple(nb[2:]):
        return False
    if (a["cuda"] is None) != (b["cuda"] is None):
        return False
    return a["cuda"] is None or all(torch.equal(x, y) for x, y in zip(a["cuda"], b["cuda"]))


# ===================================================================================== stand-in env
class StandInEnv:
    """The env an EXTRA runner is built on: ``get_observations`` returns zeros of the captured shapes WITHOUT
    computing an observation (no observation-manager call, so no history push and no noise draw); every other
    attribute is the real env's."""

    def __init__(self, real, template: dict):
        object.__setattr__(self, "_real", real)
        object.__setattr__(self, "_template", template)

    def get_observations(self):
        import torch  # noqa: PLC0415

        obs = torch.zeros_like(self._template["obs"])
        groups = {k: (torch.zeros_like(v) if torch.is_tensor(v) else v) for k, v in self._template["groups"].items()}
        return obs, {"observations": groups}

    def step(self, *_a, **_k):
        raise RuntimeError("p2r: an extra runner must never step the env")

    def reset(self, *_a, **_k):
        raise RuntimeError("p2r: an extra runner must never reset the env")

    def __getattr__(self, name):
        return getattr(self._real, name)


def build_with_template(orig_factory, env, train_cfg, log_dir, device):
    """S1's runner via the ORIGINAL factory on the real env, unchanged; its one ``get_observations`` call is passed
    through and its shapes captured (zeros on the env's device) for the stand-in env."""
    import torch  # noqa: PLC0415

    template: dict = {}
    had_own = "get_observations" in vars(env)
    orig_get = env.get_observations

    def _capture():
        obs, extras = orig_get()
        template["obs"] = torch.zeros_like(obs)
        template["groups"] = {
            k: (torch.zeros_like(v) if torch.is_tensor(v) else v) for k, v in (extras.get("observations") or {}).items()
        }
        return obs, extras

    env.get_observations = _capture
    try:
        runner = orig_factory(env, train_cfg, log_dir=log_dir, device=device)
    finally:
        if had_own:
            env.get_observations = orig_get
        else:
            del env.get_observations
    if "obs" not in template:
        raise RuntimeError("p2r: S1's runner never called env.get_observations(); no stand-in template")
    return runner, template


def policy_sigma(module):
    """S1's per-dim sigma: ``effective_std()`` (the crab actor's post-clamp / soft-mapped std), else std / exp(log_std)."""
    import torch  # noqa: PLC0415

    if hasattr(module, "effective_std"):
        s = module.effective_std()
    elif getattr(module, "std", None) is not None:
        s = module.std
    else:
        s = torch.exp(module.log_std)
    return s.detach().reshape(-1).clone()


def arm_slot_latch(cmd_term, num_envs: int, device) -> bool:
    """Arm the command term's eval slot latch at runtime (``crab_hex_omni_command.py`` ``__init__`` :471-474,
    ``schedule_hook`` :943-946). Returns whether it was ALREADY armed (a stand-keyed knob on)."""
    import torch  # noqa: PLC0415

    for attr in ("stand_mask", "schedule_hook", "slot_id", "speed_cmd", "vel_command_b"):
        if not hasattr(cmd_term, attr):
            raise SystemExit(f"p2r: the base_velocity term has no {attr!r} (not the omni command term)")
    pre = bool(getattr(cmd_term, "_slot_latch", False)) and getattr(cmd_term, "_hook_turning", None) is not None
    if not pre:
        cmd_term._slot_latch = True
        cmd_term._hook_turning = torch.zeros(int(num_envs), dtype=torch.bool, device=device)
    return pre


# ===================================================================================== the mode wrapper
class P2RMode:
    """The harness's ``policy``: ``mode(obs, hist_encoding=True) -> applied actions``, plus the per-pass record.

    ``s1_policy`` is S1's inference callable exactly as the harness would have used it; ``s1_module`` its actor-critic
    (the anchor path's ``act_inference(..., hist_encoding=False)`` and ``effective_std``). ``stand`` = the specialist
    ``(policy, estimator)`` (st2); ``probes`` = ``{name: actor-critic}`` (probe). ``priv`` = (start, end) of the
    priv_explicit columns, ``num_prop`` the estimator's input width.
    """

    def __init__(self, variant, cmd_term, unwrapped, *, priv, num_prop, stand=None, probes=None,
                 bank_max=DEFAULT_BANK_MAX, stand_labels=(STAND_LABEL,), restance=None, mu_bias=None,
                 stand_bank=False, observe=False):
        if variant not in VARIANTS:
            raise ValueError(variant)
        self.variant = variant
        self.cmd_term = cmd_term
        self.unwrapped = unwrapped
        self.p0, self.p1 = int(priv[0]), int(priv[1])
        self.num_prop = int(num_prop)
        self.stand = stand
        self.probes = dict(probes or {})
        if variant == "st2" and stand is None:
            raise ValueError("st2 needs the specialist")
        if variant == "probe" and not self.probes:
            raise ValueError("probe needs probes")
        # NOTE(p2r Z2, 2026-10-10): restance = (the by-path controller module, its RestanceParams)
        if (variant == "restance") != (restance is not None):
            raise ValueError("restance (module, params) is given exactly with the restance variant")
        if variant != "restance" and (mu_bias is not None or stand_bank or observe):
            raise ValueError("mu_bias / stand_bank / observe are restance-only")
        if observe and mu_bias is not None:
            raise ValueError("observe runs the policy's own action; mu_bias is refused")
        self.closed_loop = bool(observe)
        self.restance = restance
        self.mu_bias = None if mu_bias is None else [float(v) for v in mu_bias]
        self.stand_bank = bool(stand_bank)
        self.bank_max = int(bank_max)
        self.stand_labels = set(stand_labels)
        self.s1_policy = None
        self.s1_module = None
        self.s1_sigma = None
        self.parts: list[dict] = []
        self._cur = None
        self._last_obs = None
        self._true_priv = None
        self.alive = None
        self.calls = 0
        self.resets_rederived = 0
        # the bank (probe only): a reservoir over every recorded non-stand step, private generator
        self.bank_seen = 0
        self.bank_obs = None
        self.bank_label = None
        self.bank_cat = None
        self._bank_rng = None
        # the stand bank (restance --p2r-stand-bank only): a reservoir over every recorded alive stand-mode step
        self.sbank_seen = 0
        self.sbank = None
        self._sbank_rng = None
        self._ep_offset = 0      # episodes of the closed passes (the stacked series' episode axis)

    # ---------------------------------------------------------------- wiring
    def bind_s1(self, s1_policy, s1_module) -> None:
        self.s1_policy = s1_policy
        self.s1_module = s1_module
        self.s1_sigma = policy_sigma(s1_module)

    def begin_pass(self, compiled) -> None:
        """A new record pass (the single pass, or one suite wave): called right after every schedule install."""
        import numpy as np  # noqa: PLC0415

        self._close_pass()
        seg = np.asarray(compiled.segment_id)
        labels = list(compiled.hold_labels)
        lab_arr = np.array(labels + [PROLOGUE_LABEL])
        seg_idx = np.where(seg >= 0, seg, len(labels)).astype(np.int64)
        nonstand = np.array([lab not in self.stand_labels for lab in labels] + [False])[seg_idx]
        self._cur = {"T": int(compiled.n_steps), "t": 0, "stand": None, "dmu": {}, "seg_idx": seg_idx,
                     "lab_arr": lab_arr, "nonstand": nonstand}
        self.alive = None

    def _close_pass(self) -> None:
        if self._cur is None or self._cur["stand"] is None:
            self._cur = None
            return
        c = self._cur
        part = {RAW_STAND: c["stand"].detach().cpu().numpy().copy()}
        for name, v in c["dmu"].items():
            part[RAW_DMU_PREFIX + name] = v.detach().cpu().numpy().astype("float32").copy()
        for key, v in c.get("rst", {}).items():
            part[key] = v.detach().cpu().numpy().copy()
        self.parts.append(part)
        self._ep_offset += int(c["stand"].shape[1])
        self._cur = None

    def observe(self, obs) -> None:
        """Every obs the env hands the harness: keep the reference and a copy of its TRUE priv_explicit columns."""
        self._last_obs = obs
        self._true_priv = obs[:, self.p0:self.p1].clone()

    def on_step(self, dones) -> None:
        if self.alive is not None:
            self.alive = self.alive & ~dones.bool().reshape(-1)

    # ---------------------------------------------------------------- mode rule
    def rederive_on_reset(self):
        """Episode reset -> a FRESH command: the turning bit is re-derived from the raw relay output (the hook's own
        rule for a fresh target), for exactly the rows with ``episode_length_buf == 0``. Rebinds (no in-place write)."""
        import torch  # noqa: PLC0415

        ct = self.cmd_term
        reset = self.unwrapped.episode_length_buf.reshape(-1) == 0
        if bool(reset.any()):
            turning = ct.vel_command_b[:, 2] != 0.0
            moving = ct.speed_cmd > 0.0
            ct._hook_turning = torch.where(reset, turning, ct._hook_turning)
            ct.slot_id = torch.where(reset, moving.long() + 2 * ct._hook_turning.long(), ct.slot_id)
            self.resets_rederived += int(reset.sum())
        return ct.stand_mask().reshape(-1).bool()

    def _true_obs(self, obs):
        import torch  # noqa: PLC0415

        if self._true_priv is None or obs is not self._last_obs:
            raise RuntimeError("p2r: the policy was handed an obs the env did not just return; the true priv_explicit "
                               "snapshot does not belong to it")
        return torch.cat([obs[:, :self.p0], self._true_priv, obs[:, self.p1:]], dim=1)

    def _clone_with_own_estimate(self, obs, estimator):
        import torch  # noqa: PLC0415

        est = estimator.inference(obs[:, :self.num_prop])
        return torch.cat([obs[:, :self.p0], est, obs[:, self.p1:]], dim=1)

    @staticmethod
    def dmu(mu_probe, mu_s1, sigma):
        """``0.5 * sum_d ((mu_probe - mu_S1) / sigma_d)^2`` per row."""
        return 0.5 * (((mu_probe - mu_s1) / sigma.reshape(1, -1)) ** 2).sum(dim=-1)

    def __call__(self, obs, hist_encoding=True, **kwargs):
        import torch  # noqa: PLC0415

        if kwargs:
            raise RuntimeError(f"p2r: unexpected policy kwargs {sorted(kwargs)} (student / canon paths are refused)")
        if self.s1_policy is None:
            raise RuntimeError("p2r: S1's policy was never bound")
        c = self._cur
        if c is None:
            raise SystemExit("p2r: no world-frame schedule was installed before the first step (omni world rows only)")
        n = int(obs.shape[0])
        if c["stand"] is None:
            c["stand"] = torch.zeros(c["T"], n, dtype=torch.bool, device=obs.device)
            for name in self.probes:
                c["dmu"][name] = torch.zeros(c["T"], n, dtype=torch.float32, device=obs.device)
            if self.variant == "restance":
                rst = self.restance[0]
                c["rst"] = {
                    RAW_BRANCH: torch.full((c["T"], n, len(rst.LEGS)), -1, dtype=torch.int8, device=obs.device),
                    RAW_TARGET: torch.zeros(c["T"], n, rst.ACTION_DIM, dtype=torch.float32, device=obs.device),
                    RAW_OBS_PROXY: torch.zeros(c["T"], n, len(rst.PROXY_OBS_COLS), dtype=torch.float32,
                                               device=obs.device),
                }
        if self.alive is None:
            self.alive = torch.ones(n, dtype=torch.bool, device=obs.device)
        t = c["t"]
        if t >= c["T"]:
            raise RuntimeError(f"p2r: step {t} past the pass's {c['T']} planned steps")

        stand = self.rederive_on_reset()
        a_s1 = self.s1_policy(obs, hist_encoding=hist_encoding)
        if self.variant == "hold":
            actions = torch.where(stand.unsqueeze(1), torch.zeros_like(a_s1), a_s1)
        elif self.variant == "st2":
            st_policy, st_est = self.stand
            a_st = st_policy(self._clone_with_own_estimate(obs, st_est), hist_encoding=True)
            actions = torch.where(stand.unsqueeze(1), a_st.to(a_s1.dtype), a_s1)
        elif self.variant == "restance":
            actions = self._restance_step(t, obs, stand, a_s1)
        else:
            actions = a_s1
            obs_true = self._true_obs(obs)
            mu_s1 = self.s1_module.act_inference(obs_true.clone(), hist_encoding=False)
            sigma = self.s1_sigma.to(mu_s1.device)
            for name, mod in self.probes.items():
                mu_p = mod.act_inference(obs_true.clone(), hist_encoding=False)
                c["dmu"][name][t] = self.dmu(mu_p, mu_s1, sigma).float()
            self._bank_add(t, obs_true)
        c["stand"][t] = stand
        c["t"] = t + 1
        self.calls += 1
        return actions

    def _restance_step(self, t: int, obs, stand, a_s1):
        """NOTE(p2r Z2, 2026-10-10): R(obs) (+ the residual bias) on stand rows, S1's action elsewhere; the three raw
        series; the stand bank. ``obs`` is the harness's own tensor (never written)."""
        import torch  # noqa: PLC0415

        rst, params = self.restance
        tgt, branch = rst.restance_target(obs, params)
        applied = tgt
        if self.mu_bias is not None:
            applied = tgt + torch.tensor(self.mu_bias, dtype=tgt.dtype, device=tgt.device).reshape(1, -1)
        sm = stand.unsqueeze(1)
        # NOTE(p2r Z2c diagnosis, 2026-10-10): observe = the closed loop, the policy's own action on every row
        actions = a_s1 if self.closed_loop else torch.where(sm, applied.to(a_s1.dtype), a_s1)
        rec = self._cur["rst"]
        rec[RAW_BRANCH][t] = torch.where(sm, branch.to(torch.int8), torch.full_like(rec[RAW_BRANCH][t], -1))
        rec[RAW_TARGET][t] = torch.where(sm, tgt.float(), torch.zeros_like(rec[RAW_TARGET][t]))
        proxy = obs[:, list(rst.PROXY_OBS_COLS)].float()
        rec[RAW_OBS_PROXY][t] = proxy
        if self.stand_bank:
            self._stand_bank_add(t, self._true_obs(obs), stand, tgt, branch, proxy, mu=actions)
        return actions

    # ---------------------------------------------------------------- bank
    @staticmethod
    def _reservoir_pick(seen: int, n_cand: int, cap: int, rng):
        """Algorithm R over ``n_cand`` new items (``seen`` before them, capacity ``cap``) -> ``(cand_pos, slots)``:
        the candidate positions kept and their bank slots, a slot hit twice in one step keeping the LATER item (the
        order Algorithm R applies replacements in); ``(None, None)`` when nothing is kept."""
        import numpy as np  # noqa: PLC0415

        idx = seen + np.arange(n_cand)
        slot = np.where(idx < cap, idx, -1)
        late = idx >= cap
        if late.any():
            j = rng.integers(0, idx[late] + 1)
            slot[late] = np.where(j < cap, j, -1)
        keep = slot >= 0
        if not keep.any():
            return None, None
        pos, slots = np.nonzero(keep)[0], slot[keep]
        _, last = np.unique(slots[::-1], return_index=True)
        pick = slots.size - 1 - last
        return pos[pick], slots[pick]

    def _stand_bank_add(self, t: int, obs_true, stand, tgt, branch, proxy, mu=None) -> None:
        """Reservoir over every recorded ALIVE stand-mode step (restance ``--p2r-stand-bank``): the TRUE obs, R's
        target / branch, the policy-visible proxy, hold label, category, step and episode index."""
        import numpy as np  # noqa: PLC0415
        import torch  # noqa: PLC0415

        c = self._cur
        alive = self.alive.detach().cpu().numpy().astype(bool)
        st = stand.detach().cpu().numpy().astype(bool)
        cand = np.nonzero(alive & st[: alive.shape[0]])[0]
        if cand.size == 0:
            return
        if self.sbank is None:
            k = self.bank_max
            self.sbank = {"obs": np.zeros((k, int(obs_true.shape[1])), np.float32),
                          "target": np.zeros((k, int(tgt.shape[1])), np.float32),
                          "branch": np.zeros((k, int(branch.shape[1])), np.int8),
                          "proxy": np.zeros((k, int(proxy.shape[1])), np.float32),
                          "mu": np.zeros((k, int(tgt.shape[1])), np.float32),
                          "hold_label": np.empty(k, dtype=object), "category": np.empty(k, dtype=object),
                          "t": np.zeros(k, np.int32), "episode": np.zeros(k, np.int32)}
            self._sbank_rng = np.random.default_rng(BANK_SEED)
        pos, slots = self._reservoir_pick(self.sbank_seen, cand.size, self.bank_max, self._sbank_rng)
        self.sbank_seen += int(cand.size)
        if pos is None:
            return
        rows = cand[pos]
        rows_t = torch.as_tensor(rows, device=obs_true.device)
        b = self.sbank
        b["obs"][slots] = obs_true[rows_t].detach().float().cpu().numpy()
        b["target"][slots] = tgt[rows_t].detach().float().cpu().numpy()
        b["branch"][slots] = branch[rows_t].detach().cpu().numpy().astype(np.int8)
        b["proxy"][slots] = proxy[rows_t].detach().float().cpu().numpy()
        if mu is not None:
            b["mu"][slots] = mu[rows_t].detach().float().cpu().numpy()
        b["hold_label"][slots] = c["lab_arr"][c["seg_idx"][t, rows]]
        cat = self.cmd_term.telemetry_category().reshape(-1)[rows_t].detach().cpu().numpy()
        b["category"][slots] = np.array(list(self.cmd_term.TELEMETRY_CATEGORIES), dtype=object)[cat]
        b["t"][slots] = t
        b["episode"][slots] = self._ep_offset + rows

    def stand_bank_arrays(self) -> dict | None:
        """The stand bank's npz payload (restance ``--p2r-stand-bank``), else None."""
        import json  # noqa: PLC0415

        import numpy as np  # noqa: PLC0415

        if self.variant != "restance" or not self.stand_bank:
            return None
        rst, params = self.restance
        n = min(self.sbank_seen, self.bank_max)
        if self.sbank is None:
            raise RuntimeError("p2r: --p2r-stand-bank is on but the run recorded no alive stand-mode step")
        b = self.sbank
        out = {k: b[k][:n] for k in ("obs", "target", "branch", "proxy", "t", "episode")}
        out["hold_label"] = np.array([str(x) for x in b["hold_label"][:n]], dtype=str)
        out["category"] = np.array([str(x) for x in b["category"][:n]], dtype=str)
        out["s1_sigma"] = self.s1_sigma.detach().cpu().numpy().astype(np.float32)
        out["restance"] = np.array(rst.format_restance(params))
        out["restance_json"] = np.array(json.dumps(params.to_dict(), sort_keys=True))
        out["mu_bias"] = np.asarray(self.mu_bias if self.mu_bias is not None else [0.0] * rst.ACTION_DIM,
                                    dtype=np.float32)
        out["has_residual"] = np.array(self.mu_bias is not None)
        if self.closed_loop:   # NOTE(p2r Z2c diagnosis, 2026-10-10): only an observe bank carries the applied action
            out["mu"] = b["mu"][:n]
            out["observe"] = np.array(True)
        return out

    def _bank_add(self, t: int, obs_true) -> None:
        import numpy as np  # noqa: PLC0415
        import torch  # noqa: PLC0415

        c = self._cur
        alive = self.alive.detach().cpu().numpy().astype(bool)
        cand = np.nonzero(alive & c["nonstand"][t, : alive.shape[0]])[0]
        if cand.size == 0:
            return
        if self.bank_obs is None:
            self.bank_obs = np.zeros((self.bank_max, int(obs_true.shape[1])), dtype=np.float32)
            self.bank_label = np.empty(self.bank_max, dtype=object)
            self.bank_cat = np.empty(self.bank_max, dtype=object)
            self._bank_rng = np.random.default_rng(BANK_SEED)
        # Algorithm R applies replacements in order: for a slot hit twice in one step the LATER item wins
        pos, slots = self._reservoir_pick(self.bank_seen, cand.size, self.bank_max, self._bank_rng)
        self.bank_seen += int(cand.size)
        if pos is None:
            return
        rows = cand[pos]
        rows_t = torch.as_tensor(rows, device=obs_true.device)
        self.bank_obs[slots] = obs_true[rows_t].detach().float().cpu().numpy()
        self.bank_label[slots] = c["lab_arr"][c["seg_idx"][t, rows]]
        cat = self.cmd_term.telemetry_category().reshape(-1)[rows_t].detach().cpu().numpy()
        names = np.array(list(self.cmd_term.TELEMETRY_CATEGORIES), dtype=object)
        self.bank_cat[slots] = names[cat]

    def bank_arrays(self) -> dict | None:
        import numpy as np  # noqa: PLC0415

        if self.variant != "probe":
            return None
        n = min(self.bank_seen, self.bank_max)
        width = self.bank_obs.shape[1] if self.bank_obs is not None else 0
        return {
            "obs": (self.bank_obs[:n] if self.bank_obs is not None else np.zeros((0, width), np.float32)),
            "hold_label": np.array([str(x) for x in (self.bank_label[:n] if n else [])], dtype=str),
            "category": np.array([str(x) for x in (self.bank_cat[:n] if n else [])], dtype=str),
            "s1_sigma": self.s1_sigma.detach().cpu().numpy().astype(np.float32),
        }

    # ---------------------------------------------------------------- record
    def series(self) -> dict:
        """The (T, episodes) raw series in the harness's host layout (waves concatenated on the episode axis)."""
        import numpy as np  # noqa: PLC0415

        self._close_pass()
        if not self.parts:
            raise RuntimeError("p2r: the mode wrapper recorded no step")
        return {k: np.concatenate([p[k] for p in self.parts], axis=1) for k in self.parts[0]}


def summarize(series: dict, n_steps, *, compiled=None, settle_label=None) -> dict:
    """The run_meta numbers from the stacked series: share over RECORDED env-steps, switches per episode, the settle
    share per suite wave (``settle_label`` given) and the share per hold label."""
    import numpy as np  # noqa: PLC0415

    st = np.asarray(series[RAW_STAND], dtype=bool)
    n_steps = [int(n) for n in n_steps]
    if st.shape[1] != len(n_steps):
        raise RuntimeError(f"p2r: series has {st.shape[1]} episodes, run_meta {len(n_steps)}")
    rec = np.arange(st.shape[0])[:, None] < np.asarray(n_steps)[None, :]
    share = float(st[rec].mean()) if rec.any() else 0.0
    sw = [int(np.count_nonzero(st[1:n, e] != st[: max(n - 1, 0), e])) if n > 1 else 0 for e, n in enumerate(n_steps)]
    out = {"stand_mode_share": share, "switches_per_episode_mean": float(np.mean(sw)) if sw else 0.0,
           "settle_mode_share_by_wave": None, "stand_mode_share_by_hold": None}
    if compiled is not None:
        seg = np.asarray(compiled.segment_id)
        by_hold = {}
        for i, lab in enumerate(compiled.hold_labels):
            m = rec & (seg == i)
            if m.any():
                by_hold.setdefault(lab, []).append(st[m])
        out["stand_mode_share_by_hold"] = {k: float(np.concatenate(v).mean()) for k, v in by_hold.items()}
        if settle_label is not None and getattr(compiled, "wave_of_env", None) is not None:
            si = list(compiled.hold_labels).index(settle_label)
            wave = np.asarray(compiled.wave_of_env)
            res = []
            for w in range(int(wave.max()) + 1):
                m = rec & (seg == si) & (wave == w)[None, :]
                res.append(float(st[m].mean()) if m.any() else None)
            out["settle_mode_share_by_wave"] = res
    return out


def restance_stats(series: dict, n_steps, branch_names) -> dict:
    """NOTE(p2r Z2, 2026-10-10): R's branch shares over the RECORDED stand-mode leg-rows (``p2r_branch`` >= 0), and
    the share of recorded stand-mode steps whose target has any |component| > 1e-6 (the PPO side's
    ``Loss/restance_nonzero`` statistic, here on the probe's own stands)."""
    import numpy as np  # noqa: PLC0415

    br = np.asarray(series[RAW_BRANCH])
    tgt = np.asarray(series[RAW_TARGET])
    st = np.asarray(series[RAW_STAND], dtype=bool)
    rec = np.arange(st.shape[0])[:, None] < np.asarray([int(n) for n in n_steps])[None, :]
    on = rec & st
    legs = br[on]                                   # (rows, 6)
    n = int(legs.size)
    shares = {name: (float((legs == k).sum()) / n if n else None) for k, name in enumerate(branch_names)}
    nonzero = float((np.abs(tgt[on]) > 1e-6).any(axis=-1).mean()) if on.any() else None
    return {"branch_shares_on_stand": shares, "target_nonzero_share": nonzero, "stand_rows": int(on.sum())}


def restance_swap_live(refs_stamp, rst_stamp) -> dict:
    """NOTE(p2r Z2, 2026-10-10): the restance variant's swap-live proof, read from the raws alone (the HOLD proof's
    twin, ``p2_read.p2r_hold_swap_live``): (a) on every ``p2r_stand_mode`` step the applied action is exactly
    ``p2r_target`` + the stamp's residual bias (float32, bitwise), and ``p2r_branch`` is a code (>= 0); off stand,
    ``p2r_branch`` is -1 and ``p2r_target`` 0; (b) stand-mode share > 0; (c) actions BEFORE each episode's first
    stand entry equal REFS bitwise on the same episode index (no stand-row leak into walking rows or history --
    non-vacuous on walkstop, whose first stand follows walking; the suite enters stand at its settle, step 0).
    NOTE(p2r Z2 cams-only, 2026-10-10): ``p2r_obs_proxy`` is still recorded but no longer required here (R reads no
    proxy); a stamp needs only the stand / branch / target series."""
    import json  # noqa: PLC0415

    import numpy as np  # noqa: PLC0415

    out = {"stand_actions_equal_target": None, "off_stand_series_clean": None, "stand_mode_share": None,
           "pre_entry_matches_refs": None, "pre_entry_episodes_checked": 0, "fails": []}
    if not rst_stamp:
        return dict(out, fails=["no restance stamp"])
    rst_stamp = Path(rst_stamp)
    meta = json.loads((rst_stamp / "run_meta.json").read_text())
    blk = meta.get("p2r_mode") or {}
    if blk.get("variant") != "restance":
        return dict(out, fails=[f"{rst_stamp}: p2r_mode.variant is {blk.get('variant')!r}, not 'restance'"])
    res = (blk.get("restance") or {}).get("residual")
    bias = np.asarray(res["mu_bias"] if res else [0.0] * 18, dtype=np.float32)
    ref_files = {p.name: p for p in sorted((Path(refs_stamp) / "raw").glob("episode_*.npz"))} if refs_stamp else {}
    eq_ok, clean_ok, pre_ok, shares, checked = True, True, True, [], 0
    for f in sorted((rst_stamp / "raw").glob("episode_*.npz")):
        d = np.load(f)
        if not {RAW_STAND, RAW_BRANCH, RAW_TARGET} <= set(d.files):
            out["fails"].append(f"{f.name}: a restance raw series is missing")
            continue
        mode = np.asarray(d[RAW_STAND]).astype(bool)
        shares.append(float(mode.mean()))
        acts, tgt, br = np.asarray(d["actions"]), np.asarray(d[RAW_TARGET]), np.asarray(d[RAW_BRANCH])
        if mode.any():
            want = (tgt[mode] + bias[None, :]) if res else tgt[mode]
            if not np.array_equal(acts[mode], want.astype(acts.dtype)):
                eq_ok = False
                out["fails"].append(f"{f.name}: a stand-mode action differs from p2r_target + bias")
            if (br[mode] < 0).any():
                eq_ok = False
                out["fails"].append(f"{f.name}: a stand-mode step carries no branch code")
        if (br[~mode] != -1).any() or (tgt[~mode] != 0).any():
            clean_ok = False
            out["fails"].append(f"{f.name}: p2r_branch / p2r_target set on an off-stand step")
        first_on = int(np.argmax(mode)) if mode.any() else 0
        if first_on and f.name in ref_files:
            ra = np.asarray(np.load(ref_files[f.name])["actions"])
            n = min(first_on, acts.shape[0], ra.shape[0])
            checked += 1
            if n and not np.array_equal(acts[:n], ra[:n]):
                pre_ok = False
                out["fails"].append(f"{f.name}: pre-entry actions (steps 0:{n}) differ from REFS")
    out.update(stand_actions_equal_target=eq_ok, off_stand_series_clean=clean_ok, pre_entry_matches_refs=pre_ok,
               pre_entry_episodes_checked=checked)
    out["stand_mode_share"] = float(np.mean(shares)) if shares else None
    if out["stand_mode_share"] is not None and out["stand_mode_share"] <= 0:
        out["fails"].append("stand-mode share is not > 0 on a stamp with a stand hold")
    return out


# ===================================================================================== install
class _Ctx:
    def __init__(self, opts, harness: Path):
        self.opts = opts
        self.harness = Path(harness)
        self.mode: P2RMode | None = None
        self.scenario = None
        self.s1_path = None
        self.s1_sha = None
        self.specialist = None
        self.probe_meta: dict = {}
        self.rng_restored = None
        self.latch_preexisting = None
        self.restance_meta = None


def _refuse_harness_flags(args_cli) -> None:
    if args_cli is None:
        return
    if getattr(args_cli, "zero_actions", False):
        raise SystemExit("p2r: --zero-actions loads no policy; refused")
    if getattr(args_cli, "sample_actions", False):
        raise SystemExit("p2r: --sample-actions bypasses runner.get_inference_policy; refused")
    if getattr(args_cli, "canon", "off") not in (None, "off"):
        raise SystemExit("p2r: --canon feeds the policy a mirrored obs; refused")
    if getattr(args_cli, "policy_role", "auto") == "student":
        raise SystemExit("p2r: the depth student path is refused (teacher heads only)")
    if not getattr(args_cli, "save_raw", True):
        raise SystemExit("p2r: --no-save-raw drops the p2r raw series; refused")


def _load_extra(orig_factory, standin, cfg, path, device):
    runner = orig_factory(standin, copy.deepcopy(cfg), log_dir=None, device=device)
    runner.load(str(path), load_optimizer=False)
    policy = runner.get_inference_policy(device=device)
    estimator = runner.get_estimator_inference_policy(device=device)
    return runner, policy, estimator


def make_factory(ctx: _Ctx, orig_factory):
    """The patched ``make_on_policy_runner`` (see the module docstring)."""

    def factory(env, train_cfg, log_dir=None, device="cpu"):
        if ctx.mode is not None:
            raise RuntimeError("p2r: make_on_policy_runner called twice")
        pristine = copy.deepcopy(train_cfg)
        if (pristine.get("algorithm") or {}).get("class_name") == "DistillationWithExtractor":
            raise SystemExit("p2r: distillation (student) runners are refused")
        if pristine.get("empirical_normalization"):
            raise SystemExit("p2r: empirical_normalization is not supported (the anchor path reads raw obs)")
        est = pristine["estimator"]
        p0 = int(est["num_prop"]) + int(est["num_scan"])
        priv = (p0, p0 + int(est["num_priv_explicit"]))
        runner, template = build_with_template(orig_factory, env, train_cfg, log_dir, device)

        opts = ctx.opts
        stand = None
        probes = {}
        if opts.mode in ("st2", "probe"):
            standin = StandInEnv(env, template)
            before = save_rng()
            try:
                if opts.mode == "st2":
                    _, st_pol, st_est = _load_extra(orig_factory, standin, pristine, opts.stand_ckpt, device)
                    stand = (st_pol, st_est)
                    ctx.specialist = {"path": str(opts.stand_ckpt), "sha256": sha256_file(opts.stand_ckpt)}
                else:
                    for name, path in opts.probes.items():
                        r, _, _ = _load_extra(orig_factory, standin, pristine, path, device)
                        probes[name] = r.alg.policy
                        ctx.probe_meta[name] = {"path": str(path), "sha256": sha256_file(path)}
            finally:
                restore_rng(before)
            ctx.rng_restored = rng_equal(before, save_rng())
            if not ctx.rng_restored:
                raise RuntimeError("p2r: RNG state not restored after building the extra runners")
        else:
            ctx.rng_restored = True  # nothing extra was built

        unwrapped = env.unwrapped
        cmd_term = unwrapped.command_manager.get_term("base_velocity")
        ctx.latch_preexisting = arm_slot_latch(cmd_term, int(unwrapped.num_envs), cmd_term.slot_id.device)
        stand_labels = {STAND_LABEL}
        suite = getattr(ctx.scenario, "suite", None)
        if suite is not None:
            stand_labels.add(suite.settle_label)
        restance = mu_bias = None
        if opts.mode == "restance":
            # NOTE(p2r Z2, 2026-10-10): R's module by path + the params validate_opts parsed from KRABBY_RESTANCE
            rst = load_restance()
            restance = (rst, opts.restance)
            residual = getattr(opts, "residual", None)
            mu_bias = residual["mu_bias"] if residual else None
            ctx.restance_meta = {"params": opts.restance.to_dict(), "knob": rst.format_restance(opts.restance),
                                 "module": str(RESTANCE_MODULE), "module_sha256": sha256_file(RESTANCE_MODULE),
                                 "residual": residual, "observe": bool(getattr(opts, "observe", False))}
        mode = P2RMode(opts.mode, cmd_term, unwrapped, priv=priv, num_prop=int(est["num_prop"]), stand=stand,
                       probes=probes, bank_max=opts.bank_max, stand_labels=stand_labels, restance=restance,
                       mu_bias=mu_bias, stand_bank=bool(getattr(opts, "stand_bank", False)),
                       observe=bool(getattr(opts, "observe", False)))
        ctx.mode = mode

        orig_step = env.step
        orig_get = env.get_observations

        def step(actions):
            out = orig_step(actions)
            mode.observe(out[0])
            mode.on_step(out[2])
            return out

        def get_observations():
            out = orig_get()
            mode.observe(out[0])
            return out

        env.step = step
        env.get_observations = get_observations

        orig_load = runner.load

        def load(path, load_optimizer=True):
            ctx.s1_path = str(path)
            ctx.s1_sha = sha256_file(path)
            return orig_load(path, load_optimizer=load_optimizer)

        orig_gip = runner.get_inference_policy

        def get_inference_policy(device=None):
            pol = orig_gip(device=device)
            mode.bind_s1(pol, runner.alg.policy)
            return mode

        runner.load = load
        runner.get_inference_policy = get_inference_policy
        return runner

    return factory


class _ReportProxy:
    """``gait_eval.report`` with ``write_run`` wrapped (the module object itself is untouched)."""

    def __init__(self, real, write_run):
        self._real = real
        self.write_run = write_run

    def __getattr__(self, name):
        return getattr(self._real, name)


def p2r_block(ctx: _Ctx, stats: dict, *, bank_rel, rst_stats=None, stand_bank_rel=None) -> dict:
    mode = ctx.mode
    block = {
        "variant": mode.variant,
        "enter_rule": ENTER_RULE,
        "launcher_sha256": sha256_file(_THIS),
        "harness_sha256": sha256_file(ctx.harness),
        "s1_checkpoint_sha256": ctx.s1_sha,
        "specialist": ctx.specialist,
        "rng_restored": bool(ctx.rng_restored),
        "stand_mode_share": stats["stand_mode_share"],
        "switches_per_episode_mean": stats["switches_per_episode_mean"],
        "settle_mode_share_by_wave": stats["settle_mode_share_by_wave"],
        "bank": bank_rel,
        # reported extras (not part of the contract's required key set)
        "stand_mode_share_by_hold": stats["stand_mode_share_by_hold"],
        "priv_cols": [mode.p0, mode.p1],
        "raw_series": [RAW_STAND] + [RAW_DMU_PREFIX + n for n in mode.probes],
        "bank_rows": (min(mode.bank_seen, mode.bank_max) if mode.variant == "probe" else None),
        "bank_seen": (mode.bank_seen if mode.variant == "probe" else None),
        "latch_preexisting": ctx.latch_preexisting,
        "resets_rederived": mode.resets_rederived,
    }
    if mode.variant == "probe":
        block["probes"] = dict(ctx.probe_meta)
    if mode.variant == "restance":
        # NOTE(p2r Z2, 2026-10-10): restance-only keys (the other variants' blocks are unchanged)
        block["raw_series"] = [RAW_STAND, RAW_BRANCH, RAW_TARGET, RAW_OBS_PROXY]
        block["restance"] = dict(ctx.restance_meta or {}, **(rst_stats or {}))
        block["stand_bank"] = stand_bank_rel
        block["stand_bank_rows"] = (min(mode.sbank_seen, mode.bank_max) if mode.stand_bank else None)
        block["stand_bank_seen"] = (mode.sbank_seen if mode.stand_bank else None)
    return block


def make_write_run(ctx: _Ctx, orig_write_run):
    def write_run(run_dir, *args, **kwargs):
        import numpy as np  # noqa: PLC0415

        mode = ctx.mode
        if mode is None or mode.calls == 0:
            raise SystemExit("p2r: the mode wrapper never acted (the harness did not use runner.get_inference_policy); "
                             "refusing to write a stamp without the p2r record")
        run_meta = kwargs["run_meta"]
        raw = kwargs.get("raw")
        if raw is None:
            raise SystemExit("p2r: no raw buffers (save_raw off); refusing")
        if ctx.s1_sha is None or ctx.s1_sha != run_meta.get("checkpoint_sha256"):
            raise RuntimeError(f"p2r: S1 sha {ctx.s1_sha} != run_meta checkpoint_sha256 {run_meta.get('checkpoint_sha256')}")
        series = mode.series()
        T = int(np.asarray(raw["actions"]).shape[0])
        for k, v in series.items():
            # the restance series carry a trailing per-leg / per-dim axis: (T, episodes, k)
            if v.shape[:2] != (T, len(run_meta["n_steps_per_env"])):
                raise RuntimeError(f"p2r: series {k} shape {v.shape} != host ({T}, {len(run_meta['n_steps_per_env'])})")
        rst_stats = stand_bank_rel = None
        if mode.variant == "restance":
            env_vars = run_meta.get("env_vars")
            # NOTE(p2r Z2 cams-only, 2026-10-10): the stand bit only (R reads no proxy; STAND_PROXY is not required)
            if env_vars is not None and env_vars.get(STAND_OBS_VAR) != "clock":
                raise RuntimeError(f"p2r: restance stamp records {STAND_OBS_VAR}={env_vars.get(STAND_OBS_VAR)!r}; the "
                                   "env wrote no clock-column stand bit")
            rst_stats = restance_stats(series, run_meta["n_steps_per_env"], mode.restance[0].BRANCH_NAMES)
            if mode.stand_bank:
                sbank = mode.stand_bank_arrays()
                Path(run_dir).mkdir(parents=True, exist_ok=True)
                # before the real write_run, like the probe bank (complete.json stays the stamp's last file)
                np.savez_compressed(Path(run_dir) / STAND_BANK_NAME, **sbank)
                stand_bank_rel = STAND_BANK_NAME
        suite = getattr(ctx.scenario, "suite", None)
        stats = summarize(series, run_meta["n_steps_per_env"], compiled=kwargs.get("compiled"),
                          settle_label=(suite.settle_label if suite is not None else None))
        bank_rel = None
        if mode.variant == "probe":
            bank = mode.bank_arrays()
            Path(run_dir).mkdir(parents=True, exist_ok=True)
            # before the real write_run: its complete.json (suite rows) stays the LAST file of the stamp
            np.savez_compressed(Path(run_dir) / BANK_NAME, **bank)
            bank_rel = BANK_NAME
        run_meta["p2r_mode"] = p2r_block(ctx, stats, bank_rel=bank_rel, rst_stats=rst_stats,
                                         stand_bank_rel=stand_bank_rel)
        kwargs["raw"] = {**raw, **series}
        print(f"[P2R] variant={mode.variant} stand_mode_share={stats['stand_mode_share']:.4f} "
              f"switches/ep={stats['switches_per_episode_mean']:.3f} settle_by_wave={stats['settle_mode_share_by_wave']} "
              f"bank={bank_rel}", flush=True)
        if rst_stats is not None:
            print(f"[P2R] restance branches={rst_stats['branch_shares_on_stand']} "
                  f"nonzero={rst_stats['target_nonzero_share']} stand_bank={stand_bank_rel}", flush=True)
        return orig_write_run(run_dir, *args, **kwargs)

    return write_run


def install(hg: dict, ctx: _Ctx) -> None:
    """Patch the harness's REAL globals (``g["main"].__globals__``)."""
    _refuse_harness_flags(hg.get("args_cli"))
    hg["make_on_policy_runner"] = make_factory(ctx, hg["make_on_policy_runner"])

    orig_resolve = hg["_resolve_scenario"]

    def _resolve_scenario():
        ctx.scenario = orig_resolve()
        return ctx.scenario

    hg["_resolve_scenario"] = _resolve_scenario

    orig_world = hg["_install_schedule_override_world"]

    def _install_schedule_override_world(cmd_term, compiled, device, hold_labels):
        state = orig_world(cmd_term, compiled, device, hold_labels)
        if ctx.mode is None:
            raise RuntimeError("p2r: a schedule was installed before the runner factory ran")
        ctx.mode.begin_pass(compiled)
        return state

    hg["_install_schedule_override_world"] = _install_schedule_override_world

    def _install_schedule_override(*_a, **_k):
        raise SystemExit("p2r: body-frame rows are refused (the stand slot is the omni command term's; world rows only)")

    hg["_install_schedule_override"] = _install_schedule_override
    real_r = hg["R"]
    hg["R"] = _ReportProxy(real_r, make_write_run(ctx, real_r.write_run))


def main(argv: list[str] | None = None, *, harness: Path | str = HARNESS, environ=None) -> None:
    argv = list(sys.argv if argv is None else argv)
    opts, rest = strip_p2r_args(argv[1:], environ)
    validate_opts(opts, environ=environ)
    print(f"[P2R] variant={opts.mode} specialist={opts.stand_ckpt} probes={opts.probes} bank_max={opts.bank_max}",
          flush=True)
    if opts.mode == "restance":
        print(f"[P2R] restance params={opts.restance.to_dict()} residual={opts.residual} "
              f"stand_bank={opts.stand_bank} observe={opts.observe}", flush=True)
    sys.argv = [argv[0]] + rest
    g = runpy.run_path(str(harness), run_name="p2r_harness")
    hg = g["main"].__globals__
    simulation_app = hg.get("simulation_app")
    try:
        ctx = _Ctx(opts, Path(harness))
        install(hg, ctx)
        hg["main"]()
    except BaseException:
        traceback.print_exc()
        sys.stdout.flush()
        sys.stderr.flush()
        raise
    finally:
        if simulation_app is not None:
            simulation_app.close()


if __name__ == "__main__":
    main()
