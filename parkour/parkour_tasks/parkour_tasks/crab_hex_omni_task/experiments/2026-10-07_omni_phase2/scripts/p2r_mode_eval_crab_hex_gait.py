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

RECORD (the shared contract). Raw npz per episode: ``p2r_stand_mode`` (T,) bool, and (probe) ``p2r_dmu_<NAME>`` (T,)
float32. ``run_meta.json`` key ``p2r_mode``: variant, enter_rule, launcher/harness/S1 shas, specialist, probes
(probe variant only), rng_restored, stand_mode_share (recorded env-steps), switches_per_episode_mean,
settle_mode_share_by_wave (suite rows, else null), bank (relative path or null) -- plus the reported extras
``stand_mode_share_by_hold``, ``priv_cols``, ``raw_series``, ``bank_rows``.

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
``P2R_PROBES`` (``NAME=PATH`` separated by commas), ``P2R_BANK_MAX``, used only when the CLI flag is absent -- set them
through ``P2_ENV_EXTRA``. None of them is a ``KRABBY_*`` variable, so ``run_meta.env_vars`` is unchanged.

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

VARIANTS = ("hold", "st2", "probe")
ENTER_RULE = "latched_stand_slot"
DEFAULT_BANK_MAX = 20000
BANK_NAME = "p2r_bank.npz"
BANK_SEED = 0
STAND_LABEL = "stand"
PROLOGUE_LABEL = "prologue"
RAW_STAND = "p2r_stand_mode"
RAW_DMU_PREFIX = "p2r_dmu_"
_PROBE_NAME = re.compile(r"^[A-Za-z0-9_]+$")
_ENV_KEYS = {"mode": "P2R_MODE", "stand_ckpt": "P2R_STAND_CKPT", "probes": "P2R_PROBES", "bank_max": "P2R_BANK_MAX"}


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
    stray = [a for a in rest if a.startswith("--p2r")]
    if stray:
        raise SystemExit(f"p2r: unrecognised p2r argument(s) {stray}")
    return opts, rest


def validate_opts(opts: argparse.Namespace, *, check_files: bool = True) -> argparse.Namespace:
    """Normalise ``opts.probes`` to an ordered ``{name: path}`` and refuse every inconsistent combination."""
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
                 bank_max=DEFAULT_BANK_MAX, stand_labels=(STAND_LABEL,)):
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
        self.parts.append(part)
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

    # ---------------------------------------------------------------- bank
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
        idx = self.bank_seen + np.arange(cand.size)
        slot = np.where(idx < self.bank_max, idx, -1)
        late = idx >= self.bank_max
        if late.any():
            j = self._bank_rng.integers(0, idx[late] + 1)
            slot[late] = np.where(j < self.bank_max, j, -1)
        self.bank_seen += int(cand.size)
        keep = slot >= 0
        if not keep.any():
            return
        rows, slots = cand[keep], slot[keep]
        # Algorithm R applies replacements in order: for a slot hit twice in one step the LATER item wins
        _, last = np.unique(slots[::-1], return_index=True)
        pick = slots.size - 1 - last
        rows, slots = rows[pick], slots[pick]
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
        if opts.mode != "hold":
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
        mode = P2RMode(opts.mode, cmd_term, unwrapped, priv=priv, num_prop=int(est["num_prop"]), stand=stand,
                       probes=probes, bank_max=opts.bank_max, stand_labels=stand_labels)
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


def p2r_block(ctx: _Ctx, stats: dict, *, bank_rel) -> dict:
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
            if v.shape != (T, len(run_meta["n_steps_per_env"])):
                raise RuntimeError(f"p2r: series {k} shape {v.shape} != host ({T}, {len(run_meta['n_steps_per_env'])})")
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
        run_meta["p2r_mode"] = p2r_block(ctx, stats, bank_rel=bank_rel)
        kwargs["raw"] = {**raw, **series}
        print(f"[P2R] variant={mode.variant} stand_mode_share={stats['stand_mode_share']:.4f} "
              f"switches/ep={stats['switches_per_episode_mean']:.3f} settle_by_wave={stats['settle_mode_share_by_wave']} "
              f"bank={bank_rel}", flush=True)
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
    validate_opts(opts)
    print(f"[P2R] variant={opts.mode} specialist={opts.stand_ckpt} probes={opts.probes} bank_max={opts.bank_max}",
          flush=True)
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
