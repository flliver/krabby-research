#!/usr/bin/env python3
"""Stand-network campaign (``2026-10-11_stand_network``) -- the ONE shared stand-net input function + small helpers.

PLAN.md D1: "One function, called identically by the trainer and the launcher." IMPLEMENTATION_MAP.md section 1.
Both the ``stand_net`` launcher (``sn_stand_net_eval_crab_hex_gait.py``, step 3) and the DAgger trainer (step 4)
import THIS file by path and build the stand network's actor input with :func:`stand_net_input`, nothing else:

    x = stand_net_input(obs, stand_estimator, p0=p0, p1=p1, num_prop=num_prop, mask=stand_mask)
    a_stand = stand_policy(x, hist_encoding=True)          # act_inference: the D1 forward path (never the priv path)
    actions = route_actions(stand, a_stand, a_s1)          # torch.where on the latched stand rows

* the estimate is computed on the UNMASKED prop ``obs[:, :num_prop]`` (D1: the estimator reads cols 65-74, which
  the mask writes -- map risk L3), written into a NEW tensor at the priv-explicit columns ``[p0, p1)`` (the caller's
  ``obs`` is never written -- risk L2), and only then, iff ``mask``, ``stand_mask_obs(x, rows=None, inplace=True)``
  masks the last-action / contact-proxy blocks of the current step and the 10 real history frames on the fresh
  tensor (``rows=None``: every row of the clone; rows outside stand are discarded by :func:`route_actions`, so the
  call takes no stand vector and the trainer and launcher make literally the same call -- map section 9 C2);
* ``p0``, ``p1``, ``num_prop`` always come from the estimator cfg (the launcher: P2L's ``priv`` / ``num_prop``; the
  trainer: ``train_cfg["estimator"]``) -- never hard-coded "real layout" numbers;
* R (the scripted re-stance controller, ``mdp/crab_hex_omni_restance.py``) is loaded BY PATH and only inside
  functions; its constants are the validated D3 string :data:`RESTANCE_D3` -- never ``"default"``, never
  ``RestanceParams()`` (map risk T4). :func:`r_params` refuses a spelling that does not round-trip.

LAZY IMPORTS ONLY (map risk L6): the launcher loads this module before Kit starts, the harness imports torch only after
AppLauncher, and the R module imports torch at its top. Nothing below imports torch, numpy or the R module at module
level; :func:`load_rst` runs on the first call that needs R (post-Kit in the launcher).

Also here (shared so the launcher, the trainer and the readers cannot drift): the run_meta field names of the
``stand_net`` head-validity contract (``sn_calib.stand_net_reasons``: ``p2r_mode.variant == "stand_net"``,
``p2r_mode.specialist.sha256``, ``p2r_mode.stand_mask`` an INT 0 / 1), the stand_net raw-series keys, and the D5
obs-parity environment rule (:func:`env_reasons`).
"""
from __future__ import annotations

import contextlib
import hashlib
import importlib.util
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
#: scripts -> 2026-10-11_stand_network -> experiments -> crab_hex_omni_task
CAMPAIGN = HERE.parent
TASK = HERE.parents[2]
#: R's module (pure torch; loaded by path, never through the task package, whose ``__init__`` pulls in Isaac)
RESTANCE_MODULE = TASK / "mdp" / "crab_hex_omni_restance.py"

#: PLAN.md D3: R's validated constants, the FULL string (the P2R_RSTg_s18 / SN_RSTg_s18 probe's constants).
RESTANCE_D3 = ("lift=-0.5,rho=1,up_frac=0.5,down_frac=0.4,k_c=2,k_h=1,trim=0.1,eps_c=0.1,eps_air=0.05,"
               "tilt_max=0.1,eps_step=0.3")

# ------------------------------------------------------------------ run_meta contract (sn_calib.stand_net_reasons)
VARIANT = "stand_net"
#: ``run_meta.p2r_mode.<MASK_KEY>``: the --stand-mask flag as an INT 0 / 1 (a bool is malformed for sn_calib)
MASK_KEY = "stand_mask"
#: ``run_meta.p2r_mode.specialist.sha256``: the stand checkpoint's sha256 (P2L's st2 ``specialist`` record)
CKPT_KEY = ("specialist", "sha256")
#: ``run_meta.p2r_mode.<BLOCK_KEY>``: the campaign's own provenance block (wrapper / common / R module shas, ...)
BLOCK_KEY = "stand_net"
#: the extra per-episode raw series a stand_net stamp carries (beside P2L's ``p2r_stand_mode``)
RAW_STAND_ACTION = "sn_stand_action"    # (T, 18) float32: the stand net's action on stand rows, 0 elsewhere
RAW_S1_ACTION = "sn_s1_action"          # (T, 18) float32: S1's (counterfactual) action on stand rows, 0 elsewhere
RAW_R_TARGET = "sn_r_target"            # (T, 18) float32: R's target on stand rows (D3 constants), 0 elsewhere
RAW_R_BRANCH = "sn_r_branch"            # (T, 6) int8: R's per-leg branch code on stand rows, -1 elsewhere
RAW_KEYS = (RAW_STAND_ACTION, RAW_S1_ACTION, RAW_R_TARGET, RAW_R_BRANCH)
MASK_FN = "crab_hex_omni_restance.stand_mask_obs"
MASK_ROWS = "all"
FORWARD = "act_inference(stand_net_input(obs, stand_estimator, mask=stand_mask), hist_encoding=True)"
ACTION_DIM = 18
NUM_LEGS = 6

# ------------------------------------------------------------------ D5 obs parity (environment)
STAND_OBS_VAR = "KRABBY_OMNI_STAND_OBS"
STAND_MASK_VAR = "KRABBY_OMNI_STAND_MASK"
STAND_PROXY_VAR = "KRABBY_OMNI_STAND_PROXY"
RESTANCE_VAR = "KRABBY_RESTANCE"
#: env-level stand-slot knobs REFS_S1_s18 runs off ("0"); a stand_net run must too (the stand net does the standing)
STAND_SLOT_KNOBS = ("KRABBY_OMNI_STAND_HOLD", "KRABBY_OMNI_STAND_PARK")
#: p2_read.ENV4 (the S1 EVAL_CARRY values every phase-2 head stamp must carry; pinned equal by the tests)
ENV4 = {"KRABBY_OMNI_CLOCK_HIDE": "1", "KRABBY_STD_PARAM": "soft", "KRABBY_OMNI_SPEED_ELLIPSE": "0.35:0.35:0.25",
        "KRABBY_OMNI_FACING_OBS_CLIP": "0.6"}
_OFF = ("", "0")


def _env(environ, k: str) -> str | None:
    v = environ.get(k)
    return None if v is None else str(v).strip()


def env_reasons(environ, *, env4: bool = True) -> list[str]:
    """PLAN.md D5 obs parity + map risk L10, for a process that runs S1 beside the stand net (fail closed):
    ``KRABBY_OMNI_STAND_OBS`` exactly ``"0"`` (absent fails: col 15 = 0, as S1 / REFS / HOLD); the env-level stand
    mask and held proxy off (D1: the stand net masks ACTOR-side, S1 never sees masked frames); ``KRABBY_RESTANCE``
    off (R's constants live here, not in an env knob -- an R / Z stamp's env is the wrong env); the env-level stand
    hold / park off (as REFS); with ``env4`` (the eval launcher) the four ENV4 values exactly as REFS
    (``p2_read.ENV4``)."""
    out = []
    v = _env(environ, STAND_OBS_VAR)
    if v != "0":
        out.append(f"{STAND_OBS_VAR}={v!r}: want '0' (D5 obs parity: col 15 = 0; an R / Z stamp's 'clock' env is "
                   "the wrong env)")
    for k in (STAND_MASK_VAR, STAND_PROXY_VAR, RESTANCE_VAR) + STAND_SLOT_KNOBS:
        val = _env(environ, k)
        if val is not None and val.lower() not in _OFF:
            out.append(f"{k}={val!r}: want it off (absent / '' / '0')")
    for k, want in (ENV4.items() if env4 else ()):
        if _env(environ, k) != want:
            out.append(f"{k}={_env(environ, k)!r}: want {want!r} (the S1 EVAL_CARRY block, p2_read.ENV4)")
    return out


# ------------------------------------------------------------------ files / modules
def sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


@contextlib.contextmanager
def no_bytecode():
    """Suppress ``__pycache__`` writes for a by-path load only (the process flag is restored afterwards, so the eval
    process's state matches the plain p2r launcher path that reproduced REFS bitwise)."""
    old = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    try:
        yield
    finally:
        sys.dont_write_bytecode = old


def load_by_path(path, name: str):
    """A module loaded by path into a PRIVATE instance (not registered in ``sys.modules``; no bytecode written)."""
    spec = importlib.util.spec_from_file_location(name, str(path))
    if spec is None or spec.loader is None:
        raise ImportError(f"sn_common: cannot load {path}")
    mod = importlib.util.module_from_spec(spec)
    with no_bytecode():
        spec.loader.exec_module(mod)
    return mod


_RST: dict = {}
_PARAMS: dict = {}


def load_rst(path=RESTANCE_MODULE):
    """R's module, loaded by path once per process (imports torch: call post-Kit only)."""
    key = str(Path(path).resolve())
    if key not in _RST:
        _RST[key] = load_by_path(key, "_sn_restance")
    return _RST[key]


def r_params():
    """R's D3 params: ``parse_restance(RESTANCE_D3)``, refused unless ``format_restance`` round-trips the exact string
    and the result differs from the class defaults (``"default"`` / ``RestanceParams()`` is a different R)."""
    if "p" not in _PARAMS:
        rst = load_rst()
        p = rst.parse_restance(RESTANCE_D3)
        if p is None or rst.format_restance(p) != RESTANCE_D3:
            raise RuntimeError(f"sn_common: R's D3 constants do not round-trip: {RESTANCE_D3!r} -> "
                               f"{None if p is None else rst.format_restance(p)!r}")
        if p == rst.RestanceParams():
            raise RuntimeError("sn_common: the D3 constants equal RestanceParams() defaults -- refused (map T4)")
        _PARAMS["p"] = p
    return _PARAMS["p"]


def r_label(obs):
    """R on the visited state: ``(target (N, 18) in [-1, 1], branch (N, 6) int64)`` with the D3 constants. R reads
    only the current step's cols 3-4, 18-28 (cams) and 29-34 (hip pitch): identical on the raw, own-estimate or
    masked obs. Pure (no RNG)."""
    return load_rst().restance_target(obs, r_params())


# ------------------------------------------------------------------ the D1 forward input
def _mask_flag(mask) -> bool:
    if isinstance(mask, bool):
        return mask
    if type(mask) is int and mask in (0, 1):
        return bool(mask)
    raise TypeError(f"sn_common: mask must be a bool or an int 0 / 1, got {mask!r}")


def stand_net_input(obs, estimator, *, p0: int, p1: int, num_prop: int, mask):
    """The stand network's actor input (PLAN.md D1; the ONE function the launcher and the trainer both call).

    ``est = estimator.inference(obs[:, :num_prop])`` on the UNMASKED prop -> a NEW tensor
    ``cat(obs[:, :p0], est, obs[:, p1:])`` (bitwise the harness's in-place ``obs[:, p0:p1] = est`` when ``obs[:, p0:p1]``
    already holds this estimate) -> iff ``mask``, ``stand_mask_obs(x, rows=None, inplace=True)`` on that fresh tensor.
    ``obs`` is never written. ``mask``: bool or int 0 / 1 (anything else raises)."""
    import torch  # noqa: PLC0415

    m = _mask_flag(mask)
    if obs.dim() != 2:
        raise ValueError(f"stand_net_input: obs must be (N, W), got {tuple(obs.shape)}")
    p0, p1, num_prop = int(p0), int(p1), int(num_prop)
    if not 0 < num_prop <= p0 < p1 <= int(obs.shape[1]):
        raise ValueError(f"stand_net_input: need 0 < num_prop <= p0 < p1 <= width, got num_prop={num_prop} "
                         f"p0={p0} p1={p1} width={obs.shape[1]} (the estimate must not read the columns it replaces)")
    est = estimator.inference(obs[:, :num_prop])
    if tuple(est.shape) != (int(obs.shape[0]), p1 - p0) or est.dtype != obs.dtype:
        raise ValueError(f"stand_net_input: the estimate is {tuple(est.shape)} {est.dtype}, want "
                         f"({obs.shape[0]}, {p1 - p0}) {obs.dtype}")
    x = torch.cat([obs[:, :p0], est, obs[:, p1:]], dim=1)
    if m:
        load_rst().stand_mask_obs(x, None, inplace=True)
    return x


def route_actions(stand, a_stand, a_s1):
    """``torch.where(stand[:, None], a_stand, a_s1)``: the stand net acts on the latched stand rows, S1 elsewhere
    (the p2r st2 rule, P2L:596 -- both networks run on every row)."""
    import torch  # noqa: PLC0415

    if stand.dtype != torch.bool or stand.dim() != 1 or int(stand.shape[0]) != int(a_s1.shape[0]):
        raise ValueError(f"route_actions: stand must be an ({a_s1.shape[0]},) bool tensor, got {stand.dtype} "
                         f"{tuple(stand.shape)}")
    if tuple(a_stand.shape) != tuple(a_s1.shape):
        raise ValueError(f"route_actions: action shapes differ {tuple(a_stand.shape)} vs {tuple(a_s1.shape)}")
    return torch.where(stand.unsqueeze(1), a_stand.to(a_s1.dtype), a_s1)


def state_dicts_equal(a: dict, b: dict) -> tuple[bool, str]:
    """``(equal, why)``: same key set, and every tensor equal in shape, dtype and value (CPU copies, ``torch.equal``)."""
    import torch  # noqa: PLC0415

    ka, kb = set(a), set(b)
    if ka != kb:
        return False, f"keys differ: only-first {sorted(ka - kb)[:5]}, only-second {sorted(kb - ka)[:5]}"
    for k in sorted(ka):
        x, y = a[k], b[k]
        if not (torch.is_tensor(x) and torch.is_tensor(y)):
            if x != y:
                return False, f"{k}: non-tensor values differ"
            continue
        if x.shape != y.shape or x.dtype != y.dtype:
            return False, f"{k}: {tuple(x.shape)} {x.dtype} vs {tuple(y.shape)} {y.dtype}"
        if not torch.equal(x.detach().cpu(), y.detach().cpu()):
            return False, f"{k}: values differ"
    return True, ""
