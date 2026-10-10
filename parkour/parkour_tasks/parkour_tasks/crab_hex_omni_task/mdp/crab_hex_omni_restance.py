"""Omni phase 2, stage p2r Z2 -- the scripted re-stance controller R (``P2R_Z2_PLAN.md``, 2026-10-10).

PURE TORCH, NO ISAAC IMPORT: the controller-level probe (``p2r_mode_eval_crab_hex_gait.py --p2r-mode
restance``), the PPO statue prior (``KRABBY_STAND_PRIOR_TARGET=restance``) and the CPU tools (dose step,
fidelity scorer) all call the SAME :func:`restance_target`, so the distilled target is bit-for-bit the
controller the probe proved. CPU tools load this file by path (``importlib.util.spec_from_file_location``)
to stay off the task package's Isaac imports.

WHAT R DOES (a memoryless function of the CURRENT step's policy observation; no last-action terms, every
target restoring and absolute, so it stays bounded under training noise):

* inputs per leg ``i`` (legs in articulation order FL FR ML MR RL RR):
    ``c_i`` = cam angle, obs column ``18 + 2 i`` (``joint_pos - default``, already ``wrap_to_pi``);
    ``h_i`` = contact proxy, obs column ``83 + i`` -- under ``KRABBY_OMNI_STAND_PROXY=hold`` the
              sample-and-hold reading on stand rows (the last fully knee-gated value; -0.5 = unknown).
* ``loaded_i = h_i >= theta_L``.
* branches (``BRANCH_*``):
    HOLD  loaded, ``|c| <= eps_c``      cam: restoring trim ``-clip(k_h c, +-trim)``; rods: seat law
    WAIT  loaded, ``|c| > eps_c``, its tripod not selected   cam 0; rods: seat law
    LIFT  loaded, ``|c| > eps_c``, its tripod selected       cam 0; rods: lift
    HOME  unloaded, ``|c| > eps_air``, the OTHER tripod fully loaded
                                                             cam: ``-clip(k_c c, +-1)``; rods: lift
    SEAT  unloaded, otherwise                                cam: ``-clip(k_c c, +-1)``; rods: seat law
  (NOTE(p2r Z2 probe, 2026-10-10, CHARTER 12.1): HOME needs the other tripod fully loaded, exactly like LIFT. The
  plan's first rule lifted ANY unloaded un-homed leg, so a planted-tripod leg that lost load while the stepping
  tripod was in the air lifted too -- two adjacent legs of different tripods up, a forward pitch-over (probe pass
  P2R_RSTt1_s18: 50 stand falls in 231 walkstop stands). An unloaded leg without support now SEATS instead.)
  (``eps_air < eps_c``: a leg homes in the air to a tighter band than the loaded one, so a cam coast after
  landing stays inside ``eps_c`` and the trim holds it there.)
* tripods ``A = (FL, MR, RL)``, ``B = (FR, ML, RR)``. A tripod MAY step when it has a loaded leg with
  ``|c| > eps_c`` and every leg of the other tripod is loaded; if both may, the one with the larger sum of
  ``|c|`` over those legs steps (exact tie -> A, a measure-zero break of the left-right symmetry).
* rods: the hip-pitch command ``u`` is ``lift`` (< 0 lifts the foot: body-relative foot height falls with
  the hip-pitch angle on every leg) or the seat law ``kappa * clamp(theta_S - h, 0, dmax)`` (>= 0 extends a
  light leg until it reads ``theta_S``). The knee moves WITH the hip in every lift and press,
  ``knee = -rho * side_i * u`` (``side`` +1 left, -1 right: increasing |knee angle| raises the foot, and the
  knee's action sign flips across the body -- its left-right mirror sign is -1), so every load reading R makes
  has the knee rod moving, on the one knee-inclusive proxy scale training sees (user decision 2026-10-10).

MIRROR. R is equivariant under both planes provided ``lift``/``kappa``/... are the same on every leg (hip-pitch
mirror sign +1 in both planes) and the knee carries ``side`` (left-right knee sign -1, fore-aft +1); the cam
law is odd in ``c`` (cam sign -1 in both planes); the proxy is a leg-permuted magnitude. Pinned by the tests.
"""
# No ``from __future__ import annotations``: with string annotations ``dataclasses`` looks the module up in
# ``sys.modules``, which a by-path load (the CPU tools) need not have registered.
import dataclasses
import math

import torch

#: The knob carrying R's constants (``k=v,k=v``; ``default`` / ``1`` = the seed constants; blank / ``0`` = off).
RESTANCE_VAR = "KRABBY_RESTANCE"

#: Policy-observation layout (current step's proprio head; pinned against the obs term by the tests).
CAM_OBS_COLS = tuple(18 + 2 * i for i in range(6))
PROXY_OBS_COLS = tuple(83 + i for i in range(6))
#: The columns R reads; the obs must be at least this wide.
MIN_OBS_WIDTH = 89
LEGS = ("FL", "FR", "ML", "MR", "RL", "RR")
#: +1 left, -1 right (the knee's per-side sign).
SIDE = (1.0, -1.0, 1.0, -1.0, 1.0, -1.0)
TRIPOD_A = (0, 3, 4)
TRIPOD_B = (1, 2, 5)
#: Action layout (``run_meta.action_joint_names``): 6 cam velocities, 6 hip-pitch, 6 knee; legs FL..RR each.
CAM_ACT = slice(0, 6)
HIP_ACT = slice(6, 12)
KNEE_ACT = slice(12, 18)
ACTION_DIM = 18

BRANCH_HOLD, BRANCH_WAIT, BRANCH_LIFT, BRANCH_HOME, BRANCH_SEAT = range(5)
BRANCH_NAMES = ("hold", "wait", "lift", "home", "seat")


@dataclasses.dataclass(frozen=True)
class RestanceParams:
    """R's constants. Seeds (P2R_Z2_PLAN step 1); step 3's calibration pass fixes ``theta_L``/``theta_S``,
    ``lift`` and ``rho`` -- tuning changes this knob, never code."""

    lift: float = -0.5        #: hip-pitch lift command (action units; < 0 lifts)
    rho: float = 0.5          #: knee-with-hip ratio (knee = -rho * side * hip)
    theta_L: float = 0.35     #: proxy at/above which a leg counts as loaded
    theta_S: float = 0.75     #: proxy the seat law extends a light leg towards
    kappa: float = 0.5        #: seat-law gain (action units per proxy unit)
    dmax: float = 0.6         #: seat-law clamp on (theta_S - h), proxy units
    k_c: float = 2.0          #: in-air cam homing gain (action units per rad)
    k_h: float = 0.3          #: loaded in-band trim gain
    trim: float = 0.03        #: loaded in-band trim authority (user decision 2026-10-10)
    eps_c: float = 0.10       #: loaded home band, rad (the homed gate's threshold)
    eps_air: float = 0.05     #: in-air home band, rad (tighter, so a landing coast stays in eps_c)

    def __post_init__(self):
        for f in dataclasses.fields(self):
            v = getattr(self, f.name)
            if not isinstance(v, (int, float)) or not math.isfinite(float(v)):
                raise ValueError(f"{RESTANCE_VAR}: {f.name}={v!r} is not a finite number")
        if not self.lift < 0.0:
            raise ValueError(f"{RESTANCE_VAR}: lift={self.lift} must be < 0 (negative hip-pitch lifts the foot)")
        if not -1.0 <= self.lift:
            raise ValueError(f"{RESTANCE_VAR}: lift={self.lift} is outside the action clip [-1, 1]")
        if not -0.5 <= self.theta_L < self.theta_S <= 1.0:
            raise ValueError(f"{RESTANCE_VAR}: need -0.5 <= theta_L < theta_S <= 1.0 (the proxy's range), "
                             f"got theta_L={self.theta_L}, theta_S={self.theta_S}")
        for name in ("kappa", "dmax", "k_c", "k_h", "trim", "eps_c", "eps_air"):
            if not float(getattr(self, name)) > 0.0:
                raise ValueError(f"{RESTANCE_VAR}: {name} must be > 0, got {getattr(self, name)}")
        if self.rho < 0.0:
            raise ValueError(f"{RESTANCE_VAR}: rho={self.rho} must be >= 0 (the knee assists the hip)")
        if not self.eps_air <= self.eps_c:
            raise ValueError(f"{RESTANCE_VAR}: eps_air={self.eps_air} must be <= eps_c={self.eps_c}")
        if self.kappa * self.dmax > 1.0 or abs(self.lift) * max(1.0, self.rho) > 1.0 or self.rho * self.kappa * self.dmax > 1.0:
            raise ValueError(f"{RESTANCE_VAR}: a rod command would leave the action clip [-1, 1] "
                             f"(kappa*dmax={self.kappa * self.dmax}, lift={self.lift}, rho={self.rho})")
        if self.trim > 1.0:
            raise ValueError(f"{RESTANCE_VAR}: trim={self.trim} exceeds the action clip")

    def to_dict(self) -> dict:
        return {f.name: float(getattr(self, f.name)) for f in dataclasses.fields(self)}

    @classmethod
    def from_dict(cls, d: dict) -> "RestanceParams":
        names = {f.name for f in dataclasses.fields(cls)}
        unknown = set(d) - names
        if unknown:
            raise ValueError(f"{RESTANCE_VAR}: unknown constant(s) {sorted(unknown)}; known {sorted(names)}")
        return cls(**{k: float(v) for k, v in d.items()})


def parse_restance(text) -> RestanceParams | None:
    """``KRABBY_RESTANCE`` -> params, or None for OFF (blank / ``0``). ``default`` / ``1`` = the seeds;
    otherwise ``k=v,k=v`` overriding the seeds (unknown keys and repeats raise)."""
    if text is None:
        return None
    token = str(text).strip()
    if token in ("", "0"):
        return None
    if token in ("1", "default"):
        return RestanceParams()
    out: dict = {}
    for part in token.split(","):
        part = part.strip()
        if not part:
            continue
        if "=" not in part:
            raise ValueError(f"{RESTANCE_VAR}={text!r}: expected k=v pairs, got {part!r}")
        k, v = (s.strip() for s in part.split("=", 1))
        if k in out:
            raise ValueError(f"{RESTANCE_VAR}={text!r}: {k!r} given twice")
        try:
            out[k] = float(v)
        except ValueError as exc:
            raise ValueError(f"{RESTANCE_VAR}={text!r}: {k}={v!r} is not a number") from exc
    return RestanceParams.from_dict(out)


def format_restance(p: RestanceParams) -> str:
    """The canonical ``k=v,...`` spelling of ``p`` (round-trips through :func:`parse_restance`)."""
    return ",".join(f"{k}={v:g}" for k, v in p.to_dict().items())


def _side(device, dtype) -> torch.Tensor:
    return torch.tensor(SIDE, device=device, dtype=dtype)


def restance_target(obs: torch.Tensor, p: RestanceParams) -> tuple[torch.Tensor, torch.Tensor]:
    """``(target (N, 18), branch (N, 6) int64)`` for the policy observations ``obs`` (N, >= 89).

    Only the current step's columns :data:`CAM_OBS_COLS` / :data:`PROXY_OBS_COLS` are read. ``target`` is
    an action-mean target inside the clip [-1, 1]; ``branch`` the per-leg ``BRANCH_*`` code.
    """
    if obs.dim() != 2 or obs.shape[-1] < MIN_OBS_WIDTH:
        raise ValueError(f"restance_target: obs must be (N, >= {MIN_OBS_WIDTH}), got {tuple(obs.shape)}")
    dev, dt = obs.device, obs.dtype
    c = obs[:, list(CAM_OBS_COLS)]
    h = obs[:, list(PROXY_OBS_COLS)]
    loaded = h >= p.theta_L
    out_c = c.abs() > p.eps_c          # outside the loaded home band
    out_air = c.abs() > p.eps_air      # outside the in-air home band

    # ---- tripod selection (loaded, out-of-band legs step; the other tripod must be fully loaded)
    a_idx = torch.tensor(TRIPOD_A, device=dev)
    b_idx = torch.tensor(TRIPOD_B, device=dev)
    want = loaded & out_c
    mag = torch.where(want, c.abs(), torch.zeros_like(c))
    a_may = want[:, a_idx].any(-1) & loaded[:, b_idx].all(-1)
    b_may = want[:, b_idx].any(-1) & loaded[:, a_idx].all(-1)
    sum_a = mag[:, a_idx].sum(-1)
    sum_b = mag[:, b_idx].sum(-1)
    sel_a = a_may & (~b_may | (sum_a >= sum_b))
    sel_b = b_may & ~sel_a
    in_a = torch.zeros(6, dtype=torch.bool, device=dev)
    in_a[a_idx] = True
    selected = (sel_a[:, None] & in_a[None, :]) | (sel_b[:, None] & ~in_a[None, :])

    # ---- branches
    branch = torch.full(c.shape, BRANCH_HOLD, dtype=torch.long, device=dev)
    branch = torch.where(loaded & out_c & ~selected, torch.full_like(branch, BRANCH_WAIT), branch)
    branch = torch.where(loaded & out_c & selected, torch.full_like(branch, BRANCH_LIFT), branch)
    # the support rule: a leg may be (or stay) in the air for homing only while the OTHER tripod is fully loaded
    other_loaded = torch.where(in_a[None, :], loaded[:, b_idx].all(-1, keepdim=True),
                               loaded[:, a_idx].all(-1, keepdim=True))
    home = ~loaded & out_air & other_loaded
    branch = torch.where(home, torch.full_like(branch, BRANCH_HOME), branch)
    branch = torch.where(~loaded & ~home, torch.full_like(branch, BRANCH_SEAT), branch)

    # ---- cam velocity
    cam_air = -(p.k_c * c).clamp(-1.0, 1.0)
    cam_trim = -(p.k_h * c).clamp(-p.trim, p.trim)
    zero = torch.zeros_like(c)
    cam = torch.where(~loaded, cam_air, torch.where(out_c, zero, cam_trim))

    # ---- rods
    lifting = (branch == BRANCH_LIFT) | (branch == BRANCH_HOME)
    seat = p.kappa * (p.theta_S - h).clamp(0.0, p.dmax)
    hip = torch.where(lifting, torch.full_like(c, p.lift), seat)
    knee = -p.rho * _side(dev, dt)[None, :] * hip

    target = torch.cat([cam, hip, knee], dim=-1).clamp(-1.0, 1.0)
    return target, branch


def branch_shares(branch: torch.Tensor) -> dict[str, float]:
    """Share of leg-rows in each branch (``{name: share}``; empty input -> zeros)."""
    n = branch.numel()
    return {name: (float((branch == k).sum()) / n if n else 0.0) for k, name in enumerate(BRANCH_NAMES)}
