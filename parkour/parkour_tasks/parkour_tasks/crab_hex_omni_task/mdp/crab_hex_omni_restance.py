"""Omni phase 2, stage p2r Z2 -- the scripted re-stance controller R (``P2R_Z2_PLAN.md`` + its 2026-10-10
amendment: CAMS-ONLY, KINEMATIC re-step).

PURE TORCH, NO ISAAC IMPORT: the controller-level probe (``p2r_mode_eval_crab_hex_gait.py --p2r-mode
restance``), the PPO statue prior (``KRABBY_STAND_PRIOR_TARGET=restance``) and the CPU tools (dose step,
fidelity scorer) all call the SAME :func:`restance_target`, so the distilled target is bit-for-bit the
controller the probe proved. CPU tools load this file by path (``importlib.util.spec_from_file_location``)
to stay off the task package's Isaac imports.

WHY KINEMATIC (CHARTER 12.1, 2026-10-10). The first R seated each leg on the current-sense contact proxy held on
stand rows. The probe's premise STOP showed that a gated current reading cannot carry a still stand: a parked leg
cannot refresh it (71 % of the legs stuck "unloaded" at stand end carried a median 494 N; the stale holds deadlocked
116 / 294 walkstop stands) and a live reading made the press chatter (37 taps/s on the suite). The user chose
cams-only: R now reads NO load signal; a leg's phase comes from its own hip-pitch POSITION, and six-foot loading is
left to the static-indeterminacy lottery (as Z1's statue).

WHAT R DOES (a memoryless function of the CURRENT step's policy observation; every rod target absolute, the cam
laws restoring, so it stays bounded under training noise):

* inputs per leg ``i`` (articulation order FL FR ML MR RL RR):
    ``c_i`` = cam angle, obs column ``18 + 2 i`` (``joint_pos - default``, already ``wrap_to_pi``);
    ``p_i`` = hip-pitch angle, obs column ``29 + i`` (``joint_pos - default``, rad).
* ``down_i = p_i >= down_frac * 0.25 * lift`` (near default; a loaded leg sags ~0.018 rad below its target, inside
  the band), ``up_i = p_i <= up_frac * 0.25 * lift`` (near the lift target; ``0.25`` = the rod action scale).
* tripods ``A = (FL, MR, RL)``, ``B = (FR, ML, RR)``. A tripod MAY step when it has a DOWN leg with
  ``|c| > eps_step`` and every leg of the OTHER tripod is down; if both may, the one with the larger sum of ``|c|``
  over those legs steps (exact tie -> A, a measure-zero break of the left-right symmetry).
* branches (``BRANCH_*``):
    HOLD  down, ``|c| <= eps_step``                         rods 0;    cam: restoring trim ``-clip(k_h c, +-trim)``
    WAIT  down, ``|c| > eps_step``, its tripod not selected rods 0;    cam 0
    LIFT  down, ``|c| > eps_step``, its tripod selected     rods lift; cam 0 (it homes once up)
    HOME  not down, ``|c| > eps_air``, the other tripod all down
                                                            rods lift; cam ``-clip(k_c c, +-1)`` once UP, else 0
    SEAT  not down, otherwise (homed in the air, or no support)
                                                            rods 0;    cam ``-clip(k_c c, +-1)`` while UP, else 0
  (``eps_air < eps_c``: a leg homes in the air to a tighter band than the down one, so a cam coast after landing
  stays inside ``eps_c`` and the trim holds it. A cam turns only while its leg is UP or, inside the band, as the
  +-trim -- never a large turn with the foot down.)
* RE-STEP DEAD BAND (NOTE(p2r Z2c, 2026-10-10), ``P2R_Z2C_PLAN.md``): a DOWN leg re-steps only when
  ``|c| > eps_step`` (``eps_c <= eps_step``); between ``eps_c`` and ``eps_step`` it stays HOLD with the trim, so a
  small drift cannot start a re-step and the trim (acting on the whole HOLD band) pulls it back inside ``eps_c``.
  ``eps_c`` keeps only its role as the homed gate's band (the reads) and ``eps_step``'s lower bound: R no longer
  branches on it. The default ``eps_step = 0.10`` equals the default ``eps_c``, so every knob string written
  before ``eps_step`` existed (all carry ``eps_c=0.1``) parses to the old R bit for bit.
* TILT GUARD (NOTE(p2r Z2 probe, 2026-10-10, CHARTER 12.1)): while the body's |roll| or |pitch| (obs columns 3, 4,
  the current step's IMU) exceeds ``tilt_max``, no tripod may step and HOME becomes SEAT -- every lifted leg comes down.
  The full probe set (P2R_RST_s18) fell 4 / 296 walkstop es2 and 8 / 100 field_turn stands by tipping over the support
  tripod while the other tripod homed in the air; a tighter DOWN band stopped the falls (1 / 100) but the hip sag of
  loaded legs then read as "not down" and deadlocked the homing (homed 23 %).
* knee: ``knee = -rho * side_i * hip`` (``side`` +1 left, -1 right: increasing |knee angle| raises the foot and the
  knee's action sign flips across the body -- its left-right mirror sign is -1), so a lift raises the foot with both
  rods and every rest target is 0 on both.

MIRROR. R is equivariant under both planes provided its constants are the same on every leg (hip-pitch mirror sign +1
in both planes, so ``p`` and the hip target are mirror-invariant per leg) and the knee carries ``side`` (left-right
knee sign -1, fore-aft +1); the cam laws are odd in ``c`` (cam sign -1 in both planes). Pinned by the tests.

THE STAND-ROW INPUT MASK (NOTE(p2r Z2d, 2026-10-11), ``P2R_Z2D_PLAN.md``). Z2c's closed loop carried a stand-onset
crouch through two inputs: the contact proxy (columns 83-88) and the network's own last action (columns 65-82).
:func:`stand_mask_obs` sets both blocks to constants on the given rows -- last action 0, proxy -0.5 ("parked") --
in the CURRENT step's 89-wide ``obs_buf`` and, on a full-width (>= 1149) row, in each of the 10 REAL history
frames (89 wide at ``259 + 89 j``; the actor's own 75-wide history view is misaligned with them, so the mask follows
the real frames -- defence in depth, the history latent being constant on this fork). One pure function shared by
the env (``KRABBY_OMNI_STAND_MASK=1``, obs param ``stand_mask``), the cloning bank loader (``bc_stand_mask``) and the
CPU tools, so the bank and env masks cannot drift. Mirror-invariant: the constants are the same on every leg.
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
HIP_POS_OBS_COLS = tuple(29 + i for i in range(6))
#: The contact-proxy columns (no longer read by R since the 2026-10-10 amendment; kept for the CPU tools).
PROXY_OBS_COLS = tuple(83 + i for i in range(6))
#: The IMU roll / pitch columns (``wrap_to_pi`` rad) the tilt guard reads.
ROLL_OBS_COL, PITCH_OBS_COL = 3, 4
#: The columns R reads lie inside the current step's 89-wide proprio head.
MIN_OBS_WIDTH = 89
#: The rod action scale (rad per action unit; the hip-pitch target is ``default + ROD_SCALE * a``).
ROD_SCALE = 0.25
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

# NOTE(p2r Z2d, 2026-10-11): the stand-row input mask's layout and constants (see the module docstring).
#: The ``obs_buf`` last-action block ``[start, stop)`` (``17 + 2 * 24`` joints .. + 18 actions).
STAND_MASK_LAST_ACT_COLS = (65, 83)
#: The ``obs_buf`` contact-proxy block ``[start, stop)`` (= :data:`PROXY_OBS_COLS`).
STAND_MASK_PROXY_COLS = (83, 89)
#: The masked proxy value: "parked" (a still rod's reading).
STAND_MASK_PROXY_VALUE = -0.5
#: The masked last-action value.
STAND_MASK_LAST_ACT_VALUE = 0.0
#: The per-step ``obs_buf`` width; a history frame is one ``obs_buf``.
OBS_BUF_WIDTH = 89
#: The first REAL history frame's column: ``obs_buf`` 89 + heights 108 + priv_explicit 9 + priv_latent 53.
HISTORY_OFFSET = 259
#: The real history frames, ``OBS_BUF_WIDTH`` wide at ``HISTORY_OFFSET + OBS_BUF_WIDTH * j``.
HISTORY_FRAMES = 10


@dataclasses.dataclass(frozen=True)
class RestanceParams:
    """R's constants. Seeds (2026-10-10 amendment); step 3's probe passes tune them -- tuning changes this knob,
    never code."""

    lift: float = -0.5        #: hip-pitch lift command (action units; < 0 lifts)
    rho: float = 1.0          #: knee-with-hip ratio (knee = -rho * side * hip): lift clearance
    up_frac: float = 0.6      #: a leg is UP once its hip has travelled this share of the lift
    down_frac: float = 0.4    #: a leg is DOWN while its hip is within this share of the lift from default
    k_c: float = 2.0          #: in-air cam homing gain (action units per rad)
    k_h: float = 0.3          #: down in-band trim gain
    trim: float = 0.03        #: down in-band trim authority (user decision 2026-10-10)
    eps_c: float = 0.10       #: down home band, rad (the homed gate's threshold)
    eps_air: float = 0.05     #: in-air home band, rad (tighter, so a landing coast stays in eps_c)
    tilt_max: float = 0.10    #: |roll| or |pitch| (rad) beyond which nothing lifts and lifted legs come down
    #: NOTE(p2r Z2c, 2026-10-10): the re-step dead band, rad: a DOWN leg steps only beyond it (>= eps_c; the
    #: default = the default eps_c, i.e. no dead band -- the pre-Z2c R)
    eps_step: float = 0.10

    def __post_init__(self):
        for f in dataclasses.fields(self):
            v = getattr(self, f.name)
            if not isinstance(v, (int, float)) or not math.isfinite(float(v)):
                raise ValueError(f"{RESTANCE_VAR}: {f.name}={v!r} is not a finite number")
        if not -1.0 <= self.lift < 0.0:
            raise ValueError(f"{RESTANCE_VAR}: lift={self.lift} must be in [-1, 0) (negative hip-pitch lifts the foot)")
        if not 0.0 < self.down_frac < self.up_frac < 1.0:
            raise ValueError(f"{RESTANCE_VAR}: need 0 < down_frac < up_frac < 1, got down_frac={self.down_frac}, "
                             f"up_frac={self.up_frac}")
        for name in ("k_c", "k_h", "trim", "eps_c", "eps_air", "tilt_max", "eps_step"):
            if not float(getattr(self, name)) > 0.0:
                raise ValueError(f"{RESTANCE_VAR}: {name} must be > 0, got {getattr(self, name)}")
        if self.rho < 0.0:
            raise ValueError(f"{RESTANCE_VAR}: rho={self.rho} must be >= 0 (the knee assists the hip)")
        if abs(self.lift) * self.rho > 1.0:
            raise ValueError(f"{RESTANCE_VAR}: the knee lift rho*|lift|={self.rho * abs(self.lift)} leaves the action clip")
        if not self.eps_air <= self.eps_c:
            raise ValueError(f"{RESTANCE_VAR}: eps_air={self.eps_air} must be <= eps_c={self.eps_c}")
        if not self.eps_c <= self.eps_step:
            raise ValueError(f"{RESTANCE_VAR}: eps_c={self.eps_c} must be <= eps_step={self.eps_step} (the re-step "
                             "dead band starts at the homed band)")
        if self.trim > 1.0:
            raise ValueError(f"{RESTANCE_VAR}: trim={self.trim} exceeds the action clip")

    @property
    def p_up(self) -> float:
        """The hip-pitch angle (obs, rad) at or below which a leg is UP."""
        return self.up_frac * ROD_SCALE * self.lift

    @property
    def p_down(self) -> float:
        """The hip-pitch angle (obs, rad) at or above which a leg is DOWN."""
        return self.down_frac * ROD_SCALE * self.lift

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

    Only the current step's columns :data:`CAM_OBS_COLS` / :data:`HIP_POS_OBS_COLS` are read. ``target`` is an
    action-mean target inside the clip [-1, 1]; ``branch`` the per-leg ``BRANCH_*`` code.
    """
    if obs.dim() != 2 or obs.shape[-1] < MIN_OBS_WIDTH:
        raise ValueError(f"restance_target: obs must be (N, >= {MIN_OBS_WIDTH}), got {tuple(obs.shape)}")
    dev, dt = obs.device, obs.dtype
    c = obs[:, list(CAM_OBS_COLS)]
    hp = obs[:, list(HIP_POS_OBS_COLS)]
    down = hp >= p.p_down
    up = hp <= p.p_up
    # NOTE(p2r Z2c, 2026-10-10): a down leg re-steps only beyond the dead band eps_step (>= eps_c); inside it, HOLD
    out_step = c.abs() > p.eps_step    # outside the re-step dead band (down legs)
    out_air = c.abs() > p.eps_air      # outside the in-air home band

    a_idx = torch.tensor(TRIPOD_A, device=dev)
    b_idx = torch.tensor(TRIPOD_B, device=dev)
    in_a = torch.zeros(6, dtype=torch.bool, device=dev)
    in_a[a_idx] = True
    # ---- per leg: is every leg of the OTHER tripod down?
    a_down = down[:, a_idx].all(-1, keepdim=True)
    b_down = down[:, b_idx].all(-1, keepdim=True)
    other_down = torch.where(in_a[None, :], b_down, a_down)

    # ---- tripod selection (down legs beyond the dead band step; the other tripod must be all down)
    want = down & out_step
    mag = torch.where(want, c.abs(), torch.zeros_like(c))
    level = (obs[:, ROLL_OBS_COL].abs() <= p.tilt_max) & (obs[:, PITCH_OBS_COL].abs() <= p.tilt_max)
    a_may = want[:, a_idx].any(-1) & b_down[:, 0] & level
    b_may = want[:, b_idx].any(-1) & a_down[:, 0] & level
    sum_a = mag[:, a_idx].sum(-1)
    sum_b = mag[:, b_idx].sum(-1)
    sel_a = a_may & (~b_may | (sum_a >= sum_b))
    sel_b = b_may & ~sel_a
    selected = (sel_a[:, None] & in_a[None, :]) | (sel_b[:, None] & ~in_a[None, :])

    # ---- branches
    home = ~down & out_air & other_down & level[:, None]
    branch = torch.full(c.shape, BRANCH_HOLD, dtype=torch.long, device=dev)
    branch = torch.where(down & out_step & ~selected, torch.full_like(branch, BRANCH_WAIT), branch)
    branch = torch.where(down & out_step & selected, torch.full_like(branch, BRANCH_LIFT), branch)
    branch = torch.where(home, torch.full_like(branch, BRANCH_HOME), branch)
    branch = torch.where(~down & ~home, torch.full_like(branch, BRANCH_SEAT), branch)

    # ---- cam velocity
    zero = torch.zeros_like(c)
    cam_air = -(p.k_c * c).clamp(-1.0, 1.0)
    cam_trim = -(p.k_h * c).clamp(-p.trim, p.trim)
    cam = torch.where(down, torch.where(out_step, zero, cam_trim), torch.where(up, cam_air, zero))

    # ---- rods (absolute targets)
    lifting = (branch == BRANCH_LIFT) | (branch == BRANCH_HOME)
    hip = torch.where(lifting, torch.full_like(c, p.lift), torch.zeros_like(c))
    knee = -p.rho * _side(dev, dt)[None, :] * hip

    target = torch.cat([cam, hip, knee], dim=-1).clamp(-1.0, 1.0)
    return target, branch


def stand_mask_columns(width: int) -> tuple[list[int], list[float]]:
    """``(columns, values)`` the stand-row mask writes on a row ``width`` wide (NOTE(p2r Z2d, 2026-10-11)).

    The current step's last-action + proxy blocks (columns 65..88) always; on a full-width row (``width >=
    HISTORY_OFFSET + HISTORY_FRAMES * OBS_BUF_WIDTH`` = 1149) also the same two blocks of each real history frame.
    ``width < OBS_BUF_WIDTH`` raises.
    """
    width = int(width)
    if width < OBS_BUF_WIDTH:
        raise ValueError(f"stand_mask: a row must be >= {OBS_BUF_WIDTH} wide (the obs_buf), got {width}")
    starts = [0]
    if width >= HISTORY_OFFSET + HISTORY_FRAMES * OBS_BUF_WIDTH:
        starts += [HISTORY_OFFSET + OBS_BUF_WIDTH * j for j in range(HISTORY_FRAMES)]
    a0, a1 = STAND_MASK_LAST_ACT_COLS
    p0, p1 = STAND_MASK_PROXY_COLS
    cols: list[int] = []
    values: list[float] = []
    for s in starts:
        cols += list(range(s + a0, s + a1)) + list(range(s + p0, s + p1))
        values += [STAND_MASK_LAST_ACT_VALUE] * (a1 - a0) + [STAND_MASK_PROXY_VALUE] * (p1 - p0)
    return cols, values


#: ``(width, device, dtype) -> (index, values)`` device tensors, built once (the env masks every step).
_STAND_MASK_TENSORS: dict = {}


def _stand_mask_tensors(width: int, device, dtype) -> tuple[torch.Tensor, torch.Tensor]:
    key = (int(width), str(device), dtype)
    if key not in _STAND_MASK_TENSORS:
        cols, values = stand_mask_columns(width)
        _STAND_MASK_TENSORS[key] = (torch.tensor(cols, dtype=torch.long, device=device),
                                    torch.tensor(values, dtype=dtype, device=device))
    return _STAND_MASK_TENSORS[key]


def stand_mask_obs(obs: torch.Tensor, rows: torch.Tensor | None = None, *, inplace: bool = False) -> torch.Tensor:
    """The stand-row input mask (NOTE(p2r Z2d, 2026-10-11)): on ``rows`` of ``obs`` (N, >= 89) the last-action
    columns read :data:`STAND_MASK_LAST_ACT_VALUE` and the proxy columns :data:`STAND_MASK_PROXY_VALUE`, in the
    current step and (full width) every real history frame -- :func:`stand_mask_columns`.

    ``rows``: None = every row, else an (N,) bool tensor; the other rows are bitwise unchanged, as is every other
    column. A new tensor unless ``inplace`` (then ``obs`` is written and returned). dtype / device never change.
    """
    if obs.dim() != 2:
        raise ValueError(f"stand_mask_obs: obs must be (N, >= {OBS_BUF_WIDTH}), got {tuple(obs.shape)}")
    idx, val = _stand_mask_tensors(obs.shape[1], obs.device, obs.dtype)
    out = obs if inplace else obs.clone()
    if rows is None:
        out[:, idx] = val.expand(out.shape[0], -1)
        return out
    rows = torch.as_tensor(rows, device=obs.device)
    if rows.dtype != torch.bool or tuple(rows.shape) != (obs.shape[0],):
        raise ValueError(f"stand_mask_obs: rows must be a ({obs.shape[0]},) bool tensor, got "
                         f"{rows.dtype} {tuple(rows.shape)}")
    out[:, idx] = torch.where(rows[:, None], val[None, :], out[:, idx])
    return out


def branch_shares(branch: torch.Tensor) -> dict[str, float]:
    """Share of leg-rows in each branch (``{name: share}``; empty input -> zeros)."""
    n = branch.numel()
    return {name: (float((branch == k).sum()) / n if n else 0.0) for k, name in enumerate(BRANCH_NAMES)}
