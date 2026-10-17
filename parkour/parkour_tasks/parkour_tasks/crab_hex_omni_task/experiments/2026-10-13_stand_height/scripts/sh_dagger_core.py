#!/usr/bin/env python3
"""Stand-height campaign (``2026-10-13_stand_height``), PLAN H4 / H5 / H6 + step 5 -- the height DAgger trainer's TORCH
core. COPIED from the closed predecessor ``2026-10-11_stand_network/scripts/sn_dagger_core.py`` (sha256 737f2e97...;
never edited) and adapted; everything that decides WHAT is trained lives here so the CPU tests pin it without Isaac.

LOADED POST-KIT ONLY (``sh_dagger.py`` loads it after ``AppLauncher``; it imports torch at its top). Kept unchanged from
the predecessor: private RNG streams (T5), S1 + its estimator frozen, only ``actor.actor_backbone`` trainable (OPEN-1),
Adam over all student parameters in named order (T11), the capped reservoir of EXACT actor inputs compacted to the
columns the hist-path actor reads, per-leg inverse-branch-frequency weights + LIFT-onset oversampling (OPEN-3), beta
1 -> 0 over the first third per episode, the 10 % R-executed control slice, per-episode DART sigma, the first-draw stand
tracker (OPEN-7 amendment), the checkpoint format (rsl_rl, CPU tensors, S1's estimator verbatim).

WHAT CHANGED (PLAN H1-H6; every height semantic is ``sh_core``'s, imported, never re-implemented):

* ONE PER-STEP ORDER (``sh_core.HeightController``): ``step(stand_slot, h_cmd, obs)`` before the actions -> the limited
  command ``h_lim`` (q, column 15, R_h), the EFFECTIVE stand vector ``slot | restoring`` (routing, labels, reservoir) and
  the restore rows (input zeroing); ``reset(done rows, start kind codes, h_meas)`` after ``env.step``: walk-handover
  episodes at 1.064, table / ground / random-height starts at ``clamp(h_meas)`` measured on the reset row;
* the label is R_h's RESIDUAL ``R_h - q(h_lim)`` (``sh_core.rh_label``, raw clip 4.8), on the visited state, on every
  effective stand row (restore rows included: "restore steps are stand-network steps", user 2026-10-13);
* the executed stand action is ``sh_core.stand_action(a_res, h_lim) = q + clamp(a_res, +-1)`` (rods +-4.8, cams +-1),
  ``a_res`` = the student's unclipped mean (student episodes) or R_h's residual (R episodes), + DART sigma noise on the
  residual; an R episode at sigma 0 executes R_h's target itself; S1's rows are ``clamp(S1, +-1)``
  (``sh_core.route``); the env wrapper clips at ``sh_core.HARNESS_CLIP`` = 4.8 (runtime, no config edit);
* the stand-net input is ``sh_core.stand_net_input(obs, est, mask=1, h_cmd_limited=h_lim, restoring=...)``; the MSE
  fits the unclipped mean to the residual label on that input;
* ``--deadband d`` (``RunOpts.deadband``; default 0 = off, every output bitwise the run without the option; CHARTER
  2026-10-14 "probe a residual deadband" -> "train it in"): the STUDENT-executed stand action only (student episodes,
  any DART sigma, restore rows included) is ``sh_core.stand_action(a_res, h_lim, deadband=d)`` -- the clamped residual's
  hip / knee entries below d zeroed before ``+ q``; the labels stay R_h's residuals, the R_h control slice and the
  beta-R episodes execute exactly as without it; recorded in run.json (``opts.deadband``,
  ``sh_dagger.residual_deadband`` + its clip record), status.json, the checkpoint infos (``infos.sh_dagger``) and the
  smoke (``student_deadband_realised``: no realised student hip / knee residual inside (0, d); forward parity with the
  launcher's ``--sh-deadband d``);
* ``--trim-branch`` (``RunOpts.trim_branch``; default off = the weights over R_h's 5 branches, every output bitwise the
  run without the option; CHARTER 2026-10-14 "retrain the pilot with a trim weight"): every effective stand-row leg
  (restore rows included) whose R_h branch is HOLD and whose R_h cam label ``|.| > trim_min`` (``SPEC.TRIM_LABEL_MIN``
  1e-6 unless ``--trim-min``) enters the reservoir with the code :data:`BRANCH_TRIM`, a 6th category of the SAME per-leg
  inverse-branch-frequency weights (``branch_weight_table``, cap W_MAX; HOLD's frequency then counts the non-trim HOLD
  entries only); labels, LIFT onsets (R_h's codes), sampling and the loss are unchanged; recorded in run.json
  (``opts.trim_branch`` / ``opts.trim_min``, ``sh_dagger.trim_branch``), status.json, the checkpoint infos and the smoke
  (``trim_branch_weights``; the student branch shares report 'trim');
* the trim DIAGNOSTIC (reported, every run; ``Trainer._trim_diag``): per iteration, on student-executed sigma-0 stand
  rows, the legs R_h HOLDs with ``|c|`` in TRIM_BAND and the foot loaded -- histograms of the executed cam command and
  of R_h's label, toward home (``train_log.jsonl`` ``trim_diag``; read by ``SPEC.trim_diag_read`` in ``--read``);
* warm start: A-mask s17 with ``W[:, 15]`` zeroed (``sh_core.load_warm_start``; its estimator must equal S1's);
* the env side is ``sh_starts.StartState`` (start modes, h_cmd draws, forced first stands, table, pushes, hip-contact);
  the trainer reads its per-env fields after every ``env.step`` and a probe (root z, chassis lowest point, ``h_meas``,
  feet loaded, table top, the applied action);
* records: ``stands.jsonl`` (+ start kind, h_cmd / h_init / |h - h_cmd| / STOOD on the first draw),
  ``episodes.jsonl`` (one per finished episode: kind, pose, top, length, end cause, first stand, re-arm / park / up
  times), ``train_log.jsonl`` (+ restore events by executor x start kind); checkpoints record ``stand_mask`` 1, the
  height-command encoding and the h range.
"""
from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import math
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, str(path))
    mod = importlib.util.module_from_spec(spec)
    old = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    try:
        spec.loader.exec_module(mod)
    finally:
        sys.dont_write_bytecode = old
    return mod


#: the ONE height core (q, R_h, limiter, restore, input, router, warm start) and the torch-free spec / env composition
SH = _load(HERE / "sh_core.py", "_sh_dagger_core_sh")
SPEC = _load(HERE / "sh_dagger_spec.py", "_sh_dagger_core_spec")
STARTS = _load(HERE / "sh_starts.py", "_sh_dagger_core_starts")
SN = SH.SN

NUM_OBS, NUM_ACT, NUM_LEGS = 1149, 18, 6
CAM_COLS = tuple(18 + 2 * i for i in range(NUM_LEGS))
HIP_COLS = tuple(29 + i for i in range(NUM_LEGS))
SIDE = (1.0, -1.0, 1.0, -1.0, 1.0, -1.0)
BRANCH_HOLD = 0
BRANCH_LIFT = 2
N_BRANCH = 5
#: the trim branch (CHARTER 2026-10-14 "retrain the pilot with a trim weight"; ``RunOpts.trim_branch``): a 6th WEIGHT
#: category, never one of R's codes -- stored in the reservoir's branch column only when the option is on
BRANCH_TRIM = 5
N_BRANCH_TRIM = 6
BRANCH_NAMES = ("hold", "wait", "lift", "home", "seat")
BRANCH_NAMES_TRIM = BRANCH_NAMES + ("trim",)
GEN_NAMES = ("episode", "dart", "reservoir", "minibatch", "init_len", "starts", "hcmd")
ENTRY_KINDS = ("spawn", "handover", "run_start")
N_KINDS = len(SPEC.START_KINDS)
N_EX = len(SPEC.EXECUTORS)
N_EV = len(SPEC.RESTORE_EVENTS)
N_SG = len(SPEC.RESTORE_SIGMA)


def sha256_file(path) -> str:
    return SN.sha256_file(path)


# ===================================================================================== RNG (T5)
def make_generators(seed: int, device) -> dict:
    """Private generators, one per draw kind, seeded from (run seed, name): independent of each other and of the
    global streams the env's command sampler / pushes / terrain use. ``starts`` / ``hcmd`` are handed to the env-side
    ``StartState`` before ``gym.make``."""
    out = {}
    for name in GEN_NAMES:
        h = int.from_bytes(hashlib.sha256(f"sh_dagger:{int(seed)}:{name}".encode()).digest()[:8], "little")
        g = torch.Generator(device=device)
        g.manual_seed(h & ((1 << 63) - 1))
        out[name] = g
    return out


def save_rng() -> dict:
    st = {"torch": torch.get_rng_state(), "numpy": np.random.get_state(), "python": random.getstate(), "cuda": None}
    if torch.cuda.is_available() and torch.cuda.is_initialized():
        st["cuda"] = torch.cuda.get_rng_state_all()
    return st


def restore_rng(st: dict) -> None:
    torch.set_rng_state(st["torch"])
    np.random.set_state(st["numpy"])
    random.setstate(st["python"])
    if st["cuda"] is not None:
        torch.cuda.set_rng_state_all(st["cuda"])


def rng_equal(a: dict, b: dict) -> bool:
    if not torch.equal(a["torch"], b["torch"]) or a["python"] != b["python"]:
        return False
    na, nb = a["numpy"], b["numpy"]
    if na[0] != nb[0] or not np.array_equal(na[1], nb[1]) or tuple(na[2:]) != tuple(nb[2:]):
        return False
    if (a["cuda"] is None) != (b["cuda"] is None):
        return False
    return a["cuda"] is None or all(torch.equal(x, y) for x, y in zip(a["cuda"], b["cuda"]))


# ===================================================================================== networks (T6, OPEN-1, H4 warm start)
class Nets:
    """S1 (frozen), S1's estimator (frozen; the stand net's estimator too) and the student."""

    def __init__(self, s1, est, student, *, p0, p1, num_prop, num_hist, s1_estimator_state, s1_path, s1_sha, warm):
        self.s1, self.est, self.student = s1, est, student
        self.p0, self.p1, self.num_prop, self.num_hist = int(p0), int(p1), int(num_prop), int(num_hist)
        self.s1_estimator_state = s1_estimator_state
        self.s1_path, self.s1_sha = str(s1_path), s1_sha
        self.warm = warm

    @property
    def hist_start(self) -> int:
        return NUM_OBS - self.num_hist * self.num_prop


def priv_cols(train_cfg: dict) -> tuple[int, int, int, int]:
    """``(p0, p1, num_prop, num_hist)`` from the ESTIMATOR / actor cfg (never hard-coded layout numbers)."""
    est = train_cfg["estimator"]
    p0 = int(est["num_prop"]) + int(est["num_scan"])
    return p0, p0 + int(est["num_priv_explicit"]), int(est["num_prop"]), int(train_cfg["policy"]["actor"]["num_hist"])


def build_networks(policy_cls, est_cls, train_cfg: dict, s1_path, device, *, warm_path=None,
                   warm_sha: str | None = SH.STAND_NET_SHA256) -> Nets:
    """S1 + its estimator (strict loads of S1's checkpoint), the student = a copy of S1's network with the WARM START
    loaded strictly (``sh_core.load_warm_start``: A-mask s17, sha checked, ``W[:, 15]`` zeroed), whose estimator must
    equal S1's bitwise; only ``actor.actor_backbone`` trainable. Global RNG saved and restored around the build."""
    warm_path = SH.STAND_NET_CKPT if warm_path is None else warm_path
    before = save_rng()
    try:
        def _new():
            pc = copy.deepcopy(train_cfg["policy"])
            pc.pop("class_name", None)
            return policy_cls(NUM_OBS, NUM_ACT, **pc)

        ec = copy.deepcopy(train_cfg["estimator"])
        ec.pop("class_name", None)
        ck = torch.load(str(s1_path), map_location="cpu", weights_only=False)     # S1's pickle holds cuda:0 storages
        s1 = _new()
        s1.load_state_dict(ck["model_state_dict"], strict=True)
        est = est_cls(**ec)
        est.load_state_dict(ck["estimator_state_dict"], strict=True)
        s1, est = s1.to(device).eval(), est.to(device).eval()
        s1.requires_grad_(False)
        est.requires_grad_(False)
        student = copy.deepcopy(s1)
        warm = SH.load_warm_start(student, warm_path, sha256=warm_sha)
        warm_est = warm.pop("estimator_state_dict")
        est_state = {k: v.detach().cpu().clone() for k, v in ck["estimator_state_dict"].items()}
        ok, why = SN.state_dicts_equal(warm_est, est_state)
        if not ok:
            raise RuntimeError(f"build_networks: the warm start's estimator differs from S1's ({why})")
        student.requires_grad_(False)
        student.actor.actor_backbone.requires_grad_(True)                         # OPEN-1: the backbone only
        student.eval()
    finally:
        restore_rng(before)
    if not rng_equal(before, save_rng()):
        raise RuntimeError("sh_dagger: the global RNG was not restored after building the networks")
    p0, p1, num_prop, num_hist = priv_cols(train_cfg)
    nets = Nets(s1, est, student, p0=p0, p1=p1, num_prop=num_prop, num_hist=num_hist,
                s1_estimator_state=est_state, s1_path=s1_path, s1_sha=sha256_file(s1_path), warm=warm)
    # NOTE(M18 Task 4 step 8, 2026-10-17): the sha the warm start was pinned to -- the A-mask s17 default, or the
    # --warm-sha override (a pipeline SH stage warms from its own run's SN head). The smoke / premise check
    # ``warm_start_zeroed`` compares against THIS, not the constant. A Nets attribute only (never saved), so the
    # default run's records and checkpoints are unchanged.
    nets.warm_expected_sha256 = warm_sha if warm_sha else SH.STAND_NET_SHA256
    return nets


def trainable_params(student) -> list:
    return [p for p in student.parameters() if p.requires_grad]


def make_optimizer(student, lr: float):
    """Adam over ALL student parameters in named order (frozen ones keep ``grad=None``): PPO's single group (T11)."""
    return torch.optim.Adam(student.parameters(), lr=float(lr))


def w15_is_zero(student) -> bool:
    return bool((student.actor.actor_backbone[0].weight[:, SH.HEIGHT_COL] == 0).all())


# ===================================================================================== the forward path (T1)
def s1_actions(obs, nets: Nets):
    """S1 exactly as the harness runs it (its own estimate in the priv columns, a NEW tensor)."""
    x = SN.stand_net_input(obs, nets.est, p0=nets.p0, p1=nets.p1, num_prop=nets.num_prop, mask=False)
    return nets.s1.act_inference(x, hist_encoding=True)


def stand_input(obs, nets: Nets, h_lim, restoring):
    """The stand net's actor input -- ``sh_core.stand_net_input`` (mask 1, column 15 = the limited command, restore
    rows' command columns zeroed), the launcher's call."""
    return SH.stand_net_input(obs, nets.est, p0=nets.p0, p1=nets.p1, num_prop=nets.num_prop, mask=SPEC.STAND_MASK,
                              h_cmd_limited=h_lim, restoring=restoring)


def student_mean(student, x):
    """The D1 forward: the actor mean on the hist path (UNCLIPPED). Execution (through the router) and the MSE."""
    return student.act_inference(x, hist_encoding=True)


def trainer_actions(obs, stand_eff, restoring, h_lim, nets: Nets, *, deadband: float = 0.0):
    """The routed actions of a STUDENT-executed, sigma-0 step (the parity function): ``router_actions(stand_eff,
    student(stand_input), S1, h_lim, deadband=deadband)`` (the run's deadband: student rows execute it)."""
    a_s1 = s1_actions(obs, nets)
    mu = student_mean(nets.student, stand_input(obs, nets, h_lim, restoring)).to(a_s1.dtype)
    return SH.router_actions(stand_eff, mu, a_s1, h_lim, deadband=deadband).actions


def launcher_actions(obs, stand_eff, restoring, h_lim, s1, s1_est, st_pol, st_est, *, p0, p1, num_prop,
                     deadband: float = 0.0):
    """What a stand-height launcher computes for the same obs / controller state: the harness writes S1's estimate in
    place on ITS obs, S1 acts on it, the stand net acts on ``sh_core.stand_net_input(harness obs, its own estimator,
    mask 1, h_lim, restoring)``, ``sh_core.router_actions`` (at the launcher's ``--sh-deadband``)."""
    h = obs.clone()
    h[:, p0:p1] = s1_est.inference(h[:, :num_prop])
    a_s1 = s1.act_inference(h, hist_encoding=True)
    x = SH.stand_net_input(h, st_est, p0=p0, p1=p1, num_prop=num_prop, mask=SPEC.STAND_MASK, h_cmd_limited=h_lim,
                           restoring=restoring)
    a_net = st_pol.act_inference(x, hist_encoding=True).to(a_s1.dtype)
    return SH.router_actions(stand_eff, a_net, a_s1, h_lim, deadband=deadband).actions


# ===================================================================================== labels, onsets, weights (copied)
def lift_onset(branch, prev_branch, prev_valid):
    """``(N,)`` bool: some leg's R_h branch is LIFT on this row and was not LIFT on the env's previous row of the same
    effective stand (``prev_valid``)."""
    lift = branch.long() == BRANCH_LIFT
    was = prev_branch.long() == BRANCH_LIFT
    return prev_valid & (lift & ~was).any(dim=1)


def weight_branch(branch, label, trim_min=None):
    """The per-leg code the inverse-branch-frequency weights read (CHARTER 2026-10-14, ``--trim-branch``). ``trim_min``
    None (the option off): ``branch`` itself, unchanged. Otherwise a leg whose R_h branch is HOLD and whose R_h cam
    label ``|label[:, leg]|`` (residual cam column = R_h's cam command, q's cams are 0) exceeds ``trim_min`` -- the
    in-band trim acting -- is :data:`BRANCH_TRIM`; every other leg keeps R_h's code. Labels, LIFT onsets and the
    executors never read this code: only the weights do."""
    if trim_min is None:
        return branch
    cam = label[:, SH.CAM_ACT].abs() > float(trim_min)
    return torch.where((branch.long() == BRANCH_HOLD) & cam, torch.full_like(branch, BRANCH_TRIM), branch)


def branch_weight_table(branch, w_max: float = SPEC.W_MAX, n_branch: int = N_BRANCH):
    """``(6, n_branch)``: leg l's weight for branch b = 1 / (K_l f_lb) over the K_l branches present on leg l, capped.
    ``n_branch`` 5 = R_h's branches (the default; the table without the trim option, bitwise); 6 = + the trim branch
    (:data:`BRANCH_TRIM`: HOLD's frequency then counts only the non-trim HOLD entries)."""
    b = branch.long()
    n = int(b.shape[0])
    counts = torch.stack([torch.bincount(b[:, leg], minlength=n_branch)[:n_branch] for leg in range(NUM_LEGS)])
    counts = counts.to(torch.float64)
    if n == 0:
        return torch.zeros(NUM_LEGS, n_branch, dtype=torch.float32, device=branch.device)
    f = counts / n
    present = counts > 0
    k = present.sum(dim=1, keepdim=True).clamp_min(1).to(torch.float64)
    w = torch.where(present, 1.0 / (k * f.clamp_min(1e-300)), torch.zeros_like(f))
    return w.clamp(max=float(w_max)).to(torch.float32)


def element_weights(table, branch):
    w_leg = table[torch.arange(NUM_LEGS, device=table.device)[None, :], branch.long()]
    return torch.cat([w_leg, w_leg, w_leg], dim=1)


def dagger_loss(student, x, tgt, w):
    """Weighted MSE between the UNCLIPPED actor mean and the residual label ``R_h - q`` (T8)."""
    mu = student_mean(student, x)
    err = (mu - tgt).pow(2)
    return (w * err).sum() / w.sum().clamp_min(1e-12), err.mean()


# ===================================================================================== the reservoir (copied)
class Reservoir:
    """A capped reservoir (Algorithm R, a PRIVATE generator) of effective stand rows: the EXACT actor input compacted
    to [0, p1) + [hist_start, W) (column 15 = the height command is kept), the residual label (18), the R_h branch (6)
    and the LIFT-onset flag."""

    def __init__(self, cap_rows: int, *, p1: int, hist_start: int, device, gen, width: int = NUM_OBS):
        if not 0 < p1 <= hist_start <= width:
            raise ValueError(f"Reservoir: need 0 < p1 <= hist_start <= width, got {p1}, {hist_start}, {width}")
        self.cap = int(cap_rows)
        if self.cap <= 0:
            raise ValueError("Reservoir: cap must be > 0")
        self.width, self.p1, self.hist_start = int(width), int(p1), int(hist_start)
        self.n_keep = self.p1 + (self.width - self.hist_start)
        self.device, self.gen = device, gen
        self.x = torch.zeros(self.cap, self.n_keep, dtype=torch.float32, device=device)
        self.tgt = torch.zeros(self.cap, NUM_ACT, dtype=torch.float32, device=device)
        self.branch = torch.zeros(self.cap, NUM_LEGS, dtype=torch.int8, device=device)
        self.onset = torch.zeros(self.cap, dtype=torch.bool, device=device)
        self.size = 0
        self.seen = 0

    @staticmethod
    def row_bytes(n_keep: int) -> int:
        return 4 * int(n_keep) + 4 * NUM_ACT + NUM_LEGS + 1

    @classmethod
    def rows_for(cls, gb: float, n_keep: int) -> int:
        return int(float(gb) * 1e9) // cls.row_bytes(n_keep)

    @property
    def nbytes(self) -> int:
        return sum(t.element_size() * t.numel() for t in (self.x, self.tgt, self.branch, self.onset))

    def compact(self, x):
        return torch.cat([x[:, :self.p1], x[:, self.hist_start:]], dim=1)

    def expand_rows(self, xc):
        out = torch.zeros(xc.shape[0], self.width, dtype=xc.dtype, device=xc.device)
        out[:, :self.p1] = xc[:, :self.p1]
        out[:, self.hist_start:] = xc[:, self.p1:]
        return out

    def expand(self, idx):
        return self.expand_rows(self.x[idx])

    def add(self, x, tgt, branch, onset) -> int:
        k = int(x.shape[0])
        if k == 0:
            return 0
        pos = self.seen + torch.arange(k, device=self.device, dtype=torch.int64)
        u = torch.rand(k, generator=self.gen, device=self.device, dtype=torch.float64)
        r = torch.floor(u * (pos + 1).to(torch.float64)).to(torch.int64)
        slot = torch.where(pos < self.cap, pos, r)
        keep = slot < self.cap
        j = torch.nonzero(keep).flatten()
        if j.numel():
            s = slot[j]
            uniq, inv = torch.unique(s, return_inverse=True)
            last = torch.full((uniq.numel(),), -1, dtype=torch.int64, device=self.device)
            last = last.scatter_reduce(0, inv, j, reduce="amax", include_self=True)
            self.x[uniq] = self.compact(x[last]).to(self.x.dtype)
            self.tgt[uniq] = tgt[last].to(self.tgt.dtype)
            self.branch[uniq] = branch[last].to(torch.int8)
            self.onset[uniq] = onset[last].bool()
        self.seen += k
        self.size = min(self.cap, self.seen)
        return int(j.numel())

    def onset_index(self):
        return torch.nonzero(self.onset[:self.size]).flatten()

    def sample(self, batch: int, onset_share: float, gen, on=None):
        n = self.size
        if n == 0:
            raise RuntimeError("Reservoir.sample: empty")
        on = self.onset_index() if on is None else on
        n_on = int(round(float(onset_share) * batch)) if on.numel() else 0
        parts = [torch.randint(0, n, (batch - n_on,), generator=gen, device=self.device)]
        if n_on:
            parts.append(on[torch.randint(0, on.numel(), (n_on,), generator=gen, device=self.device)])
        return torch.cat(parts)


def train_iteration(student, opt, buf: Reservoir, gens: dict, *, minibatches: int, minibatch: int,
                    onset_share: float, w_max: float, max_grad_norm: float, n_branch: int = N_BRANCH) -> dict:
    """Fixed gradient steps per iteration: ``minibatches`` x Adam on ``min(minibatch, size)``-row samples.
    ``n_branch``: the weight categories in the reservoir's branch column (5; 6 with the trim branch)."""
    if buf.size == 0:
        return {"steps": 0}
    table = branch_weight_table(buf.branch[:buf.size], w_max, n_branch)
    params = trainable_params(student)
    b = min(int(minibatch), buf.size)
    on = buf.onset_index()
    acc = {k: torch.zeros((), dtype=torch.float64, device=buf.device) for k in ("loss", "mse", "gn", "onset",
                                                                               "onset_w")}
    for _ in range(int(minibatches)):
        idx = buf.sample(b, onset_share, gens["minibatch"], on)
        x = buf.expand(idx)
        tgt = buf.tgt[idx]
        w = element_weights(table, buf.branch[idx])
        loss, mse = dagger_loss(student, x, tgt, w)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(params, max_norm=float(max_grad_norm))
        opt.step()
        acc["loss"] += loss.detach().double()
        acc["mse"] += mse.detach().double()
        acc["gn"] += gn.detach().double()
        acc["onset"] += buf.onset[idx].double().mean()
        wr = w.double().sum(dim=1)
        acc["onset_w"] += (wr * buf.onset[idx].double()).sum() / wr.sum().clamp_min(1e-300)
    m = int(minibatches)
    out = {k: float(v.item()) / m for k, v in acc.items()}
    out["steps"] = m
    out["batch"] = b
    out["weight_table"] = table.cpu().tolist()
    return out


# ===================================================================================== checkpoints (copied + infos)
REQUIRED_CKPT_KEYS = ("model_state_dict", "estimator_state_dict", "iter", "infos", "optimizer_state_dict")


def _to_cpu(obj):
    if torch.is_tensor(obj):
        return obj.detach().cpu().clone()
    if isinstance(obj, dict):
        return {k: _to_cpu(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_to_cpu(v) for v in obj]
    if isinstance(obj, tuple):
        return tuple(_to_cpu(v) for v in obj)
    return obj


def height_command_record(ctrl=None) -> dict:
    """The height channel a checkpoint was trained with (``infos.sh_dagger.height_command``)."""
    out = {"column": SH.HEIGHT_COL, "encoding": f"(h_cmd_limited - {SH.H_SETTLED}) / {SH.HEIGHT_OBS_SCALE}",
           "h_settled": SH.H_SETTLED, "h_range": [SH.H_MIN, SH.H_MAX], "rate_m_s": SH.HEIGHT_RATE_M_S,
           "restore_zero_cols": list(SH.RESTORE_ZERO_COLS), "stand_mask": SPEC.STAND_MASK,
           "action": "q18(h_lim) + clamp(a_net, +-1); rods +-4.8, cams +-1; S1 rows clamp(S1, +-1)",
           "harness_clip_actions": SH.HARNESS_CLIP}
    if ctrl is not None:
        out["restore"] = ctrl.record()
    return out


def save_checkpoint(path, nets: Nets, opt, it: int, infos: dict) -> str:
    """rsl_rl format, CPU tensors: every S1 key of the student (critic and ``std_raw`` included), S1's estimator
    VERBATIM, ``iter``, ``infos``, ``optimizer_state_dict``. Written atomically."""
    path = Path(path)
    payload = {"model_state_dict": {k: v.detach().cpu().clone() for k, v in nets.student.state_dict().items()},
               "estimator_state_dict": {k: v.detach().cpu().clone() for k, v in nets.s1_estimator_state.items()},
               "iter": int(it), "infos": infos, "optimizer_state_dict": _to_cpu(opt.state_dict())}
    tmp = path.with_name(path.name + ".tmp")
    torch.save(payload, str(tmp))
    os.replace(tmp, path)
    return str(path)


def load_strict(path, policy_cls, est_cls, train_cfg: dict, device="cpu"):
    """The launcher's load path (``torch.load`` with NO map_location, strict model + estimator loads)."""
    ck = torch.load(str(path), weights_only=False)
    missing = [k for k in ("model_state_dict", "estimator_state_dict", "iter", "infos") if k not in ck]
    if missing:
        raise KeyError(f"load_strict: {path} lacks {missing}")
    pc = copy.deepcopy(train_cfg["policy"])
    pc.pop("class_name", None)
    ec = copy.deepcopy(train_cfg["estimator"])
    ec.pop("class_name", None)
    before = save_rng()
    try:
        pol = policy_cls(NUM_OBS, NUM_ACT, **pc)
        est = est_cls(**ec)
    finally:
        restore_rng(before)
    pol.load_state_dict(ck["model_state_dict"], strict=True)
    est.load_state_dict(ck["estimator_state_dict"], strict=True)
    return pol.to(device).eval(), est.to(device).eval(), ck


def checkpoint_reasons(path, nets: Nets, policy_cls, est_cls, train_cfg: dict, device="cpu") -> list[str]:
    """Fail closed: keys, strict load, estimator == S1's, the student's state == the saved one, and the infos carry
    ``sn_dagger.stand_mask`` 1 (an int) + ``sh_dagger.height_command`` with the frozen h range."""
    out = []
    try:
        pol, est, ck = load_strict(path, policy_cls, est_cls, train_cfg, device)
    except Exception as exc:  # noqa: BLE001
        return [f"strict load failed: {type(exc).__name__}: {exc}"]
    missing = [k for k in REQUIRED_CKPT_KEYS if k not in ck]
    if missing:
        out.append(f"missing keys {missing}")
    ok, why = SN.state_dicts_equal(ck["estimator_state_dict"], nets.s1_estimator_state)
    if not ok:
        out.append(f"estimator differs from S1's: {why}")
    if set(ck["model_state_dict"]) != set(nets.s1.state_dict()):
        out.append("model_state_dict keys differ from S1's")
    ok, why = SN.state_dicts_equal(pol.state_dict(), nets.student.state_dict())
    if not ok:
        out.append(f"loaded policy differs from the student: {why}")
    infos = ck.get("infos") or {}
    sm = (infos.get("sn_dagger") or {}).get("stand_mask")
    if type(sm) is not int or sm != SPEC.STAND_MASK:
        out.append(f"infos.sn_dagger.stand_mask {sm!r}: want the int {SPEC.STAND_MASK}")
    hc = (infos.get("sh_dagger") or {}).get("height_command") or {}
    if hc.get("h_range") != [SH.H_MIN, SH.H_MAX] or hc.get("column") != SH.HEIGHT_COL:
        out.append(f"infos.sh_dagger.height_command {hc!r}: want column 15 and h_range [{SH.H_MIN}, {SH.H_MAX}]")
    return out


# ===================================================================================== the stand tracker (OPEN-7 + height)
class StandTracker:
    """Per-env stand bookkeeping (a stand = a maximal run of stand-SLOT rows of one episode; scored on its first
    draw), with the height fields: the stand's command / starting limiter state at entry, h_meas / feet on the first
    draw's last row, and over the first draw's last 3 s the feet >= 4 count, the count of rows whose ROOT has risen
    >= 0.10 m above its spawn root z (``rise`` as the trainer passes it) and the minimum of that rise. The train-time
    STOOD reads the rise like K3 (sh_table_core.classify): root-based, on >= 95 % of the window."""

    def __init__(self, n: int, device, *, dt: float, max_ep: int, r_params=None, exit_window_steps: int | None = None,
                 stood_window_steps: int | None = None):
        self.n, self.device, self.dt, self.max_ep = int(n), device, float(dt), int(max_ep)
        self.p = r_params
        self.xf_steps = int(exit_window_steps if exit_window_steps is not None
                            else round(SPEC.EXIT_FALL_WINDOW_S / self.dt))
        self.w = int(stood_window_steps if stood_window_steps is not None else round(SPEC.STOOD_WINDOW_S / self.dt))
        z = lambda dt=torch.int64: torch.zeros(self.n, dtype=dt, device=device)  # noqa: E731
        f64 = torch.float64
        self.in_stand = z(torch.bool)
        self.entry_ep, self.entry_it, self.sched_end, self.counter = z(), z(), z(), z()
        self.sched_end0 = z()
        self.kind, self.entry_kind, self.n_merges = z(), z(), z()
        self.prev_kind = torch.full((self.n,), -1, dtype=torch.int64, device=device)
        self.sigma = torch.zeros(self.n, dtype=torch.float32, device=device)
        self.last_ep = z()
        self.last_c = torch.zeros(self.n, NUM_LEGS, dtype=torch.float32, device=device)
        self.last_hip = torch.zeros(self.n, NUM_LEGS, dtype=torch.float32, device=device)
        self.c0 = torch.zeros(self.n, NUM_LEGS, dtype=torch.float32, device=device)
        self.hip0 = torch.zeros(self.n, NUM_LEGS, dtype=torch.float32, device=device)
        self.terrain = torch.full((self.n,), -1, dtype=torch.int64, device=device)
        self.xf_left = z()
        # height / start fields
        self.skind = z()
        self.pose = torch.full((self.n,), -1, dtype=torch.int64, device=device)
        self.top = torch.full((self.n,), float("nan"), dtype=f64, device=device)
        self.first_st = z(torch.bool)
        self.h_cmd0 = torch.zeros(self.n, dtype=f64, device=device)
        self.h_init0 = torch.zeros(self.n, dtype=f64, device=device)
        self.hm0 = torch.full((self.n,), float("nan"), dtype=f64, device=device)
        self.feet0 = z()
        self.win_n, self.win_f4, self.win_rise = z(), z(), z()
        self.rise_min = torch.full((self.n,), float("inf"), dtype=f64, device=device)
        self.pending: dict[int, dict] = {}
        self.done: list[dict] = []
        self.counts = {"exits": 0, "falls": 0, "cut": 0}

    def pre(self, stand, obs, time_left, counter, ep_len, it: int, ep_kind, ep_sigma, terrain=None,
            first: bool = False, *, h_meas=None, feet=None, rise=None, h_cmd=None, h_init=None, start_kind=None,
            top=None, pose=None):
        """Entries / merges / last-row fields for the stand-SLOT rows about to act. Returns ``(entering,
        prev_valid)``. ``entry_kind``: 0 spawn (episode step 0), 1 handover, 2 run start."""
        stand = stand.bool()
        prev_valid = self.in_stand & stand
        entering = stand & ~self.in_stand
        counter = counter.to(torch.int64)
        merge = stand & self.in_stand & (counter != self.counter)
        rows = torch.ceil(time_left.to(torch.float64) / self.dt - 1e-3).to(torch.int64).clamp_min(1)
        ep_len = ep_len.to(torch.int64)
        self.sched_end = torch.where(entering | merge, ep_len + rows, self.sched_end)
        self.sched_end0 = torch.where(entering, ep_len + rows, self.sched_end0)
        self.counter = torch.where(stand, counter, self.counter)
        self.entry_ep = torch.where(entering, ep_len, self.entry_ep)
        self.entry_it = torch.where(entering, torch.full_like(self.entry_it, int(it)), self.entry_it)
        ek = torch.full_like(ep_len, 2) if first else (ep_len > 0).to(torch.int64)
        self.entry_kind = torch.where(entering, ek, self.entry_kind)
        self.n_merges = torch.where(entering, torch.zeros_like(self.n_merges), self.n_merges + merge.to(torch.int64))
        self.kind = torch.where(entering, ep_kind.to(torch.int64), self.kind)
        self.sigma = torch.where(entering, ep_sigma.to(torch.float32), self.sigma)
        if terrain is not None:
            self.terrain = torch.where(entering, terrain.to(torch.int64), self.terrain)
        self.in_stand = self.in_stand | entering
        c = obs[:, list(CAM_COLS)].to(torch.float32)
        hip = obs[:, list(HIP_COLS)].to(torch.float32)
        self.last_ep = torch.where(stand, ep_len, self.last_ep)
        self.last_c = torch.where(stand[:, None], c, self.last_c)
        self.last_hip = torch.where(stand[:, None], hip, self.last_hip)
        in_draw0 = stand & (ep_len < self.sched_end0)
        self.c0 = torch.where(in_draw0[:, None], c, self.c0)
        self.hip0 = torch.where(in_draw0[:, None], hip, self.hip0)
        # height / start fields (entry values, then the first draw's last-row / last-3-s reads)
        f64 = torch.float64
        nanv = torch.full_like(self.hm0, float("nan"))
        sk = torch.zeros_like(ep_len) if start_kind is None else start_kind.to(torch.int64)
        self.skind = torch.where(entering, sk, self.skind)
        if pose is not None:
            self.pose = torch.where(entering, pose.to(torch.int64), self.pose)
        if top is not None:
            self.top = torch.where(entering, top.to(f64), self.top)
        self.first_st = torch.where(entering, (ep_len == 0) & (sk != SPEC.SK_WALK) & (not first), self.first_st)
        if h_cmd is not None:
            self.h_cmd0 = torch.where(entering, h_cmd.to(f64), self.h_cmd0)
        if h_init is not None:
            self.h_init0 = torch.where(entering, h_init.to(f64), self.h_init0)
        self.hm0 = torch.where(entering, nanv, self.hm0)
        self.feet0 = torch.where(entering, torch.zeros_like(self.feet0), self.feet0)
        self.win_n = torch.where(entering, torch.zeros_like(self.win_n), self.win_n)
        self.win_f4 = torch.where(entering, torch.zeros_like(self.win_f4), self.win_f4)
        self.win_rise = torch.where(entering, torch.zeros_like(self.win_rise), self.win_rise)
        self.rise_min = torch.where(entering, torch.full_like(self.rise_min, float("inf")), self.rise_min)
        if h_meas is not None:
            self.hm0 = torch.where(in_draw0, h_meas.to(f64), self.hm0)
        if feet is not None:
            fe = feet.to(torch.int64)
            self.feet0 = torch.where(in_draw0, fe, self.feet0)
            win = in_draw0 & (ep_len >= self.sched_end0 - self.w)
            self.win_n = self.win_n + win.to(torch.int64)
            self.win_f4 = self.win_f4 + (win & (fe >= SPEC.STOOD_MIN_FEET)).to(torch.int64)
            if rise is not None:
                r = torch.nan_to_num(rise.to(f64), nan=-math.inf)
                self.rise_min = torch.where(win, torch.minimum(self.rise_min, r), self.rise_min)
                self.win_rise = self.win_rise + (win & (r >= SPEC.UP_RISE_M - 1e-12)).to(torch.int64)
        return entering, prev_valid

    def post(self, dones, fell, stand_next, it: int, timeout=None) -> list[dict]:
        """Ends (exit / fall / timeout-cut), exit-fall windows. Returns the records that became FINAL on this step. A
        done that is neither ``crab_failure`` nor the timeout counts as a FALL (conservative)."""
        dones = dones.bool().reshape(-1)
        fell = fell.bool().reshape(-1) & dones
        timeout = (dones & ~fell) if timeout is None else (timeout.bool().reshape(-1) & dones & ~fell)
        term = dones & ~timeout
        stand_next = stand_next.bool().reshape(-1)
        ended_fall = self.in_stand & term
        ended_cut = self.in_stand & timeout
        ended_exit = self.in_stand & ~dones & ~stand_next
        ended = ended_fall | ended_cut | ended_exit
        open_ = self.xf_left > 0
        xf_fall = open_ & term
        xf_cens = open_ & timeout
        left = torch.where(open_ & ~dones, self.xf_left - 1, torch.where(dones, torch.zeros_like(self.xf_left),
                                                                         self.xf_left))
        xf_clean = open_ & ~dones & (left == 0)
        self.xf_left = left
        event = ended | xf_fall | xf_cens | xf_clean
        final: list[dict] = []
        rows = torch.nonzero(event).flatten()
        if rows.numel():
            final = self._host(rows, ended_fall, ended_cut, ended_exit, xf_fall, xf_cens, xf_clean, it, fell)
        self.prev_kind = torch.where(ended_exit, self.kind, self.prev_kind)
        self.prev_kind = torch.where(dones, torch.full_like(self.prev_kind, -1), self.prev_kind)
        self.xf_left = torch.where(ended_exit, torch.full_like(self.xf_left, self.xf_steps), self.xf_left)
        self.in_stand = self.in_stand & ~ended
        return final

    def _host(self, rows, ended_fall, ended_cut, ended_exit, xf_fall, xf_cens, xf_clean, it, fell):
        idx = rows
        cols = {"crab_failure": fell[idx], "fall": ended_fall[idx], "cut_end": ended_cut[idx], "exit": ended_exit[idx],
                "xf_fall": xf_fall[idx], "xf_cens": xf_cens[idx], "xf_clean": xf_clean[idx],
                "entry_ep": self.entry_ep[idx], "entry_it": self.entry_it[idx], "sched_end": self.sched_end[idx],
                "kind": self.kind[idx], "entry_kind": self.entry_kind[idx], "prev_kind": self.prev_kind[idx],
                "sigma": self.sigma[idx], "last_ep": self.last_ep[idx], "last_c": self.last_c[idx],
                "last_hip": self.last_hip[idx], "terrain": self.terrain[idx], "n_merges": self.n_merges[idx],
                "sched_end0": self.sched_end0[idx], "c0": self.c0[idx], "hip0": self.hip0[idx],
                "skind": self.skind[idx], "pose": self.pose[idx], "top": self.top[idx], "first_st": self.first_st[idx],
                "h_cmd0": self.h_cmd0[idx], "h_init0": self.h_init0[idx], "hm0": self.hm0[idx],
                "feet0": self.feet0[idx], "win_n": self.win_n[idx], "win_f4": self.win_f4[idx],
                "win_rise": self.win_rise[idx], "rise_min": self.rise_min[idx]}
        h = {k: v.detach().cpu().tolist() for k, v in cols.items()}
        envs = idx.detach().cpu().tolist()
        p_down = float(self.p.p_down) if self.p is not None else -0.05
        final = []
        fin = lambda v: None if v is None or not math.isfinite(float(v)) else float(v)  # noqa: E731
        for j, e in enumerate(envs):
            if h["xf_fall"][j] or h["xf_cens"][j] or h["xf_clean"][j]:
                rec = self.pending.pop(e, None)
                if rec is not None:
                    rec["exit_fall"] = bool(h["xf_fall"][j])
                    rec["exit_censored"] = bool(h["xf_cens"][j])
                    if h["xf_fall"][j]:
                        rec["exit_fall_cause"] = "crab_failure" if h["crab_failure"][j] else "termination"
                    final.append(rec)
            if h["fall"][j] or h["cut_end"][j] or h["exit"][j]:
                end = "fall" if h["fall"][j] else ("exit" if h["exit"][j] else "timeout")
                last_ep = int(h["last_ep"][j])
                end_ep = last_ep + 1
                entry_ep = int(h["entry_ep"][j])
                sched_end = int(h["sched_end"][j])
                absc = [abs(float(v)) for v in h["last_c"][j]]
                down = [float(v) >= p_down for v in h["last_hip"][j]]
                lo, hi = SPEC.TRIM_BAND
                homed = all(a <= SPEC.HOMED_EPS for a in absc) and end != "fall"
                pk = int(h["prev_kind"][j])
                sched_end0 = int(h["sched_end0"][j])
                fall0 = end == "fall" and last_ep < sched_end0
                reached0 = not fall0 and (end == "exit" or last_ep >= sched_end0 - 1)
                end0 = "fall" if fall0 else ("reached" if reached0 else end)
                absc0 = [abs(float(v)) for v in h["c0"][j]]
                down0 = [float(v) >= p_down for v in h["hip0"][j]]
                sk = SPEC.START_KINDS[int(h["skind"][j])]
                pz = int(h["pose"][j])
                hc0, hi0, hm0 = float(h["h_cmd0"][j]), float(h["h_init0"][j]), fin(h["hm0"][j])
                dh0 = None if hm0 is None else abs(hm0 - hc0)
                wn, wf, rmin = int(h["win_n"][j]), int(h["win_f4"][j]), fin(h["rise_min"][j])
                wr = int(h["win_rise"][j])
                need = min(self.w, max(1, sched_end0 - entry_ep))
                share_need = math.ceil(SPEC.STOOD_FEET_SHARE * wn - 1e-9)
                stood0 = None
                if sk in ("table", "ground"):
                    stood0 = bool(reached0 and wn >= need and wf >= share_need and wr >= share_need
                                  and dh0 is not None and dh0 <= SPEC.STOOD_DH_M)
                rec = {
                    "mode": "dagger", "it_entry": int(h["entry_it"][j]), "it_end": int(it), "env": int(e),
                    "entry_ep": entry_ep, "last_row_ep": last_ep, "end_ep": end_ep, "sched_end_ep": sched_end,
                    "max_ep": self.max_ep, "sched_len_s": (sched_end - entry_ep) * self.dt,
                    "length_s": (end_ep - entry_ep) * self.dt, "cut": bool(sched_end >= self.max_ep), "end": end,
                    "kind": SPEC.EXECUTORS[int(h["kind"][j])], "sigma": float(h["sigma"][j]), "take_ep": entry_ep,
                    "r_time_s": 0.0, "homed": bool(homed), "max_abs_c": max(absc), "abs_c": absc, "down": down,
                    "n_down": int(sum(down)), "trim_legs": int(sum(1 for a, d in zip(absc, down) if d and lo < a <= hi)),
                    "exit_fall": None, "exit_censored": None, "entry_kind": ENTRY_KINDS[int(h["entry_kind"][j])],
                    "prev_kind": None if pk < 0 else SPEC.EXECUTORS[pk], "fell_before_takeover": False,
                    "stuck_leg": -1, "stuck_sign": 0.0, "take_hip": None, "take_c": None,
                    "terrain_level": int(h["terrain"][j]), "n_merges": int(h["n_merges"][j]),
                    "fall_cause": (("crab_failure" if h["crab_failure"][j] else "termination") if end == "fall"
                                   else None),
                    "sched_end0_ep": sched_end0, "sched_len0_s": (sched_end0 - entry_ep) * self.dt,
                    "cut0": bool(sched_end0 >= self.max_ep), "end0": end0, "fall0": bool(fall0),
                    "homed0": bool(reached0 and all(a <= SPEC.HOMED_EPS for a in absc0)),
                    "abs_c0": absc0, "max_abs_c0": max(absc0), "n_down0": int(sum(down0)),
                    "trim_legs0": int(sum(1 for a, d in zip(absc0, down0) if d and lo < a <= hi)), "r_time0_s": 0.0,
                    # height / start fields
                    "start_kind": sk, "pose": None if pz < 0 else SPEC.POSES[pz], "table_top": fin(h["top"][j]),
                    "first_stand": bool(h["first_st"][j]), "h_cmd0": hc0, "h_init0": hi0, "h_meas_end0": hm0,
                    "abs_dh0": dh0, "feet_end0": int(h["feet0"][j]), "win_n0": wn, "win_feet4_0": wf,
                    "rise_min_win0": rmin, "win_rise0": wr, "stood0": stood0,
                    "ramp0_s": abs(hc0 - hi0) / SH.HEIGHT_RATE_M_S,
                    "dt": self.dt,
                }
                if end == "exit":
                    old = self.pending.pop(e, None)
                    if old is not None:
                        old["exit_fall"] = False
                        old["exit_censored"] = True
                        old["exit_window_superseded"] = True
                        final.append(old)
                    self.pending[e] = rec
                    self.counts["exits"] += 1
                else:
                    self.counts["falls" if end == "fall" else "cut"] += 1
                    final.append(rec)
        self.done.extend(final)
        return final

    def flush(self) -> list[dict]:
        out = []
        for e in sorted(self.pending):
            rec = self.pending.pop(e)
            rec["exit_fall"] = None
            rec["exit_censored"] = True
            out.append(rec)
        self.done.extend(out)
        return out


# ===================================================================================== the run loop
check_trim_min = SPEC.check_trim_min


class RunOpts:
    """Plain options (no argparse objects inside the core)."""

    def __init__(self, *, mode="dagger", arm="mask", seed, iterations, stop_after=None, num_envs=SPEC.NUM_ENVS,
                 steps_per_iter=SPEC.STEPS_PER_ITER, lr=SPEC.LR, minibatches=SPEC.MINIBATCHES,
                 minibatch=SPEC.MINIBATCH, buffer_gb=SPEC.BUFFER_GB, buffer_rows=None, onset_share=SPEC.ONSET_SHARE,
                 w_max=SPEC.W_MAX, max_grad_norm=SPEC.MAX_GRAD_NORM, ckpt_every=SPEC.CKPT_EVERY,
                 control_share=SPEC.CONTROL_SHARE, init_random_ep_len=True, smoke=False, log_every=10,
                 deadband=0.0, trim_branch=False, trim_min=None):
        if mode not in SPEC.MODES:
            raise ValueError(f"mode {mode!r}: this trainer runs {SPEC.MODES} only")
        if arm not in SPEC.ARMS:
            raise ValueError(f"arm {arm!r}: PLAN H4 trains one arm, {tuple(SPEC.ARMS)}")
        self.mode, self.arm, self.seed = mode, arm, int(seed)
        self.stand_mask = SPEC.ARMS[arm]
        self.iterations = int(iterations)
        self.stop_after = int(iterations if stop_after is None else stop_after)
        if not 0 < self.stop_after <= self.iterations:
            raise ValueError(f"stop_after {self.stop_after} must be in (0, iterations={self.iterations}]")
        self.num_envs, self.steps_per_iter = int(num_envs), int(steps_per_iter)
        self.lr, self.minibatches, self.minibatch = float(lr), int(minibatches), int(minibatch)
        self.buffer_gb, self.buffer_rows = float(buffer_gb), (None if buffer_rows is None else int(buffer_rows))
        self.onset_share, self.w_max, self.max_grad_norm = float(onset_share), float(w_max), float(max_grad_norm)
        self.ckpt_every, self.control_share = int(ckpt_every), float(control_share)
        self.init_random_ep_len, self.smoke, self.log_every = bool(init_random_ep_len), bool(smoke), int(log_every)
        #: the residual deadband of the STUDENT-executed stand action (0 = off; labels / R executors never get it)
        self.deadband = SH.check_deadband(deadband)
        #: the trim branch (CHARTER 2026-10-14): off = the weights without the option, bitwise; on = HOLD legs whose
        #: R_h cam label |.| > trim_min are a 6th weight category (``weight_branch``)
        if not isinstance(trim_branch, bool):
            raise ValueError(f"trim_branch must be a bool, got {trim_branch!r}")
        self.trim_branch = trim_branch
        if not trim_branch and trim_min is not None:
            raise ValueError("trim_min is set but trim_branch is off (it would not be applied)")
        self.trim_min = check_trim_min(SPEC.TRIM_LABEL_MIN if trim_min is None else trim_min) if trim_branch else None

    @property
    def n_branch(self) -> int:
        """The weight categories: R_h's 5 branches (+ the trim branch when on)."""
        return N_BRANCH_TRIM if self.trim_branch else N_BRANCH

    def to_dict(self) -> dict:
        return dict(vars(self))


def _jsonl(path, rows):
    if not rows:
        return
    with open(path, "a") as fh:
        for r in rows:
            fh.write(json.dumps(r, sort_keys=True) + "\n")


def _fin(v):
    return None if v is None or not math.isfinite(float(v)) else float(v)


class Trainer:
    """One DAgger run on ANY env with the wrapper interface (``get_observations()``, ``step(a)`` clipping at
    ``clip_actions``, ``num_envs``, ``unwrapped.{command_manager, termination_manager, episode_length_buf,
    max_episode_length, step_dt}``), the env-side ``StartState`` the env was composed with and a ``probe()`` callable
    (``sh_starts.KitProbe`` under Kit). The CPU tests drive it with a fake env and the real networks."""

    def __init__(self, env, nets: Nets, opts: RunOpts, out_dir, *, state, probe, gens=None, train_cfg=None,
                 policy_cls=None, est_cls=None, infos=None, terrain_fn=None, config_clip=1.0, log=print):
        self.env, self.nets, self.o, self.out = env, nets, opts, Path(out_dir)
        self.state, self.probe = state, probe
        self.train_cfg, self.policy_cls, self.est_cls = train_cfg, policy_cls, est_cls
        self.infos, self.terrain_fn, self.log = dict(infos or {}), terrain_fn, log
        self.config_clip = float(config_clip)
        self.out.mkdir(parents=True, exist_ok=True)
        u = env.unwrapped
        self.n = int(env.num_envs)
        self.device = u.episode_length_buf.device
        self.dt, self.max_ep = float(u.step_dt), int(u.max_episode_length)
        self.cmd = u.command_manager.get_term("base_velocity")
        self.gens = make_generators(opts.seed, self.device) if gens is None else gens
        if state.gen_start is not self.gens["starts"] or state.gen_hcmd is not self.gens["hcmd"]:
            raise RuntimeError("Trainer: the StartState must draw from this run's 'starts' / 'hcmd' generators")
        self.r_params = SH.r_params()
        self.ctrl = SH.HeightController(self.n, self.device, dt=self.dt)
        self.tracker = StandTracker(self.n, self.device, dt=self.dt, max_ep=self.max_ep, r_params=self.r_params)
        self.opt = make_optimizer(nets.student, opts.lr)
        n_keep = nets.p1 + (NUM_OBS - nets.hist_start)
        cap = opts.buffer_rows or Reservoir.rows_for(opts.buffer_gb, n_keep)
        self.buf = Reservoir(cap, p1=nets.p1, hist_start=nets.hist_start, device=self.device,
                             gen=self.gens["reservoir"])
        if self.buf.nbytes > opts.buffer_gb * 1e9 + 1:
            raise RuntimeError(f"reservoir {self.buf.nbytes} B exceeds the {opts.buffer_gb} GB cap")
        z = lambda dt=torch.int64, v=0: torch.full((self.n,), v, dtype=dt, device=self.device)  # noqa: E731
        self.ep_kind = z(torch.int64, SPEC.EX_CONTROL)
        self.ep_sigma = z(torch.float32, 0.0)
        self.prev_branch = torch.zeros(self.n, NUM_LEGS, dtype=torch.int64, device=self.device)
        self.prev_eff = z(torch.bool, False)
        self.ep_start_it = z(torch.int64, 0)
        self.ep_run_start = z(torch.bool, True)
        self.ep_h_meas0 = z(torch.float64, float("nan"))
        self.ep_h_init = z(torch.float64, float("nan"))
        self.meas = None
        self.opt_steps = 0
        self.student_forwards = 0
        self.stop_requested = False
        self.paths = {k: self.out / v for k, v in (("log", "train_log.jsonl"), ("stands", "stands.jsonl"),
                                                   ("episodes", "episodes.jsonl"), ("status", "status.json"))}
        self.ckpts: list[str] = []
        self.episodes: list[dict] = []
        self.acc = self._new_acc()
        self.restore_it = torch.zeros(N_EX, N_SG, N_KINDS, N_EV, dtype=torch.float64, device=self.device)
        #: the trim diagnostic's per-iteration histograms (row 0 the executed cam command, row 1 R_h's cam label, both
        #: toward home; SPEC.TRIM_DIAG_*), written to every train_log row and zeroed
        self.trim_it = torch.zeros(2, SPEC.TRIM_DIAG_NB, dtype=torch.float64, device=self.device)
        self.trim_loaded = None
        self.capture = None
        self.restore_capture = None
        self.s1_det = None
        self.first_step = True
        self.mb_onset, self.mb_onset_w = [], []
        self.terrain_errors = 0
        self.terrain_error = None
        self.w15_zero_at_start = w15_is_zero(nets.student)

    # ---------------------------------------------------------------- accumulators (smoke + log)
    def _new_acc(self):
        d = self.device
        z = lambda *s, dt=torch.float64, v=0.0: torch.full(s, v, dtype=dt, device=d)  # noqa: E731
        return {"rows": z(), "eff_rows": z(), "slot_rows": z(), "restore_rows": z(), "rows_by_kind": z(3),
                "eff_rows_by_start": z(N_KINDS), "branch_student": z(NUM_LEGS, self.o.n_branch),
                "dart_n": z(3), "dart_ss_pre": z(3), "dart_ss_post": z(3), "sigma0_maxdev": z(),
                "onset_rows": z(), "onset_rows_student": z(), "lastrow_mismatch": z(),
                "ep_new": z(), "ep_ctrl": z(), "ep_beta_r": z(), "ep_beta_sum": z(), "ep_noncontrol": z(),
                "entries_spawn": z(), "entries_handover": z(), "entries_run_start": z(),
                "label_absmax": z(), "hlim_step_max": z(), "hlim_min": z(v=math.inf), "hlim_max": z(v=-math.inf),
                "init_n": z(N_KINDS), "init_violations": z(), "restore_zero_viol": z(), "col15_viol": z(),
                "restore_rows_reservoir": z(), "restore": z(N_EX, N_SG, N_KINDS, N_EV),
                "s1_applied_absmax": z(), "eff_hip_min": z(v=math.inf), "eff_hip_below_m1_rows": z(),
                "eff_rod_absmax": z(), "eff_cam_absmax": z(), "clip_mismatch": z(), "applied_seen": z(),
                "applied_rows_live": z(), "applied_rows_reset_skipped": z(),
                "est_absmax_off_height": z(max(1, self.nets.p1 - self.nets.p0)),
                "est_absmax_s1_rows": z(max(1, self.nets.p1 - self.nets.p0)), "est_rows_off_height": z(),
                "table_top_err_max": z(), "table_top_n": z(), "spawn_h_err_max": z(N_KINDS), "spawn_h_n": z(N_KINDS),
                "spawn_below_hmin": z(N_KINDS), "db_rows": z(), "db_viol_rows": z(), "db_zero_entries": z(),
                "db_kept_entries": z(), "db_entries": z()}

    # ---------------------------------------------------------------- episode draws (per episode, private)
    def _episode_draws(self, new_ep, it: int):
        u = torch.rand(3, self.n, generator=self.gens["episode"], device=self.device, dtype=torch.float64)
        b = SPEC.beta(it, self.o.iterations)
        ctrl = u[0] < self.o.control_share
        beta_r = ~ctrl & (u[1] < b)
        kind = torch.where(ctrl, torch.full_like(self.ep_kind, SPEC.EX_CONTROL),
                           torch.where(beta_r, torch.full_like(self.ep_kind, SPEC.EX_BETA_R),
                                       torch.full_like(self.ep_kind, SPEC.EX_STUDENT)))
        acc = torch.cumsum(torch.tensor([p for _, p in SPEC.SIGMA_TABLE], dtype=torch.float64, device=self.device), 0)
        sig_idx = torch.searchsorted(acc, u[2], right=True).clamp(max=len(SPEC.SIGMA_TABLE) - 1)
        sig = torch.tensor([s for s, _ in SPEC.SIGMA_TABLE], dtype=torch.float32, device=self.device)[sig_idx]
        sig = torch.where(ctrl, torch.zeros_like(sig), sig)
        self.ep_kind = torch.where(new_ep, kind, self.ep_kind)
        self.ep_sigma = torch.where(new_ep, sig, self.ep_sigma)

    def _terrain(self):
        if self.terrain_fn is None:
            self._terrain_fail("no terrain accessor (terrain_fn is None)")
            return None
        try:
            t = self.terrain_fn()
        except Exception as exc:  # noqa: BLE001
            self._terrain_fail(f"{type(exc).__name__}: {exc}")
            return None
        if t is None:
            self._terrain_fail("the terrain accessor returned None")
            return None
        return t.reshape(-1)

    def _terrain_fail(self, why: str) -> None:
        self.terrain_errors += 1
        if self.terrain_error is None:
            self.terrain_error = why
            self.log(f"[SH] WARNING terrain levels unavailable ({why}); stand records get terrain_level -1")

    def _measure(self) -> dict:
        m = self.probe()
        for k in ("root_z", "lowest_z", "h_meas", "feet_loaded"):
            if k not in m:
                raise KeyError(f"probe() lacks {k!r}")
        return m

    # ---------------------------------------------------------------- one env step
    def _step(self, obs, slot, new_ep, it: int, capture: bool = False):
        u = self.env.unwrapped
        o, nets, st, acc = self.o, self.nets, self.state, self.acc
        self._episode_draws(new_ep, it)
        ep_len = u.episode_length_buf.reshape(-1).clone()
        m = self.meas
        # 1) the height controller: the ONE per-step order (before the actions)
        h_prev = self.ctrl.h_lim.clone()
        hs = self.ctrl.step(slot, st.h_cmd, obs)
        h_lim, eff, rest = hs.h_lim, hs.stand_eff, hs.restoring
        rise = m["root_z"].double() - st.spawn_root_z     # the train-time STOOD's rise: the ROOT (K3's rule)
        entering, _ = self.tracker.pre(slot, obs, self.cmd.time_left.reshape(-1), self.cmd.command_counter.reshape(-1),
                                       ep_len, it, self.ep_kind, self.ep_sigma, self._terrain(),
                                       first=self.first_step, h_meas=m["h_meas"], feet=m["feet_loaded"], rise=rise,
                                       h_cmd=st.h_cmd, h_init=h_prev, start_kind=st.kind, top=st.top, pose=st.pose)
        # 2) S1, R_h, the student (all rows)
        a_s1 = s1_actions(obs, nets)
        if self.s1_det is None:
            self.s1_det = bool(torch.equal(s1_actions(obs, nets), a_s1))
        lab = SH.rh_label(obs, h_lim, check=False)
        label, tgt = lab.label.to(a_s1.dtype), lab.target.to(a_s1.dtype)
        x = stand_input(obs, nets, h_lim, rest)
        mu = student_mean(nets.student, x).to(a_s1.dtype)
        self.student_forwards += 1
        use_r = self.ep_kind != SPEC.EX_STUDENT
        sig = self.ep_sigma.to(a_s1.dtype)
        noise = torch.randn(self.n, NUM_ACT, generator=self.gens["dart"], device=self.device, dtype=a_s1.dtype)
        clean = torch.where(use_r[:, None], label, mu)
        a_res = torch.where((sig > 0)[:, None], clean + sig[:, None] * noise, clean)
        a_st, _q = SH.stand_action(a_res, h_lim)
        if o.deadband > 0.0:
            # the residual deadband on the STUDENT-executed rows only (labels and the R executors unchanged)
            a_db, _ = SH.stand_action(a_res, h_lim, deadband=o.deadband)
            a_st = torch.where(use_r[:, None], a_st, a_db)
        r_exact = use_r & (sig == 0)
        a_stand = torch.where(r_exact[:, None], tgt, a_st)
        actions = SH.route(eff, a_stand, a_s1)
        # 3) the reservoir: every effective stand row (restore rows included), R_h's residual on the visited state
        onset = lift_onset(lab.branch, self.prev_branch, self.prev_eff & eff)
        rows = torch.nonzero(eff).flatten()
        # the weight category (--trim-branch: HOLD legs whose R_h cam label is non-zero -> BRANCH_TRIM; off: R_h's
        # branch itself, the same tensor) -- the onsets above and the labels never read it
        wb = weight_branch(lab.branch, label, o.trim_min)
        self.buf.add(x[rows], label[rows], wb[rows], onset[rows])
        self.prev_branch = lab.branch.long()
        # 4) accumulators
        effd = eff.to(torch.float64)
        kind = self.ep_kind
        acc["rows"] += self.n
        acc["eff_rows"] += effd.sum()
        acc["slot_rows"] += slot.double().sum()
        acc["restore_rows"] += rest.double().sum()
        for k in range(3):
            acc["rows_by_kind"][k] += (eff & (kind == k)).sum()
        acc["eff_rows_by_start"] += torch.bincount(st.kind, weights=effd, minlength=N_KINDS)[:N_KINDS]
        stu = eff & (kind == SPEC.EX_STUDENT)
        acc["onset_rows"] += (onset & eff).sum()
        acc["onset_rows_student"] += (onset & stu).sum()
        oh = torch.nn.functional.one_hot(wb.long(), o.n_branch).to(torch.float64)
        acc["branch_student"] += (oh * stu.to(torch.float64)[:, None, None]).sum(0)
        self._trim_diag(obs, m, lab.branch, label, a_stand, stu & (self.ep_sigma == 0))
        dev_pre = (a_res - clean).double()
        dev_post = (a_res.clamp(-1, 1) - clean.clamp(-1, 1)).double()
        for k, s in enumerate((0.1, 0.3)):
            mk = (eff & (self.ep_sigma == s)).to(torch.float64)[:, None]
            acc["dart_n"][k] += mk.sum() * NUM_ACT
            acc["dart_ss_pre"][k] += (dev_pre.pow(2) * mk).sum()
            acc["dart_ss_post"][k] += (dev_post.pow(2) * mk).sum()
        m0 = eff & (self.ep_sigma == 0)
        acc["sigma0_maxdev"] = torch.maximum(
            acc["sigma0_maxdev"], torch.where(m0[:, None], dev_pre.abs(), torch.zeros_like(dev_pre)).max().double())
        new_rows = new_ep & ~(self.ep_kind == SPEC.EX_CONTROL)
        acc["ep_new"] += new_ep.sum()
        acc["ep_ctrl"] += (new_ep & (self.ep_kind == SPEC.EX_CONTROL)).sum()
        acc["ep_beta_r"] += (new_rows & (self.ep_kind == SPEC.EX_BETA_R)).sum()
        acc["ep_beta_sum"] += new_rows.sum() * SPEC.beta(it, o.iterations)
        acc["ep_noncontrol"] += new_rows.sum()
        lab_abs = torch.where(eff[:, None], label.abs(), torch.zeros_like(label)).max().double()
        acc["label_absmax"] = torch.maximum(acc["label_absmax"], lab_abs)
        acc["hlim_step_max"] = torch.maximum(acc["hlim_step_max"], (h_lim - h_prev).abs().max())
        acc["hlim_min"] = torch.minimum(acc["hlim_min"], h_lim.min())
        acc["hlim_max"] = torch.maximum(acc["hlim_max"], h_lim.max())
        zc = list(SH.RESTORE_ZERO_COLS)
        acc["restore_zero_viol"] += (rest & (x[:, zc] != 0).any(dim=1)).sum().double()
        enc = SH.encode_height(h_lim, dtype=x.dtype, device=x.device)
        acc["col15_viol"] += (x[:, SH.HEIGHT_COL] != enc).sum().double()
        # REPORTED (review 2026-10-13, accepted, not changed): the frozen estimator (S1's, shared) reads the unmasked
        # prop, whose last-action block carries q(h) + residual (beyond +-1) on stand rows at h != 1.064 -- S1 never
        # saw that; the per-column |estimate| range on those rows vs on the S1 rows shows how far it extrapolates
        est = x[:, nets.p0:nets.p1].abs().double()
        off = eff & (h_lim != SH.H_SETTLED)
        acc["est_rows_off_height"] += off.sum().double()
        acc["est_absmax_off_height"] = torch.maximum(acc["est_absmax_off_height"], torch.where(
            off[:, None], est, torch.zeros_like(est)).amax(dim=0))
        acc["est_absmax_s1_rows"] = torch.maximum(acc["est_absmax_s1_rows"], torch.where(
            (~eff)[:, None], est, torch.zeros_like(est)).amax(dim=0))
        acc["restore_rows_reservoir"] += rest[rows].double().sum()  # the restore rows among the rows just offered
        nk = N_EX * N_SG * N_KINDS
        key = ((kind * N_SG + (self.ep_sigma > 0).long()) * N_KINDS + st.kind).long()
        for e, name in enumerate(("engaged", "completed", "timed_out", "merged")):
            ev = getattr(hs, name).to(torch.float64)                 # weighted counts: no host sync
            cnt = torch.bincount(key, weights=ev, minlength=nk)[:nk].reshape(N_EX, N_SG, N_KINDS)
            self.restore_it[..., e] += cnt
            acc["restore"][..., e] += cnt
        if capture and self.capture is None and bool(eff.any()):
            self.capture = {"obs": obs.detach().clone(), "stand_slot": slot.detach().clone(), "stand_eff": eff.clone(),
                            "restoring": rest.clone(), "h_lim": h_lim.clone(), "it": int(it)}
        if self.restore_capture is None and bool(rest.any()):
            rr = torch.nonzero(rest).flatten()
            self.restore_capture = {"obs": obs[rr].detach().clone(), "h_lim": h_lim[rr].clone(),
                                    "label": label[rr].detach().clone(), "x": x[rr].detach().clone(), "it": int(it)}
        pre_c = obs[:, list(CAM_COLS)].to(torch.float32)
        # 5) the env step (the wrapper clips at clip_actions = 4.8)
        obs_next, _, dones, _ = self.env.step(actions)
        m_next = self._measure()
        fell = u.termination_manager.get_term("crab_failure").bool().reshape(-1)
        dones = dones.bool().reshape(-1)
        slot_next = self.cmd.stand_mask().reshape(-1).bool()
        ending = self.tracker.in_stand & (dones | ~slot_next)
        acc["lastrow_mismatch"] += (ending[:, None] & (self.tracker.last_c != pre_c)).any(dim=1).sum()
        app = m_next.get("applied_action")
        if app is not None:
            # the applied action is read AFTER env.step: on a row the env just reset, ActionManager.reset zeroed it
            # (IsaacLab action_manager.py: _action[env_ids] = 0), so only rows that did NOT reset carry this step's
            # applied action -- every applied-action accumulator reads the live rows only
            live = ~dones
            app = app.to(actions.dtype)
            hc = float(self.env.clip_actions)
            acc["applied_seen"] += 1.0
            acc["applied_rows_live"] += live.sum().double()
            acc["applied_rows_reset_skipped"] += dones.sum().double()
            acc["clip_mismatch"] += ((app != actions.clamp(-hc, hc)).any(dim=1) & live).sum().double()
            ne = ~eff & live
            acc["s1_applied_absmax"] = torch.maximum(
                acc["s1_applied_absmax"], torch.where(ne[:, None], app.abs(), torch.zeros_like(app)).max().double())
            el = eff & live
            hip = app[:, SH.HIP_ACT]
            acc["eff_hip_min"] = torch.minimum(acc["eff_hip_min"], torch.where(
                el[:, None], hip, torch.full_like(hip, math.inf)).min().double())
            acc["eff_hip_below_m1_rows"] += (el & (hip < -1.0).any(dim=1)).sum().double()
            rods = app[:, SH.HIP_ACT.start:]
            acc["eff_rod_absmax"] = torch.maximum(acc["eff_rod_absmax"], torch.where(
                el[:, None], rods.abs(), torch.zeros_like(rods)).max().double())
            cams = app[:, SH.CAM_ACT]
            acc["eff_cam_absmax"] = torch.maximum(acc["eff_cam_absmax"], torch.where(
                el[:, None], cams.abs(), torch.zeros_like(cams)).max().double())
            if o.deadband > 0.0:
                # REALISED: the applied hip / knee action minus the posture on live student-executed stand rows must
                # be 0 or >= the deadband (DEADBAND_TOL: float32 rounding of (q + r) - q)
                sl = el & ~use_r
                res = (app[:, SH.DEADBAND_ACT] - _q[:, SH.DEADBAND_ACT].to(app.dtype)).abs()
                inside = (res > DEADBAND_TOL) & (res < o.deadband - DEADBAND_TOL)
                slm = sl[:, None]
                acc["db_rows"] += sl.sum().double()
                acc["db_viol_rows"] += (sl & inside.any(dim=1)).sum().double()
                acc["db_zero_entries"] += (slm & (res <= DEADBAND_TOL)).sum().double()
                acc["db_kept_entries"] += (slm & (res >= o.deadband - DEADBAND_TOL)).sum().double()
                acc["db_entries"] += sl.sum().double() * res.shape[1]
        timeout = dones & (ep_len + 1 >= self.max_ep)
        final = self.tracker.post(dones, fell, slot_next, it, timeout=timeout)
        eps = self._episode_ends(dones, fell, timeout, ep_len, it) if bool(dones.any()) else []
        # 6) resets: the limiter init per start kind (walk 1.064; else clamp(h_meas) on the reset row)
        cut = self.ctrl.reset(dones, st.kind, h_meas=m_next["h_meas"])
        ck = ((kind * N_SG + (self.ep_sigma > 0).long()) * N_KINDS + st.prev["kind"]).long()
        cnt = torch.bincount(ck, weights=cut.to(torch.float64), minlength=nk)[:nk].reshape(N_EX, N_SG, N_KINDS)
        self.restore_it[..., 4] += cnt
        acc["restore"][..., 4] += cnt
        self._reset_checks(dones, m_next, it)
        if self.first_step:
            acc["entries_run_start"] += entering.sum()
            self.first_step = False
        else:
            acc["entries_spawn"] += (entering & (ep_len == 0)).sum()
            acc["entries_handover"] += (entering & (ep_len > 0)).sum()
        self.prev_eff = eff & ~dones
        self.meas = m_next
        return obs_next, slot_next, dones, final, eps

    def _trim_diag(self, obs, m, branch, label, a_stand, rows) -> None:
        """The REPORTED trim diagnostic (CHARTER 2026-10-14; ``SPEC.TRIM_DIAG_*``): on ``rows`` (student-executed
        sigma-0 stand rows) the legs whose R_h branch is HOLD with ``|c|`` in ``SPEC.TRIM_BAND`` and the foot loaded
        (the probe's per-foot ``foot_loaded``, measured on the state ``obs`` describes; a probe without it -> HOLD's
        own "down" only, recorded): histograms of the executed cam command and R_h's cam label, toward home. No host
        sync, no RNG: the training is the same with or without it."""
        c = obs[:, list(CAM_COLS)].to(torch.float64)
        ac = c.abs()
        lo, hi = SPEC.TRIM_BAND
        sel = rows[:, None] & (branch.long() == BRANCH_HOLD) & (ac > lo) & (ac <= hi)
        fl = m.get("foot_loaded") if m is not None else None
        if self.trim_loaded is None:
            self.trim_loaded = ("probe foot_loaded (contact >= FOOT_LOADED_N, footpad <= FOOT_FLOOR_M above the floor)"
                                if fl is not None else "no per-foot flag from the probe: R_h's HOLD (leg down) only")
        if fl is not None:
            sel = sel & fl.to(sel.device).bool().reshape(sel.shape)
        w = sel.reshape(-1).to(torch.float64)
        toward = -torch.sign(c)
        nb = SPEC.TRIM_DIAG_NB
        for j, v in enumerate((a_stand[:, SH.CAM_ACT], label[:, SH.CAM_ACT])):
            u = (toward * v.to(torch.float64)).clamp(-1.0, 1.0)
            k = torch.round((u + 1.0) / SPEC.TRIM_DIAG_BIN).long().clamp(0, nb - 1)
            self.trim_it[j] += torch.bincount(k.reshape(-1), weights=w, minlength=nb)[:nb]

    def _reset_checks(self, dones, m, it: int) -> None:
        """Realised doses on the reset rows: the limiter init rule, the measured spawn height vs the analytic one, the
        realised (PhysX) table top vs its draw; the per-env episode buffers."""
        st, acc = self.state, self.acc
        h = self.ctrl.h_lim
        hm = m["h_meas"].double()
        want = torch.where(st.kind == SPEC.SK_WALK, torch.full_like(hm, SH.H_SETTLED), hm.clamp(SH.H_MIN, SH.H_MAX))
        acc["init_violations"] += (dones & (h != want)).sum().double()
        acc["init_n"] += torch.bincount(st.kind, weights=dones.to(torch.float64), minlength=N_KINDS)[:N_KINDS]
        nw = dones & (st.kind != SPEC.SK_WALK)
        err = (hm - st.h_spawn).abs()
        for code in range(1, N_KINDS):
            mk = nw & (st.kind == code)
            acc["spawn_h_err_max"][code] = torch.maximum(acc["spawn_h_err_max"][code], torch.where(
                mk, err, torch.zeros_like(err)).max())
            acc["spawn_h_n"][code] += mk.sum().double()
            acc["spawn_below_hmin"][code] += (mk & (hm < SH.H_MIN)).sum().double()
        tt = m.get("table_top_w")
        if tt is not None:
            mt = dones & (st.kind == SPEC.SK_TABLE)
            e = (tt.double() - st.top_w).abs()
            acc["table_top_err_max"] = torch.maximum(acc["table_top_err_max"], torch.where(
                mt, torch.nan_to_num(e, nan=math.inf), torch.zeros_like(e)).max())
            acc["table_top_n"] += mt.sum().double()
        self.ep_start_it = torch.where(dones, torch.full_like(self.ep_start_it, int(it)), self.ep_start_it)
        self.ep_run_start = self.ep_run_start & ~dones
        self.ep_h_meas0 = torch.where(dones, hm, self.ep_h_meas0)
        self.ep_h_init = torch.where(dones, h, self.ep_h_init)

    def _episode_ends(self, dones, fell, timeout, ep_len, it: int) -> list[dict]:
        """One record per finished episode (host): the env-side snapshot of the outgoing episode + the trainer's."""
        rows = torch.nonzero(dones).flatten()
        snap = self.state.snapshot_rows(rows)
        cause = self.state.term_cause[rows].tolist()
        fl, to = fell[rows].tolist(), timeout[rows].tolist()
        L = (ep_len[rows] + 1).tolist()
        ex, sg = self.ep_kind[rows].tolist(), self.ep_sigma[rows].tolist()
        rs, hm0, hin = self.ep_run_start[rows].tolist(), self.ep_h_meas0[rows].tolist(), self.ep_h_init[rows].tolist()
        dt = self.dt
        out = []
        for j, e in enumerate(rows.tolist()):
            s = snap[j]
            if not s["valid"]:
                continue
            pz = int(s["pose"])
            end = "fall" if fl[j] else ("timeout" if to[j] else "term")
            ep_s = lambda k: None if int(s[k]) < 0 else int(s[k]) * dt  # noqa: E731
            out.append({"it": int(it), "env": int(e), "kind": SPEC.START_KINDS[int(s["kind"])],
                        "pose": None if pz < 0 else SPEC.POSES[pz], "table_top": _fin(s["top"]),
                        "length_s": int(L[j]) * dt, "end": end,
                        "cause": {0: None, 1: "tilt", 2: "hip"}[int(cause[j])] if fl[j] else None,
                        "run_start": bool(rs[j]), "executor": SPEC.EXECUTORS[int(ex[j])], "sigma": float(sg[j]),
                        "first_len_s": _fin(s["first_len_s"]), "change_s": ep_s("change_ep"), "park_s": ep_s("park_ep"),
                        "rearm_s": ep_s("rearm_ep"), "up_s": ep_s("up_ep"),
                        "clear_at_first_check": None if int(s["clear_first"]) < 0 else bool(int(s["clear_first"])),
                        "h_spawn": _fin(s["h_spawn"]), "spawn_root_z": _fin(s["spawn_root_z"]),
                        "h_meas_spawn": _fin(hm0[j]), "h_init": _fin(hin[j])})
        self.episodes.extend(out)
        return out

    # ---------------------------------------------------------------- the run
    def run(self) -> dict:
        o, u = self.o, self.env.unwrapped
        t_start = time.time()
        if o.init_random_ep_len:
            u.episode_length_buf = torch.randint(0, self.max_ep, (self.n,), generator=self.gens["init_len"],
                                                 device=self.device, dtype=u.episode_length_buf.dtype)
        obs, _ = self.env.get_observations()                      # ONCE (T6); afterwards only env.step's obs
        slot = self.cmd.stand_mask().reshape(-1).bool()
        self.meas = self._measure()
        # the episodes in progress at the start: limiter init per their start kind (walk 1.064; else clamp(h_meas))
        self.ctrl.reset(torch.ones(self.n, dtype=torch.bool, device=self.device), self.state.kind,
                        h_meas=self.meas["h_meas"])
        self.ep_h_meas0 = self.meas["h_meas"].double().clone()
        self.ep_h_init = self.ctrl.h_lim.clone()
        self.compaction_guard(obs)
        new_ep = torch.ones(self.n, dtype=torch.bool, device=self.device)
        status = "running"
        last_it = -1
        for it in range(o.stop_after):
            t0 = time.time()
            n_final, n_eps = [], []
            capture = o.smoke and it == o.stop_after - 1
            with torch.inference_mode():
                for _ in range(o.steps_per_iter):
                    obs, slot, new_ep, final, eps = self._step(obs, slot, new_ep, it, capture=capture)
                    n_final.extend(final)
                    n_eps.extend(eps)
            t1 = time.time()
            tr = train_iteration(self.nets.student, self.opt, self.buf, self.gens, minibatches=o.minibatches,
                                 minibatch=o.minibatch, onset_share=o.onset_share, w_max=o.w_max,
                                 max_grad_norm=o.max_grad_norm, n_branch=o.n_branch)
            self.opt_steps += int(tr.get("steps", 0))
            if tr.get("steps"):
                self.mb_onset.append(tr["onset"])
                self.mb_onset_w.append(tr["onset_w"])
            t2 = time.time()
            _jsonl(self.paths["stands"], n_final)
            _jsonl(self.paths["episodes"], n_eps)
            row = self._log_row(it, n_final, n_eps, tr, t1 - t0, t2 - t1)
            _jsonl(self.paths["log"], [row])
            if o.log_every and (it % o.log_every == 0 or it == o.stop_after - 1):
                self.log(self._log_line(row))
            done_its = it + 1
            last_it = it
            if done_its % o.ckpt_every == 0 or done_its == o.stop_after:
                self.save(done_its)
            if self.stop_requested:
                status = "interrupted"
                break
        else:
            status = "done"
        _jsonl(self.paths["stands"], self.tracker.flush())
        res = {"status": status, "mode": o.mode, "arm": o.arm, "iterations_run": last_it + 1,
               "iterations_schedule": o.iterations, "checkpoints": list(self.ckpts),
               "final_checkpoint": self.ckpts[-1] if self.ckpts else None, "opt_steps": self.opt_steps,
               "student_forwards": self.student_forwards, "wall_s": time.time() - t_start,
               SH.RESIDUAL_DEADBAND_KEY: o.deadband, "trim_branch": o.trim_branch, "trim_min": o.trim_min,
               "trim_diag_loaded": self.trim_loaded,
               "s_per_iter": (time.time() - t_start) / max(1, last_it + 1)}
        self.paths["status"].write_text(json.dumps(res, indent=1, sort_keys=True))
        return res

    def compaction_guard(self, obs) -> None:
        """Fail closed: the student's forward on the reservoir's compacted + re-expanded rows equals the forward on the
        full rows, bitwise (the hist path never reads [p1, hist_start); column 15 is inside [0, p1))."""
        with torch.inference_mode():
            x = stand_input(obs, self.nets, self.ctrl.h_lim, torch.zeros(self.n, dtype=torch.bool,
                                                                         device=self.device))
            a = student_mean(self.nets.student, x)
            b = student_mean(self.nets.student, self.buf.expand_rows(self.buf.compact(x)))
        if not torch.equal(a, b):
            raise RuntimeError("sh_dagger: the reservoir's column compaction changes the student's action; refused")

    def checkpoint_infos(self, done_its: int) -> dict:
        infos = copy.deepcopy(self.infos)
        sn = dict(infos.get("sn_dagger") or {})
        sn.update(stand_mask=int(SPEC.STAND_MASK), s1_sha256=self.nets.s1_sha, arm=self.o.arm, seed=self.o.seed,
                  iterations=self.o.iterations, r_knob=SN.RESTANCE_D3, iterations_done=int(done_its),
                  opt_steps=int(self.opt_steps), buffer_seen=int(self.buf.seen), buffer_size=int(self.buf.size))
        sh = dict(infos.get("sh_dagger") or {})
        sh.update(height_command=height_command_record(self.ctrl), warm_start=self.nets.warm,
                  label="R_h residual: sh_core.rh_label(obs, h_lim).label = R_h - q18(h_lim)",
                  start_shares=dict(zip(SPEC.START_KINDS, SPEC.START_SHARES)),
                  pose_shares=dict(zip(SPEC.POSES, SPEC.POSE_SHARES)), iterations_done=int(done_its))
        sh[SH.RESIDUAL_DEADBAND_KEY] = self.o.deadband      # the student-executed action's deadband (0 = off)
        sh["trim_branch"], sh["trim_min"] = self.o.trim_branch, self.o.trim_min   # the weights' trim category
        infos["sn_dagger"], infos["sh_dagger"] = sn, sh
        return infos

    def save(self, done_its: int) -> str:
        p = save_checkpoint(self.out / f"model_{int(done_its)}.pt", self.nets, self.opt, done_its,
                            self.checkpoint_infos(done_its))
        self.ckpts.append(p)
        return p

    def _log_row(self, it, final, eps, tr, t_roll, t_train) -> dict:
        row = {"it": int(it), "beta": SPEC.beta(it, self.o.iterations), "t_rollout_s": t_roll, "t_train_s": t_train,
               "n_final_stands": len(final), "n_episodes": len(eps), "buffer_size": int(self.buf.size),
               "buffer_seen": int(self.buf.seen), **{f"train_{k}": v for k, v in tr.items() if k != "weight_table"}}
        ri = self.restore_it.cpu().tolist()
        row["restore"] = {ex: {sg: {k: {e: int(ri[a][g][b][c]) for c, e in enumerate(SPEC.RESTORE_EVENTS)}
                                    for b, k in enumerate(SPEC.START_KINDS)} for g, sg in enumerate(SPEC.RESTORE_SIGMA)}
                          for a, ex in enumerate(SPEC.EXECUTORS)}
        self.restore_it.zero_()
        ti = self.trim_it.cpu().tolist()
        row["trim_diag"] = {"n": int(round(sum(ti[0]))),
                            "exec": {str(k): int(round(v)) for k, v in enumerate(ti[0]) if v},
                            "rh": {str(k): int(round(v)) for k, v in enumerate(ti[1]) if v}}
        self.trim_it.zero_()
        for kind in SPEC.START_KINDS:
            for ex in SPEC.EXECUTORS:
                g = [r for r in final if r["kind"] == ex and r.get("start_kind") == kind]
                hold = [r for r in g if SPEC.premise_scored(r)]
                stood = [r for r in g if SPEC.stood_scored(r)]
                sc = SPEC.fall_scored_set(g)
                row[f"{ex}_{kind}_hold"] = len(hold)
                row[f"{ex}_{kind}_homed"] = sum(1 for r in hold if SPEC.homed_scored(r))
                row[f"{ex}_{kind}_stood_n"] = len(stood)
                row[f"{ex}_{kind}_stood"] = sum(1 for r in stood if SPEC.stood_outcome(r))
                row[f"{ex}_{kind}_falls"] = sum(1 for r in sc if SPEC.fall_scored(r))
                row[f"{ex}_{kind}_scored"] = len(sc)
            row[f"episodes_{kind}"] = sum(1 for e in eps if e["kind"] == kind)
            row[f"episode_falls_{kind}"] = sum(1 for e in eps if e["kind"] == kind and e["end"] == "fall")
        row.update({f"count_{k}": v for k, v in self.tracker.counts.items()})
        a = self.acc
        row["eff_share_cum"] = float(a["eff_rows"].item()) / max(1.0, float(a["rows"].item()))
        row["restore_share_cum"] = float(a["restore_rows"].item()) / max(1.0, float(a["rows"].item()))
        row["rows_by_executor_cum"] = dict(zip(SPEC.EXECUTORS, (int(x) for x in a["rows_by_kind"].tolist())))
        return row

    def _log_line(self, row) -> str:
        parts = [f"[SH] it {row['it'] + 1}/{self.o.stop_after} beta {row['beta']:.3f}",
                 f"buf {row['buffer_size']} loss {row.get('train_loss', float('nan')):.5f}"]
        for kind in SPEC.START_KINDS:
            parts.append(f"{kind} homed {row[f'student_{kind}_homed']}/{row[f'student_{kind}_hold']} "
                         f"stood {row[f'student_{kind}_stood']}/{row[f'student_{kind}_stood_n']}")
        parts.append(f"roll {row['t_rollout_s']:.2f}s train {row['t_train_s']:.2f}s")
        return " | ".join(parts)

    def smoke_measures(self) -> dict:
        return {k: (v.detach().cpu().tolist() if torch.is_tensor(v) else v) for k, v in self.acc.items()}


# ===================================================================================== step-5 smoke (PLAN step 5)
def _chk(name, value, line, ok, **extra) -> dict:
    return {"name": name, "value": value, "line": line, "pass": bool(ok), **extra}


def obs_layout_reasons(obs) -> list[str]:
    """D5 on captured RAW obs rows: 1149 wide; the clock / stand-bit columns 15-16 read exactly 0 in the current step
    and in the 10 real history frames (the height command is written on the stand-net clone only)."""
    out = []
    if int(obs.shape[1]) != NUM_OBS:
        return [f"obs width {obs.shape[1]} != {NUM_OBS}"]
    cols = [15, 16] + [259 + 89 * j + k for j in range(10) for k in (15, 16)]
    nz = int((obs[:, cols] != 0).sum())
    if nz:
        out.append(f"{nz} non-zero entries in the clock / stand-bit columns {cols[:2]} (+ history copies)")
    return out


PARITY_GPU_ATOL = 1e-5
#: the realised-deadband smoke check's float slack on (q + r) - q (float32 rounding, ~5e-7 at |q| <= 4.8)
DEADBAND_TOL = 1e-5


def forward_parity(tr: "Trainer", ckpt_path) -> dict:
    """In-process (same device, same batch, same controller state): the trainer's routed action of a student-executed
    sigma-0 step vs :func:`launcher_actions` with the SAVED checkpoint loaded on the launcher's path."""
    cap = tr.capture
    if cap is None:
        return {"equal": False, "why": "no captured stand batch", "pass": False}
    pol, est, _ = load_strict(ckpt_path, tr.policy_cls, tr.est_cls, tr.train_cfg, tr.device)
    n = tr.nets
    db = tr.o.deadband
    with torch.inference_mode():
        a_tr = trainer_actions(cap["obs"], cap["stand_eff"], cap["restoring"], cap["h_lim"], n, deadband=db)
        a_l = launcher_actions(cap["obs"], cap["stand_eff"], cap["restoring"], cap["h_lim"], n.s1, n.est, pol, est,
                               p0=n.p0, p1=n.p1, num_prop=n.num_prop, deadband=db)
    eq = bool(torch.equal(a_tr, a_l))
    diff = float((a_tr - a_l).abs().max().item())
    on_gpu = torch.device(tr.device).type == "cuda"
    out = tr.out / "smoke_capture.pt"
    torch.save({"obs": cap["obs"].detach().cpu(), "stand_slot": cap["stand_slot"].cpu(),
                "stand_eff": cap["stand_eff"].cpu(), "restoring": cap["restoring"].cpu(), "h_lim": cap["h_lim"].cpu(),
                "actions_trainer": a_tr.detach().cpu(), "stand_mask": int(SPEC.STAND_MASK), "ckpt": str(ckpt_path),
                "ckpt_sha256": sha256_file(ckpt_path), "s1": n.s1_path, "s1_sha256": n.s1_sha,
                "device": str(tr.device), "it": cap["it"], "priv": [n.p0, n.p1], "num_prop": n.num_prop,
                SH.RESIDUAL_DEADBAND_KEY: db},
               str(out))
    ok = eq or (on_gpu and diff <= PARITY_GPU_ATOL)
    return {"equal": eq, "max_abs_diff": diff, "device": str(tr.device), "rule": "CPU torch.equal; GPU atol 1e-5",
            "pass": bool(ok), "rows": int(cap["obs"].shape[0]), "stand_rows": int(cap["stand_eff"].sum()),
            "restore_rows": int(cap["restoring"].sum()), "capture": str(out), SH.RESIDUAL_DEADBAND_KEY: db}


def _share_check(counts: list[float], want: tuple, names: tuple) -> tuple[dict, bool]:
    """Every share within max(0.05, 3 SE) of its expected value; a share-0 entry (never drawn: the step-4 decision's
    ground starts / rod_min poses) is asserted ABSENT -- exactly 0 draws, no tolerance."""
    n = float(sum(counts))
    out, ok = {"n": n}, n > 0
    for c, p, name in zip(counts, want, names):
        got = c / n if n else None
        if float(p) == 0.0:
            out[name] = {"k": c, "share": got, "expected": 0.0, "tol": 0.0, "asserted_absent": True}
            ok = ok and float(c) == 0.0
            continue
        tol = SPEC.share_tol(p, int(n)) if n else None
        out[name] = {"k": c, "share": got, "expected": p, "tol": tol}
        ok = ok and got is not None and abs(got - p) <= tol
    return out, ok


def _kind_drawn(kind: str) -> bool:
    """A start kind with share > 0 (``SPEC.DRAWN_START_KINDS``); a share-0 kind is asserted absent by the smoke."""
    return kind in SPEC.DRAWN_START_KINDS


def first_stand_rows(tr: "Trainer", kind: str) -> list[dict]:
    recs = list(tr.tracker.done) + list(tr.tracker.pending.values())
    return [r for r in recs if r.get("first_stand") and r.get("start_kind") == kind and r.get("entry_kind") != "run_start"]


def first_stand_ok(r: dict) -> tuple[bool, float]:
    """The forced first stand's scheduled first draw covers ramp (from the limiter state it started from) + settle +
    3 s (a reset entry holds the draw - dt: one dt of slack)."""
    need = SPEC.first_stand_seconds(r["h_cmd0"], r["h_init0"], margin=False)
    return bool(float(r["sched_len0_s"]) + float(r["dt"]) >= need - 1e-6), float(r["sched_len0_s"]) - need


def trim_branch_check(tr: "Trainer") -> tuple[dict, bool]:
    """The ``trim_branch_weights`` smoke check (CHARTER 2026-10-14): with ``--trim-branch``, on the reservoir as the
    last iteration trained on it -- the weight table (``branch_weight_table`` over the 6 categories, cap W_MAX), the
    leg-entry counts and shares per category, and the mean weight over the trim entries vs over the non-trim HOLD
    entries. PASS: trim entries present, non-trim HOLD entries present, trim's mean weight > HOLD's, the table has the
    trim column. Off: "not applied" (pass)."""
    o = tr.o
    if not o.trim_branch:
        return {"trim_branch": False, "note": "trim branch off: not applied (the weights over R_h's 5 branches)"}, True
    b = tr.buf.branch[:tr.buf.size].long()
    table = branch_weight_table(b, o.w_max, o.n_branch)
    w = table[torch.arange(NUM_LEGS, device=table.device)[None, :], b].double()
    n = int(b.numel())
    counts = {name: int((b == k).sum()) for k, name in enumerate(BRANCH_NAMES_TRIM)}

    def mean_w(code):
        m = b == code
        k = int(m.sum())
        return (float((w * m.double()).sum()) / k) if k else None

    wt, wh = mean_w(BRANCH_TRIM), mean_w(BRANCH_HOLD)
    tab = table.cpu().tolist()
    v = {"trim_branch": True, "trim_min": o.trim_min, "w_max": o.w_max, "reservoir_rows": int(tr.buf.size),
         "leg_entries": n, "entries": counts,
         "shares": {k: (c / n if n else None) for k, c in counts.items()},
         "mean_weight_trim": wt, "mean_weight_hold_non_trim": wh,
         "weight_table": {f"leg{leg}": dict(zip(BRANCH_NAMES_TRIM, tab[leg])) for leg in range(NUM_LEGS)},
         "weight_table_columns": list(BRANCH_NAMES_TRIM)}
    ok = (counts["trim"] > 0 and counts["hold"] > 0 and wt is not None and wh is not None and wt > wh
          and len(tab[0]) == N_BRANCH_TRIM)
    return v, ok


def smoke_checks(tr: "Trainer", *, env_parity: list[str], ckpt: str, env_clip: float) -> list[dict]:
    """PLAN step 5's dose list (``SPEC.SMOKE_DAGGER``) on the run ``tr`` just finished."""
    a = tr.smoke_measures()
    sc = tr.state.counts()
    o = tr.o
    checks = []
    # -- start modes, poses, tops, spawns
    ep = [sc[f"episodes_{k}"] for k in SPEC.START_KINDS]
    v, ok = _share_check(ep, SPEC.START_SHARES, SPEC.START_KINDS)
    checks.append(_chk("start_mode_shares", v, f"every share within max(0.05, 3 SE) of {SPEC.START_SHARES} "
                       "(walk / table / ground / random); a share-0 kind (ground, step-4 decision) asserted ABSENT: "
                       "0 draws", ok))
    pz = [x + y for x, y in zip(sc["poses_table"], sc["poses_ground"])]
    v, ok = _share_check(pz, SPEC.POSE_SHARES, SPEC.POSES)
    v["fold_fallbacks"] = sc["fold_fallbacks"]
    checks.append(_chk("retracted_pose_shares", v, f"table + ground pose shares within max(0.05, 3 SE) of "
                       f"{SPEC.POSE_SHARES} (knee_folded / rod_min / random fold); a share-0 pose (rod_min, step-4 "
                       "decision) asserted ABSENT: 0 draws (a random fold no cell allows falls back to knee_folded, "
                       "counted)", ok))
    n_top = sc["top_n"]
    mean = sc["top_sum"] / n_top if n_top else None
    mid = 0.5 * (SPEC.TOP_MIN_M + SPEC.TOP_MAX_M)
    sd = (SPEC.TOP_MAX_M - SPEC.TOP_MIN_M) / math.sqrt(12.0)
    tol = max(0.02, 4 * sd / math.sqrt(n_top)) if n_top else None
    v = {"n": n_top, "min": sc["top_min"], "max": sc["top_max"], "mean": mean, "expected_mean": mid, "tol": tol,
         "realised_top_err_max_m": a["table_top_err_max"], "realised_n": a["table_top_n"]}
    ok = (n_top > 0 and sc["top_min"] >= SPEC.TOP_MIN_M - 1e-9 and sc["top_max"] <= SPEC.TOP_MAX_M + 1e-9
          and abs(mean - mid) <= tol and a["table_top_n"] > 0 and a["table_top_err_max"] <= 0.01)
    checks.append(_chk("table_tops", v, "tops in [top_min, top_max], mean within max(0.02, 4 SE) of the midpoint; "
                       "realised (PhysX) top within 1 cm of the draw on every table reset", ok))
    sk = {SPEC.START_KINDS[c]: {"n": a["spawn_h_n"][c], "h_meas_vs_analytic_err_max_m": a["spawn_h_err_max"][c],
                                "below_h_min": a["spawn_below_hmin"][c]} for c in range(1, N_KINDS)}
    absent = [k for k in sk if not _kind_drawn(k)]
    v = {"by_kind": sk, "floor_fallbacks": sc["floor_fallbacks"], "asserted_absent": absent}
    ok = (a["spawn_h_n"][SPEC.SK_TABLE] > 0 and all(sk[k]["n"] == 0 for k in absent)
          and all(x["n"] == 0 or x["h_meas_vs_analytic_err_max_m"] <= 0.03 for x in sk.values())
          and sc["floor_fallbacks"] == 0)
    checks.append(_chk("ground_spawns", v, "table (and random) resets: measured h_meas within 3 cm of the analytic "
                       "spawn height, >= 1 table spawn measured; every floor from the raycast; a share-0 kind (ground, "
                       "step-4 decision) asserted ABSENT: 0 spawns", ok))
    # -- commands, limiter, labels
    nf, na = sc["hcmd_free_n"], sc["hcmd_free_atom"]
    nn = nf - na
    m_ = sc["hcmd_free_sum"] / nn if nn else None
    var = (sc["hcmd_free_sumsq"] / nn - m_ * m_) if nn else None
    mid_h = 0.5 * (SH.H_MIN + SH.H_MAX)
    sd_h = (SH.H_MAX - SH.H_MIN) / math.sqrt(12.0)
    tol_m = max(0.02, 4 * sd_h / math.sqrt(nn)) if nn else None
    atom_p = na / nf if nf else None
    v = {"free_draws": nf, "atom_share": atom_p, "atom_expected": SPEC.H_CMD_ATOM_P,
         "atom_tol": SPEC.share_tol(SPEC.H_CMD_ATOM_P, int(nf)) if nf else None, "uniform_mean": m_,
         "uniform_sd": None if var is None else math.sqrt(max(var, 0.0)), "expected_mean": mid_h, "mean_tol": tol_m,
         "uniform_min": sc["hcmd_free_min"], "uniform_max": sc["hcmd_free_max"], "table_draws": sc["hcmd_table_n"],
         "table_atom": sc["hcmd_table_atom"], "table_margin_min_m": sc["hcmd_table_margin_min"],
         "table_violations": sc["hcmd_table_violations"], "ground_first_draws": sc["hcmd_ground_first_n"],
         "ground_first_bounded": sc["hcmd_ground_first_bounded"],
         "ground_first_violations": sc["hcmd_ground_first_violations"],
         "never_up": {k: {"ended": sc[f"ended_{k}"], "never_up": sc[f"never_up_{k}"]} for k in ("table", "ground")}}
    ok = (nf > 0 and nn > 0 and abs(atom_p - SPEC.H_CMD_ATOM_P) <= v["atom_tol"] and abs(m_ - mid_h) <= tol_m
          and sc["hcmd_free_min"] >= SH.H_MIN - 1e-12 and sc["hcmd_free_max"] <= SH.H_MAX + 1e-12
          and sc["hcmd_table_n"] > 0 and sc["hcmd_table_violations"] == 0
          and sc["hcmd_ground_first_violations"] == 0)
    checks.append(_chk("h_cmd_distribution", v, "free draws: 1.064 share within max(0.05, 3 SE) of 0.35, the rest in "
                       "[h_min, h_max] with mean within max(0.02, 4 SE) of the midpoint; table draws >= top + 0.264; "
                       "ground first draws >= max(h_min, h_spawn + 0.14625) (bounded ones counted apart); never-up "
                       "episodes reported", ok))
    lim_step = SH.HEIGHT_RATE_M_S * tr.dt
    v = {"step_max_m": a["hlim_step_max"], "step_line_m": lim_step, "h_lim_min": a["hlim_min"],
         "h_lim_max": a["hlim_max"], "init_by_kind": dict(zip(SPEC.START_KINDS, a["init_n"])),
         "init_violations": a["init_violations"], "controller": tr.ctrl.record()}
    ok = (a["hlim_step_max"] <= lim_step + 1e-12 and a["hlim_min"] >= SH.H_MIN - 1e-12
          and a["hlim_max"] <= SH.H_MAX + 1e-12 and a["init_violations"] == 0 and a["init_n"][SPEC.SK_WALK] > 0
          and sum(a["init_n"][1:]) > 0)
    checks.append(_chk("limiter", v, "|dh_lim| <= 0.08 dt per step, h_lim in [h_min, h_max], init = 1.064 on walk "
                       "resets and clamp(h_meas) on table / ground / random resets", ok))
    checks.append(_chk("residual_labels_within_1", {"label_absmax": a["label_absmax"], "tol": SH.LABEL_TOL},
                       "max |R_h - q| over effective stand rows <= 1 (+ 1e-5)",
                       0.0 < a["label_absmax"] <= 1.0 + SH.LABEL_TOL))
    # -- restores (relative to the R_h control slice)
    rs = a["restore"]
    rt = {ex: {sg: {k: {e: rs[i][g][b][j] for j, e in enumerate(SPEC.RESTORE_EVENTS)}
                    for b, k in enumerate(SPEC.START_KINDS)} for g, sg in enumerate(SPEC.RESTORE_SIGMA)}
          for i, ex in enumerate(SPEC.EXECUTORS)}
    pooled = SPEC.restore_pooled(rt)
    pooled_all = SPEC.restore_pooled(rt, None)
    shares = {ex: SPEC.timeout_share(pooled[ex]) for ex in SPEC.EXECUTORS}
    ctl = shares["control"]
    lines, over = {}, {}
    for ex in ("student", "beta_r"):
        s = shares[ex]
        if not s["n"]:
            lines[ex] = over[ex] = None
            continue
        if ctl["n"]:
            se = math.sqrt(s["p"] * (1 - s["p"]) / s["n"] + ctl["p"] * (1 - ctl["p"]) / ctl["n"])
            line = ctl["p"] + max(0.01, 2 * se)
        else:
            line = 0.0
        lines[ex] = line
        over[ex] = bool(s["p"] > line + 1e-12)
    # GATED (mechanics, R_h-executed slices only): the R_h control slice's sigma-0 restores complete in the sim, and
    # the beta-R sigma-0 slice (the SAME executor: R_h's exact target) times out no more than the control slice + the
    # line. The STUDENT's share is REPORTED against the same line: at the smoke the student is the A-mask warm start
    # (W[:, 15] = 0, ~100 iterations of mostly-R episodes), so its restores are a policy outcome, not a dose; the
    # student-vs-R_h restore-timeout comparison is the step-6 premise (SPEC.premise_read: > 2 x R_h's + 0.01 = STOP).
    # absolute bound on the control slice itself (re-review): a wrong rod tolerance in PhysX would make R_h's own
    # restores time out and hand S1 off-posture rows while the relative line still passes
    ctl_abs_ok = (ctl["p"] <= SPEC.RESTORE_CTL_TIMEOUT_MAX_SHARE if ctl["n"] >= 20
                  else pooled["control"]["timed_out"] <= 1)
    ok = (pooled["control"]["completed"] > 0 and sum(pooled_all[ex]["completed"] for ex in SPEC.EXECUTORS) > 0
          and over["beta_r"] is not True and ctl_abs_ok)
    checks.append(_chk("restores_complete", {"pooled_sigma0": pooled, "timeout_share_sigma0": shares,
                                             "lines": lines, "over_line": over,
                                             "student_reported_not_gated": {"share": shares["student"],
                                                                            "line": lines["student"],
                                                                            "over_line": over["student"]},
                                             "pooled_all_sigma": pooled_all,
                                             "timeout_share_all_sigma": {ex: SPEC.timeout_share(pooled_all[ex])
                                                                         for ex in SPEC.EXECUTORS}, "by_kind": rt},
                       "the R_h control slice's sigma-0 restores completed > 0 (timeouts counted); the beta-R sigma-0 "
                       "slice (same executor) times out <= the control slice's share + max(0.01, 2 SE of the "
                       "difference) (0 when the slice has no ended restore); the student's share is reported against "
                       "the same line, gated at the step-6 premise; the control slice's own timeout share <= 0.05 (<= 1 "
                       "timed-out restore when it has < 20 ended restores)", ok))
    # -- forced first stands
    t_rows = first_stand_rows(tr, "table")
    bad_len = [r for r in t_rows if not first_stand_ok(r)[0]]
    bad_cmd = [r for r in t_rows if r.get("table_top") is None
               or r["h_cmd0"] < r["table_top"] + SPEC.TABLE_CMD_MARGIN_M - 1e-9]
    v = {"first_stands": len(t_rows), "too_short": len(bad_len), "h_cmd_below_top_plus_0p264": len(bad_cmd),
         "min_slack_s": min((first_stand_ok(r)[1] for r in t_rows), default=None),
         "forced_table": sc["forced_table"], "episodes_table": sc["episodes_table"]}
    checks.append(_chk("table_first_stand", v, "every table first stand: scheduled >= ramp + 4 s settle + 3 s and h_cmd "
                       ">= top + 0.264; one forced stand per table episode", bool(t_rows) and not bad_len
                       and not bad_cmd and sc["forced_table"] == sc["episodes_table"]))
    gr = {k: first_stand_rows(tr, k) for k in ("ground", "random")}
    v = {k: {"first_stands": len(rs_), "too_short": sum(1 for r in rs_ if not first_stand_ok(r)[0]),
             "min_slack_s": min((first_stand_ok(r)[1] for r in rs_), default=None),
             "forced": sc[f"forced_{k}"], "episodes": sc[f"episodes_{k}"]} for k, rs_ in gr.items()}
    v["forced_walk"] = sc["forced_walk"]
    v["asserted_absent"] = [k for k in gr if not _kind_drawn(k)]
    ok = (all((v[k]["first_stands"] > 0 and v[k]["too_short"] == 0 and v[k]["forced"] == v[k]["episodes"])
              if _kind_drawn(k) else (v[k]["first_stands"] == 0 and v[k]["forced"] == 0 and v[k]["episodes"] == 0)
              for k in gr) and sc["forced_walk"] == 0)
    checks.append(_chk("forced_first_stand_ground_random", v, "every ground / random-height first stand scheduled >= "
                       "ramp + settle + 3 s, one forced stand per episode, none on walk episodes; a share-0 kind "
                       "(ground, step-4 decision) asserted ABSENT: 0 episodes, 0 forced / first stands", ok))
    v = {"parks": sc["parks"], "first_checks": sc["first_checks"], "clear_at_first_check": sc["first_checks_clear"],
         "parks_before_change": sc["parks_before_change"], "parks_not_clear": sc["parks_not_clear"],
         "clear_first_not_parked": sc["clear_first_not_parked"], "delay_steps_min": sc["park_delay_min"],
         "delay_steps_max": sc["park_delay_max"],
         "delay_steps_mean": sc["park_delay_sum"] / sc["parks"] if sc["parks"] else None}
    checks.append(_chk("table_parked_at_slot_change", v, "tables parked > 0, never before the first command change, "
                       "never with the belly not clear, always at the first check after the change when clear",
                       sc["parks"] > 0 and sc["parks_before_change"] == 0 and sc["parks_not_clear"] == 0
                       and sc["clear_first_not_parked"] == 0))
    v = {k: sc[k] for k in ("push_events", "push_applied", "push_suppressed", "push_restore_mismatch", "ups")}
    checks.append(_chk("pushes_off_until_up", v, "pushes fired, some suppressed on envs not up, every suppressed env's "
                       "velocity restored exactly, pushes applied on envs up",
                       sc["push_events"] > 0 and sc["push_suppressed"] > 0 and sc["push_applied"] > 0
                       and sc["push_restore_mismatch"] == 0))
    g_eps = [e for e in tr.episodes if e["kind"] == "ground" and not e["run_start"]]
    early = [e for e in g_eps if e["first_len_s"] is not None and e["length_s"] < e["first_len_s"] - 1e-9]
    early_bad = [e for e in early if e["end"] == "timeout" or e["cause"] == "hip"]
    hip_before = [e for e in g_eps if e["cause"] == "hip" and e["rearm_s"] is None]
    g_first = gr["ground"]
    v = {"hip_term_while_disarmed": sc["hip_term_disarmed"], "hip_would_fire_disarmed": sc["hip_would_fire_disarmed"],
         "rearms": sc["rearms"], "ground_episodes_ended": len(g_eps), "ended_before_first_stand": len(early),
         "ended_before_first_stand_by_a_fall": sum(1 for e in early if e["end"] != "timeout" and e["cause"] != "hip"),
         "ended_before_first_stand_by_hip_or_timeout": len(early_bad), "hip_ends_before_rearm": len(hip_before),
         "ground_first_stands_reached": sum(1 for r in g_first if r["end0"] == "reached"),
         "episodes_ground": sc["episodes_ground"], "ground_first_stands": len(g_first),
         "ground_asserted_absent": not _kind_drawn("ground")}
    if _kind_drawn("ground"):
        ok = (sc["hip_term_disarmed"] == 0 and not early_bad and not hip_before
              and v["ground_first_stands_reached"] > 0)
    else:   # share 0 (step-4 decision): no ground episode at all, so no disarmed hip clause either
        ok = (sc["hip_term_disarmed"] == 0 and sc["episodes_ground"] == 0 and not g_eps and not g_first
              and sc["rearms"] == 0)
    checks.append(_chk("ground_episodes_hip_rearm", v, "0 hip-contact terminations before re-arm (wrapper output and "
                       "episode records); a ground episode ends before its first stand only by a fall (never the hip clause / a "
                       "time-out); >= 1 "
                       "ground first stand reached its end; with the ground share 0 (step-4 decision): ground ASSERTED "
                       "ABSENT (0 ground episodes / records / first stands, 0 re-arms, 0 disarmed hip terminations)",
                       ok))
    # -- restore rows on the stand-net path
    v = {"restore_rows": a["restore_rows"], "zeroed_col_violations": a["restore_zero_viol"],
         "col15_violations": a["col15_viol"],
         "estimate_range_reported": {
             "rule": "per-column max |estimate| (the frozen S1 estimator on the unmasked prop) on effective stand rows "
                     "at h_lim != 1.064 vs on S1 rows; reported, not gated (the last-action block carries q + residual "
                     "beyond +-1 there)",
             "rows_off_height": a["est_rows_off_height"], "absmax_off_height": a["est_absmax_off_height"],
             "absmax_s1_rows": a["est_absmax_s1_rows"],
             "max_ratio": (max(o / s_ for o, s_ in zip(a["est_absmax_off_height"], a["est_absmax_s1_rows"])
                               if s_ > 0) if any(s_ > 0 for s_ in a["est_absmax_s1_rows"]) else None)}}
    checks.append(_chk("restore_rows_stand_net_inputs", v, "restore rows > 0; their input cols 5, 8, 9, 10 == 0; col 15 "
                       "== (h_lim - 1.064) / 0.4 on every row", a["restore_rows"] > 0 and a["restore_zero_viol"] == 0
                       and a["col15_viol"] == 0))
    rc = tr.restore_capture
    lab_ok, why = False, "no restore row captured"
    if rc is not None:
        with torch.inference_mode():
            want = SH.rh_label(rc["obs"], rc["h_lim"], check=False).label.to(rc["label"].dtype)
        lab_ok = bool(torch.equal(want, rc["label"]))
        why = "bitwise" if lab_ok else f"max diff {float((want - rc['label']).abs().max())}"
    v = {"restore_rows_added": a["restore_rows_reservoir"], "buffer_seen": tr.buf.seen, "capture_rows":
         None if rc is None else int(rc["obs"].shape[0]), "labels_vs_rh_label": why}
    checks.append(_chk("restore_rows_in_reservoir_with_rh_labels", v, "restore rows enter the reservoir; their labels "
                       "== sh_core.rh_label(obs, h_lim).label bitwise", a["restore_rows_reservoir"] > 0 and lab_ok))
    # -- clip semantics
    checks.append(_chk("s1_rows_pm1", {"applied_absmax_non_stand_rows": a["s1_applied_absmax"],
                                       "applied_steps": a["applied_seen"]},
                       "every applied action on a non-stand row (rows that did not reset) within +-1",
                       a["applied_seen"] > 0
                       and a["s1_applied_absmax"] <= 1.0))
    checks.append(_chk("stand_hip_targets_below_minus1", {"min_hip": a["eff_hip_min"], "rows_below_m1":
                                                          a["eff_hip_below_m1_rows"], "eff_rows": a["eff_rows"]},
                       "applied hip action < -1 on some effective stand rows", a["eff_hip_below_m1_rows"] > 0
                       and a["eff_hip_min"] < -1.0))
    v = {"env_clip_actions": env_clip, "config_clip_actions": tr.config_clip, "clip_mismatch_rows": a["clip_mismatch"],
         "applied_rows_live": a["applied_rows_live"], "applied_rows_reset_skipped": a["applied_rows_reset_skipped"],
         "stand_rod_absmax": a["eff_rod_absmax"], "stand_cam_absmax": a["eff_cam_absmax"],
         "record": SH.clip_record(o.deadband)}
    checks.append(_chk("harness_clip_4p8", v, "wrapper clip 4.8 at runtime with the config's 1.0 untouched; applied == "
                       "clamp(routed, +-4.8) on every row that did not reset this step (a reset row's applied action "
                       "is zeroed by ActionManager.reset); stand rods beyond +-1 occur, stand cams within +-1",
                       float(env_clip) == SH.HARNESS_CLIP and tr.config_clip == 1.0 and a["clip_mismatch"] == 0
                       and a["applied_rows_live"] > 0 and a["eff_rod_absmax"] > 1.0 and a["eff_cam_absmax"] <= 1.0))
    par = forward_parity(tr, ckpt)
    checks.append(_chk("forward_parity", par, "same device, same batch: CPU torch.equal, GPU atol 1e-5 (both at the "
                       "run's residual deadband)", par.get("pass") is True))
    if o.deadband > 0.0:
        v = {SH.RESIDUAL_DEADBAND_KEY: o.deadband, "student_rows_live": a["db_rows"],
             "rows_with_residual_inside_0_d": a["db_viol_rows"], "hip_knee_entries": a["db_entries"],
             "entries_zero": a["db_zero_entries"], "entries_kept_ge_d": a["db_kept_entries"],
             "tol": DEADBAND_TOL, "student_rows_executor": SPEC.EXECUTORS[SPEC.EX_STUDENT]}
        ok = a["db_rows"] > 0 and a["db_viol_rows"] == 0 and a["db_zero_entries"] > 0
    else:
        v = {SH.RESIDUAL_DEADBAND_KEY: 0.0, "note": "deadband 0: not applied (the action path without the option)"}
        ok = True
    checks.append(_chk("student_deadband_realised", v, "deadband d > 0: on live student-executed stand rows the "
                       "applied hip / knee action minus q(h_lim) is 0 or >= d (never inside (0, d); 1e-5 float "
                       "slack), some entries zeroed; d = 0: not applied", ok))
    v, ok = trim_branch_check(tr)
    checks.append(_chk("trim_branch_weights", v, "--trim-branch: trim entries (HOLD legs with a non-zero R_h cam "
                       "label) in the reservoir, their mean weight > the non-trim HOLD entries' mean weight (both "
                       "present), the weight table reports a trim column; off: not applied", ok))
    w = tr.nets.warm or {}
    checks.append(_chk("warm_start_zeroed", {"w15_zero_at_start": tr.w15_zero_at_start, "record": w},
                       "student W[:, 15] == 0 at the start; the warm start's sha is the pinned one (A-mask s17, or "
                       "the --warm-sha override)",
                       tr.w15_zero_at_start
                       and w.get("sha256") == getattr(tr.nets, "warm_expected_sha256", SH.STAND_NET_SHA256)
                       and float(w.get("w15_norm_before") or 0) > 0))
    # -- the predecessor's generic checks
    steps = max(1.0, a["rows"])
    ent = {"handover": a["entries_handover"], "spawn": a["entries_spawn"], "run_start": a["entries_run_start"],
           "exits": tr.tracker.counts["exits"], "falls_in_stand": tr.tracker.counts["falls"],
           "per_1000_env_steps": {"handover": 1000 * a["entries_handover"] / steps,
                                  "exits": 1000 * tr.tracker.counts["exits"] / steps},
           "eff_share": a["eff_rows"] / steps, "slot_share": a["slot_rows"] / steps,
           "restore_share": a["restore_rows"] / steps}
    checks.append(_chk("handovers", ent, "walk->stand handovers > 0 and stand exits > 0",
                       a["entries_handover"] > 0 and tr.tracker.counts["exits"] > 0))
    bs = a["branch_student"]
    tot = sum(sum(r) for r in bs)
    names_b = BRANCH_NAMES_TRIM if o.trim_branch else BRANCH_NAMES
    shares_b = ({name: sum(bs[l][k] for l in range(NUM_LEGS)) / tot for k, name in enumerate(names_b)}
                if tot else None)
    checks.append(_chk("branch_shares_student_rows", shares_b, "student-executed stand rows exist (shares reported; "
                       "with --trim-branch 'hold' is the non-trim HOLD and 'trim' its own share)",
                       tot > 0, student_rows=a["rows_by_kind"][SPEC.EX_STUDENT],
                       rows_by_executor=dict(zip(SPEC.EXECUTORS, a["rows_by_kind"])), trim_branch=o.trim_branch))
    mb = (sum(tr.mb_onset) / len(tr.mb_onset)) if tr.mb_onset else None
    mbw = (sum(tr.mb_onset_w) / len(tr.mb_onset_w)) if tr.mb_onset_w else None
    stu_rows = a["rows_by_kind"][SPEC.EX_STUDENT]
    dose = {"onset_rows_student": a["onset_rows_student"], "student_stand_rows": stu_rows,
            "pre_oversampling_share_student_rows": (a["onset_rows_student"] / stu_rows) if stu_rows else None,
            "pre_oversampling_share_all_stand_rows": (a["onset_rows"] / a["eff_rows"]) if a["eff_rows"] else None,
            "minibatch_row_share_reported": mb, "loss_weight_share_after_balancing": mbw,
            "buffer_onset_share": (float(tr.buf.onset[:tr.buf.size].float().mean().item()) if tr.buf.size else None)}
    checks.append(_chk("lift_onset_after_balancing", dose, "LIFT-onset rows on student-visited stand rows > 0 and >= "
                       "0.01 of the minibatch loss weight after balancing",
                       a["onset_rows_student"] > 0 and mbw is not None and mbw >= 0.01))
    it_n = o.iterations
    grid = [SPEC.beta(i, it_n) for i in range(it_n)]
    fn_ok = (SPEC.beta(0, it_n) == 1.0 and abs(SPEC.beta(it_n // 6, it_n) - (1 - (it_n // 6) / (it_n / 3))) < 1e-12
             and SPEC.beta(-(-it_n // 3), it_n) == 0.0 and SPEC.beta(it_n - 1, it_n) == 0.0
             and all(x >= y for x, y in zip(grid, grid[1:])))
    real = (a["ep_beta_r"] / a["ep_noncontrol"]) if a["ep_noncontrol"] else None
    expect = (a["ep_beta_sum"] / a["ep_noncontrol"]) if a["ep_noncontrol"] else None
    se = math.sqrt(expect * (1 - expect) / a["ep_noncontrol"]) if a["ep_noncontrol"] and expect is not None else None
    ctrl = (a["ep_ctrl"] / a["ep_new"]) if a["ep_new"] else None
    real_ok = real is not None and abs(real - expect) <= max(0.02, 4 * (se or 0))
    checks.append(_chk("beta_schedule", {"beta(0)": SPEC.beta(0, it_n), "beta(T/6)": SPEC.beta(it_n // 6, it_n),
                                         "beta(T/3)": SPEC.beta(-(-it_n // 3), it_n), "realized_beta_r_share": real,
                                         "expected_beta_r_share": expect, "control_share": ctrl},
                       "beta = 1 -> 0 linearly over the first third, then 0 (function); realized R share of non-control "
                       "episode draws within max(0.02, 4 SE) of the mean beta at draw", fn_ok and real_ok))
    dart, ok = {}, True
    for k, s in enumerate((0.1, 0.3)):
        n_ = a["dart_n"][k]
        pre = math.sqrt(a["dart_ss_pre"][k] / n_) if n_ else None
        post = math.sqrt(a["dart_ss_post"][k] / n_) if n_ else None
        dart[str(s)] = {"n_entries": n_, "sigma_pre_clip": pre, "sigma_post_clip": post}
        ok = ok and pre is not None and abs(pre - s) <= 0.1 * s
    dart["0.0"] = {"max_abs_deviation": a["sigma0_maxdev"]}
    checks.append(_chk("dart_sigma_realized", dart, "realized pre-clip sigma (on the residual) within 10 % of 0.1 / "
                       "0.3; sigma-0 rows exactly clean", ok and a["sigma0_maxdev"] == 0.0))
    checks.append(_chk("obs_parity_env", env_parity, "no obs-affecting difference from the REFS eval stamp",
                       not env_parity))
    lay = obs_layout_reasons(tr.capture["obs"]) if tr.capture is not None else ["no captured stand batch"]
    checks.append(_chk("obs_parity_cols", lay, "1149 wide; raw clock / stand-bit cols 15-16 (+ history copies) == 0",
                       not lay))
    checks.append(_chk("s1_deterministic", tr.s1_det, "S1 act_inference twice on one obs -> equal", tr.s1_det is True))
    per_iter = tr.buf.seen / max(1, o.stop_after)
    proj = per_iter * o.iterations
    checks.append(_chk("buffer_projection", {"allocated_bytes": tr.buf.nbytes, "cap_bytes": int(o.buffer_gb * 1e9),
                                             "cap_rows": tr.buf.cap, "rows_per_iter": per_iter,
                                             "projected_rows_seen": proj,
                                             "projected_fill": min(1.0, proj / tr.buf.cap),
                                             "cuda_max_allocated": (torch.cuda.max_memory_allocated()
                                                                    if torch.cuda.is_available() else None)},
                       "the pre-allocated reservoir <= the cap", tr.buf.nbytes <= o.buffer_gb * 1e9))
    rs_ = checkpoint_reasons(ckpt, tr.nets, tr.policy_cls, tr.est_cls, tr.train_cfg, tr.device)
    checks.append(_chk("ckpt_loads", {"ckpt": ckpt, "reasons": rs_}, "strict load, all keys, estimator == S1's, infos "
                       "stand_mask 1 + the height channel", not rs_))
    recs = list(tr.tracker.done) + list(tr.tracker.pending.values())
    missing = sum(1 for r in recs if int(r.get("terrain_level", -1)) < 0)
    checks.append(_chk("terrain_level_recorded", {"records": len(recs), "terrain_level_missing": missing,
                                                  "accessor_failures": tr.terrain_errors,
                                                  "first_error": tr.terrain_error},
                       "every stand record has terrain_level >= 0 and the accessor never failed",
                       len(recs) > 0 and missing == 0 and tr.terrain_errors == 0))
    return checks
