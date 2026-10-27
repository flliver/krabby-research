#!/usr/bin/env python3
"""Stand-network campaign, PLAN.md step 4 -- the DAgger trainer's TORCH core (modes ``dagger`` and ``recover_probe``).

LOADED POST-KIT ONLY: ``sn_dagger.py`` loads this file by path after ``AppLauncher`` (it imports torch at its top, as
the harness does after Kit). The CPU tests load it directly. Everything that decides WHAT is trained lives here, so
the tests can pin it without Isaac:

* the D1 forward path (PLAN D1 "one rule", map risk T1): the student's input is ``sn_common.stand_net_input(obs,
  S1's estimator, p0, p1, num_prop, mask)`` -- the SAME call the ``stand_net`` launcher makes -- and both the action it
  executes and the output its MSE fits are ``student.act_inference(x, hist_encoding=True)`` (the UNCLIPPED mean, map
  T8: the env wrapper clips the executed action to +-1 exactly as in deployment, T7). S1 runs
  ``act_inference(stand_net_input(obs, est, mask=False), hist_encoding=True)`` == the harness's in-place estimate +
  policy call. Routing: ``sn_common.route_actions(cmd.stand_mask(), a_stand, a_s1)`` (both nets on every row, T12).
* labels: R with the D3 constants on the VISITED state (``sn_common.r_label``, never ``"default"`` -- T4); DART noise
  only on executed stand rows, labels always clean.
* RNG discipline (T5): every trainer draw (episode executor / sigma, DART noise, reservoir, minibatches, the random
  initial episode lengths of OPEN-6, A1 assignment / window / leg / sign, chatter noise) comes from a PRIVATE
  ``torch.Generator`` seeded from the run seed and the draw's name, so no trainer draw ever touches the global
  streams (the env's command sampler / pushes / terrain) and a same-seed run replays bitwise. Most draws are full
  ``(N,)`` vectors per step (a fixed count); the RESERVOIR stream is the exception: ``Reservoir.add`` draws one
  uniform per stand row added, so its position depends on the stand mix (harmless -- its own generator, still a
  deterministic function of the run). Network construction runs inside a global-RNG save / restore (``rng_equal``
  asserted).
* networks (T6: no PPO runner, no extra ``get_observations``): S1 + its estimator frozen; the student a deepcopy of S1
  with ONLY ``actor.actor_backbone`` trainable (CHARTER OPEN-1); Adam (lr 3e-4) over ALL student parameters in named
  order (frozen ones never get a grad), so ``optimizer_state_dict`` loads into PPO's single-group Adam (T11).
* the capped reservoir (OPEN-2, <= 4 GB on the device): it stores the EXACT actor input, compacted to the columns the
  hist-path actor reads ([0, p1) and the last num_hist * num_prop), plus R's target / branch and a LIFT-onset flag;
  per-leg inverse-branch-frequency weights on each leg's three action columns + LIFT-onset oversampling (OPEN-3).
* the stand tracker (OPEN-7): a stand = a maximal run of stand-slot rows of one episode; scheduled end = entry +
  ``ceil(time_left / dt)`` (+-1 step), re-extended at every stand->stand resample (``command_counter`` change);
  ``|c|`` read on the LAST stand row (the row the stand acted on before the exit; never a post-reset row, T9); exits /
  falls / timeouts; the exit-fall window (100 steps, p2_read's 2 s); one record per stand (``stands.jsonl``).
  SCORING IS ON THE FIRST DRAW (review 2026-10-12, an amendment to OPEN-7 for CHARTER section 3): the merge
  extension happens only to stands that SURVIVED their first draw, so a length floor on the merged length includes a
  short-first-draw stand only if it did not fall -- outcome-dependent inclusion that drops early spoiler falls. Each
  record therefore also carries the FIRST draw's schedule (``sched_end0_ep`` = entry + the entry draw's rows, never
  re-extended; ``sched_len0_s``, ``cut0``) and outcome: ``fall0`` (a fall before ``sched_end0``), ``end0``
  (``fall`` / ``reached`` / the actual end otherwise), ``|c|`` / hip on the first draw's last stand row
  (``abs_c0``, ``homed0``, ``n_down0``, ``trim_legs0``) and ``r_time0_s``; the readers score those (the merged fields
  stay, reported only).
* A1 ``recover_probe`` (PLAN Amendment 1 + its audit + CHARTER OPEN-7): beta = 1 (R) on every stand, no student
  forward, no optimizer, sigma 0; per stand (private generator) control 1/2, statue / chatter / stuck lift 1/6 each;
  the spoiler acts for U[1, 4] s (50..200 steps) after entry, then R; statue = 0; chatter = R + 0.3 N(0, 1) on all 18
  channels; stuck lift = R with one random leg's hip at R's lift, its knee at -rho * side * hip and its cam by R's
  saturated in-air P-law shifted to +-0.4 rad.
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


#: the ONE shared stand-net input module (also loaded by the stand_net launcher) and the torch-free spec
SN = _load(HERE / "sn_common.py", "_sn_dagger_common")
SPEC = _load(HERE / "sn_dagger_spec.py", "_sn_dagger_core_spec")

NUM_OBS, NUM_ACT, NUM_LEGS = 1149, 18, 6
CAM_COLS = tuple(18 + 2 * i for i in range(NUM_LEGS))
HIP_COLS = tuple(29 + i for i in range(NUM_LEGS))
SIDE = (1.0, -1.0, 1.0, -1.0, 1.0, -1.0)
BRANCH_LIFT = 2
N_BRANCH = 5
#: action column -> leg (cam i, hip 6 + i, knee 12 + i)
LEG_OF_COL = tuple(list(range(NUM_LEGS)) * 3)
GEN_NAMES = ("episode", "dart", "reservoir", "minibatch", "init_len", "a1", "chatter")
ENTRY_KINDS = ("spawn", "handover", "run_start")


def sha256_file(path) -> str:
    return SN.sha256_file(path)


# ===================================================================================== RNG (T5)
def make_generators(seed: int, device) -> dict:
    """Private generators, one per draw kind, seeded from (run seed, name) -- independent of each other and of the
    global streams the env's command sampler / pushes / terrain draws use."""
    out = {}
    for name in GEN_NAMES:
        h = int.from_bytes(hashlib.sha256(f"sn_dagger:{int(seed)}:{name}".encode()).digest()[:8], "little")
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


# ===================================================================================== networks (T6, OPEN-1)
class Nets:
    """S1 (frozen), S1's estimator (frozen; the stand net's estimator too, D1) and the student."""

    def __init__(self, s1, est, student, *, p0, p1, num_prop, num_hist, s1_estimator_state, s1_path, s1_sha):
        self.s1, self.est, self.student = s1, est, student
        self.p0, self.p1, self.num_prop, self.num_hist = int(p0), int(p1), int(num_prop), int(num_hist)
        self.s1_estimator_state = s1_estimator_state
        self.s1_path, self.s1_sha = str(s1_path), s1_sha

    @property
    def hist_start(self) -> int:
        return NUM_OBS - self.num_hist * self.num_prop


def priv_cols(train_cfg: dict) -> tuple[int, int, int, int]:
    """``(p0, p1, num_prop, num_hist)`` from the ESTIMATOR / actor cfg (never hard-coded layout numbers)."""
    est = train_cfg["estimator"]
    p0 = int(est["num_prop"]) + int(est["num_scan"])
    return p0, p0 + int(est["num_priv_explicit"]), int(est["num_prop"]), int(train_cfg["policy"]["actor"]["num_hist"])


def build_networks(policy_cls, est_cls, train_cfg: dict, s1_path, device) -> Nets:
    """Fresh deep copies of the cfg for every build (the builders pop ``class_name`` / ``channel_size``); strict
    loads of S1's ``model_state_dict`` / ``estimator_state_dict``; the global RNG saved and restored around the build
    (``nn.Linear`` init draws it, the env's command sampler uses it)."""
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
        student.requires_grad_(False)
        student.actor.actor_backbone.requires_grad_(True)                         # OPEN-1: the backbone only
        student.eval()
        est_state = {k: v.detach().cpu().clone() for k, v in ck["estimator_state_dict"].items()}
    finally:
        restore_rng(before)
    if not rng_equal(before, save_rng()):
        raise RuntimeError("sn_dagger: the global RNG was not restored after building the networks")
    p0, p1, num_prop, num_hist = priv_cols(train_cfg)
    return Nets(s1, est, student, p0=p0, p1=p1, num_prop=num_prop, num_hist=num_hist,
                s1_estimator_state=est_state, s1_path=s1_path, s1_sha=sha256_file(s1_path))


def trainable_params(student) -> list:
    return [p for p in student.parameters() if p.requires_grad]


def make_optimizer(student, lr: float):
    """Adam over ALL student parameters in named order (frozen ones keep ``grad=None`` and are skipped), so the saved
    ``optimizer_state_dict`` has PPO's single param group of the same length (T11)."""
    return torch.optim.Adam(student.parameters(), lr=float(lr))


# ===================================================================================== the D1 forward path (T1)
def s1_actions(obs, nets: Nets):
    """S1 exactly as the harness runs it: its own estimate in the priv columns (a NEW tensor, bitwise the harness's
    in-place write), ``act_inference(..., hist_encoding=True)``."""
    x = SN.stand_net_input(obs, nets.est, p0=nets.p0, p1=nets.p1, num_prop=nets.num_prop, mask=False)
    return nets.s1.act_inference(x, hist_encoding=True)


def stand_input(obs, nets: Nets, mask: int):
    """The stand net's actor input -- the launcher's call, verbatim (``sn_common.stand_net_input``)."""
    return SN.stand_net_input(obs, nets.est, p0=nets.p0, p1=nets.p1, num_prop=nets.num_prop, mask=bool(mask))


def student_mean(student, x):
    """The D1 forward: the actor mean on the hist path (UNCLIPPED). Execution and the MSE both use this."""
    return student.act_inference(x, hist_encoding=True)


def trainer_actions(obs, stand, nets: Nets, mask: int):
    """The routed actions of a STUDENT-executed, sigma-0 step (the parity function the tests compare with the
    launcher's ``StandNetMode``): ``route_actions(stand, student(stand_input), S1)``."""
    a_s1 = s1_actions(obs, nets)
    a_st = student_mean(nets.student, stand_input(obs, nets, mask))
    return SN.route_actions(stand, a_st, a_s1)


def launcher_actions(obs, stand, s1, s1_est, st_pol, st_est, *, p0, p1, num_prop, mask: int):
    """What the stand_net launcher computes for the same obs (``StandNetMode.__call__``): the harness writes S1's
    estimate in place on ITS obs, S1 acts on it, the stand net acts on ``stand_net_input(harness obs, its own
    estimator, mask)``. Used by the in-process smoke parity and the offline replay; pinned against the real
    ``StandNetMode`` by the tests."""
    h = obs.clone()
    h[:, p0:p1] = s1_est.inference(h[:, :num_prop])
    a_s1 = s1.act_inference(h, hist_encoding=True)
    x = SN.stand_net_input(h, st_est, p0=p0, p1=p1, num_prop=num_prop, mask=bool(mask))
    a_st = st_pol.act_inference(x, hist_encoding=True).to(a_s1.dtype)
    return SN.route_actions(stand, a_st, a_s1)


# ===================================================================================== A1 spoilers
def spoiler_actions(kind, tgt, obs, leg, sign, chatter_noise, p):
    """Per-row spoiler action (PLAN A1 + audit fixes). ``kind`` (N,) long in ``SPEC.AS_*``; ``tgt`` R's target on
    the row; ``leg`` (N,) long; ``sign`` (N,) +-1; ``chatter_noise`` (N, 18) N(0, 1); ``p`` R's D3 params.

    statue: 0 (18). chatter: R + 0.3 * noise. stuck lift (leg l, sign s): R, except hip l = lift, knee l =
    -rho * side_l * lift (R's knee coupling) and cam l = clamp(-k_c * (c_l - s * 0.4), -1, 1). control: R."""
    c = obs[:, list(CAM_COLS)]
    onehot = torch.nn.functional.one_hot(leg.long(), NUM_LEGS).bool()
    side = torch.tensor(SIDE, device=tgt.device, dtype=tgt.dtype)[None, :].expand_as(c)
    cam_law = (-(p.k_c * (c - sign.to(c.dtype)[:, None] * SPEC.STUCK_CAM_TARGET))).clamp(-1.0, 1.0)
    hip = torch.full_like(c, float(p.lift))
    knee = -float(p.rho) * side * float(p.lift)
    stuck = torch.cat([torch.where(onehot, cam_law.to(tgt.dtype), tgt[:, 0:6]),
                       torch.where(onehot, hip, tgt[:, 6:12]),
                       torch.where(onehot, knee, tgt[:, 12:18])], dim=1)
    chatter = tgt + SPEC.CHATTER_SIGMA * chatter_noise.to(tgt.dtype)
    k = kind.long()[:, None]
    out = torch.where(k == SPEC.AS_STATUE, torch.zeros_like(tgt), tgt)
    out = torch.where(k == SPEC.AS_CHATTER, chatter, out)
    out = torch.where(k == SPEC.AS_STUCK, stuck, out)
    return out


# ===================================================================================== labels, onsets, weights
def lift_onset(branch, prev_branch, prev_valid):
    """``(N,)`` bool: some leg's R branch is LIFT on this row and was not LIFT on the env's PREVIOUS row of the SAME
    stand (``prev_valid``: the env was in this stand on its previous row)."""
    lift = branch.long() == BRANCH_LIFT
    was = prev_branch.long() == BRANCH_LIFT
    return prev_valid & (lift & ~was).any(dim=1)


def branch_weight_table(branch, w_max: float = SPEC.W_MAX):
    """``(6, 5)``: leg l's weight for R branch b = 1 / (K_l * f_lb) over the K_l branches present on leg l in
    ``branch`` (n, 6) -- each present branch gets equal total weight and ``sum_b f_lb w_lb = 1`` before the cap
    ``w_max`` (OPEN-3); 0 for an absent branch (never sampled)."""
    b = branch.long()
    n = int(b.shape[0])
    counts = torch.stack([torch.bincount(b[:, leg], minlength=N_BRANCH)[:N_BRANCH] for leg in range(NUM_LEGS)])
    counts = counts.to(torch.float64)
    if n == 0:
        return torch.zeros(NUM_LEGS, N_BRANCH, dtype=torch.float32, device=branch.device)
    f = counts / n
    present = counts > 0
    k = present.sum(dim=1, keepdim=True).clamp_min(1).to(torch.float64)
    w = torch.where(present, 1.0 / (k * f.clamp_min(1e-300)), torch.zeros_like(f))
    return w.clamp(max=float(w_max)).to(torch.float32)


def element_weights(table, branch):
    """``(B, 18)``: each action column carries its leg's weight for that row's branch (cam i, hip 6+i, knee 12+i)."""
    w_leg = table[torch.arange(NUM_LEGS, device=table.device)[None, :], branch.long()]
    return torch.cat([w_leg, w_leg, w_leg], dim=1)


def dagger_loss(student, x, tgt, w):
    """Weighted MSE between the UNCLIPPED actor mean on the D1 path and R's target (map section 3.5 / T8)."""
    mu = student_mean(student, x)
    err = (mu - tgt).pow(2)
    return (w * err).sum() / w.sum().clamp_min(1e-12), err.mean()


# ===================================================================================== the reservoir (OPEN-2)
class Reservoir:
    """A capped reservoir (Algorithm R, a PRIVATE generator, float64 uniforms) of stand rows on the device.

    Stores the EXACT ``act_inference`` input compacted to ``keep`` = [0, p1) + [hist_start, W) (the hist-path actor
    never reads [p1, hist_start): ACE:97-120), R's target (18) and branch (6, int8) and the LIFT-onset flag. Within one
    add, two accepted rows aimed at the same slot resolve deterministically (the later stream position wins)."""

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
        """Compacted rows -> full-width actor inputs (zeros in the never-read [p1, hist_start))."""
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
        """``batch`` indices with replacement: ``round(onset_share * batch)`` drawn from the LIFT-onset rows (when any
        exist), the rest uniform over the reservoir (OPEN-3 oversampling). ``on``: :meth:`onset_index`, if known."""
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
                    onset_share: float, w_max: float, max_grad_norm: float) -> dict:
    """Fixed gradient steps per iteration (OPEN-2): ``minibatches`` x Adam steps on ``min(minibatch, size)``-row
    samples; branch weights recomputed from the whole reservoir once per iteration."""
    if buf.size == 0:
        return {"steps": 0}
    table = branch_weight_table(buf.branch[:buf.size], w_max)
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
        acc["onset"] += buf.onset[idx].double().mean()                                  # rows (oversampling only)
        wr = w.double().sum(dim=1)
        acc["onset_w"] += (wr * buf.onset[idx].double()).sum() / wr.sum().clamp_min(1e-300)   # + per-leg weights
    m = int(minibatches)
    out = {k: float(v.item()) / m for k, v in acc.items()}
    out["steps"] = m
    out["batch"] = b
    out["weight_table"] = table.cpu().tolist()
    return out


# ===================================================================================== checkpoints (map section 3.7, T11)
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


def save_checkpoint(path, nets: Nets, opt, it: int, infos: dict) -> str:
    """rsl_rl format, CPU tensors (loads with or without CUDA; ``runner.load`` passes no map_location): every S1 key
    of the student (critic and ``std_raw`` included, no prefixes), S1's estimator VERBATIM (the launcher's estimator
    identity refuses anything else), ``iter``, ``infos`` (provenance), ``optimizer_state_dict``. ``alg_state`` is
    omitted (rebuilt from ``iter``). Written atomically."""
    path = Path(path)
    payload = {"model_state_dict": {k: v.detach().cpu().clone() for k, v in nets.student.state_dict().items()},
               "estimator_state_dict": {k: v.detach().cpu().clone() for k, v in nets.s1_estimator_state.items()},
               "iter": int(it), "infos": infos, "optimizer_state_dict": _to_cpu(opt.state_dict())}
    tmp = path.with_name(path.name + ".tmp")
    torch.save(payload, str(tmp))
    os.replace(tmp, path)
    return str(path)


def load_strict(path, policy_cls, est_cls, train_cfg: dict, device="cpu"):
    """The launcher's load path (``runner.load``: ``torch.load`` with NO map_location, strict model + estimator
    loads) into fresh networks built from a deepcopy of ``train_cfg``. Returns ``(policy, estimator, ckpt dict)``."""
    ck = torch.load(str(path), weights_only=False)
    missing = [k for k in ("model_state_dict", "estimator_state_dict", "iter", "infos") if k not in ck]
    if missing:
        raise KeyError(f"load_strict: {path} lacks {missing} (OPR:761-804 reads all four)")
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
    """Fail-closed checks on a saved checkpoint: the keys (incl. ``optimizer_state_dict``), a strict load, the
    estimator equal to S1's (key / shape / dtype / value), the student's state equal to the saved one."""
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
    ok, why = SN.state_dicts_equal(est.state_dict(), nets.est.state_dict())
    if not ok:
        out.append(f"loaded estimator differs from S1's in-memory estimator: {why}")
    if set(ck["model_state_dict"]) != set(nets.s1.state_dict()):
        out.append("model_state_dict keys differ from S1's")
    ok, why = SN.state_dicts_equal(pol.state_dict(), nets.student.state_dict())
    if not ok:
        out.append(f"loaded policy differs from the student: {why}")
    return out


# ===================================================================================== the stand tracker (OPEN-7)
class StandTracker:
    """Per-env stand bookkeeping on the device; host records only when a stand ends or an exit-fall window closes."""

    def __init__(self, n: int, device, *, dt: float, max_ep: int, mode: str, gen=None, r_params=None,
                 exit_window_steps: int | None = None):
        if mode not in SPEC.MODES:
            raise ValueError(mode)
        self.n, self.device, self.dt, self.max_ep, self.mode = int(n), device, float(dt), int(max_ep), mode
        self.gen = gen
        self.p = r_params
        self.xf_steps = int(exit_window_steps if exit_window_steps is not None
                            else round(SPEC.EXIT_FALL_WINDOW_S / self.dt))
        z = lambda dt=torch.int64: torch.zeros(self.n, dtype=dt, device=device)  # noqa: E731
        self.in_stand = z(torch.bool)
        self.entry_ep, self.entry_it, self.sched_end, self.counter = z(), z(), z(), z()
        self.sched_end0 = z()                    # the FIRST draw's scheduled end (set on entry only, never merged)
        self.kind, self.entry_kind, self.n_merges = z(), z(), z()
        self.prev_kind = torch.full((self.n,), -1, dtype=torch.int64, device=device)
        self.sigma = torch.zeros(self.n, dtype=torch.float32, device=device)
        self.take_ep, self.leg = z(), z()
        self.sign = torch.ones(self.n, dtype=torch.float32, device=device)
        self.last_ep = z()
        self.last_c = torch.zeros(self.n, NUM_LEGS, dtype=torch.float32, device=device)
        self.last_hip = torch.zeros(self.n, NUM_LEGS, dtype=torch.float32, device=device)
        self.c0 = torch.zeros(self.n, NUM_LEGS, dtype=torch.float32, device=device)     # on the first draw's last row
        self.hip0 = torch.zeros(self.n, NUM_LEGS, dtype=torch.float32, device=device)
        self.take_c = torch.full((self.n,), float("nan"), device=device)
        self.take_hip = torch.full((self.n,), float("nan"), device=device)
        self.reached_take = z(torch.bool)
        self.terrain = torch.full((self.n,), -1, dtype=torch.int64, device=device)
        self.xf_left = z()
        self.pending: dict[int, dict] = {}
        self.done: list[dict] = []
        self.counts = {"exits": 0, "falls": 0, "cut": 0}

    # ---------------------------------------------------------------- before the routed step
    def pre(self, stand, obs, time_left, counter, ep_len, it: int, ep_kind=None, ep_sigma=None, terrain=None,
            first: bool = False):
        """Entries / merges / last-row fields for the rows about to act. Returns ``(entering, prev_valid)``.
        ``entry_kind``: 0 spawn (episode step 0), 1 handover (walk -> stand), 2 run start (``first``: the stands
        already in progress when the run begins)."""
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
        if self.mode == "dagger":
            self.kind = torch.where(entering, ep_kind.to(torch.int64), self.kind)
            self.sigma = torch.where(entering, ep_sigma.to(torch.float32), self.sigma)
            self.take_ep = torch.where(entering, ep_len, self.take_ep)
        else:
            u = torch.rand(4, self.n, generator=self.gen, device=self.device, dtype=torch.float64)
            acc = torch.cumsum(torch.tensor(SPEC.ASSIGN_PROBS, dtype=torch.float64, device=self.device), 0)
            kind = torch.searchsorted(acc, u[0], right=True).clamp(max=len(SPEC.ASSIGNS) - 1)
            lo, hi = (int(round(s / self.dt)) for s in SPEC.SPOIL_WINDOW_S)
            window = lo + torch.floor(u[1] * (hi - lo + 1)).to(torch.int64)
            take = torch.where(kind == SPEC.AS_CONTROL, ep_len, ep_len + window)
            leg = torch.floor(u[2] * NUM_LEGS).to(torch.int64).clamp(max=NUM_LEGS - 1)
            sign = torch.where(u[3] < 0.5, -1.0, 1.0).to(torch.float32)
            self.kind = torch.where(entering, kind, self.kind)
            self.sigma = torch.where(entering, torch.zeros_like(self.sigma), self.sigma)
            self.take_ep = torch.where(entering, take, self.take_ep)
            self.leg = torch.where(entering, leg, self.leg)
            self.sign = torch.where(entering, sign, self.sign)
        self.take_c = torch.where(entering, torch.full_like(self.take_c, float("nan")), self.take_c)
        self.take_hip = torch.where(entering, torch.full_like(self.take_hip, float("nan")), self.take_hip)
        self.reached_take = self.reached_take & ~entering
        if terrain is not None:
            self.terrain = torch.where(entering, terrain.to(torch.int64), self.terrain)
        self.in_stand = self.in_stand | entering
        c = obs[:, list(CAM_COLS)].to(torch.float32)
        hip = obs[:, list(HIP_COLS)].to(torch.float32)
        self.last_ep = torch.where(stand, ep_len, self.last_ep)
        self.last_c = torch.where(stand[:, None], c, self.last_c)
        self.last_hip = torch.where(stand[:, None], hip, self.last_hip)
        # the first draw's stand-end read: every stand row before sched_end0 overwrites it, so it holds the LAST row
        # the stand acted on within its first draw (== last_c when the stand ends at / before sched_end0)
        in_draw0 = stand & (ep_len < self.sched_end0)
        self.c0 = torch.where(in_draw0[:, None], c, self.c0)
        self.hip0 = torch.where(in_draw0[:, None], hip, self.hip0)
        at_take = stand & (ep_len == self.take_ep)
        held = self.leg[:, None]
        self.take_c = torch.where(at_take, c.gather(1, held).squeeze(1), self.take_c)
        self.take_hip = torch.where(at_take, hip.gather(1, held).squeeze(1), self.take_hip)
        self.reached_take = self.reached_take | at_take
        return entering, prev_valid

    def in_window(self, stand, ep_len):
        """A1: the stand rows still inside their spoiler window (never for a control stand)."""
        return stand.bool() & (ep_len.to(torch.int64) < self.take_ep)

    # ---------------------------------------------------------------- after env.step
    def post(self, dones, fell, stand_next, it: int, timeout=None) -> list[dict]:
        """Ends (exit / fall / timeout-cut), exit-fall windows, the per-episode carry-over. Returns the records that
        became FINAL on this step. ``fell``: the ``crab_failure`` term; ``timeout``: the episode-length timeout
        (``episode step + 1 >= max``; None = every non-``crab_failure`` done). A done that is neither (the omni
        ``total_terminates`` also cuts at |roll| / |pitch| > 1.5 rad and a height floor) counts as a FALL --
        conservative: a terminated stand scores homed 0 and never leaves the scored set."""
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
        # carry-over and resets
        self.prev_kind = torch.where(ended_exit, self.kind, self.prev_kind)
        self.prev_kind = torch.where(dones, torch.full_like(self.prev_kind, -1), self.prev_kind)
        self.xf_left = torch.where(ended_exit, torch.full_like(self.xf_left, self.xf_steps), self.xf_left)
        self.in_stand = self.in_stand & ~ended
        return final

    def _host(self, rows, ended_fall, ended_cut, ended_exit, xf_fall, xf_cens, xf_clean, it, fell):
        idx = rows
        cols = {"crab_failure": fell[idx],
            "fall": ended_fall[idx], "cut_end": ended_cut[idx], "exit": ended_exit[idx], "xf_fall": xf_fall[idx],
            "xf_cens": xf_cens[idx], "xf_clean": xf_clean[idx], "entry_ep": self.entry_ep[idx],
            "entry_it": self.entry_it[idx], "sched_end": self.sched_end[idx], "kind": self.kind[idx],
            "entry_kind": self.entry_kind[idx], "prev_kind": self.prev_kind[idx], "sigma": self.sigma[idx],
            "take_ep": self.take_ep[idx], "leg": self.leg[idx], "sign": self.sign[idx], "last_ep": self.last_ep[idx],
            "last_c": self.last_c[idx], "last_hip": self.last_hip[idx], "take_c": self.take_c[idx],
            "take_hip": self.take_hip[idx], "reached_take": self.reached_take[idx], "terrain": self.terrain[idx],
            "n_merges": self.n_merges[idx], "sched_end0": self.sched_end0[idx], "c0": self.c0[idx],
            "hip0": self.hip0[idx],
        }
        h = {k: v.detach().cpu().tolist() for k, v in cols.items()}
        envs = idx.detach().cpu().tolist()
        p_down = float(self.p.p_down) if self.p is not None else -0.05
        names = SPEC.EXECUTORS if self.mode == "dagger" else SPEC.ASSIGNS
        final = []
        for j, e in enumerate(envs):
            # 1) an exit-fall window of an EARLIER exit closes on this step
            if h["xf_fall"][j] or h["xf_cens"][j] or h["xf_clean"][j]:
                rec = self.pending.pop(e, None)
                if rec is not None:
                    rec["exit_fall"] = bool(h["xf_fall"][j])
                    rec["exit_censored"] = bool(h["xf_cens"][j])
                    if h["xf_fall"][j]:
                        rec["exit_fall_cause"] = "crab_failure" if h["crab_failure"][j] else "termination"
                    final.append(rec)
            # 2) a stand ends on this step
            if h["fall"][j] or h["cut_end"][j] or h["exit"][j]:
                end = "fall" if h["fall"][j] else ("exit" if h["exit"][j] else "timeout")
                last_ep = int(h["last_ep"][j])
                end_ep = last_ep + 1
                entry_ep = int(h["entry_ep"][j])
                sched_end = int(h["sched_end"][j])
                take_ep = int(h["take_ep"][j])
                absc = [abs(float(v)) for v in h["last_c"][j]]
                down = [float(v) >= p_down for v in h["last_hip"][j]]
                lo, hi = SPEC.TRIM_BAND
                homed = all(a <= SPEC.HOMED_EPS for a in absc) and end != "fall"
                pk = int(h["prev_kind"][j])
                tc, th = float(h["take_c"][j]), float(h["take_hip"][j])
                # the FIRST draw (the scored stand; see the module docstring)
                sched_end0 = int(h["sched_end0"][j])
                fall0 = end == "fall" and last_ep < sched_end0
                reached0 = not fall0 and (end == "exit" or last_ep >= sched_end0 - 1)
                end0 = "fall" if fall0 else ("reached" if reached0 else end)
                absc0 = [abs(float(v)) for v in h["c0"][j]]
                down0 = [float(v) >= p_down for v in h["hip0"][j]]
                rec = {
                    "mode": self.mode, "it_entry": int(h["entry_it"][j]), "it_end": int(it), "env": int(e),
                    "entry_ep": entry_ep, "last_row_ep": last_ep, "end_ep": end_ep, "sched_end_ep": sched_end,
                    "max_ep": self.max_ep, "sched_len_s": (sched_end - entry_ep) * self.dt,
                    "length_s": (end_ep - entry_ep) * self.dt, "cut": bool(sched_end >= self.max_ep), "end": end,
                    "kind": names[int(h["kind"][j])], "sigma": float(h["sigma"][j]), "take_ep": take_ep,
                    "r_time_s": max(0, end_ep - take_ep) * self.dt, "homed": bool(homed),
                    "max_abs_c": max(absc), "abs_c": absc, "down": down, "n_down": int(sum(down)),
                    "trim_legs": int(sum(1 for a, d in zip(absc, down) if d and lo < a <= hi)),
                    "exit_fall": None, "exit_censored": None,
                    "entry_kind": ENTRY_KINDS[int(h["entry_kind"][j])],
                    "prev_kind": None if pk < 0 else names[pk],
                    "fell_before_takeover": bool(end == "fall" and last_ep < take_ep),
                    "stuck_leg": int(h["leg"][j]), "stuck_sign": float(h["sign"][j]),
                    "reached_take": bool(h["reached_take"][j]),
                    "take_hip": None if math.isnan(th) else th, "take_c": None if math.isnan(tc) else tc,
                    "terrain_level": int(h["terrain"][j]), "n_merges": int(h["n_merges"][j]),
                    "fall_cause": (("crab_failure" if h["crab_failure"][j] else "termination") if end == "fall"
                                   else None),
                    # first-draw fields (scored by sn_dagger_spec.scored / premise_scored / group_stats)
                    "sched_end0_ep": sched_end0, "sched_len0_s": (sched_end0 - entry_ep) * self.dt,
                    "cut0": bool(sched_end0 >= self.max_ep), "end0": end0, "fall0": bool(fall0),
                    "homed0": bool(reached0 and all(a <= SPEC.HOMED_EPS for a in absc0)),
                    "abs_c0": absc0, "max_abs_c0": max(absc0), "n_down0": int(sum(down0)),
                    "trim_legs0": int(sum(1 for a, d in zip(absc0, down0) if d and lo < a <= hi)),
                    "r_time0_s": max(0, min(end_ep, sched_end0) - take_ep) * self.dt,
                }
                if end == "exit":
                    old = self.pending.pop(e, None)
                    if old is not None:                      # cannot happen (stands and walks last >= 3 s); kept safe
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
        """At the end of the run: every still-open exit window is censored."""
        out = []
        for e in sorted(self.pending):
            rec = self.pending.pop(e)
            rec["exit_fall"] = None
            rec["exit_censored"] = True
            out.append(rec)
        self.done.extend(out)
        return out


# ===================================================================================== the run loop
class RunOpts:
    """Plain options (no argparse objects inside the core)."""

    def __init__(self, *, mode, arm=None, seed, iterations, stop_after=None, num_envs=SPEC.NUM_ENVS,
                 steps_per_iter=SPEC.STEPS_PER_ITER, lr=SPEC.LR, minibatches=SPEC.MINIBATCHES,
                 minibatch=SPEC.MINIBATCH, buffer_gb=SPEC.BUFFER_GB, buffer_rows=None, onset_share=SPEC.ONSET_SHARE,
                 w_max=SPEC.W_MAX, max_grad_norm=SPEC.MAX_GRAD_NORM, ckpt_every=SPEC.CKPT_EVERY,
                 control_share=SPEC.CONTROL_SHARE, init_random_ep_len=True, smoke=False, log_every=10):
        if mode not in SPEC.MODES:
            raise ValueError(f"mode {mode!r}")
        if mode == "dagger" and arm not in SPEC.ARMS:
            raise ValueError(f"arm {arm!r} (dagger needs mask|full)")
        if mode == "recover_probe" and arm is not None:
            raise ValueError("recover_probe runs no student: no arm")
        self.mode, self.arm, self.seed = mode, arm, int(seed)
        self.stand_mask = SPEC.ARMS[arm] if arm is not None else 0
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

    def to_dict(self) -> dict:
        return dict(vars(self))


def _jsonl(path, rows):
    if not rows:
        return
    with open(path, "a") as fh:
        for r in rows:
            fh.write(json.dumps(r, sort_keys=True) + "\n")


class Trainer:
    """One run (dagger or recover_probe) on ANY env with the wrapper interface: ``get_observations()``, ``step(a)``
    (clips to +-clip_actions), ``num_envs``, ``unwrapped.{command_manager, termination_manager, episode_length_buf,
    max_episode_length, step_dt}``. The CPU tests drive it with a fake env and the real S1 networks."""

    def __init__(self, env, nets: Nets, opts: RunOpts, out_dir, *, train_cfg=None, policy_cls=None, est_cls=None,
                 infos=None, terrain_fn=None, log=print):
        self.env, self.nets, self.o, self.out = env, nets, opts, Path(out_dir)
        self.train_cfg, self.policy_cls, self.est_cls = train_cfg, policy_cls, est_cls
        self.infos, self.terrain_fn, self.log = dict(infos or {}), terrain_fn, log
        self.out.mkdir(parents=True, exist_ok=True)
        u = env.unwrapped
        self.n = int(env.num_envs)
        self.device = u.episode_length_buf.device
        self.dt, self.max_ep = float(u.step_dt), int(u.max_episode_length)
        self.cmd = u.command_manager.get_term("base_velocity")
        self.gens = make_generators(opts.seed, self.device)
        self.r_params = SN.r_params()
        self.tracker = StandTracker(self.n, self.device, dt=self.dt, max_ep=self.max_ep, mode=opts.mode,
                                    gen=self.gens["a1"], r_params=self.r_params)
        self.dagger = opts.mode == "dagger"
        self.opt = make_optimizer(nets.student, opts.lr) if self.dagger else None
        if self.dagger:
            n_keep = nets.p1 + (NUM_OBS - nets.hist_start)
            cap = opts.buffer_rows or Reservoir.rows_for(opts.buffer_gb, n_keep)
            self.buf = Reservoir(cap, p1=nets.p1, hist_start=nets.hist_start, device=self.device,
                                 gen=self.gens["reservoir"])
            if self.buf.nbytes > opts.buffer_gb * 1e9 + 1:
                raise RuntimeError(f"reservoir {self.buf.nbytes} B exceeds the {opts.buffer_gb} GB cap")
        else:
            self.buf = None
        self.ep_kind = torch.full((self.n,), SPEC.EX_CONTROL, dtype=torch.int64, device=self.device)
        self.ep_sigma = torch.zeros(self.n, dtype=torch.float32, device=self.device)
        self.prev_branch = torch.zeros(self.n, NUM_LEGS, dtype=torch.int64, device=self.device)
        self.opt_steps = 0
        self.student_forwards = 0
        self.stop_requested = False
        self.paths = {k: self.out / v for k, v in (("log", "train_log.jsonl"), ("stands", "stands.jsonl"),
                                                   ("status", "status.json"))}
        self.ckpts: list[str] = []
        self.smoke_acc = self._new_acc()
        self.capture = None
        self.s1_det = None
        self.first_step = True
        self.mb_onset = []
        self.mb_onset_w = []
        self.terrain_errors = 0
        self.terrain_error = None

    # ---------------------------------------------------------------- accumulators (smoke + per-iteration log)
    def _new_acc(self):
        d = self.device
        z = lambda *s, dt=torch.float64: torch.zeros(*s, dtype=dt, device=d)  # noqa: E731
        return {"rows": z(()), "stand_rows": z(()), "rows_by_kind": z(3), "branch_student": z(NUM_LEGS, N_BRANCH),
                "dart_n": z(3), "dart_ss_pre": z(3), "dart_ss_post": z(3), "sigma0_maxdev": z(()),
                "statue_rows": z(()), "statue_max": z(()), "chatter_rows": z(()), "chatter_sum": z(NUM_ACT),
                "chatter_ss": z(NUM_ACT), "chatter_ss_post": z(NUM_ACT), "exec_mismatch": z(()),
                "sigma_max": z(()), "onset_rows": z(()), "a1_window_rows": z(()), "lastrow_mismatch": z(()),
                "onset_rows_student": z(()),
                "ep_new": z(()), "ep_ctrl": z(()), "ep_beta_r": z(()), "ep_beta_sum": z(()), "ep_noncontrol": z(()),
                "entries_spawn": z(()), "entries_handover": z(()), "entries_run_start": z(())}

    # ---------------------------------------------------------------- episode draws (per episode, private)
    def _episode_draws(self, new_ep, it: int):
        u = torch.rand(3, self.n, generator=self.gens["episode"], device=self.device, dtype=torch.float64)
        if not self.dagger:
            return
        b = SPEC.beta(it, self.o.iterations)
        ctrl = u[0] < self.o.control_share
        beta_r = ~ctrl & (u[1] < b)
        kind = torch.where(ctrl, torch.full_like(self.ep_kind, SPEC.EX_CONTROL),
                           torch.where(beta_r, torch.full_like(self.ep_kind, SPEC.EX_BETA_R),
                                       torch.full_like(self.ep_kind, SPEC.EX_STUDENT)))
        acc = torch.cumsum(torch.tensor([p for _, p in SPEC.SIGMA_TABLE], dtype=torch.float64, device=self.device), 0)
        sig_idx = torch.searchsorted(acc, u[2], right=True).clamp(max=len(SPEC.SIGMA_TABLE) - 1)
        sig = torch.tensor([s for s, _ in SPEC.SIGMA_TABLE], dtype=torch.float32, device=self.device)[sig_idx]
        sig = torch.where(ctrl, torch.zeros_like(sig), sig)                       # OPEN-5: sigma 0 on the control slice
        self.ep_kind = torch.where(new_ep, kind, self.ep_kind)
        self.ep_sigma = torch.where(new_ep, sig, self.ep_sigma)

    def _terrain(self):
        """The per-env terrain levels (stamped on each stand at entry). A missing / failing accessor leaves
        ``terrain_level`` -1 on the records; it is counted here and FAILS the smoke's ``terrain_level_recorded`` (the
        step-7 / A1 "same terrain levels" claim must stay auditable)."""
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
            self.log(f"[SN] WARNING terrain levels unavailable ({why}); stand records get terrain_level -1 "
                     "(the smoke's terrain_level_recorded check fails)")

    # ---------------------------------------------------------------- one env step
    def _step(self, obs, stand, new_ep, it: int, capture: bool = False):
        u = self.env.unwrapped
        o, nets = self.o, self.nets
        self._episode_draws(new_ep, it)
        ep_len = u.episode_length_buf.reshape(-1).clone()
        entering, prev_valid = self.tracker.pre(stand, obs, self.cmd.time_left.reshape(-1),
                                                self.cmd.command_counter.reshape(-1), ep_len, it,
                                                self.ep_kind, self.ep_sigma, self._terrain(),
                                                first=self.first_step)
        a_s1 = s1_actions(obs, nets)
        if self.s1_det is None:
            self.s1_det = bool(torch.equal(s1_actions(obs, nets), a_s1))           # S1 is deterministic
        tgt, branch = SN.r_label(obs)
        tgt = tgt.to(a_s1.dtype)
        acc = self.smoke_acc
        st_f = stand.to(torch.float64)
        if self.dagger:
            x_st = stand_input(obs, nets, o.stand_mask)
            mu = student_mean(nets.student, x_st).to(a_s1.dtype)
            self.student_forwards += 1
            use_r = self.tracker.kind != SPEC.EX_STUDENT
            clean = torch.where(use_r[:, None], tgt, mu)
            noise = torch.randn(self.n, NUM_ACT, generator=self.gens["dart"], device=self.device, dtype=a_s1.dtype)
            sig = self.tracker.sigma.to(a_s1.dtype)
            a_stand = torch.where((sig > 0)[:, None], clean + sig[:, None] * noise, clean)
            # buffer: every stand row, labelled with R on the visited state (unmasked obs; R reads 3-4, 18-34)
            onset = lift_onset(branch, self.prev_branch, prev_valid)
            rows = torch.nonzero(stand).flatten()
            self.buf.add(x_st[rows], tgt[rows], branch[rows], onset[rows])
            acc["onset_rows"] += (onset & stand).sum()
            kind = self.tracker.kind
            for k in range(3):
                acc["rows_by_kind"][k] += (stand & (kind == k)).sum()
            stu = stand & (kind == SPEC.EX_STUDENT)
            acc["onset_rows_student"] += (onset & stu).sum()
            oh = torch.nn.functional.one_hot(branch.long(), N_BRANCH).to(torch.float64)
            acc["branch_student"] += (oh * stu.to(torch.float64)[:, None, None]).sum(0)
            dev_pre = (a_stand - clean).double()
            dev_post = (a_stand.clamp(-1, 1) - clean.clamp(-1, 1)).double()
            for k, s in enumerate((0.1, 0.3)):
                m = (stand & (self.tracker.sigma == s)).to(torch.float64)[:, None]
                acc["dart_n"][k] += m.sum() * NUM_ACT
                acc["dart_ss_pre"][k] += (dev_pre.pow(2) * m).sum()
                acc["dart_ss_post"][k] += (dev_post.pow(2) * m).sum()
            m0 = stand & (self.tracker.sigma == 0)
            acc["sigma0_maxdev"] = torch.maximum(
                acc["sigma0_maxdev"], torch.where(m0[:, None], dev_pre.abs(), torch.zeros_like(dev_pre)).max().double())
            new_rows = new_ep & ~(self.ep_kind == SPEC.EX_CONTROL)
            acc["ep_new"] += new_ep.sum()
            acc["ep_ctrl"] += (new_ep & (self.ep_kind == SPEC.EX_CONTROL)).sum()
            acc["ep_beta_r"] += (new_rows & (self.ep_kind == SPEC.EX_BETA_R)).sum()
            acc["ep_beta_sum"] += new_rows.sum() * SPEC.beta(it, o.iterations)
            acc["ep_noncontrol"] += new_rows.sum()
            if capture and self.capture is None and bool(stand.any()):
                self.capture = {"obs": obs.detach().clone(), "stand": stand.detach().clone(), "it": int(it)}
        else:
            win = self.tracker.in_window(stand, ep_len)
            chatter = torch.randn(self.n, NUM_ACT, generator=self.gens["chatter"], device=self.device,
                                  dtype=a_s1.dtype)
            spoil = spoiler_actions(self.tracker.kind, tgt, obs, self.tracker.leg, self.tracker.sign, chatter,
                                    self.r_params)
            a_stand = torch.where(win[:, None], spoil, tgt)
            kind = self.tracker.kind
            ws = win & (kind == SPEC.AS_STATUE)
            acc["statue_rows"] += ws.sum()
            ex = a_stand.clamp(-1, 1).abs()
            acc["statue_max"] = torch.maximum(acc["statue_max"],
                                              torch.where(ws[:, None], ex, torch.zeros_like(ex)).max().double())
            wc = win & (kind == SPEC.AS_CHATTER)
            wcf = wc.to(torch.float64)[:, None]
            d = (a_stand - tgt).double()
            dp = (a_stand.clamp(-1, 1) - tgt).double()
            acc["chatter_rows"] += wc.sum()
            acc["chatter_sum"] += (d * wcf).sum(0)
            acc["chatter_ss"] += (d.pow(2) * wcf).sum(0)
            acc["chatter_ss_post"] += (dp.pow(2) * wcf).sum(0)
            acc["a1_window_rows"] += win.sum()
            acc["sigma_max"] = torch.maximum(acc["sigma_max"], (self.tracker.sigma * st_f.float()).max().double())
        self.prev_branch = branch.long()
        actions = SN.route_actions(stand, a_stand, a_s1)
        if not self.dagger:
            # smoke (A1): every ROUTED stand row is R's target or its stand's own spoiler; every other row S1's
            want = torch.where(stand[:, None], torch.where(win[:, None], spoil, tgt), a_s1)
            acc["exec_mismatch"] += (actions != want).any(dim=1).sum()
        acc["rows"] += self.n
        acc["stand_rows"] += st_f.sum()
        pre_c = obs[:, list(CAM_COLS)].to(torch.float32)
        obs_next, _, dones, extras = self.env.step(actions)       # the wrapper clips to +-clip_actions (T7)
        fell = u.termination_manager.get_term("crab_failure").bool().reshape(-1)
        dones = dones.bool().reshape(-1)
        stand_next = self.cmd.stand_mask().reshape(-1).bool()
        ending = self.tracker.in_stand & (dones | ~stand_next)
        acc["lastrow_mismatch"] += (ending[:, None] & (self.tracker.last_c != pre_c)).any(dim=1).sum()
        timeout = dones & (ep_len + 1 >= self.max_ep)            # the executed row was the episode's last
        final = self.tracker.post(dones, fell, stand_next, it, timeout=timeout)
        if self.first_step:
            acc["entries_run_start"] += entering.sum()
            self.first_step = False
        else:
            acc["entries_spawn"] += (entering & (ep_len == 0)).sum()
            acc["entries_handover"] += (entering & (ep_len > 0)).sum()
        return obs_next, stand_next, dones, final

    # ---------------------------------------------------------------- the run
    def run(self) -> dict:
        o, u = self.o, self.env.unwrapped
        t_start = time.time()
        if o.init_random_ep_len:                                  # OPEN-6, a private generator
            u.episode_length_buf = torch.randint(0, self.max_ep, (self.n,), generator=self.gens["init_len"],
                                                 device=self.device, dtype=u.episode_length_buf.dtype)
        obs, _ = self.env.get_observations()                      # ONCE (T6); afterwards only env.step's obs
        stand = self.cmd.stand_mask().reshape(-1).bool()
        if self.dagger:
            self.compaction_guard(obs)
        new_ep = torch.ones(self.n, dtype=torch.bool, device=self.device)
        status = "running"
        last_it = -1
        for it in range(o.stop_after):
            t0 = time.time()
            n_final = []
            capture = o.smoke and it == o.stop_after - 1
            with torch.inference_mode():
                for _ in range(o.steps_per_iter):
                    obs, stand, new_ep, final = self._step(obs, stand, new_ep, it, capture=capture)
                    n_final.extend(final)
            t1 = time.time()
            tr = {}
            if self.dagger:
                tr = train_iteration(self.nets.student, self.opt, self.buf, self.gens, minibatches=o.minibatches,
                                     minibatch=o.minibatch, onset_share=o.onset_share, w_max=o.w_max,
                                     max_grad_norm=o.max_grad_norm)
                self.opt_steps += int(tr.get("steps", 0))
                if tr.get("steps"):
                    self.mb_onset.append(tr["onset"])
                    self.mb_onset_w.append(tr["onset_w"])
            t2 = time.time()
            _jsonl(self.paths["stands"], n_final)
            row = self._log_row(it, n_final, tr, t1 - t0, t2 - t1)
            _jsonl(self.paths["log"], [row])
            if o.log_every and (it % o.log_every == 0 or it == o.stop_after - 1):
                self.log(self._log_line(row))
            done_its = it + 1
            last_it = it
            if self.dagger and (done_its % o.ckpt_every == 0 or done_its == o.stop_after):
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
               "s_per_iter": (time.time() - t_start) / max(1, last_it + 1)}
        self.paths["status"].write_text(json.dumps(res, indent=1, sort_keys=True))
        return res

    def compaction_guard(self, obs) -> None:
        """Fail closed: the student's D1 forward on the reservoir's compacted + re-expanded rows equals the forward on
        the full rows, bitwise, on this device and batch (the hist path never reads [p1, hist_start))."""
        with torch.inference_mode():
            x = stand_input(obs, self.nets, self.o.stand_mask)
            a = student_mean(self.nets.student, x)
            b = student_mean(self.nets.student, self.buf.expand_rows(self.buf.compact(x)))
        if not torch.equal(a, b):
            raise RuntimeError("sn_dagger: the reservoir's column compaction changes the student's action; refused")

    def save(self, done_its: int) -> str:
        infos = dict(self.infos)
        infos["sn_dagger"] = dict(infos.get("sn_dagger") or {}, iterations_done=int(done_its),
                                  opt_steps=int(self.opt_steps), buffer_seen=int(self.buf.seen),
                                  buffer_size=int(self.buf.size))
        p = save_checkpoint(self.out / f"model_{int(done_its)}.pt", self.nets, self.opt, done_its, infos)
        self.ckpts.append(p)
        return p

    def _log_row(self, it, final, tr, t_roll, t_train) -> dict:
        row = {"it": int(it), "beta": SPEC.beta(it, self.o.iterations) if self.dagger else 1.0,
               "t_rollout_s": t_roll, "t_train_s": t_train, "n_final_stands": len(final)}
        if self.dagger:
            row.update(buffer_size=int(self.buf.size), buffer_seen=int(self.buf.seen),
                       **{f"train_{k}": v for k, v in tr.items() if k != "weight_table"})
        names = SPEC.EXECUTORS if self.dagger else SPEC.ASSIGNS
        keep = SPEC.premise_scored if self.dagger else (lambda r: SPEC.scored(r, SPEC.A1_MIN_SCHED_S))
        for name in names:
            g = [r for r in final if r["kind"] == name and keep(r)]
            row[f"{name}_scored"] = len(g)
            row[f"{name}_homed"] = sum(1 for r in g if SPEC.homed_scored(r))
            row[f"{name}_falls"] = sum(1 for r in g if SPEC.fall_scored(r))
            xs = [r for r in final if r["kind"] == name and r["end"] == "exit" and r["exit_fall"] is not None]
            row[f"{name}_exit_n"] = len(xs)
            row[f"{name}_exit_falls"] = sum(1 for r in xs if r["exit_fall"])
        row.update({f"count_{k}": v for k, v in self.tracker.counts.items()})
        a = self.smoke_acc
        row["stand_share_cum"] = float(a["stand_rows"].item()) / max(1.0, float(a["rows"].item()))
        row["entries_handover_cum"] = int(a["entries_handover"].item())
        row["entries_spawn_cum"] = int(a["entries_spawn"].item())
        if self.dagger:
            row["rows_by_executor_cum"] = dict(zip(SPEC.EXECUTORS, (int(x) for x in a["rows_by_kind"].tolist())))
        return row

    def _log_line(self, row) -> str:
        names = SPEC.EXECUTORS if self.dagger else SPEC.ASSIGNS
        parts = [f"[SN] it {row['it'] + 1}/{self.o.stop_after} beta {row['beta']:.3f}"]
        if self.dagger:
            parts.append(f"buf {row['buffer_size']} loss {row.get('train_loss', float('nan')):.5f} "
                         f"onset {row.get('train_onset', float('nan')):.3f}")
        parts += [f"{n} {row[f'{n}_homed']}/{row[f'{n}_scored']}" for n in names]
        parts.append(f"roll {row['t_rollout_s']:.2f}s train {row['t_train_s']:.2f}s")
        return " | ".join(parts)

    # ---------------------------------------------------------------- the smoke checks (PLAN step 6)
    def smoke_measures(self) -> dict:
        a = {k: (v.detach().cpu().tolist() if torch.is_tensor(v) else v) for k, v in self.smoke_acc.items()}
        return a


# ===================================================================================== step-6 smoke (PLAN step 6 + A1 smoke)
def _chk(name, value, line, ok, **extra) -> dict:
    return {"name": name, "value": value, "line": line, "pass": bool(ok), **extra}


def obs_layout_reasons(obs) -> list[str]:
    """D5 on captured obs rows: 1149 wide; with ``STAND_OBS=0`` + ``CLOCK_HIDE=1`` the clock / stand-bit columns 15-16
    read exactly 0 in the current step AND in each of the 10 real history frames (as in the eval)."""
    out = []
    if int(obs.shape[1]) != NUM_OBS:
        return [f"obs width {obs.shape[1]} != {NUM_OBS}"]
    cols = [15, 16] + [259 + 89 * j + k for j in range(10) for k in (15, 16)]
    nz = int((obs[:, cols] != 0).sum())
    if nz:
        out.append(f"{nz} non-zero entries in the clock / stand-bit columns {cols[:2]} (+ history copies)")
    return out


PARITY_GPU_ATOL = 1e-5


def forward_parity(tr: "Trainer", ckpt_path) -> dict:
    """In-process (same device, same batch): the trainer's routed action of a student-executed sigma-0 step vs the
    stand_net launcher's computation with the SAVED checkpoint loaded on its path (``runner.load`` = strict loads, no
    map_location). Pinned against the real ``StandNetMode`` on CPU by the tests."""
    cap = tr.capture
    if cap is None:
        return {"equal": False, "why": "no captured stand batch"}
    pol, est, _ = load_strict(ckpt_path, tr.policy_cls, tr.est_cls, tr.train_cfg, tr.device)
    n = tr.nets
    with torch.inference_mode():
        a_tr = trainer_actions(cap["obs"], cap["stand"], n, tr.o.stand_mask)
        a_l = launcher_actions(cap["obs"], cap["stand"], n.s1, n.est, pol, est, p0=n.p0, p1=n.p1,
                               num_prop=n.num_prop, mask=tr.o.stand_mask)
    eq = bool(torch.equal(a_tr, a_l))
    diff = float((a_tr - a_l).abs().max().item())
    on_gpu = torch.device(tr.device).type == "cuda"
    out = tr.out / "smoke_capture.pt"
    torch.save({"obs": cap["obs"].detach().cpu(), "stand": cap["stand"].detach().cpu(),
                "actions_trainer": a_tr.detach().cpu(), "stand_mask": int(tr.o.stand_mask),
                "ckpt": str(ckpt_path), "ckpt_sha256": sha256_file(ckpt_path), "s1": n.s1_path, "s1_sha256": n.s1_sha,
                "device": str(tr.device), "it": cap["it"], "priv": [n.p0, n.p1], "num_prop": n.num_prop}, str(out))
    # CHARTER 2026-10-12: same batch shape in one process -- CPU torch.equal; GPU atol 1e-5
    ok = eq or (on_gpu and diff <= PARITY_GPU_ATOL)
    return {"equal": eq, "max_abs_diff": diff, "device": str(tr.device), "rule": "CPU torch.equal; GPU atol 1e-5",
            "pass": bool(ok), "rows": int(cap["obs"].shape[0]), "stand_rows": int(cap["stand"].sum()),
            "capture": str(out)}


def smoke_checks_dagger(tr: "Trainer", *, env_parity: list[str], ckpt: str) -> list[dict]:
    """PLAN step 6's dose list for one DAgger arm (``SPEC.SMOKE_DAGGER``), on the run ``tr`` just finished."""
    a = tr.smoke_measures()
    o = tr.o
    checks = []
    share = a["stand_rows"] / max(1.0, a["rows"])
    checks.append(_chk("stand_row_share", share, "|share - 0.5| <= 0.05", abs(share - 0.5) <= 0.05,
                       rows=a["rows"], stand_rows=a["stand_rows"]))
    steps = max(1.0, a["rows"])
    ent = {"handover": a["entries_handover"], "spawn": a["entries_spawn"], "run_start": a["entries_run_start"],
           "exits": tr.tracker.counts["exits"], "falls_in_stand": tr.tracker.counts["falls"],
           "per_1000_env_steps": {"handover": 1000 * a["entries_handover"] / steps,
                                  "exits": 1000 * tr.tracker.counts["exits"] / steps}}
    checks.append(_chk("handovers", ent, "walk->stand handovers > 0 and stand exits > 0",
                       a["entries_handover"] > 0 and tr.tracker.counts["exits"] > 0))
    bs = a["branch_student"]
    tot = sum(sum(r) for r in bs)
    shares = ({name: sum(bs[l][k] for l in range(NUM_LEGS)) / tot for k, name in enumerate(
        ("hold", "wait", "lift", "home", "seat"))} if tot else None)
    checks.append(_chk("branch_shares_student_rows", shares, "student-executed stand rows exist (shares reported)",
                       tot > 0, student_rows=a["rows_by_kind"][SPEC.EX_STUDENT],
                       rows_by_executor=dict(zip(SPEC.EXECUTORS, a["rows_by_kind"]))))
    # the DOSE (review 2026-10-12): the minibatch ROW share is >= ONSET_SHARE by construction whenever any onset row
    # exists, so it is reported only; the check binds on (i) LIFT onsets present on STUDENT-visited stand rows
    # (pre-oversampling -- the DAgger premise: a failing student generates onset labels) and (ii) their share of the
    # LOSS WEIGHT after both balancing mechanisms (oversampling + per-leg inverse-frequency weights) >= 0.01
    mb = (sum(tr.mb_onset) / len(tr.mb_onset)) if tr.mb_onset else None
    mbw = (sum(tr.mb_onset_w) / len(tr.mb_onset_w)) if tr.mb_onset_w else None
    stu_rows = a["rows_by_kind"][SPEC.EX_STUDENT]
    dose = {"onset_rows_student": a["onset_rows_student"], "student_stand_rows": stu_rows,
            "pre_oversampling_share_student_rows": (a["onset_rows_student"] / stu_rows) if stu_rows else None,
            "pre_oversampling_share_all_stand_rows": (a["onset_rows"] / a["stand_rows"]) if a["stand_rows"] else None,
            "minibatch_row_share_reported": mb, "loss_weight_share_after_balancing": mbw,
            "buffer_onset_share": (float(tr.buf.onset[:tr.buf.size].float().mean().item()) if tr.buf.size
                                   else None)}
    checks.append(_chk("lift_onset_after_balancing", dose,
                       "LIFT-onset rows on student-visited stand rows > 0 (pre-oversampling dose reported) and >= 0.01 "
                       "of the minibatch loss weight after balancing (oversampling + per-leg weights)",
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
    dart = {}
    ok = True
    for k, s in enumerate((0.1, 0.3)):
        n_ = a["dart_n"][k]
        pre = math.sqrt(a["dart_ss_pre"][k] / n_) if n_ else None
        post = math.sqrt(a["dart_ss_post"][k] / n_) if n_ else None
        dart[str(s)] = {"n_entries": n_, "sigma_pre_clip": pre, "sigma_post_clip": post}
        ok = ok and pre is not None and abs(pre - s) <= 0.1 * s
    dart["0.0"] = {"max_abs_deviation": a["sigma0_maxdev"]}
    checks.append(_chk("dart_sigma_realized", dart, "realized pre-clip sigma within 10 % of 0.1 / 0.3; sigma-0 rows "
                       "exactly clean", ok and a["sigma0_maxdev"] == 0.0))
    checks.append(_chk("obs_parity_env", env_parity, "no obs-affecting difference from the REFS eval stamp",
                       not env_parity))
    lay = obs_layout_reasons(tr.capture["obs"]) if tr.capture is not None else ["no captured stand batch"]
    checks.append(_chk("obs_parity_cols", lay, "1149 wide; clock / stand-bit cols 15-16 (+ history copies) == 0",
                       not lay))
    par = forward_parity(tr, ckpt)
    checks.append(_chk("forward_parity", par, "same device, same batch: CPU torch.equal, GPU atol 1e-5 (CHARTER)",
                       par.get("pass") is True))
    checks.append(_chk("s1_deterministic", tr.s1_det, "S1 act_inference twice on one obs -> equal",
                       tr.s1_det is True))
    per_iter = tr.buf.seen / max(1, o.stop_after)
    proj = per_iter * o.iterations
    checks.append(_chk("buffer_projection", {"allocated_bytes": tr.buf.nbytes, "cap_bytes": int(o.buffer_gb * 1e9),
                                             "cap_rows": tr.buf.cap, "rows_per_iter": per_iter,
                                             "projected_rows_seen": proj,
                                             "projected_fill": min(1.0, proj / tr.buf.cap),
                                             "cuda_max_allocated": (torch.cuda.max_memory_allocated()
                                                                    if torch.cuda.is_available() else None)},
                       "the pre-allocated reservoir <= the cap (a reservoir never grows past it)",
                       tr.buf.nbytes <= o.buffer_gb * 1e9))
    rs = checkpoint_reasons(ckpt, tr.nets, tr.policy_cls, tr.est_cls, tr.train_cfg, tr.device)
    checks.append(_chk("ckpt_loads", {"ckpt": ckpt, "reasons": rs}, "strict load, all keys, estimator == S1's",
                       not rs))
    checks.append(terrain_check(tr))
    return checks


def terrain_check(tr: "Trainer") -> dict:
    """Review 2026-10-12: every stand record carries a real terrain level (the accessor never failed), so the step-7 /
    A1 "same terrain levels" claim stays auditable (``_terrain`` turns a failing accessor into -1)."""
    recs = list(tr.tracker.done) + list(tr.tracker.pending.values())
    missing = sum(1 for r in recs if int(r.get("terrain_level", -1)) < 0)
    return _chk("terrain_level_recorded", {"records": len(recs), "terrain_level_missing": missing,
                                           "accessor_failures": tr.terrain_errors, "first_error": tr.terrain_error},
                "every stand record has terrain_level >= 0 and the accessor never failed",
                len(recs) > 0 and missing == 0 and tr.terrain_errors == 0)


def smoke_checks_a1(tr: "Trainer") -> list[dict]:
    """PLAN step 6's A1-mode smoke list (``SPEC.SMOKE_A1``) on the recover_probe run ``tr`` just finished."""
    a = tr.smoke_measures()
    recs = list(tr.tracker.done) + list(tr.tracker.pending.values())
    checks = []
    n = len(recs)
    shares, ok = {}, n > 0
    for k, name in enumerate(SPEC.ASSIGNS):
        m = sum(1 for r in recs if r["kind"] == name)
        p, q = (m / n if n else None), SPEC.ASSIGN_PROBS[k]
        tol = max(0.05, 3 * math.sqrt(q * (1 - q) / n)) if n else None
        shares[name] = {"n": m, "share": p, "expected": q, "tol": tol}
        ok = ok and p is not None and abs(p - q) <= tol
    checks.append(_chk("a1_assignment_shares", shares, "~1/2 control, ~1/6 per spoiler (max(0.05, 3 SE))", ok,
                       n_stands=n))
    checks.append(_chk("a1_executed_r_or_spoiler", {"mismatched_rows": a["exec_mismatch"],
                                                     "spoiler_window_rows": a["a1_window_rows"]},
                       "every executed stand row is R's target or its stand's spoiler",
                       a["exec_mismatch"] == 0 and a["a1_window_rows"] > 0))
    checks.append(_chk("a1_no_optimizer", {"opt_steps": tr.opt_steps, "student_forwards": tr.student_forwards,
                                            "optimizer": tr.opt is not None, "checkpoints": len(tr.ckpts)},
                       "0 optimizer steps, 0 student forwards, no checkpoint",
                       tr.opt_steps == 0 and tr.student_forwards == 0 and tr.opt is None and not tr.ckpts))
    checks.append(_chk("a1_dart_zero", a["sigma_max"], "max sigma on stand rows == 0", a["sigma_max"] == 0.0))
    checks.append(_chk("a1_statue_zero", {"rows": a["statue_rows"], "max_abs_executed": a["statue_max"]},
                       "statue window rows execute exactly 0 (after the +-1 clip)",
                       a["statue_rows"] > 0 and a["statue_max"] == 0.0))
    nc = a["chatter_rows"]
    pre = [math.sqrt(s / nc) for s in a["chatter_ss"]] if nc else None
    post = [math.sqrt(s / nc) for s in a["chatter_ss_post"]] if nc else None
    checks.append(_chk("a1_chatter_sigma", {"rows": nc, "sigma_pre_clip": pre, "sigma_post_clip_vs_R": post},
                       "every channel's pre-clip sigma in [0.27, 0.33] (post-clip realised reported)",
                       bool(nc) and all(0.27 <= s <= 0.33 for s in pre)))
    p_up = float(tr.r_params.p_up)
    st = [r for r in recs if r["kind"] == "stuck_lift" and r.get("reached_take") and r.get("take_hip") is not None]
    good = [r for r in st if r["take_hip"] <= p_up and abs(r["take_c"]) > 0.3]
    frac = len(good) / len(st) if st else None
    checks.append(_chk("a1_stuck_lift_state", {"stands_reaching_takeover": len(st), "held_up_and_off_home": len(good),
                                                "share": frac, "p_up": p_up},
                       ">= 0.5 of stuck-lift stands: held leg hip <= p_up and |c| > 0.3 at takeover",
                       frac is not None and frac >= 0.5))
    # review 2026-10-12: (i) the VALUE clause -- last_c equals the obs of the row the stand acted on when it ended
    # (counted in _step); (ii) the SCHEDULE clause -- every exit's last acted row is its scheduled end - 1 (+-1 step,
    # the time_left rounding): the OPEN-7 schedule the scoring rests on matches the env's resample; (iii) the FIRST-
    # DRAW read -- an unmerged stand that exited has its first draw's end == its end, so abs_c0 must equal abs_c
    # (two independently updated buffers)
    exits = [r for r in recs if r["end"] == "exit"]
    sched_bad = [r for r in exits if abs(int(r["last_row_ep"]) - (int(r["sched_end_ep"]) - 1)) > 1]
    sched_exact = sum(1 for r in exits if int(r["last_row_ep"]) == int(r["sched_end_ep"]) - 1)
    # (an unmerged exit one row PAST sched_end0 is the +-1 rounding case: its first-draw read is one row earlier)
    unmerged = [r for r in exits if int(r.get("n_merges", 0)) == 0
                and int(r["last_row_ep"]) <= int(r["sched_end0_ep"]) - 1]
    first_bad = [r for r in unmerged if r["abs_c0"] != r["abs_c"] or r["end0"] != "reached"]
    checks.append(_chk("a1_last_row_read", {"records": n, "exits": len(exits), "value_mismatch_rows":
                                             a["lastrow_mismatch"], "exit_schedule_off_by_more_than_1": len(sched_bad),
                                             "exit_schedule_exact": sched_exact, "unmerged_exits": len(unmerged),
                                             "first_draw_read_mismatch": len(first_bad)},
                       "|c| read on the stand's last acted row (value), every exit at its scheduled end - 1 (+-1), "
                       "the first-draw read == the stand-end read on unmerged exits",
                       n > 0 and bool(exits) and a["lastrow_mismatch"] == 0 and not sched_bad and not first_bad))
    sc = [r for r in recs if SPEC.scored(r, SPEC.A1_MIN_SCHED_S)]
    viol = [r for r in sc if r["sched_len0_s"] < SPEC.A1_MIN_SCHED_S - 1e-9 or r["cut0"]
            or r["end0"] not in ("reached", "fall") or r.get("entry_kind") == "run_start"]
    excl = SPEC.a1_read([dict(r, mode="recover_probe") for r in recs])["excluded"] if recs else {}
    checks.append(_chk("a1_scored_filter", {"scored": len(sc), "excluded": excl, "violations": len(viol)},
                       "scored on the FIRST draw = its scheduled length >= 8 s, not timeout-cut, not a run-start "
                       "stand, ended by reaching its end / a fall", len(sc) > 0 and not viol))
    checks.append(terrain_check(tr))
    return checks
