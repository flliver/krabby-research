#!/usr/bin/env python3
"""Stand-height campaign, PLAN.md steps 2 / 4 -- the ``stand_height`` launcher variant (EVAL ONLY).

The name ends in ``eval_crab_hex_gait.py`` on purpose: ``run_arm.sh``'s ``drivers_idle`` finds running evals by that
suffix (one Isaac process at a time). Its absolute path contains no spaces (``EVAL_EXTRA_ARGS`` is unquoted).

WHAT IT EXTENDS. The stand-network campaign's ``stand_net`` launcher (``../../2026-10-11_stand_network/scripts/
sn_stand_net_eval_crab_hex_gait.py``, SNL) is loaded BY PATH into a private instance and reused: its P2L loader
(``load_p2r``: the phase-2 controller-level launcher in a PRIVATE module instance, map risk L7), its ``StandNetMode``
(the estimator identity in ``bind_s1``, map L5) as the base class of the network controllers, its path rules (L9).
Nothing of either file is edited. The height semantics are ``sh_core``'s, imported, never re-implemented: the posture
map ``q18``, ``R_h`` (``rh_target``), the H6 :class:`sh_core.HeightController` (limiter + restore, the ONE per-step
order), ``stand_net_input`` (column 15 = the limited command, restore rows show a stand), the router
(``stand_action`` / ``route``: S1 rows ``clamp(S1, +-1)``, stand rows ``q + clamp(a_net, +-1)``, rods +-4.8).

PER STEP (``StandHeightMode.__call__``; the latched slot as P2L): (1) the command term's latched stand slot, re-derived
on reset rows; (2) episode resets -> ``HeightController.reset(rows, "walk", h_cmd)`` (every eval start is a walk
handover: the limiter starts at the command); (3) ``HeightController.step(slot, h_cmd, obs)`` -> the limited command
``h_lim``, the EFFECTIVE stand vector ``slot | restoring`` and the restore events; (4) S1 on the harness obs, exactly as
the harness would; (5) the stand controller (``--sh-controller``):

    stand_net     a_net = net(stand_net_input(obs, own estimator, mask, h_lim, restoring)); executed q18(h_lim) + clamp(a_net)
    amask_old     the same at h = 1.064 (no q, column 15 = 0) -- A-mask s17 with W[:, 15] zeroed, the old +-1 path
    r_h           R_h(obs, h_lim) (sh_core.rh_target; rods +-4.8)
    r_old         R(obs) (sn_common.r_label; +-1, no q)
    posture_only  q18(h_lim) (zero residual)

(6) executed = ``sh_core.route(stand_eff, a_stand, S1)``; (7) the raw series (``sh_eval_common.MODE_RAW_KEYS``).

THE CLIP (PLAN H1; "write the realised clip into run_meta"). The wrapper's ``clip_actions`` is raised from S1's 1.0 to
4.8 at RUNTIME only here (``env.clip_actions``, set in the patched runner factory; no config edit), and the router
clamps S1's rows to +-1 explicitly. The harness records the tensor the policy returns and then steps the env with it,
so this launcher keeps the harness's PRE-CLIP convention for what it records (S1 rows = S1's raw output, exactly what
REFS recorded; stand rows = the controller's pre-clamp value) and swaps in the EXECUTED router output at ``env.step``
(the outermost ``env.step`` wrapper refuses any tensor the mode did not just return). That is what lets the identity
bridge (S1 as the stand checkpoint, mask 0, zero command, restore armed) reproduce REFS_S1_s18 bitwise on EVERY raw key,
``actions`` included, while the executed path is the new one (and executed S1 rows = REFS's wrapper clamp, bitwise).
run_meta records the clip before / realised, the swap count, and the executed-action maxima on S1 and stand rows.

PER-HOLD HEIGHTS (``--sh-command``). ``zero``: every hold at 1.064 (the bridge and K5 rows; a 1.064 command never
engages the restore). ``manifest``: the harness's own ``--manifest`` must carry a ``stand_height`` block naming the
``--scenario`` row; ``sh_eval_common.height_row`` validates it BEFORE Kit and ``height_table`` maps every step's hold
(the harness's ``segment_id``) to its ``hold_heights`` entry (stand holds -- all labelled "stand", as p2s_stand
requires -- the row's height; walk holds 1.064; the prologue: the first hold's). The launcher refuses ``zero``
on a manifest height row and a resolved scenario that is not the height row (post-Kit, before the first step), and a
compiled schedule whose hold labels are not the manifest row's.

RECORD. Raw series ``sh_eval_common.MODE_RAW_KEYS`` + ``sh_h_meas`` (write time); P2L's ``p2r_stand_mode`` is the
EFFECTIVE stand vector slot | restoring (who acted: P2L's "stand mode"), ``sh_stand_slot`` the latched slot alone.
run_meta ``p2r_mode.stand_height`` (the names the K-readers ``sh_read.py`` consume): ``control`` (the controller),
``command``, ``restore`` ("armed" / "off"), ``hold_heights`` (one entry per hold: a manifest row's list; a zero
command's 1.064 on "stand" holds, null elsewhere), ``height_row`` (a manifest row's record: manifest + sha, scenario,
labels, hold_heights, gates; None on a zero command), ``controller`` (``HeightController.record()`` + ``restore``;
``engage_m`` null with the restore off), the clip, the checkpoint kind, the provenance shas, event counts and
executed-action maxima.

CLI (``run_arm.sh --row-eval`` with ``P2_EVAL=<this file>``; every harness flag passes through in order):

    --sh-controller {stand_net, amask_old, r_h, r_old, posture_only}       (env fallback SH_CONTROLLER)
    --sh-command {zero, manifest}                                         (SH_COMMAND)
    --sh-restore {armed, off}     default armed; off = hand back at height (SH_RESTORE)
    --stand-ckpt PATH --stand-mask 0|1    network controllers only        (SH_STAND_CKPT / SH_STAND_MASK)

REFUSED BEFORE KIT (SystemExit; the harness is never imported): a missing / unknown controller or command; a network
controller without, or a scripted one with, a stand checkpoint / mask; a malformed / relative / spaced / globbed /
missing checkpoint path; a repeated flag; CLI-env disagreement; leftover ``--sh*`` / ``--stand*`` / ``--sn*`` /
``--p2r*``; any ``P2R_*`` variable; the predecessor's ``SN_STAND_CKPT`` / ``SN_STAND_MASK``; the D5 env rule
(``sn_common.env_reasons``: STAND_OBS exactly 0, no env-level mask / proxy / KRABBY_RESTANCE / stand hold / park, ENV4);
a checkpoint kind that does not pair with the controller / command / mask (A-mask s17 needs mask 1; S1 only as the
stand_net zero-command bridge; amask_old only the A-mask s17 file); ``--sh-command manifest`` without a valid height row.

REFUSED POST-KIT, BEFORE THE FIRST STEP (no stamp; the app still closes): a stand estimator that differs from S1's; a
trainer checkpoint without ``infos.sh_dagger.stand_mask`` == ``--stand-mask``; a wrapper clip that was not S1's 1.0; a
zero command on a manifest height row; a compiled schedule that is not the manifest row.

Nothing here imports torch / numpy / yaml at module level (L6): torch arrives with the harness (after AppLauncher).
"""
from __future__ import annotations

import argparse
import importlib.util
import math
import os
import sys
from pathlib import Path

_THIS = Path(__file__).resolve()
SCRIPTS = _THIS.parent
#: scripts -> 2026-10-13_stand_height -> experiments
EXPERIMENTS = SCRIPTS.parents[1]
SN_LAUNCHER = EXPERIMENTS / "2026-10-11_stand_network" / "scripts" / "sn_stand_net_eval_crab_hex_gait.py"
COMMON_PATH = SCRIPTS / "sh_eval_common.py"


def _load_by_path(path, name: str):
    spec = importlib.util.spec_from_file_location(name, str(path))
    if spec is None or spec.loader is None:  # pragma: no cover
        raise ImportError(f"sh: cannot load {path}")
    mod = importlib.util.module_from_spec(spec)
    old = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    try:
        spec.loader.exec_module(mod)
    finally:
        sys.dont_write_bytecode = old
    return mod


#: the eval contract (run_meta / raw names, the schedule manifest) and, through it, THE core
EC = _load_by_path(COMMON_PATH, "_sh_eval_launcher_common")
CORE = EC.CORE
#: sn_common (the predecessor's input / R / env rules), the instance sh_core uses
SN = CORE.SN
#: the predecessor stand_net launcher, a private instance (its P2L loader, StandNetMode, path rules)
SNL = _load_by_path(SN_LAUNCHER, "_sh_eval_sn_launcher")

VARIANT = EC.VARIANT
FLAGS = {"controller": "--sh-controller", "command": "--sh-command", "restore": "--sh-restore",
         "stand_ckpt": "--stand-ckpt", "stand_mask": "--stand-mask"}
ENV_KEYS = {"controller": "SH_CONTROLLER", "command": "SH_COMMAND", "restore": "SH_RESTORE",
            "stand_ckpt": "SH_STAND_CKPT", "stand_mask": "SH_STAND_MASK"}
STRAY_PREFIXES = ("--sh", "--stand", "--sn", "--p2r")
#: the predecessor launcher's env fallbacks: never read here, so refused (a leftover would look like a setting)
FOREIGN_ENV = ("SN_STAND_CKPT", "SN_STAND_MASK")
_BAD_PATH = SNL._BAD_PATH


# ===================================================================================== argv / environment (pre-Kit)
def _pick(cli, env, flag: str, var: str):
    env = None if env is None or not str(env).strip() else str(env).strip()
    if cli is not None and env is not None and str(cli).strip() != env:
        raise SystemExit(f"sh: {flag} {cli!r} and {var}={env!r} disagree; refused (one source per run)")
    return cli if cli is not None else env


def harness_args(rest: list[str]) -> dict:
    """``--manifest`` / ``--scenario`` / ``--checkpoint`` of the harness argv (read, never removed)."""
    ap = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    for f in ("manifest", "scenario", "checkpoint"):
        ap.add_argument(f"--{f}", dest=f, default=None)
    got, _ = ap.parse_known_args(list(rest))
    return {"manifest": got.manifest, "scenario": got.scenario, "checkpoint": got.checkpoint}


def parse_sh_args(argv: list[str], environ=None) -> tuple[argparse.Namespace, list[str]]:
    """``argv`` WITHOUT the program name -> (the stand_height options, the harness's argv in its original order).
    Every refusal is a SystemExit raised before the harness (and so Kit) is touched."""
    environ = os.environ if environ is None else environ
    argv = list(argv)
    for flag in FLAGS.values():
        n = sum(1 for a in argv if a == flag or a.startswith(flag + "="))
        if n > 1:
            raise SystemExit(f"sh: {flag} given {n} times; refused")
    ap = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    for dest, flag in FLAGS.items():
        ap.add_argument(flag, dest=dest, default=None)
    opts, rest = ap.parse_known_args(argv)
    stray = [a for a in rest if a.startswith(STRAY_PREFIXES)]
    if stray:
        raise SystemExit(f"sh: unrecognised / forbidden argument(s) {stray} (the stand_height variant takes only "
                         f"{tuple(FLAGS.values())}; P2L's --p2r-* flags are set internally)")
    p2r_env = sorted(k for k in environ if str(k).startswith("P2R_"))
    if p2r_env:
        raise SystemExit(f"sh: P2R_* variable(s) {p2r_env} in the environment; refused (P2L's env fallbacks would "
                         "change the mode silently)")
    foreign = [k for k in FOREIGN_ENV if str(environ.get(k) or "").strip()]
    if foreign:
        raise SystemExit(f"sh: {foreign} set in the environment: the stand_net launcher's fallbacks, never read by "
                         f"the stand_height variant ({tuple(ENV_KEYS.values())}); refused")
    v = {d: _pick(getattr(opts, d), environ.get(ENV_KEYS[d]), FLAGS[d], ENV_KEYS[d]) for d in FLAGS}
    ctl, cmd = v["controller"], v["command"]
    if ctl is None or str(ctl).strip() not in EC.CONTROLLERS:
        raise SystemExit(f"sh: --sh-controller must be one of {EC.CONTROLLERS} (or {ENV_KEYS['controller']}), "
                         f"got {ctl!r}")
    if cmd is None or str(cmd).strip() not in EC.COMMANDS:
        raise SystemExit(f"sh: --sh-command must be one of {EC.COMMANDS} (or {ENV_KEYS['command']}), got {cmd!r}")
    rst = "armed" if v["restore"] is None else str(v["restore"]).strip()
    if rst not in EC.RESTORES:
        raise SystemExit(f"sh: --sh-restore must be one of {EC.RESTORES}, got {rst!r}")
    ctl, cmd = str(ctl).strip(), str(cmd).strip()
    net = ctl in EC.NET_CONTROLLERS
    ck, mk = v["stand_ckpt"], v["stand_mask"]
    if net:
        if ck is None:
            raise SystemExit(f"sh: --sh-controller {ctl} needs --stand-ckpt PATH (or {ENV_KEYS['stand_ckpt']})")
        if mk is None:
            raise SystemExit(f"sh: --sh-controller {ctl} needs --stand-mask 0|1 (or {ENV_KEYS['stand_mask']})")
        mk = str(mk).strip()
        if mk not in ("0", "1"):
            raise SystemExit(f"sh: --stand-mask must be 0 or 1, got {mk!r}")
        ck = str(ck)
        if _BAD_PATH.search(ck):
            raise SystemExit(f"sh: --stand-ckpt {ck!r} contains whitespace / glob / shell characters; refused "
                             "(EVAL_EXTRA_ARGS is unquoted)")
        if not Path(ck).is_absolute():
            raise SystemExit(f"sh: --stand-ckpt {ck!r} must be an absolute path (run_arm cds before launching)")
        if not Path(ck).is_file():
            raise SystemExit(f"sh: --stand-ckpt not found: {ck}")
        mk = int(mk)
    elif ck is not None or mk is not None:
        raise SystemExit(f"sh: --sh-controller {ctl} is scripted: --stand-ckpt / --stand-mask are refused")
    reasons = SN.env_reasons(environ)
    if reasons:
        raise SystemExit("sh: environment refused (D5 obs parity / map L10):\n  " + "\n  ".join(reasons))
    hx = harness_args(rest)
    kind = None
    if net:
        s1 = hx["checkpoint"]
        s1_sha = EC.sha256_file(s1) if s1 and Path(s1).is_file() else None
        kind = EC.ckpt_kind(EC.sha256_file(ck), s1_sha)
        why = EC.ckpt_kind_reasons(kind, controller=ctl, command=cmd, stand_mask=mk)
        if why:
            raise SystemExit("sh: stand checkpoint refused:\n  " + "\n  ".join(why))
    row = None
    if cmd == "manifest":
        if not hx["manifest"] or not hx["scenario"]:
            raise SystemExit("sh: --sh-command manifest needs the harness's --manifest and --scenario (the height row)")
        try:
            row = EC.height_row(hx["manifest"], hx["scenario"])
        except (OSError, ValueError) as exc:
            raise SystemExit(f"sh: --sh-command manifest: {exc}") from exc
    return argparse.Namespace(controller=ctl, command=cmd, restore=rst, stand_ckpt=(ck if net else None),
                              stand_mask=(mk if net else None), pre_kind=kind, height_row=row,
                              harness_manifest=hx["manifest"], harness_scenario=hx["scenario"]), rest


# ===================================================================================== the mode
class _HeightMixin:
    """The stand_height step on top of P2L's mode machinery (``P2RMode`` for scripted controllers, SNL's
    ``StandNetMode`` for network controllers). ``SH`` / ``INNER`` are bound by :func:`make_mode_class`."""

    SH = None
    INNER = None

    def __init__(self, variant, *args, **kwargs):
        if variant != self.INNER:
            raise ValueError(f"sh: the stand_height {self.SH.controller} mode runs on P2L's {self.INNER!r} machinery, "
                             f"got {variant!r}")
        super().__init__(variant, *args, **kwargs)
        sh = self.SH
        self.variant = VARIANT
        self.controller, self.command, self.restore = sh.controller, sh.command, sh.restore
        self.net = sh.controller in EC.NET_CONTROLLERS
        self.old_path = sh.controller in EC.OLD_PATH_CONTROLLERS
        self.height_row = sh.height_row
        if not self.net:
            self.stand_mask = None
            self.stand_ckpt = None
            self.estimator_equals_s1 = None
            self.trained_stand_mask = None
            self.stand_ckpt_kind = None
            self.s1_runner = None
            self.sn_ctx = None
        self.trained_infos_keys = None
        self.w15 = None
        self.sh_bound = False
        self.ctl = None
        self._h_np = None
        self._h_dev = None
        self._compiled_dt = None
        self._ret = None
        self._exec = None
        self.exec_swaps = 0
        self.clip_before = None
        self.clip_env = None
        self.acc = None
        self.events_all = {k: 0 for k in ("engaged", "completed", "timed_out", "merged", "reset_cut")}
        self.ctl_record = None
        self.passes = 0
        self.h_meas_record = None
        self.hold_labels_seen = None

    # ---------------------------------------------------------------- wiring
    def trained_arm(self) -> tuple:
        """``(trained_stand_mask, stand_ckpt_kind)`` (StandNetMode.bind_s1 calls it after the estimator identity):
        the kind from the hashes P2L recorded (A-mask s17 pinned sha / S1's sha), else a stand-height trainer
        checkpoint whose ``infos.sh_dagger.stand_mask`` must be an int equal to ``--stand-mask``. Every refusal is a
        SystemExit before the first step. Post-Kit; ``map_location="cpu"``, no RNG draw."""
        import torch  # noqa: PLC0415

        ctx = self.sn_ctx
        if ctx is None:
            raise RuntimeError("sh: P2L's context was never handed to the stand_height mode (make_factory not wrapped)")
        s1_sha = getattr(ctx, "s1_sha", None)
        sp = (getattr(ctx, "specialist", None) or {}).get("sha256")
        if not s1_sha or not sp:
            raise RuntimeError("sh: S1's / the stand checkpoint's sha256 was never recorded by P2L")
        kind = EC.ckpt_kind(sp, s1_sha)
        why = EC.ckpt_kind_reasons(kind, controller=self.controller, command=self.command, stand_mask=self.stand_mask)
        if why:
            raise SystemExit("sh: stand checkpoint refused before the first step:\n  " + "\n  ".join(why))
        if kind == EC.CKPT_KIND_AMASK:
            return 1, kind
        if kind == EC.CKPT_KIND_S1:
            return None, kind
        if not self.stand_ckpt:
            raise RuntimeError("sh: the stand checkpoint path was never handed to the stand_height mode")
        ck = torch.load(str(self.stand_ckpt), map_location="cpu", weights_only=False)
        infos = ck.get("infos") if isinstance(ck, dict) else None
        sd = infos.get(EC.CKPT_INFOS_KEY) if isinstance(infos, dict) else None
        if not isinstance(sd, dict):
            raise SystemExit(f"sh: the stand checkpoint {self.stand_ckpt} carries no infos.{EC.CKPT_INFOS_KEY} (not a "
                             "stand-height trainer checkpoint), is not the A-mask s17 file and is not S1; refused "
                             "before the first step")
        tm = sd.get("stand_mask")
        if type(tm) is not int or tm not in (0, 1):
            raise SystemExit(f"sh: infos.{EC.CKPT_INFOS_KEY}.stand_mask is {tm!r}, not an int 0 / 1; refused before "
                             "the first step")
        if tm != self.stand_mask:
            raise SystemExit(f"sh: --stand-mask {self.stand_mask} but the checkpoint was trained with stand_mask {tm}; "
                             "refused before the first step (an arm is evaluated with the mask it was trained with)")
        self.trained_infos_keys = sorted(str(k) for k in sd)
        return int(tm), EC.CKPT_KIND_TRAINED

    def bind_s1(self, s1_policy, s1_module) -> None:
        """S1 bound as P2L does (network controllers: + the estimator identity and :meth:`trained_arm`, SNL); then the
        A-mask s17 warm start's W[:, 15] zeroing (sh_core.warm_start_state, sha pinned) and R's D3 constants."""
        super().bind_s1(s1_policy, s1_module)
        if self.net and self.stand_ckpt_kind == EC.CKPT_KIND_AMASK:
            self.w15 = self._zero_w15()
        self.r_params = CORE.r_params()
        self.sh_bound = True

    def _zero_w15(self) -> dict:
        """Load sh_core's warm start (the A-mask s17 file, sha pinned, ``W[:, 15]`` zeroed) into the stand policy
        module P2L built (strict; no RNG, before the first step): bitwise the trainer's warm start."""
        mod = getattr(self.stand[0], "__self__", None)
        if mod is None or not hasattr(mod, "load_state_dict") or not hasattr(mod, "named_parameters"):
            raise SystemExit("sh: cannot reach the stand policy module to zero W[:, 15] (the inference policy is not a "
                             "bound act_inference); refused before the first step")
        st = CORE.warm_start_state(self.stand_ckpt)
        mod.load_state_dict(st["model_state_dict"], strict=True)
        w = dict(mod.named_parameters()).get(CORE.WARM_ZERO_KEY)
        if w is None:
            raise SystemExit(f"sh: the stand policy has no {CORE.WARM_ZERO_KEY}; refused")
        after = float(w.detach()[:, CORE.HEIGHT_COL].abs().max())
        if after != 0.0:
            raise RuntimeError(f"sh: W[:, {CORE.HEIGHT_COL}] is {after} after the zeroed warm-start load")
        ok, why = SN.state_dicts_equal(st["estimator_state_dict"], self.stand[1].state_dict())
        if not ok:
            raise SystemExit(f"sh: the A-mask s17 estimator in the file differs from the loaded one ({why}); refused")
        ws = st["warm_start"]
        return {"zeroed": ws["zeroed"], "w15_norm_before": float(ws["w15_norm_before"]), "w15_max_abs_after": after,
                "source_sha256": ws["sha256"], "via": "sh_core.warm_start_state + load_state_dict(strict)"}

    # ---------------------------------------------------------------- passes
    def begin_pass(self, compiled) -> None:
        super().begin_pass(compiled)            # closes the previous pass (its controller counts folded below)
        self.passes += 1
        self.ctl = None
        self._h_dev = None
        self._compiled_dt = getattr(compiled, "dt", None)
        labels = list(compiled.hold_labels)
        if self.hold_labels_seen is not None and labels != self.hold_labels_seen:
            raise RuntimeError(f"sh: pass hold labels {labels} differ from the first pass's {self.hold_labels_seen}")
        self.hold_labels_seen = labels
        if self.command == "manifest":
            want = list(self.height_row["labels"])
            if labels != want:
                raise RuntimeError(f"sh: the compiled schedule's hold labels {labels} are not the height row's {want} "
                                   f"({self.height_row['manifest']} {self.height_row['scenario']})")
            self._h_np = EC.height_table(compiled.segment_id, self.height_row["hold_heights"])
        else:
            self._h_np = None

    def _close_pass(self) -> None:
        if self.ctl is not None:
            for k, v in self.ctl.counts().items():
                self.events_all[k] += int(v)
            self.ctl = None
        super()._close_pass()

    def _start_pass_state(self, n: int, dev) -> None:
        import torch  # noqa: PLC0415

        dt = getattr(self.unwrapped, "step_dt", None)
        if dt is None:
            raise RuntimeError("sh: the env has no step_dt (the restore's 0.2-s settle and timeout need it)")
        dt = float(dt)
        if self._compiled_dt is not None and abs(float(self._compiled_dt) - dt) > 1e-12:
            raise RuntimeError(f"sh: compiled schedule dt {self._compiled_dt} != env step_dt {dt}")
        engage = CORE.RESTORE_ENGAGE_M if self.restore == "armed" else math.inf
        self.ctl = CORE.HeightController(n, dev, dt=dt, engage_m=engage)
        if self._h_np is not None:
            self._h_dev = torch.as_tensor(self._h_np, dtype=torch.float64, device=dev)
        self.ctl.reset(None, "walk", h_cmd=(None if self._h_dev is None else self._h_dev[0]))
        rec = dict(self.ctl.record())
        rec["restore"] = self.restore
        if self.restore == "off":
            rec["engage_m"] = None                                         # inf: the restore never engages
        self.ctl_record = rec
        if self.acc is None:
            zi = lambda: torch.zeros((), dtype=torch.int64, device=dev)  # noqa: E731
            zf = lambda: torch.zeros((), dtype=torch.float32, device=dev)  # noqa: E731
            self.acc = {"stand_rows_alive": zi(), "stand_rows_differ_from_s1": zi(), "hip_beyond_1_rows": zi(),
                        "hip_below_minus_1_rows": zi(), "restore_rows_alive": zi(), "engaged": zi(), "completed": zi(),
                        "timed_out": zi(), "merged": zi(), "leave_off_height": zi(), "s1_exec_abs_max": zf(),
                        "stand_rod_exec_abs_max": zf(), "stand_cam_exec_abs_max": zf(), "rh_minus_q_abs_max": zf()}
            if self.net:
                # REPORTED (review 2026-10-13, accepted): the stand net's estimate (S1's frozen estimator on the
                # unmasked prop, whose last-action block holds q + residual beyond +-1 on stand rows off 1.064)
                w = max(1, int(self.p1) - int(self.p0))
                self.acc.update({"est_rows_off_height": zi(),
                                 "est_abs_max_stand_off_height": torch.zeros(w, dtype=torch.float32, device=dev),
                                 "est_abs_max_s1_rows": torch.zeros(w, dtype=torch.float32, device=dev)})

    def _alloc(self, T: int, n: int, dev) -> dict:
        import torch  # noqa: PLC0415

        rec = {k: torch.zeros(T, n, dtype=torch.bool, device=dev) for k in EC.RAW_BOOL_KEYS}
        rec.update({k: torch.full((T, n), CORE.H_SETTLED, dtype=torch.float64, device=dev) for k in EC.RAW_H_KEYS})
        rec.update({k: torch.zeros(T, n, CORE.ACTION_DIM, dtype=torch.float32, device=dev) for k in EC.RAW_ACT_KEYS})
        rec[EC.RAW_RH_BRANCH] = torch.full((T, n, CORE.NUM_LEGS), -1, dtype=torch.int8, device=dev)
        return rec

    # ---------------------------------------------------------------- the step
    def __call__(self, obs, hist_encoding=True, **kwargs):
        import torch  # noqa: PLC0415

        # -- P2L:563-588 semantics
        if kwargs:
            raise RuntimeError(f"p2r: unexpected policy kwargs {sorted(kwargs)} (student / canon paths are refused)")
        if self.s1_policy is None:
            raise RuntimeError("p2r: S1's policy was never bound")
        if not self.sh_bound:
            raise RuntimeError("sh: bind_s1 never completed (no estimator / checkpoint / R check)")
        c = self._cur
        if c is None:
            raise SystemExit("p2r: no world-frame schedule was installed before the first step (omni world rows only)")
        if self._ret is not None:
            raise RuntimeError("sh: the previous step's action never reached env.step (the executed-action swap)")
        n = int(obs.shape[0])
        dev = obs.device
        if c["stand"] is None:
            c["stand"] = torch.zeros(c["T"], n, dtype=torch.bool, device=dev)
            c["rst"] = self._alloc(c["T"], n, dev)
        if self.alive is None:
            self.alive = torch.ones(n, dtype=torch.bool, device=dev)
        t = c["t"]
        if t >= c["T"]:
            raise RuntimeError(f"p2r: step {t} past the pass's {c['T']} planned steps")
        if self.ctl is None:
            self._start_pass_state(n, dev)
        ctl = self.ctl
        h_cmd = None if self._h_dev is None else self._h_dev[t]

        # -- the height state: slot, resets, the ONE per-step controller order (sh_core.HeightController)
        slot = self.rederive_on_reset()
        reset = self.unwrapped.episode_length_buf.reshape(-1) == 0
        cut = ctl.reset(reset, "walk", h_cmd=h_cmd)
        leave_off = ctl.prev_slot & ~slot & ((ctl.h_lim - CORE.H_SETTLED).abs() > CORE.RESTORE_ENGAGE_M)
        hs = ctl.step(slot, h_cmd, obs)
        stand = hs.stand_eff

        # -- S1 exactly as the harness would; the stand controller; the router
        a_s1 = self.s1_policy(obs, hist_encoding=hist_encoding)
        s1c = CORE.s1_clamp(a_s1)
        q_lim = CORE.q18(hs.h_lim, dtype=a_s1.dtype, device=dev)
        rh_t, rh_b = CORE.rh_target(obs, hs.h_lim)
        a_net = None
        if self.net:
            h_x = CORE.H_SETTLED if self.old_path else hs.h_lim
            st_policy, st_est = self.stand
            x = CORE.stand_net_input(obs, st_est, p0=self.p0, p1=self.p1, num_prop=self.num_prop,
                                     mask=bool(self.stand_mask), h_cmd_limited=h_x, restoring=hs.restoring)
            a_net = st_policy(x, hist_encoding=True).to(a_s1.dtype)       # the D1 forward path (L11)
            est = x[:, self.p0:self.p1].abs().float()
            off = stand & self.alive & (hs.h_lim != CORE.H_SETTLED) if not self.old_path else None
            a_stand, q = CORE.stand_action(a_net, h_x)
            pre = torch.where(q != 0, a_net + q, a_net)
        elif self.controller == "r_h":
            a_stand, q = rh_t.to(a_s1.dtype), q_lim
            pre = a_stand
        elif self.controller == "r_old":
            tgt, _ = SN.r_label(obs)
            a_stand, q = tgt.to(a_s1.dtype), torch.zeros_like(a_s1)
            pre = a_stand
        else:                                                              # posture_only
            a_stand, q = CORE.stand_action(torch.zeros_like(a_s1), hs.h_lim)
            pre = a_stand
        actions = CORE.route(stand, a_stand, a_s1)                         # EXECUTED (swapped in at env.step)
        ret = torch.where(stand.unsqueeze(1), pre.to(a_s1.dtype), a_s1)    # RECORDED (the harness's pre-clip key)

        # -- record (pure; no RNG)
        sm = stand.unsqueeze(1)
        rec = c["rst"]
        for key, v in ((EC.RAW_STAND_SLOT, slot), (EC.RAW_RESTORING, hs.restoring), (EC.RAW_ENGAGED, hs.engaged),
                       (EC.RAW_COMPLETED, hs.completed), (EC.RAW_TIMED_OUT, hs.timed_out),
                       (EC.RAW_MERGED, hs.merged), (EC.RAW_RESET_CUT, cut), (EC.RAW_LEAVE_OFF, leave_off)):
            rec[key][t] = v
        if h_cmd is not None:
            rec[EC.RAW_H_CMD][t] = h_cmd
        rec[EC.RAW_H_LIM][t] = hs.h_lim
        z = torch.zeros(n, CORE.ACTION_DIM, dtype=torch.float32, device=dev)
        rec[EC.RAW_Q][t] = torch.where(sm, q.float(), z)
        if a_net is not None:
            rec[EC.RAW_A_NET][t] = torch.where(sm, a_net.float(), z)
        rec[EC.RAW_STAND_ACTION][t] = torch.where(sm, a_stand.float(), z)
        rec[EC.RAW_S1_ACTION][t] = torch.where(sm, s1c.float(), z)
        rec[EC.RAW_RH_TARGET][t] = torch.where(sm, rh_t.float(), z)
        rec[EC.RAW_RH_BRANCH][t] = torch.where(sm, rh_b.to(torch.int8), torch.full_like(rec[EC.RAW_RH_BRANCH][t], -1))

        # -- run statistics (device accumulators: no host sync per step)
        acc = self.acc
        live = stand & self.alive
        s1_live = ~stand & self.alive
        lm = live.unsqueeze(1)
        hips = a_stand[:, CORE.HIP_ACT]
        acc["stand_rows_alive"] += live.sum()
        acc["restore_rows_alive"] += (hs.restoring & self.alive).sum()
        acc["stand_rows_differ_from_s1"] += (live & (a_stand.to(s1c.dtype) != s1c).any(dim=1)).sum()
        acc["hip_beyond_1_rows"] += (live & (hips.abs() > 1.0).any(dim=1)).sum()
        acc["hip_below_minus_1_rows"] += (live & (hips < -1.0).any(dim=1)).sum()
        for k in ("engaged", "completed", "timed_out", "merged"):
            acc[k] += (getattr(hs, k) & self.alive).sum()
        acc["leave_off_height"] += (leave_off & self.alive).sum()
        za = torch.zeros_like(actions, dtype=torch.float32)
        acc["s1_exec_abs_max"] = torch.maximum(acc["s1_exec_abs_max"], torch.where(
            s1_live.unsqueeze(1), actions.abs().float(), za).amax())
        acc["stand_rod_exec_abs_max"] = torch.maximum(acc["stand_rod_exec_abs_max"], torch.where(
            lm, actions.abs().float(), za)[:, CORE.HIP_ACT.start:].amax())
        acc["stand_cam_exec_abs_max"] = torch.maximum(acc["stand_cam_exec_abs_max"], torch.where(
            lm, actions.abs().float(), za)[:, CORE.CAM_ACT].amax())
        acc["rh_minus_q_abs_max"] = torch.maximum(acc["rh_minus_q_abs_max"], torch.where(
            lm, (rh_t - q_lim).abs().float(), za).amax())
        if self.net:
            ze = torch.zeros_like(est)
            if off is not None:
                acc["est_rows_off_height"] += off.sum()
                acc["est_abs_max_stand_off_height"] = torch.maximum(acc["est_abs_max_stand_off_height"], torch.where(
                    off.unsqueeze(1), est, ze).amax(dim=0))
            acc["est_abs_max_s1_rows"] = torch.maximum(acc["est_abs_max_s1_rows"], torch.where(
                s1_live.unsqueeze(1), est, ze).amax(dim=0))

        c["stand"][t] = stand                                              # p2r_stand_mode: who acted
        c["t"] = t + 1
        self.calls += 1
        self._ret, self._exec = ret, actions
        return ret

    def executed_for(self, actions, env):
        """The outermost ``env.step`` hook: ``actions`` must be the tensor this mode just returned; the executed router
        output goes to the env instead (the wrapper clip must still be the realised 4.8)."""
        if self._ret is None:
            raise RuntimeError("sh: env.step without a stand_height action (the harness bypassed the policy)")
        if actions is not self._ret:
            raise RuntimeError("sh: env.step got a tensor the stand_height mode did not return; refusing to execute it "
                               "(S1 rows would run unclamped at the 4.8 wrapper clip)")
        clip = getattr(env, "clip_actions", None)
        if clip is None or float(clip) != CORE.HARNESS_CLIP:
            raise RuntimeError(f"sh: the wrapper clip is {clip!r}, not the realised {CORE.HARNESS_CLIP}")
        ex = self._exec
        self._ret = self._exec = None
        self.exec_swaps += 1
        return ex

    # ---------------------------------------------------------------- record
    def counts(self) -> dict:
        out = {}
        for k, v in (self.acc or {}).items():
            if v.dim() > 0:
                out[k] = [float(x) for x in v.detach().cpu().tolist()]
            else:
                out[k] = float(v.item()) if v.is_floating_point() else int(v.item())
        return out

    def block_record(self, *, p2r_file) -> dict:
        stats = self.counts()
        rp = self.r_params
        clip_env = self.clip_env
        return {
            "wrapper": str(_THIS), "wrapper_sha256": EC.sha256_file(_THIS),
            "common": str(COMMON_PATH), "common_sha256": EC.sha256_file(COMMON_PATH),
            "sn_launcher": str(SN_LAUNCHER), "sn_launcher_sha256": EC.sha256_file(SN_LAUNCHER),
            "p2r_launcher": str(p2r_file), "p2r_launcher_sha256": EC.sha256_file(p2r_file),
            "core": CORE.provenance(),
            "control": self.controller, "command": self.command, "restore": self.restore,
            "inner_mode": self.INNER, "old_path": bool(self.old_path), "executor": EC.EXECUTORS[self.controller],
            "stand_ckpt": self.stand_ckpt, "stand_ckpt_kind": self.stand_ckpt_kind,
            "stand_mask": (int(self.stand_mask) if self.net else None),
            "trained_stand_mask": self.trained_stand_mask, "trained_infos_keys": self.trained_infos_keys,
            "estimator_equals_s1": self.estimator_equals_s1, "w15": self.w15,
            "forward": ("act_inference(sh_core.stand_net_input(obs, stand_estimator, mask, h_cmd_limited, restoring), "
                        "hist_encoding=True)" if self.net else None),
            "hist_encoding": True, "mask_fn": SN.MASK_FN, "mask_rows": SN.MASK_ROWS, "mask_after_estimate": True,
            "clip": {"wrapper_clip_before": self.clip_before,
                     "wrapper_clip_realised": (float(clip_env.clip_actions) if clip_env is not None else None),
                     **CORE.clip_record()},
            "recorded_actions": EC.RECORDED_ACTIONS,
            "hold_heights": (list(self.height_row["hold_heights"]) if self.height_row is not None
                             else EC.zero_hold_heights(self.hold_labels_seen or [])),
            "height_row": (None if self.height_row is None else dict(self.height_row)),
            "hold_labels": self.hold_labels_seen,
            "controller": self.ctl_record,
            "events_alive": {k: stats.get(k) for k in ("engaged", "completed", "timed_out", "merged",
                                                       "leave_off_height")},
            "events_all_rows": dict(self.events_all),
            "stats": stats,
            "h_meas": self.h_meas_record,
            "r_knob": SN.RESTANCE_D3, "r_params": (rp.to_dict() if rp is not None else None),
            "exec_swaps": self.exec_swaps, "calls": self.calls, "passes": self.passes,
            "raw_keys": list(EC.RAW_KEYS),
        }


def make_mode_class(p2r, opts: argparse.Namespace):
    """``StandHeightMode`` on the PRIVATE P2L instance ``p2r``: :class:`_HeightMixin` over SNL's ``StandNetMode`` (network
    controllers; P2L st2) or P2L's ``P2RMode`` (scripted; P2L hold). ``__call__`` is overridden in full (map L1)."""
    if getattr(p2r.P2RMode, "_sh_stand_height", False) or getattr(p2r.P2RMode, "_sn_stand_net", False):
        raise RuntimeError("sh: P2RMode on this module instance is already a patched subclass")
    net = opts.controller in EC.NET_CONTROLLERS
    base = SNL.make_mode_class(p2r, int(opts.stand_mask), opts.stand_ckpt) if net else p2r.P2RMode
    snap = argparse.Namespace(controller=opts.controller, command=opts.command, restore=opts.restore,
                              height_row=opts.height_row)

    class StandHeightMode(_HeightMixin, base):
        """The harness's ``policy`` for the stand_height variant (see the module docstring)."""

        _sh_stand_height = True
        SH = snap
        INNER = EC.INNER_MODE[opts.controller]

    return StandHeightMode


# ===================================================================================== install (private P2L instance)
def install(p2r, opts: argparse.Namespace):
    """Rebind ``P2RMode`` / ``make_factory`` / ``p2r_block`` / ``make_write_run`` on the PRIVATE P2L instance ``p2r``
    (each is resolved at call time inside P2L). Returns the mode class. Refuses a second install."""
    if getattr(p2r, "_sh_installed", False) or getattr(p2r, "_sn_installed", False):
        raise RuntimeError("sh: patches are already installed on this P2L instance")
    mode_cls = make_mode_class(p2r, opts)
    net = opts.controller in EC.NET_CONTROLLERS
    orig_make_factory, orig_block, orig_mwr = p2r.make_factory, p2r.p2r_block, p2r.make_write_run
    p2r_file = Path(p2r.__file__).resolve()

    def make_factory(ctx, orig_factory):
        inner = orig_make_factory(ctx, orig_factory)

        def factory(env, train_cfg, log_dir=None, device="cpu"):
            runner = inner(env, train_cfg, log_dir=log_dir, device=device)
            mode = ctx.mode
            if type(mode) is not mode_cls:
                raise RuntimeError(f"sh: the mode wrapper is {type(mode).__name__}, not StandHeightMode (map L1)")
            if ctx.opts.mode != mode_cls.INNER or (ctx.specialist is not None) != net:
                raise RuntimeError(f"sh: P2L ran mode {ctx.opts.mode!r} (specialist {ctx.specialist is not None}); "
                                   f"the {opts.controller} controller needs {mode_cls.INNER!r}")
            mode.s1_runner = runner
            mode.sn_ctx = ctx
            resolved = getattr(ctx.scenario, "id", None)
            scen_ids = sorted({s for s in (resolved, opts.harness_scenario) if s})
            if opts.command == "zero" and any(EC.has_height_row(opts.harness_manifest, s) for s in scen_ids):
                raise SystemExit(f"sh: {scen_ids} is a height row of {opts.harness_manifest}: it runs with "
                                 "--sh-command manifest; refused before the first step")
            if opts.command == "manifest" and resolved != opts.height_row["scenario"]:
                raise SystemExit(f"sh: the harness resolved scenario {resolved!r}, not the height row "
                                 f"{opts.height_row['scenario']!r}; refused before the first step")
            before = getattr(env, "clip_actions", None)
            if before is None or isinstance(before, bool) or float(before) != EC.S1_TRAIN_CLIP:
                raise SystemExit(f"sh: the wrapper clip_actions is {before!r}, not S1's {EC.S1_TRAIN_CLIP}; refused "
                                 "(the runtime raise to 4.8 is defined from S1's clip)")
            env.clip_actions = CORE.HARNESS_CLIP                          # runtime only (PLAN H1)
            mode.clip_before = float(before)
            mode.clip_env = env
            p2l_step = env.step

            def step(actions):
                return p2l_step(mode.executed_for(actions, env))

            env.step = step
            return runner

        return factory

    def make_write_run(ctx, orig_write_run):
        inner = orig_mwr(ctx, orig_write_run)

        def write_run(run_dir, *args, **kwargs):
            raw = kwargs.get("raw")
            mode = ctx.mode
            if raw is not None and type(mode) is mode_cls:
                hm, rec = EC.h_meas_series(raw, kwargs.get("run_meta"))
                mode.h_meas_record = rec
                kwargs["raw"] = {**raw, EC.RAW_H_MEAS: hm}
            return inner(run_dir, *args, **kwargs)

        return write_run

    def p2r_block(ctx, stats, *, bank_rel, rst_stats=None, stand_bank_rel=None):
        b = orig_block(ctx, stats, bank_rel=bank_rel, rst_stats=rst_stats, stand_bank_rel=stand_bank_rel)
        mode = ctx.mode
        if b.get("variant") != VARIANT or type(mode) is not mode_cls:
            raise RuntimeError(f"sh: p2r_mode.variant {b.get('variant')!r} is not {VARIANT!r} (map L1)")
        if not mode.sh_bound:
            raise RuntimeError("sh: bind_s1 never completed; refusing to write the stamp")
        if net and (mode.estimator_equals_s1 is not True or mode.stand_ckpt_kind not in EC.CKPT_KINDS):
            raise RuntimeError("sh: the estimator identity / checkpoint-kind check never ran; refusing to write")
        if mode.exec_swaps != mode.calls or mode._ret is not None:
            raise RuntimeError(f"sh: {mode.exec_swaps} executed-action swaps for {mode.calls} policy calls")
        if net:
            b[EC.MASK_KEY] = int(mode.stand_mask)                       # an INT, never a bool (map L4)
        b[EC.BLOCK_KEY] = mode.block_record(p2r_file=p2r_file)
        have = list(b.get("raw_series") or [])
        b["raw_series"] = have + [k for k in EC.RAW_KEYS if k not in have]
        blk = b[EC.BLOCK_KEY]
        print(f"[SH] stand_height controller={mode.controller} command={mode.command} restore={mode.restore} "
              f"ckpt_kind={mode.stand_ckpt_kind} clip {blk['clip']['wrapper_clip_before']}->"
              f"{blk['clip']['wrapper_clip_realised']} events={blk['events_alive']} "
              f"stand_rows_alive={blk['stats'].get('stand_rows_alive')} "
              f"hip_beyond_1_rows={blk['stats'].get('hip_beyond_1_rows')}", flush=True)
        return b

    p2r.P2RMode = mode_cls
    p2r.make_factory = make_factory
    p2r.p2r_block = p2r_block
    p2r.make_write_run = make_write_run
    p2r._sh_installed = True
    return mode_cls


def main(argv: list[str] | None = None, *, harness: Path | str | None = None, environ=None) -> None:
    argv = list(sys.argv if argv is None else argv)
    environ = os.environ if environ is None else environ
    opts, rest = parse_sh_args(argv[1:], environ)                     # every pre-Kit refusal
    p2r = SNL.load_p2r()
    mode_cls = install(p2r, opts)
    net = opts.controller in EC.NET_CONTROLLERS
    print(f"[SH] variant={VARIANT} controller={opts.controller} command={opts.command} restore={opts.restore} "
          f"stand_ckpt={opts.stand_ckpt} stand_mask={opts.stand_mask} kind(pre)={opts.pre_kind} "
          f"row={(opts.height_row or {}).get('scenario')} inner={mode_cls.INNER} r_knob={SN.RESTANCE_D3}", flush=True)
    inner = ["--p2r-mode", mode_cls.INNER] + (["--p2r-stand-ckpt", opts.stand_ckpt] if net else [])
    p2r.main([argv[0]] + inner + rest, harness=(p2r.HARNESS if harness is None else harness), environ=environ)


if __name__ == "__main__":
    main()
