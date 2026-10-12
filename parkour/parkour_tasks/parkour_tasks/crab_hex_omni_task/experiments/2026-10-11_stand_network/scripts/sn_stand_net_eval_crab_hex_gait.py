#!/usr/bin/env python3
"""Stand-network campaign, PLAN.md step 3 -- the ``stand_net`` launcher variant (EVAL ONLY; CHARTER section 3).

The name ends in ``eval_crab_hex_gait.py`` on purpose: ``run_arm.sh``'s ``drivers_idle`` finds running evals by that
suffix (one Isaac process at a time). Its absolute path contains no spaces (``EVAL_EXTRA_ARGS`` is unquoted).

WHAT IT DOES. The phase-2 controller-level stand-mode launcher ``p2r_mode_eval_crab_hex_gait.py`` (P2L) is loaded BY
PATH into a PRIVATE module instance (never edited, never imported by name -- map risk L7) and run as its ``st2``
machinery: S1 (``--checkpoint``) is the harness's runner, the stand checkpoint (``--stand-ckpt``) is P2L's extra
runner on the stand-in env with the RNG saved / restored around it, the command term's slot latch is armed at
runtime, entry / exit on the latched stand slot, the turning bit re-derived on reset (PLAN D5). Three things are
rebound on the private instance:

* ``P2RMode`` -> :class:`StandNetMode` (built by :func:`make_mode_class`), which overrides ``__call__`` IN FULL (map
  risk L1: the parent's ``__call__`` would route an unknown variant into its probe branch, S1 on every row). Per step:
  S1 acts on the harness's obs exactly as the harness would; the stand net acts on
  ``sn_common.stand_net_input(obs, stand_estimator, p0, p1, num_prop, mask)`` -- its own estimate on the UNMASKED
  prop, written into a NEW tensor, then iff ``--stand-mask 1`` the actor-side mask (``stand_mask_obs``, rows=None) on
  that tensor only -- with ``hist_encoding=True`` (L11); ``sn_common.route_actions`` takes the stand net's action on
  the latched stand rows and S1's elsewhere. The harness's obs is never written (S1 never sees a masked frame, L2).
  Also recorded per step (raw series, PLAN "observe pass" / OPEN-9 and the swap-live proof): ``sn_stand_action`` /
  ``sn_s1_action`` (both networks' actions on stand rows, 0 elsewhere), ``sn_r_target`` / ``sn_r_branch`` (R with the
  D3 constants on stand rows; pure, no RNG). ``bind_s1`` (the harness's ``get_inference_policy``, after S1's
  ``runner.load`` and before the first step) REFUSES a stand checkpoint whose estimator differs from S1's (D1 / L5),
  then ties the checkpoint to the mask it is evaluated with (:meth:`StandNetMode.trained_arm`, review 2026-10-12): a
  trainer checkpoint (``infos.sn_dagger`` present, ``sn_dagger.py``) must carry ``stand_mask`` == ``--stand-mask``
  (an A-mask net is never evaluated unmasked, nor an A-full net masked); a checkpoint WITHOUT ``infos.sn_dagger``
  must be S1 itself (specialist sha == S1's sha: the identity bridge / the swap-live proof), any other file is
  refused. Both run post-Kit (they need torch), before any episode; a refusal writes no stamp. It also validates R's
  D3 constants.
* ``make_factory`` -> a wrapper handing the mode S1's runner (for the estimator identity check).
* ``p2r_block`` -> a wrapper that writes ``run_meta.p2r_mode.variant = "stand_net"`` (from the mode), ``stand_mask``
  as an INT 0 / 1 (L4; ``sn_calib.stand_net_reasons`` treats a bool as malformed), and the ``stand_net`` block:
  wrapper / sn_common / R-module shas (L8: P2L's ``launcher_sha256`` names only the phase-2 file), the mask function
  and rows, ``estimator_equals_s1``, ``trained_stand_mask`` (the checkpoint's ``infos.sn_dagger.stand_mask``; None
  for S1) and ``stand_ckpt_kind`` (``sn_dagger`` / ``s1_identity``), R's knob string and params, the forward rule,
  the alive stand-row count and how many of those rows the stand net's action differs from S1's. ``specialist {path,
  sha256}`` is P2L's st2 record.

CLI (``run_arm.sh --row-eval`` with ``P2_EVAL=<this file>``; every harness flag passes through in order):

    P2_GUARD_REF=REFH_S1_s18 P2_EVAL=<abs>/sn_stand_net_eval_crab_hex_gait.py \\
    EVAL_EXTRA_ARGS='--stand-ckpt <abs path> --stand-mask <0|1>' \\
    bash <P2>/scripts/run_arm.sh --row-eval <NAME> <S1 ckpt> <REFS_S1_s18 suite es1 env stamp> <manifest> <scenario> <es>

    --stand-ckpt PATH   the stand network checkpoint (rsl_rl format; its estimator must equal S1's). Absolute, an
                        existing file, no whitespace / glob / shell characters (L9). Env fallback SN_STAND_CKPT.
    --stand-mask 0|1    the actor-side stand mask (A-full 0 / A-mask 1). Env fallback SN_STAND_MASK. (The fallbacks
                        are not KRABBY_* variables, so run_meta.env_vars is unchanged; run_arm's lr_eval passes no
                        EVAL_EXTRA_ARGS.) A CLI value and a different env value are refused, as is a repeated flag.

REFUSED BEFORE KIT (SystemExit; the harness is never imported): a missing / malformed stand ckpt or mask; any leftover
``--stand*`` / ``--sn*`` / ``--p2r*`` argument; ANY ``P2R_*`` variable in the environment (P2L's fallbacks would inject
a second mode; ``P2R_BANK_MAX`` would be accepted silently -- L9); ``KRABBY_OMNI_STAND_OBS`` != ``"0"`` or an env-level
``KRABBY_OMNI_STAND_MASK`` / ``_STAND_PROXY`` / ``KRABBY_RESTANCE`` / stand hold / park, or the ENV4 values not REFS's
(D5 / L10; ``sn_common.env_reasons``). P2L then refuses its own set (``--zero-actions``, ``--sample-actions``,
``--canon``, the student path, ``--no-save-raw``, body-frame rows).

REFUSED POST-KIT, BEFORE THE FIRST STEP (``bind_s1``; SystemExit, no stamp, the app still closes): a stand estimator
that differs from S1's; a trainer checkpoint whose ``infos.sn_dagger.stand_mask`` is not ``--stand-mask`` (or is not
an int 0 / 1); a checkpoint without ``infos.sn_dagger`` that is not S1 itself.

Nothing here imports torch at module level (L6): this file, ``sn_common`` and P2L load torch-free; torch arrives with
the harness (after AppLauncher) and every torch use below is inside a function.
"""
from __future__ import annotations

import argparse
import importlib.util
import os
import re
import sys
from pathlib import Path

_THIS = Path(__file__).resolve()
SCRIPTS = _THIS.parent
#: scripts -> 2026-10-11_stand_network -> experiments
EXPERIMENTS = SCRIPTS.parents[1]
P2R_LAUNCHER = EXPERIMENTS / "2026-10-07_omni_phase2" / "scripts" / "p2r_mode_eval_crab_hex_gait.py"
COMMON_PATH = SCRIPTS / "sn_common.py"


def _load_common():
    spec = importlib.util.spec_from_file_location("_sn_common", str(COMMON_PATH))
    mod = importlib.util.module_from_spec(spec)
    old = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    try:
        spec.loader.exec_module(mod)
    finally:
        sys.dont_write_bytecode = old
    return mod


#: the ONE shared stand-net input module (torch-free at import)
SN = _load_common()

VARIANT = SN.VARIANT
#: the mode P2L runs internally (its st2 machinery: the extra runner, ``ctx.specialist``)
INNER_MODE = "st2"
ENV_CKPT, ENV_MASK = "SN_STAND_CKPT", "SN_STAND_MASK"
FLAGS = ("--stand-ckpt", "--stand-mask")
STRAY_PREFIXES = ("--stand", "--sn", "--p2r")
#: whitespace, glob and shell metacharacters: EVAL_EXTRA_ARGS is unquoted in run_arm.sh (map risk L9)
_BAD_PATH = re.compile(r"[\s*?\[\]{}$`'\"\\;|&<>()!~#]")


# ===================================================================================== argv / environment (pre-Kit)
def _pick(cli, env, flag: str, var: str):
    env = None if env is None or not str(env).strip() else str(env).strip()
    if cli is not None and env is not None and str(cli).strip() != env:
        raise SystemExit(f"sn: {flag} {cli!r} and {var}={env!r} disagree; refused (one source per run)")
    return cli if cli is not None else env


def parse_sn_args(argv: list[str], environ=None) -> tuple[argparse.Namespace, list[str]]:
    """``argv`` WITHOUT the program name -> (``Namespace(stand_ckpt, stand_mask: int)``, the harness's argv in its
    original order). Every refusal is a SystemExit raised before the harness (and so Kit) is touched."""
    environ = os.environ if environ is None else environ
    argv = list(argv)
    for flag in FLAGS:
        n = sum(1 for a in argv if a == flag or a.startswith(flag + "="))
        if n > 1:
            raise SystemExit(f"sn: {flag} given {n} times; refused")
    ap = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    ap.add_argument("--stand-ckpt", dest="stand_ckpt", default=None)
    ap.add_argument("--stand-mask", dest="stand_mask", default=None)
    opts, rest = ap.parse_known_args(argv)
    stray = [a for a in rest if a.startswith(STRAY_PREFIXES)]
    if stray:
        raise SystemExit(f"sn: unrecognised / forbidden argument(s) {stray} (the stand_net variant takes only "
                         f"{FLAGS}; P2L's --p2r-* flags are set internally)")
    p2r_env = sorted(k for k in environ if str(k).startswith("P2R_"))
    if p2r_env:
        raise SystemExit(f"sn: P2R_* variable(s) {p2r_env} in the environment; refused (P2L's env fallbacks would "
                         "change the mode or the bank silently)")
    ck = _pick(opts.stand_ckpt, environ.get(ENV_CKPT), "--stand-ckpt", ENV_CKPT)
    mk = _pick(opts.stand_mask, environ.get(ENV_MASK), "--stand-mask", ENV_MASK)
    if ck is None:
        raise SystemExit(f"sn: the stand_net variant needs --stand-ckpt PATH (or {ENV_CKPT})")
    if mk is None:
        raise SystemExit(f"sn: the stand_net variant needs --stand-mask 0|1 (or {ENV_MASK})")
    mk = str(mk).strip()
    if mk not in ("0", "1"):
        raise SystemExit(f"sn: --stand-mask must be 0 or 1, got {mk!r}")
    ck = str(ck)
    if _BAD_PATH.search(ck):
        raise SystemExit(f"sn: --stand-ckpt {ck!r} contains whitespace / glob / shell characters; refused "
                         "(EVAL_EXTRA_ARGS is unquoted)")
    if not Path(ck).is_absolute():
        raise SystemExit(f"sn: --stand-ckpt {ck!r} must be an absolute path (run_arm cds before launching)")
    if not Path(ck).is_file():
        raise SystemExit(f"sn: --stand-ckpt not found: {ck}")
    reasons = SN.env_reasons(environ)
    if reasons:
        raise SystemExit("sn: environment refused (PLAN D5 obs parity / map L10):\n  " + "\n  ".join(reasons))
    return argparse.Namespace(stand_ckpt=ck, stand_mask=int(mk)), rest


# ===================================================================================== P2L (private instance)
def load_p2r(path: Path | str = P2R_LAUNCHER, name: str = "_sn_p2r"):
    """P2L loaded BY PATH into a fresh PRIVATE module instance (not in ``sys.modules``): the patches below never
    reach another importer (map risk L7). Torch-free at import."""
    return SN.load_by_path(path, name)


#: ``stand_net.stand_ckpt_kind``: a ``sn_dagger.py`` checkpoint, or S1 itself (identity bridge / swap-live proof)
CKPT_KIND_TRAINED, CKPT_KIND_S1 = "sn_dagger", "s1_identity"


def make_mode_class(p2r, mask_flag: int, stand_ckpt: str | None = None):
    """``StandNetMode``: ``p2r.P2RMode`` with ``__call__`` overridden IN FULL (map risk L1), bound to ``mask_flag``
    and to the stand checkpoint path (``--stand-ckpt``; read once more in ``bind_s1`` for its trained arm)."""
    if isinstance(mask_flag, bool) or mask_flag not in (0, 1):
        raise ValueError(f"sn: mask_flag must be an int 0 / 1, got {mask_flag!r}")
    base = p2r.P2RMode
    if getattr(base, "_sn_stand_net", False):
        raise RuntimeError("sn: P2RMode is already the stand_net subclass on this module instance")

    class StandNetMode(base):
        """The harness's ``policy`` for the stand_net variant (see the module docstring)."""

        _sn_stand_net = True
        STAND_MASK = int(mask_flag)
        STAND_CKPT = None if stand_ckpt is None else str(stand_ckpt)

        def __init__(self, variant, *args, **kwargs):
            if variant != INNER_MODE:
                raise ValueError(f"sn: StandNetMode runs on P2L's {INNER_MODE!r} machinery, got {variant!r}")
            if kwargs.get("stand") is None:
                raise ValueError("sn: StandNetMode needs the stand network (policy, estimator)")
            super().__init__(variant, *args, **kwargs)
            self.variant = VARIANT
            self.stand_mask = int(mask_flag)
            self.stand_ckpt = self.STAND_CKPT
            self.s1_runner = None
            self.sn_ctx = None                  # P2L's _Ctx (s1_sha, specialist), handed over by make_factory
            self.estimator_equals_s1 = None
            self.trained_stand_mask = None
            self.stand_ckpt_kind = None
            self.r_params = None
            self.stand_rows_alive = None        # device tensors, summed without a host sync per step
            self.stand_rows_differ = None

        # ---------------------------------------------------------------- wiring
        def bind_s1(self, s1_policy, s1_module) -> None:
            """The harness's ``get_inference_policy`` (after S1's ``runner.load``, before the first step): S1 bound as
            the parent does, then the estimator identity (D1 / L5), the trained arm (:meth:`trained_arm`) and R's
            D3 constants (post-Kit, pre-episode). Every refusal is a SystemExit before the first step."""
            super().bind_s1(s1_policy, s1_module)
            runner = self.s1_runner
            if runner is None:
                raise RuntimeError("sn: S1's runner was never handed to the stand_net mode (make_factory not wrapped)")
            ok, why = SN.state_dicts_equal(runner.alg.estimator.state_dict(), self.stand[1].state_dict())
            if not ok:
                raise SystemExit(f"sn: the stand checkpoint's estimator differs from S1's ({why}); refused before the "
                                 "first step (PLAN D1: the stand net's estimator is S1's, frozen; map risk L5)")
            trained, kind = self.trained_arm()
            self.estimator_equals_s1 = True
            self.trained_stand_mask, self.stand_ckpt_kind = trained, kind
            self.r_params = SN.r_params()

        def trained_arm(self) -> tuple:
            """``(trained_stand_mask, stand_ckpt_kind)`` -- ties the checkpoint to the mask it is evaluated with
            (review 2026-10-12: a valid-looking stamp of an arm under the wrong mask would read as a normal head).

            * ``infos.sn_dagger`` present (a ``sn_dagger.py`` checkpoint): its ``stand_mask`` must be an int 0 / 1
              EQUAL to ``--stand-mask`` -> ``(stand_mask, "sn_dagger")``;
            * absent: the checkpoint must be S1 itself (P2L's specialist sha == S1's sha, both hashed by P2L from the
              files it loaded) -> ``(None, "s1_identity")`` -- the identity bridge (mask 0) and the swap-live proof
              (mask 1); any other checkpoint is refused (no arm to check it against).
            Post-Kit (torch); ``map_location="cpu"``, no RNG draw, no device allocation."""
            import torch  # noqa: PLC0415

            path = self.stand_ckpt
            if not path:
                raise RuntimeError("sn: the stand checkpoint path was never handed to the stand_net mode (install)")
            ck = torch.load(str(path), map_location="cpu", weights_only=False)
            infos = ck.get("infos") if isinstance(ck, dict) else None
            sd = infos.get("sn_dagger") if isinstance(infos, dict) else None
            if sd is not None:
                tm = sd.get("stand_mask") if isinstance(sd, dict) else None
                if type(tm) is not int or tm not in (0, 1):
                    raise SystemExit(f"sn: the stand checkpoint's infos.sn_dagger.stand_mask is {tm!r}, not an int "
                                     "0 / 1; refused before the first step")
                if tm != self.stand_mask:
                    raise SystemExit(f"sn: --stand-mask {self.stand_mask} but the checkpoint was trained with "
                                     f"stand_mask {tm} (arm {sd.get('arm')!r}); refused before the first step (an arm "
                                     "is evaluated with the mask it was trained with)")
                return int(tm), CKPT_KIND_TRAINED
            ctx = self.sn_ctx
            if ctx is None:
                raise RuntimeError("sn: P2L's context was never handed to the stand_net mode (make_factory not "
                                   "wrapped)")
            s1_sha = getattr(ctx, "s1_sha", None)
            sp = (getattr(ctx, "specialist", None) or {}).get("sha256")
            if not s1_sha or sp != s1_sha:
                raise SystemExit(f"sn: the stand checkpoint {path} carries no infos.sn_dagger (not a sn_dagger.py "
                                 f"checkpoint) and is not S1 itself (sha {str(sp)[:12]} vs S1 {str(s1_sha)[:12]}); "
                                 "refused before the first step (only trainer checkpoints, or S1 for the identity "
                                 "bridge / swap-live proof)")
            return None, CKPT_KIND_S1

        # ---------------------------------------------------------------- the step
        def __call__(self, obs, hist_encoding=True, **kwargs):
            import torch  # noqa: PLC0415

            # -- P2L:563-588, verbatim semantics
            if kwargs:
                raise RuntimeError(f"p2r: unexpected policy kwargs {sorted(kwargs)} (student / canon paths are refused)")
            if self.s1_policy is None:
                raise RuntimeError("p2r: S1's policy was never bound")
            c = self._cur
            if c is None:
                raise SystemExit("p2r: no world-frame schedule was installed before the first step (omni world rows "
                                 "only)")
            n = int(obs.shape[0])
            dev = obs.device
            if c["stand"] is None:
                c["stand"] = torch.zeros(c["T"], n, dtype=torch.bool, device=dev)
                # P2L._close_pass writes every key of c["rst"] as its own raw npz key
                c["rst"] = {
                    SN.RAW_STAND_ACTION: torch.zeros(c["T"], n, SN.ACTION_DIM, dtype=torch.float32, device=dev),
                    SN.RAW_S1_ACTION: torch.zeros(c["T"], n, SN.ACTION_DIM, dtype=torch.float32, device=dev),
                    SN.RAW_R_TARGET: torch.zeros(c["T"], n, SN.ACTION_DIM, dtype=torch.float32, device=dev),
                    SN.RAW_R_BRANCH: torch.full((c["T"], n, SN.NUM_LEGS), -1, dtype=torch.int8, device=dev),
                }
            if self.alive is None:
                self.alive = torch.ones(n, dtype=torch.bool, device=dev)
            t = c["t"]
            if t >= c["T"]:
                raise RuntimeError(f"p2r: step {t} past the pass's {c['T']} planned steps")

            # -- the stand_net step (the st2 branch with the D1 input)
            stand = self.rederive_on_reset()
            a_s1 = self.s1_policy(obs, hist_encoding=hist_encoding)          # the harness's obs, S1's estimate in place
            st_policy, st_est = self.stand
            x = SN.stand_net_input(obs, st_est, p0=self.p0, p1=self.p1, num_prop=self.num_prop,
                                   mask=bool(self.stand_mask))               # a NEW tensor; obs is never written
            a_st = st_policy(x, hist_encoding=True).to(a_s1.dtype)           # the D1 forward path, never the priv path
            actions = SN.route_actions(stand, a_st, a_s1)

            # -- record (pure; no RNG): both actions + R on stand rows, the swap-live counters
            sm = stand.unsqueeze(1)
            rec = c["rst"]
            tgt, branch = SN.r_label(obs)
            rec[SN.RAW_STAND_ACTION][t] = torch.where(sm, a_st.float(), torch.zeros_like(rec[SN.RAW_STAND_ACTION][t]))
            rec[SN.RAW_S1_ACTION][t] = torch.where(sm, a_s1.float(), torch.zeros_like(rec[SN.RAW_S1_ACTION][t]))
            rec[SN.RAW_R_TARGET][t] = torch.where(sm, tgt.float(), torch.zeros_like(rec[SN.RAW_R_TARGET][t]))
            rec[SN.RAW_R_BRANCH][t] = torch.where(sm, branch.to(torch.int8), torch.full_like(rec[SN.RAW_R_BRANCH][t], -1))
            live = stand & self.alive
            n_live = live.sum()
            n_diff = (live & (a_st != a_s1).any(dim=1)).sum()
            self.stand_rows_alive = n_live if self.stand_rows_alive is None else self.stand_rows_alive + n_live
            self.stand_rows_differ = n_diff if self.stand_rows_differ is None else self.stand_rows_differ + n_diff

            c["stand"][t] = stand
            c["t"] = t + 1
            self.calls += 1
            return actions

        def counts(self) -> dict:
            def _i(v):
                return None if v is None else int(v.item())

            return {"stand_rows_alive": _i(self.stand_rows_alive),
                    "stand_rows_action_differs_from_s1": _i(self.stand_rows_differ)}

    return StandNetMode


def install(p2r, opts: argparse.Namespace):
    """Rebind ``P2RMode`` / ``make_factory`` / ``p2r_block`` on the PRIVATE P2L instance ``p2r`` (each is resolved
    at call time inside P2L). Returns the mode class. Refuses a second install on the same instance."""
    if getattr(p2r, "_sn_installed", False):
        raise RuntimeError("sn: the stand_net patches are already installed on this P2L instance")
    flag = int(opts.stand_mask)
    mode_cls = make_mode_class(p2r, flag, getattr(opts, "stand_ckpt", None))
    orig_make_factory = p2r.make_factory
    orig_block = p2r.p2r_block
    p2r_file = Path(p2r.__file__).resolve()

    def make_factory(ctx, orig_factory):
        inner = orig_make_factory(ctx, orig_factory)

        def factory(env, train_cfg, log_dir=None, device="cpu"):
            runner = inner(env, train_cfg, log_dir=log_dir, device=device)
            if type(ctx.mode) is not mode_cls:
                raise RuntimeError(f"sn: the mode wrapper is {type(ctx.mode).__name__}, not StandNetMode (map L1)")
            if ctx.opts.mode != INNER_MODE or ctx.specialist is None:
                raise RuntimeError("sn: P2L did not build the stand network as its st2 extra runner")
            ctx.mode.s1_runner = runner
            ctx.mode.sn_ctx = ctx                      # s1_sha (set by P2L's load wrapper) + specialist sha
            return runner

        return factory

    def p2r_block(ctx, stats, *, bank_rel, rst_stats=None, stand_bank_rel=None):
        b = orig_block(ctx, stats, bank_rel=bank_rel, rst_stats=rst_stats, stand_bank_rel=stand_bank_rel)
        mode = ctx.mode
        if b.get("variant") != VARIANT or type(mode) is not mode_cls:
            raise RuntimeError(f"sn: p2r_mode.variant {b.get('variant')!r} is not {VARIANT!r} (map L1)")
        if mode.estimator_equals_s1 is not True:
            raise RuntimeError("sn: the estimator identity was never checked; refusing to write the stamp")
        if mode.stand_ckpt_kind not in (CKPT_KIND_TRAINED, CKPT_KIND_S1):
            raise RuntimeError("sn: the trained-arm check never ran; refusing to write the stamp")
        rp = mode.r_params
        b[SN.MASK_KEY] = int(mode.stand_mask)                    # an INT, never a bool (map L4)
        b[SN.BLOCK_KEY] = {
            "wrapper": str(_THIS), "wrapper_sha256": SN.sha256_file(_THIS),
            "common": str(COMMON_PATH), "common_sha256": SN.sha256_file(COMMON_PATH),
            "p2r_launcher": str(p2r_file), "p2r_launcher_sha256": SN.sha256_file(p2r_file),
            "restance_module": str(SN.RESTANCE_MODULE),
            "restance_module_sha256": SN.sha256_file(SN.RESTANCE_MODULE),
            "inner_mode": INNER_MODE,
            "forward": SN.FORWARD, "hist_encoding": True,
            "mask_fn": SN.MASK_FN, "mask_rows": SN.MASK_ROWS, "mask_after_estimate": True,
            "estimator_equals_s1": True,
            "trained_stand_mask": mode.trained_stand_mask, "stand_ckpt_kind": mode.stand_ckpt_kind,
            "r_knob": SN.RESTANCE_D3, "r_params": (rp.to_dict() if rp is not None else None),
            "raw_keys": list(SN.RAW_KEYS),
            **mode.counts(),
        }
        b["raw_series"] = list(b.get("raw_series") or []) + [k for k in SN.RAW_KEYS if k not in (b.get("raw_series")
                                                                                              or [])]
        print(f"[SN] stand_net stand_mask={b[SN.MASK_KEY]} ckpt_kind={mode.stand_ckpt_kind} "
              f"trained_stand_mask={mode.trained_stand_mask} specialist={(b.get('specialist') or {}).get('sha256')} "
              f"stand_rows_alive={b[SN.BLOCK_KEY]['stand_rows_alive']} "
              f"differs_from_s1={b[SN.BLOCK_KEY]['stand_rows_action_differs_from_s1']}", flush=True)
        return b

    p2r.P2RMode = mode_cls
    p2r.make_factory = make_factory
    p2r.p2r_block = p2r_block
    p2r._sn_installed = True
    return mode_cls


def main(argv: list[str] | None = None, *, harness: Path | str | None = None, environ=None) -> None:
    argv = list(sys.argv if argv is None else argv)
    environ = os.environ if environ is None else environ
    opts, rest = parse_sn_args(argv[1:], environ)                     # every pre-Kit refusal
    p2r = load_p2r()
    install(p2r, opts)
    print(f"[SN] variant={VARIANT} stand_ckpt={opts.stand_ckpt} stand_mask={opts.stand_mask} p2r={P2R_LAUNCHER} "
          f"r_knob={SN.RESTANCE_D3}", flush=True)
    p2r.main([argv[0], "--p2r-mode", INNER_MODE, "--p2r-stand-ckpt", opts.stand_ckpt] + rest,
             harness=(p2r.HARNESS if harness is None else harness), environ=environ)


if __name__ == "__main__":
    main()
