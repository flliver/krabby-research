"""Four-stage bench harness: install → flash → bringup → motion.

Extends krabby-bench. Each stage returns its
own result so a failure names where it happened.

Dual-use Orin: prefer ``~/.venv-krabby-bench`` and ``scripts/jetson/bench-reset.sh``.
Never touches ``~/.venv-krabby``.
"""
from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

from krabby_bench._smoke import SmokeResult, run_smoke

log = logging.getLogger(__name__)

DEFAULT_BENCH_VENV = Path.home() / ".venv-krabby-bench"
# Knee/hip-lift joints for motion. PIN_REV=2 local-testing only (not future).
DEFAULT_JOINTS = ("RLKL", "RRKL", "FLKL", "FRHL", "FRKL")


@dataclass
class StageResult:
    stage: str
    ok: bool
    detail: str = ""
    command: str = ""
    stdout: str = ""
    stderr: str = ""


@dataclass
class HarnessResult:
    ok: bool
    stages: list[StageResult] = field(default_factory=list)
    failed_stage: Optional[str] = None

    def summary_lines(self) -> list[str]:
        lines = []
        for s in self.stages:
            mark = "PASS" if s.ok else "FAIL"
            lines.append(f"[{mark}] {s.stage}: {s.detail or ('ok' if s.ok else 'failed')}")
            if not s.ok and s.command:
                lines.append(f"       command: {s.command}")
        return lines


@dataclass
class HarnessConfig:
    """Knobs for a single harness run (CLI / future Actions job)."""

    bench_venv: Path = field(default_factory=lambda: DEFAULT_BENCH_VENV)
    firmware_channel: str = "release/0.2.15"
    reset: bool = True
    reset_rmi: bool = False
    skip_install: bool = False
    skip_flash: bool = False
    skip_bringup: bool = False
    skip_motion: bool = False
    joints: list[str] = field(default_factory=lambda: list(DEFAULT_JOINTS))
    jog_pwm: int = 200
    jog_seconds: float = 2.0
    bringup_timeout: float = 90.0
    pot_delta_min: int = 5
    hall_delta_min: int = 1  # PIN_REV=2 (local-only) has no hall; pot_delta is the assert
    install_no_launch: bool = True
    repo_root: Optional[Path] = None


def _repo_root() -> Path:
    """Best-effort path to krabby-research (bench/ sits one level under it)."""
    return Path(__file__).resolve().parents[2]


def _bench_reset_script(cfg: HarnessConfig) -> Path:
    root = cfg.repo_root or _repo_root()
    return root / "scripts" / "jetson" / "bench-reset.sh"


def _krabby_bin(venv: Path) -> str:
    candidate = venv / "bin" / "krabby"
    if candidate.is_file():
        return str(candidate)
    return shutil.which("krabby") or "krabby"


def _run(
    cmd: list[str],
    *,
    timeout: Optional[float] = None,
    env: Optional[dict] = None,
    cwd: Optional[Path] = None,
) -> subprocess.CompletedProcess[str]:
    log.debug("exec: %s", " ".join(cmd))
    return subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
        env=env,
        cwd=str(cwd) if cwd else None,
    )


def _sudo_env_path(venv: Path) -> dict:
    """PATH with venv bin first so sudo -E sees the bench krabby (F2)."""
    env = os.environ.copy()
    env["PATH"] = f"{venv / 'bin'}:{env.get('PATH', '')}"
    env["VIRTUAL_ENV"] = str(venv)
    return env


def stage_install(cfg: HarnessConfig) -> StageResult:
    venv = cfg.bench_venv
    reset_script = _bench_reset_script(cfg)

    if cfg.reset:
        if not reset_script.is_file():
            return StageResult(
                stage="install",
                ok=False,
                detail=f"bench-reset.sh not found at {reset_script}",
                command=str(reset_script),
            )
        reset_cmd = [str(reset_script)]
        if cfg.reset_rmi:
            reset_cmd.append("--rmi")
        r = _run(reset_cmd, timeout=120)
        if r.returncode != 0:
            return StageResult(
                stage="install",
                ok=False,
                detail=f"bench-reset.sh exit {r.returncode}",
                command=" ".join(reset_cmd),
                stdout=r.stdout,
                stderr=r.stderr,
            )
        log.info("bench-reset.sh completed")

    if venv.exists():
        shutil.rmtree(venv)
    py = sys.executable
    r = _run([py, "-m", "venv", str(venv)], timeout=60)
    if r.returncode != 0:
        return StageResult(
            stage="install",
            ok=False,
            detail=f"venv create failed (exit {r.returncode})",
            command=f"{py} -m venv {venv}",
            stdout=r.stdout,
            stderr=r.stderr,
        )

    pip = str(venv / "bin" / "pip")
    krabby = _krabby_bin(venv)
    bench_src = (cfg.repo_root or _repo_root()) / "bench"
    installs: list[tuple[str, list[str], int]] = [
        ("pip upgrade", [pip, "install", "-U", "pip"], 120),
        ("pip install krabby-launcher", [pip, "install", "krabby-launcher"], 300),
    ]
    # Install local krabby-bench so remaining stages can re-exec under this venv.
    if bench_src.is_dir():
        installs.append(
            ("pip install ./bench", [pip, "install", str(bench_src)], 180),
        )
    else:
        installs.append(
            ("pip install krabby-bench", [pip, "install", "krabby-bench"], 180),
        )
    for label, cmd, timeout in installs:
        r = _run(cmd, timeout=timeout)
        if r.returncode != 0:
            return StageResult(
                stage="install",
                ok=False,
                detail=f"{label} failed (exit {r.returncode})",
                command=" ".join(cmd),
                stdout=r.stdout,
                stderr=r.stderr,
            )

    install_cmd = [
        "sudo", "-E", "env",
        f"PATH={venv / 'bin'}:{os.environ.get('PATH', '')}",
        krabby, "install",
    ]
    if cfg.install_no_launch:
        install_cmd.append("--no-launch-on-startup")
    r = _run(install_cmd, timeout=600, env=_sudo_env_path(venv))
    if r.returncode != 0:
        return StageResult(
            stage="install",
            ok=False,
            detail=f"krabby install failed (exit {r.returncode})",
            command=" ".join(install_cmd),
            stdout=r.stdout,
            stderr=r.stderr,
        )

    return StageResult(
        stage="install",
        ok=True,
        detail=f"venv={venv} launcher installed; krabby install ok",
        command=" ".join(install_cmd),
        stdout=r.stdout,
    )


def stage_flash(cfg: HarnessConfig) -> StageResult:
    try:
        from krabby._state import installed_image, resolve_image_ref
    except ImportError as exc:
        return StageResult(
            stage="flash",
            ok=False,
            detail=f"krabby not importable in this interpreter: {exc}",
            command="python -c 'import krabby'",
        )

    image_ref = resolve_image_ref(installed_image())
    smoke: SmokeResult = run_smoke(cfg.firmware_channel, image_ref)
    if not smoke.ok:
        return StageResult(
            stage="flash",
            ok=False,
            detail=f"{smoke.step}: {smoke.detail}",
            command=f"krabby firmware update {cfg.firmware_channel} <ports>",
            stdout=smoke.stdout,
            stderr=smoke.stderr,
        )
    return StageResult(
        stage="flash",
        ok=True,
        detail=f"boards={smoke.ver_observed} expected={smoke.ver_expected}",
        command=f"run_smoke({cfg.firmware_channel!r})",
        stdout=smoke.stdout,
    )


def _container_running(name: str = "krabby") -> bool:
    r = _run(
        ["docker", "ps", "--filter", f"name=^{name}$", "--format", "{{.State}}"],
        timeout=10,
    )
    return (r.stdout or "").strip().lower() == "running"


def _docker_logs_tail(name: str = "krabby", n: int = 80) -> str:
    r = _run(["docker", "logs", "--tail", str(n), name], timeout=15)
    return (r.stdout or "") + (r.stderr or "")


def stage_bringup(cfg: HarnessConfig) -> StageResult:
    venv = cfg.bench_venv
    krabby = _krabby_bin(venv)
    env = _sudo_env_path(venv)

    _run(["docker", "rm", "-f", "krabby"], timeout=30)

    log_path = Path("/tmp/krabby-bench-bringup.log")
    if log_path.exists():
        log_path.unlink()

    cmd = [krabby, "run", "--gamepad-only"]
    cmd_str = " ".join(cmd)
    with open(log_path, "w") as logf:
        proc = subprocess.Popen(
            cmd,
            stdout=logf,
            stderr=subprocess.STDOUT,
            env=env,
            start_new_session=True,
        )

    deadline = time.time() + cfg.bringup_timeout
    mcu_ok = False
    container_ok = False
    last_logs = ""
    while time.time() < deadline:
        if proc.poll() is not None:
            body = log_path.read_text() if log_path.exists() else ""
            return StageResult(
                stage="bringup",
                ok=False,
                detail=f"krabby run exited early (code {proc.returncode})",
                command=cmd_str,
                stdout=body,
            )
        container_ok = _container_running("krabby")
        last_logs = _docker_logs_tail("krabby") if container_ok else ""
        if container_ok and (
            "MCU connected successfully" in last_logs
            or "Gamepad mode active" in last_logs
            or "Connected to /dev/tty" in last_logs
        ):
            mcu_ok = True
            break
        time.sleep(2.0)

    if not container_ok or not mcu_ok:
        host_log = log_path.read_text() if log_path.exists() else ""
        return StageResult(
            stage="bringup",
            ok=False,
            detail=(
                f"timeout {cfg.bringup_timeout}s "
                f"(container_running={container_ok}, mcu_seen={mcu_ok})"
            ),
            command=cmd_str,
            stdout=host_log + "\n--- docker logs ---\n" + last_logs,
        )

    return StageResult(
        stage="bringup",
        ok=True,
        detail=f"container running; MCU/gamepad ready (pid={proc.pid})",
        command=cmd_str,
        stdout=last_logs[-2000:],
    )


_MOTION_SCRIPT = r"""
import json, sys, time
from firmware.krabby_mcu import KrabbyMCUSDK

joints = [j for j in sys.argv[1].split(",") if j]
pwm = int(sys.argv[2])
seconds = float(sys.argv[3])

mcu = KrabbyMCUSDK()
if not mcu.connect():
    print(json.dumps({"ok": False, "detail": "connect failed"}))
    sys.exit(1)

deadline = time.time() + 12.0
while time.time() < deadline:
    if any(mcu.joints.get(j) is not None for j in joints):
        break
    time.sleep(0.05)

def peak_while_jogging(joint, signed_pwm, duration):
    # Sample pot/hall during the jog (peak-to-peak), not only start vs end.
    # Start/end alone can miss motion or read a stale pot after stop.
    pots, halls, positions = [], [], []
    mcu.send_command_jog(joint, signed_pwm)
    end_t = time.time() + duration
    while time.time() < end_t:
        jt = mcu.joints.get(joint)
        if jt is not None:
            pots.append(jt.pot)
            halls.append(jt.saf)
            positions.append(jt.pos)
        time.sleep(0.05)
    mcu.send_command_jog(joint, 0)
    # Drain a few more samples while PWM ramps down / UART catches up.
    drain_t = time.time() + 0.8
    last = mcu.joints.get(joint)
    while time.time() < drain_t:
        jt = mcu.joints.get(joint)
        if jt is not None:
            pots.append(jt.pot)
            halls.append(jt.saf)
            positions.append(jt.pos)
            last = jt
        time.sleep(0.05)
    if not pots:
        return None
    return {
        "before_pot": pots[0],
        "after_pot": pots[-1],
        "min_pot": min(pots),
        "max_pot": max(pots),
        "pot_peak": max(pots) - min(pots),
        "before_hall": halls[0],
        "after_hall": halls[-1],
        "min_hall": min(halls),
        "max_hall": max(halls),
        "hall_peak": max(halls) - min(halls),
        "before_pos": positions[0],
        "after_pos": positions[-1],
        "samples": len(pots),
        "en": list(last.en) if last else None,
        "pwm": list(last.pwm) if last else None,
    }

results = []
for joint in joints:
    # Wait for this joint specifically (followers lag FRONT over UART).
    j_deadline = time.time() + 8.0
    while time.time() < j_deadline and mcu.joints.get(joint) is None:
        time.sleep(0.05)
    before = mcu.joints.get(joint)
    if before is None:
        keys = sorted(k for k, v in mcu.joints.items() if v is not None)
        results.append({
            "ok": False,
            "joint": joint,
            "detail": f"no telemetry for {joint}",
            "joints_seen": keys,
        })
        continue

    # Use J (same as firmware GUI), not batch B — B forward to LEFT/RIGHT
    # followers has been unreliable vs J<name> <pwm> (see F4).
    best = None
    for sign in (1, -1):
        sample = peak_while_jogging(joint, sign * abs(pwm), seconds)
        if sample is None:
            continue
        if best is None or sample["pot_peak"] >= best["pot_peak"]:
            best = sample

    if best is None:
        results.append({
            "ok": False,
            "joint": joint,
            "detail": f"no samples while jogging {joint}",
        })
        continue

    results.append({"ok": True, "joint": joint, **best})

mcu.send_command_joints_hold()
mcu.close()

print(json.dumps({
    "ok": all(r.get("ok") for r in results) and bool(results),
    "joints": results,
}))
"""


def stage_motion(cfg: HarnessConfig) -> StageResult:
    """Jog each configured joint on the real MCU and assert pot or hall changed.

    Default joints: RLKL, RRKL, FLKL, FRHL, FRKL. Stops the locomotion container
    so the serial port is free. Runs **on the host** with repo ``firmware/`` on
    ``PYTHONPATH`` so local PIN_REV=2 / ``dev-local`` telemetry parses (PIN_REV=2
    is local-testing only and will not be used going forward; we are not using
    published S3 / ECR image firmware here). PIN_REV=2 has no hall sensors —
    assert on pot peak-to-peak during the jog (not only start vs end). Requires
    bench 12 V so the H-bridge can move.
    """
    root = cfg.repo_root or _repo_root()
    firmware_dir = root / "firmware"
    joints = list(cfg.joints) or list(DEFAULT_JOINTS)
    joints_label = ",".join(joints)
    if not (firmware_dir / "krabby_mcu.py").is_file():
        return StageResult(
            stage="motion",
            ok=False,
            detail=f"repo firmware not found at {firmware_dir} (pass --repo-root)",
            command=f"--repo-root {root}",
        )

    _run(["docker", "rm", "-f", "krabby"], timeout=60)
    time.sleep(1.0)

    env = os.environ.copy()
    env["PYTHONPATH"] = f"{root}{os.pathsep}{env.get('PYTHONPATH', '')}"

    # ~2 directions * (jog + drain) * N joints + connect/settle margin
    timeout = max(180.0, 30.0 + len(joints) * (2.0 * (cfg.jog_seconds + 1.0) + 5.0))
    cmd = [
        sys.executable, "-c", _MOTION_SCRIPT,
        joints_label, str(cfg.jog_pwm), str(cfg.jog_seconds),
    ]
    r = _run(cmd, timeout=timeout, env=env)
    if r.returncode != 0:
        return StageResult(
            stage="motion",
            ok=False,
            detail=f"motion script exit {r.returncode}: {(r.stdout or r.stderr or '')[-500:]}",
            command=f"{sys.executable} -c … jog {joints_label}",
            stdout=r.stdout,
            stderr=r.stderr,
        )

    try:
        lines = [ln for ln in (r.stdout or "").splitlines() if ln.strip().startswith("{")]
        data = json.loads(lines[-1]) if lines else {}
    except (json.JSONDecodeError, IndexError):
        return StageResult(
            stage="motion",
            ok=False,
            detail="could not parse motion JSON",
            command=f"jog {joints_label}",
            stdout=r.stdout,
            stderr=r.stderr,
        )

    per_joint = data.get("joints") or []
    if not per_joint:
        return StageResult(
            stage="motion",
            ok=False,
            detail=str(data.get("detail", data) or "no joint results"),
            command=f"jog {joints_label}",
            stdout=r.stdout,
        )

    details: list[str] = []
    failed: list[str] = []
    for item in per_joint:
        j = item.get("joint", "?")
        if not item.get("ok"):
            failed.append(j)
            details.append(f"{j}: {item.get('detail', 'failed')}")
            continue
        pot_peak = int(item.get("pot_peak", abs(int(item["after_pot"]) - int(item["before_pot"]))))
        hall_peak = int(item.get("hall_peak", abs(int(item["after_hall"]) - int(item["before_hall"]))))
        moved = pot_peak >= cfg.pot_delta_min or hall_peak >= cfg.hall_delta_min
        line = (
            f"{j} pot {item.get('min_pot', item['before_pot'])}…{item.get('max_pot', item['after_pot'])} "
            f"(peak Δ{pot_peak}); hall peak Δ{hall_peak}"
        )
        if item.get("en") is not None:
            line += f"; en={item.get('en')} pwm={item.get('pwm')}"
        if item.get("samples") is not None:
            line += f"; n={item.get('samples')}"
        details.append(("FAIL " if not moved else "PASS ") + line)
        if not moved:
            failed.append(j)

    detail = "; ".join(details)
    cmd_str = f"jog {joints_label} pwm={cfg.jog_pwm}"
    if failed:
        return StageResult(
            stage="motion",
            ok=False,
            detail=(
                f"failed joints: {','.join(failed)} — {detail}. "
                f"Check 12 V bench power / EN wiring (PIN_REV=2 local-only has no hall)."
            ),
            command=cmd_str,
            stdout=r.stdout,
        )
    return StageResult(
        stage="motion",
        ok=True,
        detail=detail,
        command=cmd_str,
        stdout=r.stdout,
    )


def _in_bench_venv(cfg: HarnessConfig) -> bool:
    """True if this interpreter is cfg.bench_venv (so `import krabby` is the Install venv)."""
    venv = cfg.bench_venv.resolve()
    prefix = Path(sys.prefix).resolve()
    return prefix == venv or venv in prefix.parents or str(prefix).startswith(str(venv))


def _reexec_under_bench_venv(cfg: HarnessConfig) -> HarnessResult:
    """Continue flash/bringup/motion with the bench venv's krabby-bench."""
    py = cfg.bench_venv / "bin" / "python"
    if not py.is_file():
        return HarnessResult(
            ok=False,
            stages=[
                StageResult(
                    stage="install",
                    ok=False,
                    detail=f"bench venv python missing after install: {py}",
                )
            ],
            failed_stage="install",
        )
    cmd = [
        str(py), "-m", "krabby_bench", "harness",
        "--skip-install",
        "--no-reset",
        "--bench-venv", str(cfg.bench_venv),
        "--firmware-channel", cfg.firmware_channel,
        "--joint", *list(cfg.joints),
        "--jog-pwm", str(cfg.jog_pwm),
        "--jog-seconds", str(cfg.jog_seconds),
        "--bringup-timeout", str(cfg.bringup_timeout),
    ]
    if cfg.skip_flash:
        cmd.append("--skip-flash")
    if cfg.skip_bringup:
        cmd.append("--skip-bringup")
    if cfg.skip_motion:
        cmd.append("--skip-motion")
    if cfg.repo_root:
        cmd.extend(["--repo-root", str(cfg.repo_root)])
    log.info("Re-exec under bench venv: %s", " ".join(cmd))
    r = _run(cmd, timeout=None)
    # Child prints its own summary; map exit code.
    ok = r.returncode == 0
    return HarnessResult(
        ok=ok,
        stages=[
            StageResult(
                stage="install",
                ok=True,
                detail="install done; continued under bench venv",
            ),
            StageResult(
                stage="flash+bringup+motion",
                ok=ok,
                detail=f"child exit {r.returncode}",
                command=" ".join(cmd),
                stdout=r.stdout,
                stderr=r.stderr,
            ),
        ],
        failed_stage=None if ok else "flash+bringup+motion",
    )


def run_harness(cfg: HarnessConfig) -> HarnessResult:
    """Run selected stages in order; stop on first failure."""
    results: list[StageResult] = []

    if not cfg.skip_install:
        log.info("=== stage install ===")
        try:
            result = stage_install(cfg)
        except subprocess.TimeoutExpired as exc:
            result = StageResult(
                stage="install",
                ok=False,
                detail=f"timeout after {exc.timeout}s",
                command=" ".join(map(str, exc.cmd or [])),
            )
        except Exception as exc:
            log.exception("stage install crashed")
            result = StageResult(stage="install", ok=False, detail=f"exception: {exc}")
        results.append(result)
        for line in HarnessResult(ok=result.ok, stages=[result]).summary_lines():
            log.info("%s", line)
        if not result.ok:
            return HarnessResult(ok=False, stages=results, failed_stage="install")
        # Remaining stages need the new venv's krabby import / binaries.
        if not _in_bench_venv(cfg):
            child = _reexec_under_bench_venv(cfg)
            return HarnessResult(
                ok=child.ok,
                stages=results + child.stages,
                failed_stage=child.failed_stage,
            )

    plan: list[tuple[str, bool, Callable[[HarnessConfig], StageResult]]] = [
        ("flash", cfg.skip_flash, stage_flash),
        ("bringup", cfg.skip_bringup, stage_bringup),
        ("motion", cfg.skip_motion, stage_motion),
    ]
    for name, skip, fn in plan:
        if skip:
            log.info("Skipping stage %s", name)
            results.append(StageResult(stage=name, ok=True, detail="skipped"))
            continue
        log.info("=== stage %s ===", name)
        try:
            result = fn(cfg)
        except subprocess.TimeoutExpired as exc:
            result = StageResult(
                stage=name,
                ok=False,
                detail=f"timeout after {exc.timeout}s",
                command=" ".join(map(str, exc.cmd or [])),
            )
        except Exception as exc:
            log.exception("stage %s crashed", name)
            result = StageResult(stage=name, ok=False, detail=f"exception: {exc}")
        results.append(result)
        for line in HarnessResult(ok=result.ok, stages=[result]).summary_lines():
            log.info("%s", line)
        if not result.ok:
            return HarnessResult(ok=False, stages=results, failed_stage=name)
    return HarnessResult(ok=True, stages=results)
