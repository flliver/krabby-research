#!/bin/bash
# bootstrap.sh — one-command bring-up for a fresh NVIDIA Jetson Orin.
#
# Chains the individual bring-up steps, in order, idempotently:
#   1. remove brltty        — frees USB serial boards (see note below)
#   2. install-docker.sh    — Docker Engine               (section 2 of the guide)
#   3. setup-docker-gpu.sh  — NVIDIA container runtime     (section 3 of the guide)
#   4. krabby-launcher      — python3-pip + pip install krabby-launcher
#   5. krabby install       — udev rules, dialout group, boot autostart unit
#   6. krabby-bench         — error-reporting watchdog (only when SSM IAM keys
#                             are provided; see "SSM error reporting" below)
#
# brltty: its udev rules claim CH340/Arduino-class USB serial adapters as
# Braille displays, so the Mega/MCU boards vanish from /dev/ttyACM*. Step 1
# purges it. Harmless to re-run when it's already gone.
#
# Each step self-skips when its work is already done, so re-running is safe.
#
# Usage (on the Jetson, from anywhere in the repo):
#     ./scripts/jetson/bootstrap.sh [options]
#
# Options:
#     --skip-docker         Skip steps 2 and 3 (Docker already configured).
#     --no-krabby-install   Install the krabby-launcher package but do not run
#                           'krabby install' (skips udev/dialout/boot-autostart).
#     --ssm-prefix PREFIX   SSM parameter path prefix for the watchdog
#                           (default: /krabby/bench).
#     -h, --help            Show this help and exit.
#
# SSM error reporting (step 6, optional):
#     Step 6 installs the krabby-bench watchdog so the device can report smoke-
#     test failures via SMTP/GitHub. It runs only when this device's read-only
#     IAM key is provided in the environment:
#         BENCH_AWS_KEY_ID, BENCH_AWS_SECRET_KEY
#     The watchdog fetches the shared SMTP/GitHub secrets from AWS SSM at
#     runtime. Those secrets are seeded fleet-wide, off-device, by an operator
#     (see bench/README.md and set-ssm-params.sh) — bootstrap does not seed
#     them. Without the IAM keys, step 6 is skipped with a note.
#         BENCH_AWS_KEY_ID=AKIA... BENCH_AWS_SECRET_KEY=... \
#             ./scripts/jetson/bootstrap.sh
#
# Escalates per-command via sudo; you may be prompted for your password.
# Run as your normal user (not 'sudo ./bootstrap.sh') so group memberships and
# install state target your account, not root.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

SKIP_DOCKER=false
RUN_KRABBY_INSTALL=true
SSM_PREFIX="/krabby/bench"

# Print the leading comment block (skip the shebang; stop at the first
# non-comment line) as help text.
usage() { awk 'NR==1{next} /^#/{sub(/^# ?/,""); print; next} {exit}' "${BASH_SOURCE[0]}"; }

while [[ $# -gt 0 ]]; do
    case "$1" in
        --skip-docker)       SKIP_DOCKER=true ;;
        --no-krabby-install) RUN_KRABBY_INSTALL=false ;;
        --ssm-prefix)        SSM_PREFIX="${2:?--ssm-prefix needs a value}"; shift ;;
        --ssm-prefix=*)      SSM_PREFIX="${1#*=}" ;;
        -h|--help)           usage; exit 0 ;;
        *) echo "ERROR: unknown option '$1' (try --help)" >&2; exit 2 ;;
    esac
    shift
done

# sudo when not already root; empty otherwise.
if [[ "$(id -u)" -eq 0 ]]; then
    SUDO=""
else
    SUDO="sudo"
fi

# Preflight: this targets Debian-based Jetson (Ubuntu 22.04 / jammy, arm64).
if ! command -v apt-get &> /dev/null; then
    echo "ERROR: apt-get not found. This script targets Ubuntu/Debian (Jetson)." >&2
    exit 1
fi
arch="$(uname -m)"
if [[ "$arch" != "aarch64" ]]; then
    echo "WARNING: architecture is '$arch', not 'aarch64' — this is meant for Jetson Orin." >&2
fi

# Cache sudo credentials up front so steps don't interleave password prompts.
if [[ -n "$SUDO" ]]; then
    echo "==> Priming sudo (you may be prompted for your password once)"
    sudo -v
fi

step() { echo; echo "================================================================"; echo "  $1"; echo "================================================================"; }

# --- 1: Remove brltty (grabs USB serial / MCU boards) --------------------------
# brltty's udev rules (85-brltty.rules) claim CH340/Arduino-class USB serial
# adapters as Braille displays, so the Mega/MCU boards disappear from
# /dev/ttyACM*. Purge it so the HAL boards stay attached.
step "Step 1: Remove brltty (conflicts with USB serial boards)"
if dpkg -l brltty 2>/dev/null | grep -q '^ii'; then
    $SUDO systemctl stop brltty-udev.service 2>/dev/null || true
    $SUDO apt-get purge -y brltty
    $SUDO udevadm control --reload-rules && $SUDO udevadm trigger || true
    echo "✓ brltty removed"
else
    echo "✓ brltty not installed; nothing to remove"
fi

# --- 2 & 3: Docker Engine + NVIDIA container runtime ---------------------------
if [[ "$SKIP_DOCKER" == true ]]; then
    echo "==> Skipping Docker steps (--skip-docker)"
else
    step "Step 2: Docker Engine"
    "$SCRIPT_DIR/install-docker.sh"

    step "Step 3: NVIDIA container runtime (GPU access)"
    "$SCRIPT_DIR/setup-docker-gpu.sh"
fi

# --- 4: krabby-launcher --------------------------------------------------------
step "Step 4: krabby-launcher"
if ! command -v pip3 &> /dev/null; then
    echo "==> Installing python3-pip"
    $SUDO apt-get update
    $SUDO apt-get install -y python3-pip
else
    echo "✓ pip3 already present ($(pip3 --version))"
fi

# Install system-wide (under sudo) so 'krabby' lands on root's PATH for the
# 'krabby install' step below. --upgrade keeps a re-run current and idempotent.
echo "==> Installing/upgrading krabby-launcher"
$SUDO pip3 install --upgrade krabby-launcher

# --- 5: krabby install ---------------------------------------------------------
if [[ "$RUN_KRABBY_INSTALL" == true ]]; then
    step "Step 5: krabby install"
    $SUDO krabby install
else
    echo
    echo "==> Skipping 'krabby install' (--no-krabby-install)"
    echo "    Run it later with: sudo krabby install"
fi

# --- 6: krabby-bench (SSM error-reporting watchdog) ----------------------------
# Opt-in: runs only when this device's read-only IAM key is in the environment.
# The watchdog reads the shared SMTP/GitHub secrets from SSM at runtime; those
# are seeded fleet-wide by an operator (set-ssm-params.sh), not by bootstrap.
if [[ -n "${BENCH_AWS_KEY_ID:-}" && -n "${BENCH_AWS_SECRET_KEY:-}" ]]; then
    step "Step 6: krabby-bench (error reporting via SSM)"
    echo "==> Installing/upgrading krabby-bench"
    $SUDO pip3 install --upgrade krabby-bench
    echo "==> Running krabby-bench install (--ssm-prefix ${SSM_PREFIX})"
    # Pass the IAM key through to the (root) install; sudo drops env by default,
    # so set the vars explicitly on the command. When SUDO is empty (already
    # root) this is a plain env-assignment prefix.
    $SUDO BENCH_AWS_KEY_ID="$BENCH_AWS_KEY_ID" \
          BENCH_AWS_SECRET_KEY="$BENCH_AWS_SECRET_KEY" \
          krabby-bench install --ssm-prefix "$SSM_PREFIX"
else
    step "Step 6: krabby-bench (skipped)"
    echo "==> BENCH_AWS_KEY_ID / BENCH_AWS_SECRET_KEY not set — skipping the"
    echo "    error-reporting watchdog. To enable it later:"
    echo "      sudo pip3 install krabby-bench"
    echo "      sudo BENCH_AWS_KEY_ID=AKIA... BENCH_AWS_SECRET_KEY=... \\"
    echo "        krabby-bench install --ssm-prefix ${SSM_PREFIX}"
    echo "    (Shared SMTP/GitHub secrets are seeded in SSM separately; see"
    echo "    bench/README.md and set-ssm-params.sh.)"
fi

# --- Done ----------------------------------------------------------------------
echo
echo "================================================================"
echo "✓ Bring-up complete."
echo "================================================================"
echo
echo "Next steps / reminders:"
echo "  - 'docker' group membership takes effect on next login. Run"
echo "    'newgrp docker' or log out/in before using docker without sudo."
echo "  - If the GPU verification above warned, it is usually because the"
echo "    docker group is not yet active in this shell — re-login and re-run"
echo "    scripts/jetson/setup-docker-gpu.sh to confirm GPU access."
echo "  - Replug the Mega USB boards so the new udev rule applies."
echo "  - Then update firmware and pull/run the locomotion image"
echo "    (see docs/JETSON_DEPLOYMENT.md and the top-level README)."
if [[ -n "${BENCH_AWS_KEY_ID:-}" && -n "${BENCH_AWS_SECRET_KEY:-}" ]]; then
    echo "  - Watchdog installed. Monitor it with: journalctl -fu krabby-bench"
fi
