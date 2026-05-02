#!/bin/bash
# Progress reporting subsystem with pluggable backends.
#
# Decouples script-side progress reporting from any specific transport,
# so krabby work runs identically on hosts that have a particular
# infrastructure (e.g., baeprz fleet w/ nanny-progress + MQTT dashboard)
# AND on hosts that have nothing — without the script even noticing.
#
# Backends:
#   null    — no-op (default fallback when no other backend is detected)
#   nanny   — pushes to baeprz beeprz dash via /usr/local/bin/nanny-progress
#
# Detection (on `progress_init`):
#   1. If env var KRABBY_PROGRESS_BACKEND is set, use it (allows explicit
#      override or force-null for tests).
#   2. Else auto-detect: try `nanny` (command -v nanny-progress);
#      fall back to `null` if not found.
#
# Public API:
#   progress_init <total_phases>           # install EXIT trap, pick backend
#   progress_set <phase_idx> <pct> [label] # advance to phase, set %
#   progress_phase <phase_idx>             # advance phase, leave % alone
#   progress_percent <pct>                 # update % within current phase
#   progress_clear                         # reset (auto-fires on EXIT)
#
# Usage in a script:
#   source ~/lib_progress.sh
#   progress_init 5
#   progress_set 1 0 "fetch inputs"
#   ... do work ...
#   progress_set 2 0 "transform"
#   ... etc ...
#   # progress_clear runs automatically via the EXIT trap
#
# Adding a new backend (e.g., a webhook):
#   1. Define _progress_<name>_set, _progress_<name>_phase,
#      _progress_<name>_percent, _progress_<name>_clear (each takes the
#      same args as the public function).
#   2. Add a detection branch in _progress_detect_backend.
#   That's it. Public API and existing scripts unchanged.

PROGRESS_TOTAL=1
PROGRESS_BACKEND=""


# ---------------------------------------------------------------------------
# Backend: null  (no-op fallback)
# ---------------------------------------------------------------------------

_progress_null_set()     { :; }
_progress_null_phase()   { :; }
_progress_null_percent() { :; }
_progress_null_clear()   { :; }


# ---------------------------------------------------------------------------
# Backend: nanny  (baeprz beeprz dashboard via nanny-progress)
# ---------------------------------------------------------------------------

_progress_nanny_set()     { nanny-progress set     "$1" "$2" 2>/dev/null || true; }
_progress_nanny_phase()   { nanny-progress phase   "$1"      2>/dev/null || true; }
_progress_nanny_percent() { nanny-progress percent "$1"      2>/dev/null || true; }
_progress_nanny_clear()   { nanny-progress clear              2>/dev/null || true; }


# ---------------------------------------------------------------------------
# Detection
# ---------------------------------------------------------------------------

_progress_detect_backend() {
    # Explicit override always wins.
    if [ -n "${KRABBY_PROGRESS_BACKEND:-}" ]; then
        echo "$KRABBY_PROGRESS_BACKEND"
        return
    fi
    # Auto-detect: prefer nanny when present; null otherwise.
    if command -v nanny-progress >/dev/null 2>&1; then
        echo "nanny"
    else
        echo "null"
    fi
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

progress_init() {
    PROGRESS_TOTAL="${1:-1}"
    PROGRESS_BACKEND=$(_progress_detect_backend)
    # Validate backend is implemented; fall back to null if not.
    if ! declare -F "_progress_${PROGRESS_BACKEND}_set" >/dev/null; then
        echo "[progress] unknown backend '$PROGRESS_BACKEND'; falling back to null"
        PROGRESS_BACKEND=null
    fi
    echo "[progress] backend=$PROGRESS_BACKEND total_phases=$PROGRESS_TOTAL"
    # Install the cleanup trap. Always fires on script exit (success,
    # error, Ctrl-C). Even with the null backend, the trap is harmless.
    trap progress_clear EXIT
}

progress_set() {
    local phase="${1:?usage: progress_set <phase> <percent> [label]}"
    local percent="${2:-0}"
    local label="${3:-}"
    local phase_str="$phase/$PROGRESS_TOTAL"
    echo "[progress] phase $phase_str ($percent%)${label:+ — $label}"
    "_progress_${PROGRESS_BACKEND}_set" "$phase_str" "$percent"
}

progress_phase() {
    local phase="${1:?usage: progress_phase <phase>}"
    local phase_str="$phase/$PROGRESS_TOTAL"
    "_progress_${PROGRESS_BACKEND}_phase" "$phase_str"
}

progress_percent() {
    local percent="${1:?usage: progress_percent <0..100>}"
    "_progress_${PROGRESS_BACKEND}_percent" "$percent"
}

progress_clear() {
    # Hygiene: always called from EXIT trap, even on error/Ctrl-C.
    # Avoid recursion by un-trapping first.
    trap - EXIT
    if [ -n "$PROGRESS_BACKEND" ]; then
        "_progress_${PROGRESS_BACKEND}_clear"
        echo "[progress] cleared (backend=$PROGRESS_BACKEND)"
    fi
}
