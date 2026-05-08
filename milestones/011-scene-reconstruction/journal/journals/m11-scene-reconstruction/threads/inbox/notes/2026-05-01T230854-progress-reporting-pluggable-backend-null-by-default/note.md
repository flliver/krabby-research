---
kind: note
captured: 2026-05-01T23:08:54-07:00
consolidated: false
tags: []
---
# Progress reporting: pluggable backend, NULL by default

Architectural decision recorded. Initially we built `lib_progress.sh` as a thin wrapper around `nanny-progress` (the baeprz-fleet helper). On reflection that's wrong — it would force every krabby contributor to either install the baeprz Ansible role or have their scripts log errors when calling a missing helper.

**Krabby is broader than the baeprz fleet.** A team member running on a fresh Mac, a Jetson, a cloud VM, or a friend's Linux box should be able to run any of our pipeline scripts unchanged and not care that a separate sub-organization happens to also use them on a richly-instrumented fleet.

## Design

The lib now has a pluggable backend interface. Run scripts call the public API:

```bash
source ~/lib_progress.sh
progress_init <total_phases>
progress_set <phase> <pct> [<label>]
progress_phase <phase>
progress_percent <pct>
progress_clear        # (auto-fires on EXIT)
```

The lib delegates each call to the active backend's implementation. Backend selection happens once at `progress_init`:

1. If `KRABBY_PROGRESS_BACKEND` env var is set → use that.
2. Else auto-detect: try `nanny` (`command -v nanny-progress`), else `null`.

## Built-in backends

- **`null`** — every operation is a no-op. The default fallback. Krabby members without any progress infrastructure get this automatically. Scripts run identically; they just don't broadcast progress anywhere.
- **`nanny`** — pushes to the baeprz `beeprz dash` via `nanny-progress`. Auto-detected when the helper is on PATH.

## Adding a future backend

Three lines of changes:

1. Define four functions: `_progress_<name>_set`, `_progress_<name>_phase`, `_progress_<name>_percent`, `_progress_<name>_clear`. Each takes the same args the public function takes.
2. Add a detection branch in `_progress_detect_backend` (or rely on the env-var override for explicit selection).
3. Done. No script changes anywhere.

Ideas for future backends:
- `stdout` — prettier terminal output (e.g., a single redrawn status line).
- `file` — write to a JSON status file (consumed by external tools).
- `webhook` — POST to an arbitrary URL (Discord, Slack, Healthchecks.io).
- `mqtt-direct` — bypass `nanny-progress`, talk to MQTT broker directly.

## Why the EXIT trap matters even with null backend

`progress_init` always installs a `trap progress_clear EXIT`. This is harmless for the null backend (clear is a no-op there) but **critical** for backends with retained state (the dashboard footgun). The script doesn't have to know which backend is active to do the right thing on Ctrl-C / OOM / scripted-exit — the trap handles it uniformly.

## Where the lib lives

- Workspace: `milestones/011-scene-reconstruction/workspace/lib_progress.sh` (canonical source).
- Hosts: `~/lib_progress.sh` on bbeeprz and tbeeprz. Scripts source it as `source ~/lib_progress.sh`. The path is host-portable since `$HOME` resolves correctly anywhere.

When other krabby contributors check out the workspace, they can either:
- Symlink/copy `lib_progress.sh` into `~/` on their dev machine, or
- Source it directly from the workspace path (works fine with the null backend on a personal machine).

## Status of the in-flight job at the time this note was written

`b9b9v56m3` was the curated mesh re-orient job for variants 12 / 12-strong / 16-strong. It pre-dates this lib_progress.sh refactor (and even pre-dates lib_progress.sh integration in scripts at all — I was pushing manual updates from outside, which is the anti-pattern this note exists to retire).

Going forward, every wrapper script we generate sources `~/lib_progress.sh` and calls `progress_init` + `progress_set` from inside the script.

## Pivot to the broader principle

Same pattern is worth applying elsewhere. Anywhere we'd otherwise hardcode a baeprz-fleet dependency:

- Logging (writes a log file vs. publishes to a centralized log collector)
- Error reporting (prints trace vs. sends to a central error tracker)
- Metric collection (in-script summary vs. pushes to a metrics service)

Default-NULL + drop-in-when-available keeps krabby code portable.
