---
kind: note
captured: 2026-05-01T22:31:39-07:00
consolidated: false
tags: []
---
# beeprz dash: in-flight progress reporting via nanny-progress

New baeprz-fleet capability shipped 2026-05-01. Initial form (commit a5dcbc6) required `sudo -u nanny`. Corrected later the same day (commit **107cb3c**, the canonical form going forward) to be runnable as any user without sudo. **Use the corrected form below.**

## What it is

Two new retained MQTT topics per host:

```
${prefix}/${host}/work/phase     "X/Y" or "-"   (e.g. "2/4")
${prefix}/${host}/work/percent   "0".."100" or "-"
```

The `beeprz dash` dashboard now shows PHASE + % columns (the SESSION column was dropped). Default state is `-`/`-` — nanny initializes both on startup and never touches them again.

A helper at `/usr/local/bin/nanny-progress` provides the write surface, runnable as any user (no sudo, no group membership needed):

```bash
nanny-progress set <phase> <percent>
nanny-progress phase <X/Y>
nanny-progress percent <0..100>
nanny-progress clear
```

Under the hood: world-readable MQTT credentials in `/etc/nanny/`, with the cert publish-scoped to the calling host's topic subtree only — you cannot affect other hosts' readings even if you wanted to.

**Verified installed on bbeeprz and tbeeprz as of 2026-05-01.** Tested live (push set / percent / clear; no errors).

## Why we care

Long MAtCha runs (full pipeline ≈ 15 min, post-processing adds 4-5 min more) are exactly the workload this is built for. With it deployed, the operator on the dashboard sees:

```
…   PHASE   %
tbeeprz   3/8    65    (free-Gaussians refinement)
bbeeprz   -      -     (idle)
```

instead of guessing.

## Hygiene rule (the footgun this exists to avoid)

**ALWAYS `clear` at end — success or failure path.** A leftover stale "2/3 47%" sitting on the dashboard for hours after the job died is the same shape as the stay-awake leftovers we cleaned up Tuesday.

The right pattern in shell is a `trap` so the cleanup runs even on Ctrl-C, OOM, or unexpected exit:

```bash
trap 'nanny-progress clear' EXIT
```

## Status as of capture (and update)

**Initial capture (22:31 PDT):** `nanny-progress` not yet on bbeeprz or tbeeprz; lib_progress.sh built to degrade cleanly when missing.

**Updated (22:46 PDT):** baeprz-ops shipped commit **107cb3c** (the no-sudo form) and rolled it out to both bbeeprz and tbeeprz. `command -v nanny-progress` returns `/usr/local/bin/nanny-progress` on both. Live-tested: push set/percent/clear sequence works as any user.

`lib_progress.sh` updated to drop the sudo wrapping. Subsequent runs that source it will report progress automatically.

## Our integration: `lib_progress.sh`

Lives at `milestones/011-scene-reconstruction/workspace/lib_progress.sh`. Sourceable from any runner script:

```bash
source ~/krabby/workspace/lib_progress.sh
progress_init 8                              # 8 phases total
progress_set 1 0 'scene graph'               # entering phase 1
# ...do phase 1 work...
progress_percent 50
progress_set 2 0 'chart alignment'           # entering phase 2
# ...etc...
# progress_clear runs automatically via EXIT trap installed by progress_init
```

If `nanny-progress` is missing, each helper logs to stdout only — no errors, no warnings, no failed `sudo` calls. The run still runs; the dashboard just stays at `-`/`-`.

## Standard MAtCha pipeline phase decomposition

For a full curated run (MAtCha + B1-B4 post-processing), 8 phases is a reasonable granularity:

| Phase | Stage | Approx % of total wall-clock |
|------:|-------|------:|
| 1/8 | SfM scene graph + alignment | 15% |
| 2/8 | chart alignment (1000 iters) | 20% |
| 3/8 | free-Gaussians refinement (3000 iters) | 35% |
| 4/8 | tetra mesh extraction | 10% |
| 5/8 | B1 orient + decimate | 10% |
| 6/8 | B4 project_color | 5% |
| 7/8 | B2 cull | 3% |
| 8/8 | local rsync + B3 Blender (if part of script) | 2% |

The exact phase boundaries don't matter — the operator on the dashboard mostly wants "is it running, what's the rough progress, is it stuck."

## Next time we set up matcha-build

Worth adding `nanny-progress` to the post-provisioning checklist on any host we use heavily (bbeeprz, tbeeprz). Until then, runs log progress only to their own /tmp logs.

## Related

- Operational lesson note `2026-05-01T222605-operational-lesson-cuda-disappears-from-long-running-container` — also a baeprz-side reliability concern. Both lessons could go in a future "ops" thread once we accumulate more.
