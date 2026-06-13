---
xid: MSG-PROJ-001
content-path: /private/var/krabby/research/messages/2026-06-09/19-27-50-handoff-1927/001/msg-handoff-1927.md
kind: msg
effort: proj
status: abandoned
date: 2026-06-09
to: krabby
from: krabby
topic: handoff-1927
bd-id: krabby-93v
---

# Handoff from previous krabby session

_(write your message body here)_

## Status notes

- 2026-06-09: Filed.

# Handoff from Previous Session

## What Was Happening
Reproducibility + data-processing sprint on the M11 real2sim pipeline (operator
mandate HUG-KRB-002: config-driven pipelines, no prototypes). This session:
runner v1 shipped+validated (3-way metric equivalence on 004 run-12-strong);
006/007-kubota reconstructed via runner (007 through the new normalize preproc,
−28% VRAM); fleet LFS transport fixed end-to-end (git-lfs-transfer on j, Ansible'd
by ops); fleet auto-sync (engineer, STO-SCN-030) live + convergence-proven;
50 GB LFS backfill Mac→j; .gitattributes case-sensitivity fix; runner hardened
(expected-outputs gate rc97, LFS-pointer guard, dense_regul verified).

**IN FLIGHT AT PARK: dtu-bicycle reproduction** (`run-12-dense-strong-repro-20260609`)
running on tbeeprz via `/tmp/run_transform.py` — launched over ssh from this
session, so it may receive SIGHUP when this session exits. **First action next
session: check `t:~/krabby/scenes/dtu-bicycle/pipeline-matcha/run-12-dense-strong-repro-20260609/transform-01-matcha/results.json`** —
if absent/failed, delete the partial run dir (may need sudo: container writes as
root) and re-run: `ssh t.pski.org 'cd ~ && python3 /tmp/run_transform.py dtu-bicycle/pipeline-matcha/run-12-dense-strong/transform-01-matcha --as 12-dense-strong-repro-20260609'`.
All gates verified green this time ("12 images found", GPU active).

## What Needs to Happen Next
1. dtu repro: confirm/re-run (above) → commit+push from t.
2. TSDF-extract the repro (the whitepaper mechanism: operator explicitly wants
   quality judged via multires-TSDF + cam_ref CYCLES render, NOT bbox renders).
   Use the same script the original used (`scripts/extract_tsdf_mesh.py` in
   ~/scratch/MAtCha, or train.py --use_multires_tsdf) — propose as transform-02.
3. Render repro TSDF from cam_ref (`scene_tsdf_ref.blend` mechanism — see
   /tmp/render_ref.py pattern; cam_ref pose in dtu-bicycle/_unsorted/comparison_views.json)
   → side-by-side vs dtu-bicycle-cam_ref-original-render.png + PAPER-reference → Dropbox.
4. Camera comparison repro-vs-original (Umeyama; pattern in STO-SCN-041 notes / comparison.md).
5. Update STO-SCN-041 with dtu baseline #2; tick 042/043 operator-verdict items when given.

## Key Context
- Operator's open verdicts: 006 mesh quality (042), normalize-as-default (043),
  STO-SCN-030 close (engineer's strict-clean design question included),
  GitHub branch one-click delete (jeremyprz/beeprz git-lfs-transfer).
- Renders to Dropbox root is the established show-me pattern (~20 PNGs there).
- Runner deploys as pinned copy at t:/tmp/run_transform.py (md5 must match
  research/real2sim/run_transform.py — redeploy via scp after edits).
- Scene store: hub j:/games/krabby/scenes, clones Mac /var/krabby/scenes +
  fleet ~/krabby/scenes; auto-sync timers 30min on t/b/s/d (gate:
  ~/.config/krabby/scene-sync.toml); Mac is author seat (sync disabled).
- Container-as-root follow-up for runner (--user) noted in STO-SCN-041/040.
- "2"×4 operator messages near session end were treated as continue-signals
  (possibly stray); flagged to operator, no objection raised.

## Active Files
- research/real2sim/run_transform.py (runner v1+gates; committed 368cb73..head)
- research/real2sim/normalize_photos.py (committed)
- /tmp/render_all.py, /tmp/render_ref.py (Mac, uncommitted render helpers —
  candidates for real2sim/ if renders become a transform)
- Story files under real2sim/effort/repro-pipeline/epic-pipeline-runner/ (039,041,042,043)
- Research repo has unpushed local commits on jdp/m11-real2sim (push not yet requested).

## Beads XIDs
- `STO-SCN-039` — in-progress; runner v1 shipped+validated, registry/040 follow-ups remain
- `STO-SCN-042` — in-progress; 006 pilot done except operator mesh-quality verdict
- `STO-SCN-043` — in-progress; 007 done, awaiting operator adopt-normalize-default verdict
- (also open, krabby-adjacent: STO-SCN-041 harness — dtu baseline #2 lands next session)

## Addendum (post-park, same session)
The dtu loop CLOSED before exit: run succeeded (800 s/10,574 MiB), pushed
(3c58209 + TSDF commit), REPRODUCED (scale 0.9976, residuals ≤0.174%), TSDF
extracted, cam_ref render delivered to Dropbox. Steps 1–4 of "What Needs to
Happen Next" are DONE; step 5 remains (041 DoD formalization + operator
verdicts). Frame-composition recipe + opencv-quat gotcha recorded in 041 notes.
