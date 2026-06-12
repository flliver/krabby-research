---
xid: STO-SCN-089
parent: ./epic.md
kind: story
effort: scn
size: M
status: draft
date: 2026-06-11
depends-on: []
bd-id: krabby-tmb
---

# BUG: studio-era da3 fusion 140° yaw off despite passing alignment gate — suspect npz convention drift 0.2->0.4 (blocks trusting 088 re-fusions)

## Summary

BUG (operator-caught, 2026-06-11): the studio-era da3 fusion
(2XEIMLEA5NBY, 006) renders upside-down/sideways — measured ~140°
yaw vs its sibling fusion (2NUKSRCT455R) even though BOTH fusion
records claim "matcha-oriented frame" with healthy alignment
(scale 0.347, residual 2.9% — gate passed).

## Evidence (measured)

- Mesh bboxes: both floor-true (z∈[-0.06,1.97]) → internally
  consistent fusions; principal-axis yaw differs ~140°.
- Render path exonerated: identity Procrustes (8/8 anchors, 0.0000
  residual) with the correct 8-strong gauge still renders 2XEI
  sideways; 2NUK renders correctly with the same machinery.
- The two fusions differ by producer: 2NUK from the d-run npz
  (krabby-da3 0.2 era), 2XEI from the t studio-run npz (0.4).

## CLEAN-SPECIMEN FINDINGS (2026-06-12, native 009)

The locked-#11 native rebuild reproduced the failure with zero
archaeology: cameras pair + align at 8-33 MILLIMETERS per index, yet
the fused mesh lands median 0.44 m (ICP suggests ~50deg/1m rigid-ish)
from the matcha reference in the same gauge. Ruled out by
measurement: extrinsics convention (w2c verified, c2w 76% residual),
npz depth scaling (0.2/0.4 eras byte-similar), open3d local-version
convention (fusing with inverted extrinsics lands equally off),
camera correspondence (per-index mm). Remaining suspect class:
DA3's depths are internally inconsistent with its own camera
baselines at this scale/views count — an error the camera-residual
gate STRUCTURALLY cannot see.

Mitigation shipped: `v4exec verify-frame` — fused-mesh-vs-reference
geometry gate (0.15 m median), writes measured verdict + rankable
flag in-graph. 009's fusion: FAIL -> excluded from runoff.

## Suspect

npz extrinsics convention drift between krabby-da3 0.2 and 0.4
(w2c vs c2w would keep each fusion internally consistent while
shifting its global frame — exactly the symptom). UNVERIFIED (T-002).

## Why it matters

STO-SCN-088's executor will re-fuse da3 meshes with current images —
if the convention drifted, every new fusion lands mis-framed while
passing the residual gate (the gate checks camera fit, not mesh
frame). Must be root-caused BEFORE 088 trusts its fusions.

## Mitigation applied now

2XEIMLEA5NBY flagged `rankable: false` (variance-pair evidence for
STO-SCN-075 stays valid — counts, not pose); its render removed;
scan/payload filter excludes non-rankable meshes. 006's runoff list
is clean again.

## RESOLUTION (2026-06-12): da3-fuse@2 — register onto the reference

The convention-drift suspect was WRONG (eras byte-similar, both w2c).
The real shape, established by measurement across four specimens with
the corrected metric (coarse-to-fine ICP 0.5/0.25/0.1 against the
SAME-GAUGE matcha reference — the earlier 10deg/0.17m reading of
"perfect" 006 was against a wrong-era reference):

| specimen | operator verdict | correction | fitness |
|---|---|---|---|
| 006 2NUK (8 views) | perfect | 1.3deg / 0.05 m | 0.91 |
| 007 CZP2 (9 views) | success | 2.4deg / 0.04 m | 0.84 |
| 009 (7 fwd-walk views) | wrong | NO rigid pose exists | <=0.25 |

- Fusion code exonerated: same code produced 007 near-perfect and
  009 wrong. The variable is capture geometry — forward-walk views
  condition DA3 poorly and its geometry comes out NON-RIGIDLY warped
  (75%+ of the surface >10 cm off at ANY rigid placement).
- Depth anchoring (da3-fuse@1) measured ineffective: per-view anchors
  ~=1.0 (DA3 depths already agree with sparse points at projected
  pixels); the warp lives elsewhere. Retired.
- Naive single-pass ICP at 1.0 m corr is UNSTABLE: found a degenerate
  99-deg basin on 009's corridor geometry at fitness 1.00 (renders
  proved it garbage). Shrinking schedule + physical bound
  (30deg/1m — cameras pair at mm, bigger corrections are impossible)
  guards it.

Shipped (`v4exec.py`, da3-fuse@2): fuse + c2f ICP registration onto
the matcha reference mesh (now a resolved input — in the identity
hash). Correction applied when within bounds; recorded as
`measured.self_alignment` either way (it IS DA3's per-capture quality
score). Degenerate registration => camera-aligned placement +
`rankable: false`. Honest caveat in metadata: a registered mesh is no
longer independent evidence of DA3 global accuracy — fine for the
evaluation branch, which compares geometry quality in a shared gauge.

009's 2I7XIVWBGWMY: flagged unrankable (correct — its DA3 geometry is
genuinely unalignable). KBTX (@1 anchored) retired.

## Definition of Done

- [x] Root cause verified: NOT npz convention drift (0.2/0.4
      byte-similar, w2c both); DA3 non-rigid geometry warp on weakly
      conditioned captures, invisible to the camera-residual gate.
- [x] Frame-sanity check added to fusion: in-graph c2f ICP
      registration vs reference with degeneracy guard + rankable flag
      (supersedes the flawed verify-frame median-distance gate, which
      had failed the operator-verified-perfect 006 at 0.155 m).
- [x] 2XEI retired (rankable:false); 009 fusions superseded by @2.
- [ ] OPERATOR: confirm 007 CZP2-class registered fusions read as
      aligned in the runoff; capture guidance (avoid forward-walk-only
      captures for the da3 branch) noted for future scenes.
