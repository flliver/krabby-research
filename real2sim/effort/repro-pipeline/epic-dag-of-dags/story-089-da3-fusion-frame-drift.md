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

## Definition of Done

- [ ] Root cause verified against da3_infer_gs.py exports in 0.2 vs
      0.4 (npz extrinsics convention).
- [ ] A frame-sanity check added to fusion (e.g. compare fused-mesh
      up/floor vs the anchor gauge — catches global-frame lies the
      camera-residual gate misses).
- [ ] 2XEI re-fused correctly or retired permanently.
