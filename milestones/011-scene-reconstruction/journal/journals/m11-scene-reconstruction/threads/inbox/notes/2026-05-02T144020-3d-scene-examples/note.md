---
kind: note
captured: 2026-05-02T14:40:20-07:00
consolidated: false
tags: []
---
# 3D Scene Examples

Captured today: three public benchmark datasets that sparse-view
3D-reconstruction methods (MAtCha, SuGaR, NeuS, MVSplat, etc.) commonly
evaluate against. Submitted to the OLAI research corpus and incorporated
into `orn:personal:research:::3d-reconstruction/examples`.

On disk:
`/var/olai/accounts/ADHI6MoVcbKkX_LYFtkqaA/corpora/personal.research/3d-reconstruction/examples/`

## Triggering need

Planning the chart-encoding `r` truncation experiment exposed a gap: I had
no clean reference for which datasets to validate on, what each tests, or
where to download. The MAtCha repo bundles no sample data — bring-your-own
— and the project page links to the paper but not the underlying
benchmarks. Future-me (or another agent) deserves a captured pointer.

## What each dataset is for

- **DTU** — bounded-object Chamfer benchmark (3-view sparse-view setup).
  MAtCha sets `r = 0.1` here. Best for paper-comparable extreme-sparse
  numbers.
- **Mip-NeRF 360** — unbounded-scene NVS benchmark (5-view setup).
  MAtCha sets `r = 0.4` here AND it's the dataset behind the paper's only
  ablation that touches chart-encoding architecture. The natural target
  for reproducing or extending that ablation.
- **Tanks and Temples** — mid-scale F-score MVS benchmark. Real photometric
  conditions, public ground-truth meshes. Cross-dataset robustness check.

## Decision relevant to M11

For the `r`-truncation sweep, the right validation suite is one Mip-NeRF
360 indoor scene (`bonsai` or `kitchen` — small + clean COLMAP poses
removes SfM-noise as a confound) PLUS our scene 004. That gives one
paper-comparable result and one on-target-distribution result. Decision
pending compute availability + disk budget for the ~1.5 GB scene download.

## Corpus references

- `orn:personal:research:::3d-reconstruction/examples/dtu`
  (proposal `kp-20260502-5674`, incorporated)
- `orn:personal:research:::3d-reconstruction/examples/mip-nerf-360`
  (proposal `kp-20260502-b75a`, incorporated)
- `orn:personal:research:::3d-reconstruction/examples/tanks-and-temples`
  (proposal `kp-20260502-7cf8`, incorporated)
