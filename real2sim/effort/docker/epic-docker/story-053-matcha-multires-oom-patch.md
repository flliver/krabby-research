---
xid: STO-SCN-053
parent: ./epic.md
kind: story
effort: scn
size: S
status: in-progress
date: 2026-06-10
depends-on: []
bd-id: krabby-vic
assignee: krabby
---

# krabby-matcha 0.2.2: multires-merge OOM patch baked into image

## Summary

`krabby-matcha:0.2.2-selfcontained` — same image as 0.2.1 plus
`patch_matcha_multires_oom.py` baked in, so TSDF extraction at
mesh_res 1024 no longer OOMs 16 GB fleet GPUs. Built and deployed to
dbeeprz.

## Context

Parent: [EPI-SCN-DOCKER](./epic.md). During 013-basement's first
reconstruction (STO-SCN-052), `scripts/extract_tsdf_mesh.py -c default`
OOM'd on sbeeprz (RTX 4080 16 GB) in the multires merge:
`render_multires.py` projects ALL mesh vertices into ALL cameras as one
tensor — ~43M verts x 17 cams -> a single >3 GiB allocation on top of
~13 GiB resident. Real capacity OOM (expandable_segments was set;
71 MiB reserved-unallocated). No tool flag controls it; every fleet
GPU is 16 GB, so host-hopping cannot dodge it.

## Problem

The fix was validated as a runtime bind-mount overlay on sbeeprz
(43.3M raw / 30.8M post verts, no OOM) — but a /tmp overlay on one
host is not deployment. Any other host, or s after a /tmp wipe,
re-hits the OOM. The patch must live in the image with the other five
patch scripts.

## Design

- `images/matcha/patches/patch_matcha_multires_oom.py`: chunks the
  vertex visibility test over 2M-vert slices (order-independent
  `any(dim=0)` reduction — semantically identical; transient memory
  ~0.5 GiB/slice). Anchor-asserted (hard error if upstream drifts),
  idempotent (no-op if already patched).
- Dockerfile: patch stage AFTER checkpoint download (cache-friendly);
  patchset label -> `curope,includes,tetra_cmake,torch_load,multires_oom`.
- Tag: `0.2.2-selfcontained`, built on dbeeprz (has 0.2.1 layer cache).

## Definition of Done

- [x] Patch script verified against the pristine 0.2.1 image:
      applies, compiles, idempotent.
- [ ] `krabby-matcha:0.2.2-selfcontained` built on dbeeprz.
- [ ] Deployed image label-checked (`patchset` includes multires_oom)
      and a container-run smoke test confirms the patched code is
      present.
- [ ] Distribution tail noted: s still runs 0.2.1 + /tmp overlay;
      t/b out of rotation (operator: kids home) — ship there later
      (extends the STO-SCN-038 tail).

## Status notes

- 2026-06-10: Minted after the runtime overlay fixed 013's TSDF OOM.
  Patch script + Dockerfile staged; build starting on dbeeprz.
