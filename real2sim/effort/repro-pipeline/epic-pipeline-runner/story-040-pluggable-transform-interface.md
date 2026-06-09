---
xid: STO-SCN-040
parent: ./epic.md
kind: story
effort: scn
size: M
status: open
date: 2026-06-09
depends-on: []
bd-id: krabby-31r
title: Pluggable transform interface + registry
priority: 2
assignee: devex
---

# Pluggable transform interface + registry

## Summary
Define the plugin contract a transform must satisfy (name, image, parameter schema, input/output declaration, invocation template) and a registry the runner consults — so adding mast3r/colmap/conditioning/TSDF-extract is configuration, not new runner code.

## Context
The store already names transforms (`transform-01-matcha`, `transform-01-legacy`); the conditioning chain (orient → decimate → color → cull, B1–B4) is the obvious second plugin family, turning multi-step post-processing into declared pipeline stages. Operator direction: "configuration-driven data pipelines with pluggable transformations" (HUG-KRB-002).

## Definition of Done
- [ ] Transform contract documented + schema'd; registry file in-repo
- [ ] `matcha` registered (proves reconstruction-class transforms)
- [ ] One conditioning step registered (proves CPU/post-processing-class transforms)
- [ ] Runner (039) executes both through the same interface
- [ ] Multi-transform run: spec chain executes in order with per-transform results.json
