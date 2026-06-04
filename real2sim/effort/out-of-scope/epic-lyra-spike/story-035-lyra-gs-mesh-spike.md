---
xid: STO-SCN-035
parent: ./epic.md
kind: story
effort: scn
size: M
status: open
date: 2026-06-04
depends-on: []
bd-id: krabby-84e
title: Lyra 2.0 → GS → mesh — research spike (Fletcher request)
assignee: principal
priority: 4
---

# Lyra 2.0 → GS → mesh — research spike (Fletcher request)

## Summary

**Research spike — deliberately light.** Capture and lightly assess Fletcher's proposed alternative pipeline: **Lyra 2.0 (quantized to fit a 5080) → generate Gaussian Splats (GS) → GS-to-mesh**. Output is a short go / defer / reject recommendation with rationale — **not** an implementation, and not a full pipeline bake-off.

## The request (Fletcher, 2026-06-04)

> "…at some point we would like to have ~10 scenes that have good RGB+mesh quality that the robot could walk around in. More broadly, would like to be able to take short videos of some nearby construction sites, reconstruct and show robot walking around in them… I think you could do the whole project with Lyra 2.0 quantized to fit on 5080, then run it to generate GS, then run GS to mesh… I anticipate the agent(s) will need to iterate on the epic a bit as they see issues."

**GS = 3D Gaussian Splatting** — a photoreal scene representation (millions of optimized 3D Gaussians). "GS→mesh" extracts collision geometry from it. This is the grant's **Appendix B** (3DGUT/Gaussian) "RGB for looks, mesh for physics" dual model — previously optional because M11 is depth+PPO only.

## Why this is in the TX (out-of-scope) bucket

Framing given to Fletcher (and agreed):

- **Scope creep.** It's a layer of complexity to *improve the quality* of 3D scene capture — possibly a good tech solution, but "extra" beyond what the contracted work requires (M11 prioritizes collision geometry over visual fidelity; photoreal RGB is a documented non-goal).
- **Fidelity risk.** A generative GS step may produce scenes that **don't match the reality** of the landscape that was actually captured.
- **Doesn't solve the real ceiling.** It does **not** address consumer-grade hardware being unable to regenerate *large* scenes without a multi-phase approach (camera "spines" + mesh blending) — itself out of scope and tracked separately under **EPI-SCN-SCENE-SYNC** (`STO-SCN-025`).
- **The actual pain (deltapunctum):** "needing structured processing pipelines to run the experiments within to compare the different outputs." The blocker isn't any single model — it's the **experiment harness** to fairly compare pipeline outputs. Surface this as the spike's most likely real finding.

## Definition of Done (light — do not over-build)

- [ ] Request + scope framing captured (this story)
- [ ] Quick feasibility read: does Lyra 2.0 quantize onto a 16 GB 5080? what does it emit (standard 3DGS?)? what's the GS→mesh path/quality?
- [ ] Note shared constraints it does **not** solve — monocular scale ambiguity (`STO-SCN-016` ★), large-scene fusion (`EPI-SCN-SCENE-SYNC`)
- [ ] Call out the structured experiment-comparison harness as a prerequisite (the deltapunctum point)
- [ ] One-paragraph recommendation: pursue / defer / reject, tied to scope and to the GS-vs-MAtCha(TSDF) tradeoff (cf. `STO-SCN-002`)

---
_Research spike in the TX (out-of-scope) bucket. No implementation expected. Created 2026-06-04 from a Fletcher request + operator scope framing._
