# M11 Scene Processing — the process, end to end

> **Canonical, operator-facing documentation of how M11 turns a real space into a
> ranked, simulation-ready scene.** This replaces `real2sim/RECIPES.md` as the entry
> point (RECIPES.md is now a pointer here). The goal of this doc set: the process is
> **documented**, **automated**, and **easy** — no more winging it.
>
> Grounded against the code on branch `jdp/m11-real2sim` (2026-06-15). When code and a
> doc disagree, **code wins — fix the doc** (cite the file you checked).

---

## The process is five tiers (T0–T4), seven phases

We run every scene through the same pipeline. It is organized as **five tiers T0–T4**;
the reconstruction tier (T3) has three distinct phases (pre / processing / post), so the
process is **seven documented phases** total. Each phase has its own doc:

| Tier | Phase | Doc | What it produces |
|---|---|---|---|
| **T0** | Scene Ingress & Creation | [`T0-scene-ingress.md`](T0-scene-ingress.md) | source video/images → a content-addressed image pool + `capture.json`; the scene-folder layout |
| **T1** | Scouting & Spine | [`T1-scouting-spine.md`](T1-scouting-spine.md) | a posed **spine** (full camera trajectory), segmented for long pools; the **scout** splat + verify surface |
| **T2** | View Selection | [`T2-view-selection.md`](T2-view-selection.md) | the **1…N virtual cameras** that will be rendered for comparison (operator-authored) |
| **T3a** | Reconstruction — Pre-Processing | [`T3a-reconstruction-preprocessing.md`](T3a-reconstruction-preprocessing.md) | **model choice** (matcha / DA3 / …), **model settings**, and the **best-N "best cameras" auto-selection** subset |
| **T3b** | Reconstruction — Processing | [`T3b-reconstruction-processing.md`](T3b-reconstruction-processing.md) | choosing a GPU host, **sync→host**, delegating the run, **monitoring**, **sync←outputs** to the workstation/store |
| **T3c** | Reconstruction — Post-Processing | [`T3c-reconstruction-postprocessing.md`](T3c-reconstruction-postprocessing.md) | the **N renderings** from the scene mesh / `.blend` |
| **T4** | View Ranking | [`T4-view-ranking.md`](T4-view-ranking.md) | the **studio UI** ranking — operator-judged best reconstruction |

> **Naming note (avoid a collision).** In the grant vocabulary, **T0 = SfM/pose-solving**
> and **T1 = MVS/dense-reconstruction** (Matcha is "primary T1"). Those are *task tiers* on
> a different axis; here **T0–T4 are the M11 *workflow* tiers**. The grant's SfM-T0 and
> MVS-T1 both live *inside our T3* (pre-select solve / reconstruct). Don't conflate the two.

---

## End-to-end flow

```
T0  capture.json → INGEST ──────────────► content-addressed image pool
T1  PRE-CULL → SOLVE (FastMap) → COVIS(gate) → [SPINE segment/register/fuse if long]
        → SCOUT (DA3 splat in solve gauge) → ★VERIFY (operator)         = the posed SPINE
T2  author 1…N virtual render cameras (operator, in the scout/blend)     = cameras.json
T3a SELECT best-N (voxel coverage) → FINAL-N subset · choose model+settings
T3b point primary→subset · reconstruct on a GPU host (sync→delegate→monitor→sync back)
T3c build .blend → render the N views                                    = renders/<view>.png
T4  studio (:8091) — operator ranks the variants per view                = scores.jsonl
```

★ = the human-in-the-loop verification points (T-020). Everything else is automated.

---

## Cross-cutting facts (true for every scene)

- **Store-shape v4 — content-addressed DAG-of-DAGs** (HUG-SCN-005). `v4exec.py` is the
  **sole writer**; identity = `hash(resolved inputs + tunable/frozen settings + algo@version)`,
  so re-running a node with the same inputs is a NOOP. **Never hand-edit files under
  `/var/krabby/scenes/`** — go through `v4exec`.
- **GPU-only solver/reconstruct.** Solve, scout, and reconstruct run on a GPU host over SSH
  (`--host U@H`). The fleet GPU box used here is **`tbeeprz`**.
- **Gauges.** The **solve gauge** (FastMap `sparse/0`) is the reference frame and the spine.
  DA3 gaussians live in DA3's *normalized* frame and are registered → solve gauge via
  `scout_gauge.json` (Umeyama of predicted poses, STO-SCN-105). Spine fusion composes
  **105 ∘ 098** (gs→solve→global).
- **Hard limits** (apply everywhere — see each phase doc for the rest):
  | Limit | Value | Source |
  |---|---|---|
  | SfM solve ceiling | ≈300 frames / 16 GB GPU (mast3r); FastMap scales past it | RECIPES; STO-SCN-093 |
  | **MAtCha TSDF multires-merge OOM** | **≥17 cameras × mesh_res 1024 OOMs 16 GB — fixed in matcha image ≥0.2.2** (`multires_oom`); `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` required | STO-SCN-053/056 |
  | MAtCha target view band | **24–30 frames** (sparse-view method) | `images/matcha/NOTES.md` |
  | Normalize preproc | long-edge 2048 → −28 % training VRAM (headroom for larger N) | STO-SCN-043 |
- **Reconstruct reads `primary`.** `reconstruct-matcha` / `reconstruct-da3` reconstruct
  whatever subset is tagged **`primary`** (no `--subset` flag). Re-pointing `primary` is a
  **deliberate operator act (locked #1)** — see T3b.
- **DA3 is the evaluation branch** (CC BY-NC — research evaluation only).

---

## Running a new scene — the checklist

1. **T0** — write `scenes/<scene>/capture.json`; `v4exec ingest <scene> --host tbeeprz`.
2. **T1** — `v4exec precull`, `solve`, `covis` (validity gate), [`spine` if long], `scout`,
   then **verify** in the viewer.
3. **T2** — author the comparison render cameras (`/camera-save` in the scene `.blend`).
4. **T3a** — `v4exec select … --n <N>`; pick model + settings (matcha ≤15 to dodge the OOM
   cliff unless on image ≥0.2.2; DA3 can take more frames).
5. **T3b** — point `primary` at the FINAL-N subset; reconstruct on `tbeeprz`; monitor; sync back.
6. **T3c** — build the `.blend`; render the N views.
7. **T4** — open studio (`:8091`); rank the variants.

Each step's exact command, settings, gotchas, and automation status live in the per-phase doc.

---

## Automation status — automated *per step*, operator-*orchestrated*

**The honest end-to-end picture (2026-06-15): this process is NOT push-button.** Every step is a
hardened single command, but a human chains them — there is no driver that takes a scene from
`capture.json` → ranked variants. Three categories:

| | What | Why |
|---|---|---|
| ✅ **Automated step** | `ingest`, `precull`, `solve`, `covis`, `spine-*`, `select`, `scout`, `reconstruct-*`, `reconstruct-da3-scout`, `render-missing` | each is one content-addressed `v4exec`/`v4job` command; re-runs are NOOPs |
| ⚠️ **Manual glue (automation debt)** | (1) **no orchestrator** — you run ~10 commands in order, hand-carrying `solve`/`covis`/`scout`/`subset`/rep ids between them; (2) the **`primary` re-point** before every reconstruct (a deliberately *locked* act, but still manual); (3) per-step gotchas that aren't yet self-healing (e.g. a FastMap-solve variant silently fails to render until its `cameras.json` is emitted — see [T3c](T3c-reconstruction-postprocessing.md) "Render camera contract") | the steps were hardened before the chaining was |
| 🟢 **Intentional human gate** | `capture.json` declaration (T0), **view authoring** (T2), **verify** (T1), **ranking** (T4) | T-019/T-020 — the operator *should* drive these; the goal is not 100% automation, it's removing the *incidental* toil |

**So "fully automated" is not yet true.** It would mean a **scene driver** that auto-resolves
ids and chains T0→T3c, pausing at the four intentional gates. That driver does not exist; today
the operator is the orchestrator. (Concretely: the DA3-24 variant for 001-patio was *not*
push-button — it took manual id lookup + two code fixes, STO-SCN-127.)

The per-phase "Automation status" lines mean **"this step is a single command,"** not "the
process is hands-off." Don't read them as end-to-end automation.

---

## Status of this doc set

Tracked under **`EPI-SCN-M11-PROCESS-DOCS`** (within `DES-SCN-REPRO`). Per-phase docs are
filled by STO-SCN-114…120; this index + the RECIPES.md stub is STO-SCN-113. Until a phase
doc is marked complete, treat `real2sim/RECIPES.md` (the legacy recipe) as the fallback
source for that phase.
