---
xid: HUG-SCN-005
kind: hug
effort: scn
status: active
date: 2026-06-11
author: operator
bd-id: krabby-v5h
---

# DAG of DAGs — operator direction for pipeline decomposition

## Quote

Now that I'm examining and thinking things through more carefully - I think we might need a "DAG of DAGs" solution. I want to talk this through so we don't leave anything out:
  
Generally, this is how I'm thinking of the problem space (for our M11 milestone):
  
**GIVEN**: Either a VIDEO or N_IMAGES

### Settings

I general - settings seem to fall into 3 categories:

- tunable - a truly modifiable setting typically with known enum values, range, etc
- unexplored - a possibly tunable setting that from a whitepaper's perspective is treated as a constant
- constants - the task essentially hardcodes a value. considered "execution facts"

### Identity

An 
 
- type: (str) - kebab style type
- hash: (str) - [0-9A-Z]+ hash of data

### Tasks

TASK: hash-n-images
- INPUT: N images
- SETTINGS: (none)
- OUTPUT:
  * N hashes
  * HOH: Hash of Hashes (N images)
- IDENTITY: n-images/<hash>

> NOTE (locked #4): everything below called "Pipeline" is a **task**
> in the locked vocabulary (task / graph / job; "pipeline" and
> "stage" retired as type names).

## Non Comprehensive List of Pipelines

### Pipeline: Video-To-Images

A simple pipeline that converts a video to a set of images.

- NAME: `video-to-images`
- INPUTs: 
  * scene: str
  * video: str
- DEDUCED_INPUT:
   * `scenes/<scene>/videos/<video>/video.<ext>`
- SETTINGS:
  * image_type: PNG|JPG
- OUTPUTs:
  * N images stored in: `scenes/<scene>/images/<hash>/` with:
    - `image.jpeg`
    - `metadata.json`
  * extraction info: `scenes/<scene>/videos/<video>/video-to-images/`
    - `metadata.json`
    - `conversion.log`

### Pipeline: Images Subset

Produces a subset of the images available within a scene. 
Calculate "hash" by:
- sorting all image hashes in a list
- hash those hashes to produce a single hash

- NAME: `images-subset`
- INPUTs:
  * scene: str
  * hashes: set(str) - A set of the hashes of the images making up the subset
  * mechanism: HUMAN
  * label: str - kebab label
  * primary: bool
- DEDUCED_INPUT:
  * hash: str - sort the ingress hashes and hash those
- SETTINGS:
  * (any/all settings from the viser UI)
- IDENTITY: `scenes/<scene>/images/subsets/<hash>`
- OUTPUTs:
  * `scenes/<scene>/images/subsets/<hash>/subset.json` (pointers to files)
  * `scenes/<scene>/images/subsets/primary` -> `<hash>` (symlink if primary=true)

### Pipeline: Orient Cameras

Produces a file that orients the camera (image) positions within 3D space.

pipeline-3: orient-cameras
- INPUT:
  * scene: str
- IDENTITY: `scenes/<scene>/images/subsets/primary/cameras`
  > SUPERSEDED (locked #1/#2): a mutable ref (`primary`) can never
  > appear in an identity — identities use resolved hashes; the
  > placement is `.../cameras/<solve_identity>/orient/<orient_identity>/`.
- SETTINGS: (?? ... how have we been orienting cameras)
  > ANSWERED (locked #2, see Direction): today we orient the MESH
  > (RANSAC floor-fit, STO-SCN-004) per-run and back-apply to
  > cameras. This spec inverts that: orient primary's cameras ONCE;
  > everything downstream is born oriented. Split into two tasks:
  > `solve-cameras` then `orient-cameras` (defined in Direction).
- OUTPUTs:
  * `scenes/<scene>/images/subsets/primary/cameras/oriented.json` - contains the oriented cameras

### Pipeline: Represent via Matcha


- **NAME**: `represent-via-matcha`
- INPUTs: 
  * scene: str
  * images_hash: hash - of the subset
- SETTINGS: 
  * (tunables) - (mesh_res, ...?)
    > CORRECTED (locked #6): mesh_res belongs to meshify-via-tsdf, not
    > represent. Represent-via-matcha tunables today: dense_regul,
    > n_iters; frozen: encoder, alignment_config, dense_pruning.
  * (frozen)  - (...?)
  * (pins) - (pinned facts)
- DEDUCED_INPUTs:
  * subset metadata: `scenes/<scene>/images/subsets/<images_hash>/subset.json`
  * images - (via subset metadata)
  * cameras metadata: `scenes/<scene>/images/subsets/primary/cameras/oriented.json`
    > SUPERSEDED (locked #1): resolve `primary` -> concrete solve/orient
    > identities at run time; only resolved hashes are consumed/hashed.
  * IDENTITY_HASH: Hash of SETTINGS+scene+images_hash
    > SUPERSEDED (locked #3): hash(resolved input identities + tunable
    > + frozen settings + algo@version). `scene` is a namespace, not
    > hashed; refs resolved first.
- IDENTITY
  * `scenes/<scene>/represent/matcha/<IDENTITY_HASH>`
- OUTPUTs:
  * `scenes/<scene>/represent/matcha/<IDENTITY_HASH>/<filename>` dense representation - (gaussians)
  * training records - (logs, measured VRAM peak, duration, metrics)
  * produced variables - (gaussian count, final loss, etc)
- IDENTITY: Hash of SETTINGS+HOH (produce a hash from the input HOH hashed with the SETTINGS)

  pipeline-5: meshify-via-matcha
  - INPUTs: N-
    * dense_representation (via IDENTITY HASH of dense representation provider)
    * cameras (via oriented-cameras IDENTITY)
    * images (via )
  > SUPERSEDED (locked #6): completed as TWO tasks — meshify-via-tetra
  > and meshify-via-tsdf — nested under the representation they
  > derive from. See Direction.



## Context

Authored 2026-06-11 during the EPI-SCN-PIPELINE-STUDIO review.
After the Studio MVP shipped its first reproducible-by-record run,
the operator re-thought the pipeline model: a **DAG of DAGs** with
**content-addressed identities** (recipe locked in #3), replacing
run-name paths. The Quote above is
the operator's working breakdown; Direction below records the
decisions as they lock, one issue at a time.

## Direction

### Locked #1 — `primary` semantics + the ref-resolution rule (2026-06-11)

`primary` is the scene's **canonical posed pool**, not a casual
pointer:

```
raw capture (e.g. 5000 frames)
   ↓ sample (representative)
PRIMARY subset (could be hundreds)   ← solved ONCE: 100% of cameras
   ↓ curate (per experiment)            placed in 3D, gauge fixed
reconstruction subsets ⊆ primary     (small scenes: all == primary == subset)
```

- All non-primary raw images are effectively **discarded**.
- **Solve once, inherit everywhere:** a reconstruction subset never
  solves its own cameras — its cameras are a row-selection of
  primary's solved set. Every experiment on a scene therefore lives
  in ONE shared gauge; meshes/views/renders are directly comparable
  without re-alignment.
- A subset's cameras-input identity is fully deduced:
  `hash(primary_solve_identity + subset_hash)`.

**File placement (#1):**

```
scenes/<scene>/images/<image_hash>/image.<ext>        # canonical image pool
scenes/<scene>/images/<image_hash>/metadata.json
scenes/<scene>/images/subsets/<subset_hash>/subset.json   # pointers (image hashes)
scenes/<scene>/images/subsets/<subset_hash>/metadata.json # mechanism, label, settings, resolved refs
scenes/<scene>/images/subsets/primary -> <subset_hash>    # symlink (a REF; full mutable surface listed in the flow diagram)
```

Inherited cameras for a reconstruction subset ARE materialized (downstream
tools need a file), under the same shape as any cameras product, and
**named by the source solve's identity** (locked: identity-propagation
rule — a task with zero settings and one input does NOT mint a new
identity; pure selections/projections are transparent to identity):

```
scenes/<scene>/images/subsets/<subset_hash>/cameras/<solve_identity>/cameras.json
scenes/<scene>/images/subsets/<subset_hash>/cameras/<solve_identity>/metadata.json
   # metadata.json: mechanism: inherit, source solve: <solve_identity>,
   # members filtered by subset.json
```

Same identity string under primary and under the subset = same solve,
same gauge — lineage is readable in the path. Re-derivable,
byte-stable, safe to delete and regenerate.
- **Ref-resolution rule (the git model):** refs (`primary`, labels)
  are for humans at invocation time. A task MUST resolve every ref
  to a concrete hash before running, record the resolved hash in
  DEDUCED_INPUTS, and only resolved hashes enter IDENTITY_HASH.
  Re-establishing primary (rare: better sampling, new footage) then
  never changes what past runs meant — only what future runs
  resolve to.

### Locked #2 — cameras take two tasks: `solve-cameras` → `orient-cameras` (2026-06-11)

Nothing in the original draft *produced* cameras; and orientation is
**inverted** vs today: current practice orients the mesh (RANSAC
floor-fit on dense geometry, STO-SCN-004) per-run and back-applies;
this spec orients **primary's cameras once** — all downstream
artifacts are born oriented.

**TASK `solve-cameras`** — place 100% of a subset's cameras in 3D
- INPUTS: scene, subset hash (normally primary's)
- DEDUCED: images via subset.json
- SETTINGS: tunable: none yet · frozen: solver `mast3r-sfm`,
  `sfm_config: unposed` · (>300 frames: chunk_size/overlap — the
  photo-spine folds in as this stage's big-pool strategy)
- IDENTITY: `<solve_identity> = hash(subset_hash + settings + algo@version)` (per locked #3; digest in metadata.json)
- OUTPUTS (file placement):

```
scenes/<scene>/images/subsets/<subset_hash>/cameras/<solve_identity>/cameras.json   # poses+intrinsics, ARBITRARY gauge
scenes/<scene>/images/subsets/<subset_hash>/cameras/<solve_identity>/points.ply    # sparse cloud
scenes/<scene>/images/subsets/<subset_hash>/cameras/<solve_identity>/metadata.json # resolved inputs, settings, measured (host, duration, VRAM)
scenes/<scene>/images/subsets/<subset_hash>/cameras/<solve_identity>/solve.log
```

(`cameras/<identity>/` is ONE shape regardless of producer — solve,
inherit (#1), or future solvers; `metadata.json#mechanism` names which.)

**TASK `orient-cameras`** — fix the gauge (gravity/floor, z-up)
- INPUTS: solve identity
- SETTINGS: tunable: `method` · frozen: RANSAC params
- IDENTITY: `<orient_identity> = hash(solve_identity + settings + algo@version)`
- OUTPUTS (file placement — nested under the solve it orients):

```
.../cameras/<solve_identity>/orient/<orient_identity>/transform.json   # rotation, z_shift (the gauge itself)
.../cameras/<solve_identity>/orient/<orient_identity>/oriented.json    # cameras with gauge applied
.../cameras/<solve_identity>/orient/<orient_identity>/metadata.json    # method, params, measured residuals
.../cameras/<solve_identity>/orient/<orient_identity>/orient.log
```

Two tasks, not one: orientation is re-runnable without re-solving
(10-min solve vs 2-s gauge fix), and `method` is its own experiment
axis — separate identities keep a re-orient from re-keying the solve.

**RESOLVED by measurement (STO-SCN-082, 2026-06-11):** candidate
(a) RANSAC on sparse `points.ply` is **REJECTED** — 58–166° z-axis
error vs mesh-era ground truth across 006/003/004/008, both
unconstrained (locks onto walls/objects) and with a camera-up prior
(the prior itself is unreliable: portrait vs landscape captures flip
the camera Y axis). Adopted: **(b) bootstrap-mesh** — the first
reconstruction's dense floor fit (validated, STO-SCN-004) baked back
onto primary's cameras, once per solve; (c) operator pick remains the
manual fallback. Experiment record: `real2sim/orient_sparse.py`
(kept as rejected-method evidence).


### Locked #3 — the IDENTITY_HASH recipe (2026-06-11)

```
IDENTITY_HASH = hash( resolved input identities     ← per locked #1 (refs resolved first)
                    + tunable settings
                    + frozen settings               ← constant today => same hash;
                                                      future domain-widening re-keys correctly
                    + algo@version )                ← DECLARED behavior version (option B)
```

**Code enters the hash as a declared `algo@version`, NOT the image
digest.** The exact image digest is recorded in the task's
`metadata.json` for audit, but does not key the identity.

- Rationale: images are rebuilt constantly for non-behavioral reasons
  (tool additions, lib bumps); digest-in-hash would re-key the entire
  store per release and destroy the memoization this design exists to
  provide.
- Identity therefore means **"behaviorally equivalent recipe" — a
  claim, not a bytes guarantee.** Implementers MUST bump a task's
  version on any behavior-affecting change.
- **Safety net for forgotten bumps:** the reproducibility harness
  (repro_check, STO-SCN-075). Same identity + different digest is an
  auditable pair — re-run, compare within measured tolerances; drift
  means the version was effectively lied about → flag, bump, re-key.
- Pins/execution facts ride along inside `algo@version` + the
  recorded digest; they never appear as hashable settings.

(Precedent: the t-vs-d 0.44% murkiness on 2026-06-11 was image 0.2
vs 0.4 compared under one run label — exactly the ambiguity this
rule + harness resolves.)

### Locked #4 — vocabulary: `task` / `graph` / `job` (2026-06-11)

Structurally it is DAGs all the way down — leaf and composition share
INPUTS/SETTINGS/OUTPUTS/IDENTITY. The only real distinction is
**atomicity**, so the vocabulary is two structural words + one
operational word (Dagster's op/graph/job shape — vocabulary borrowed,
software rejected by the 069 spike):

| Word | Is | Properties |
|---|---|---|
| **task** | executable leaf | one dispatch (image+entrypoint), succeeds/fails atomically, its output dir IS the cache entry, mints identity |
| **graph** | composition | DAG of tasks and/or graphs, nests freely; identity derived recursively; owns no outputs — its outputs are its members' |
| **job** | invocation, not a type | "here are variable bindings (scene, subset, setting overrides, host) — **materialize** this graph's outputs" |

- A unit can change tier without changing interface:
  `represent-via-matcha` is a task today (one container run) and
  becomes a graph if the matcha monolith is split — consumers don't
  care.
- **Jobs materialize, not execute:** the job resolves every node's
  IDENTITY_HASH first and only executes nodes whose identities don't
  exist yet. A job over a fully-cached graph is a no-op returning
  addresses. (This is the memoization payoff of locked #1/#3.)
- Mapping to the Studio A–F taxonomy: graph+bindings ≈ E
  (pipeline_instance); a job's record ≈ F (pipeline_run). The word
  "pipeline" is retired as a type name.
- "Stage" (used transiently in earlier discussion) is retired.

### Locked #5 — subset identity is content-only (2026-06-11)

A subset's identity is the HOH of its member image hashes — nothing
else. `mechanism` (HUMAN/viser, sharp-select, sample), `label`, and
the selection settings are **recorded in metadata.json but never
hashed**.

- Two mechanisms that pick the same members produce THE SAME subset:
  same identity, same downstream cache hits.
- The subset IS its members; how you arrived is provenance, not
  identity. (Consistent with the identity-propagation rule in #1.)
- Accepted cost: you cannot have two subsets with identical members
  but different "meanings" — no real case found.

### Locked #6 — meshify: two tasks, nested placement, fused execution (2026-06-11)

**Two tasks, not one task with a `method` flag** (different code,
settings, and accepted input kinds — same reason represent-via-matcha
and represent-via-da3 are separate tasks):

```
TASK: meshify-via-tetra                    TASK: meshify-via-tsdf
- INPUTS:                                  - INPUTS:
  * representation identity                  * representation identity
    (kind: gaussians+charts ONLY)              (kind: gaussians or depths)
  * cameras (oriented, via solve id)         * cameras (oriented, via solve id)
- SETTINGS:                                - SETTINGS:
  * tunable: (none exposed yet)              * tunable: mesh_res (1024 validated)
  * frozen: binary-search params             * frozen: config=default
- IDENTITY: hash(rep_id + cameras_id       - IDENTITY: same recipe
    + settings + algo@version)
```

**File placement — nest a derivative under what it derives from**
(the orient-under-solve precedent):

```
scenes/<scene>/represent/matcha/<RID>/meshify/tetra/<MID>/{mesh.ply, metadata.json, meshify.log}
scenes/<scene>/represent/matcha/<RID>/meshify/tsdf/<MID>/...
```

**Consequences locked with it:**

- **`ground` disappears as a task.** Cameras arrive oriented
  (locked #2), so meshes are BORN in the scene gauge — today's
  per-run mesh orientation step evaporates. (DA3's gauge_align
  survives only inside represent-via-da3, aligning into primary's
  frame.)
- **The tetra weld is an executor fact, not a model fact.** Until
  the matcha monolith is split, ONE container dispatch materializes
  BOTH identities (RID and tetra MID). Evaluating the graph in DAG
  order: represent executes (writing both), then meshify-via-tetra
  is inspected → its identity already exists → **NOOP**. No special
  case in the model; the planner's materialize-check handles it.
- **Unrequested outputs are never thrown away** — an output that
  falls out of a fused dispatch is a pre-warmed cache entry with a
  real identity, not garbage.
- **Known compute-waste corner:** representation cached, tetra
  requested LATER → the welded container recomputes the
  representation on its way to the mesh (idempotent, data-safe,
  GPU-wasteful). Mitigations: (1) matcha jobs request tetra eagerly
  by default (it's the ranked-#1 branch), (2) the monolith split
  removes the corner, (3) until then the planner warns.

### Locked #7 — the evaluate tier: views, viewsets, renders, scores (2026-06-11)

**(a) Views — scene globals, slot-addressed, content-identified.**
The overview views are part of the scene's canonical apparatus:
every rendering of the scene is judged against the scene's views.

```
scenes/<scene>/views/01/view.json        ← 01 is a SLOT (human label)
scenes/<scene>/views/02/view.json           a view's IDENTITY = hash of
scenes/<scene>/views/.../view.json          its view.json CONTENT
scenes/<scene>/viewset/canonical/views.json  ← MUTABLE list of slots
```

- The viewset members file is editable with no ceremony — add a
  slot, re-frame a slot. Safe because of the ref-resolution rule
  (#1): a render job resolves slots → content hashes at run time
  and records the resolved hashes.
- Views are generated by a task (`generate-views`, inputs: primary's
  oriented cameras + sparse points, settings: n_views/strategy) —
  **from gauge-level data, NEVER from a specific mesh**, or variants
  would get different views and cross-variant ranking dies.
  N==1 is fine. Manual capture can return as `strategy: human`.
- **Ingest is idempotent, refs are set-if-unset:** the ingest graph
  (video-to-images → primary → solve → orient → generate-views)
  self-NOOPs via identity checks — "generate views once per scene"
  needs no conditional machinery. A job may CREATE
  `viewset/canonical` / `primary` when none exists; it may never
  MOVE an existing ref — that is an operator act.

**(b) Renders — ordinary tasks; keyed on the VIEW, never the SET:**

```
RENDER IDENTITY = hash(mesh_id + view_content_hash + settings + algo@version)
                                 ↑ the viewset hash MUST NOT appear here
```

This is what makes evaluation incremental with zero extra machinery:

| Operation | At re-render |
|---|---|
| add a view (slot 06) | 01–05 renders: identities exist → NOOP; only (each mesh × 06) executes |
| modify a view (re-frame 02) | new content hash → (mesh × new-02) renders; old 02 renders stay addressable |
| "force recompute" | not a mode — just run the render job; the materialize pass renders exactly the missing set |

- SETTINGS: tunable: engine, resolution (today WORKBENCH 1920×1080)
- PLACEMENT (nested under the mesh, per #6 precedent):
  `.../meshify/tetra/<MID>/renders/<render_identity>/render.png + metadata.json`
- Cross-variant comparison for one view is a read-time query ("all
  renders with this view_content_hash") — the rate_renders pattern.

**(c) Scores are NOT task outputs.** Operator judgments are scores
attached to identities (attachable at ANY node: render, mesh,
representation). Append-only event log; entries reference resolved
identities (view content hash for stability, slot number for
display); never hashed into anything:

```
scenes/<scene>/scores.jsonl   # {at: <identity>, view: <content_hash>, slot: "02", rank, rater, ts}
```

(`condition` needs no new decisions: ordinary task, nests under its
mesh: `.../meshify/tetra/<MID>/condition/<CID>/`.)

### Locked #8 — graph definitions in the repo; job records in the scene (2026-06-11)

**(a) Graph definitions live in the research repo, beside task defs**
— graphs are recipes: versioned, reviewed, branched like code, zero
per-scene state. The store holds only what jobs materialize.

```
real2sim/tasks/<task>.json      # leaf defs
real2sim/graphs/<graph>.json    # nodes (task OR graph names — nesting), edges
```

**(b) A job is an EVENT, not an artifact** — two identical jobs a
week apart are two events that NOOP into the same identities. Job
records are therefore NOT content-addressed; they are append-only
and scene-scoped:

```
scenes/<scene>/jobs/<timestamp>-<short_id>/job.json
```

containing: graph name+version, bindings (subset, setting overrides,
host), resolved refs (primary→…, canonical→…, per locked #1), and a
per-node outcome: `NOOP @ <identity>` or `EXECUTED @ <identity>
(host, duration, image digest, rc)`.

**Division of labor:** `metadata.json` (per identity) = what this
artifact IS; `job.json` = what this invocation DID. An artifact can
appear in many jobs (NOOPed); its metadata never changes.

### Locked #9 — migration: full restructure, no legacy residue (2026-06-11)

**Operator decision (overruling the proposed strangler pattern):**
do NOT live beside legacy-structured data. Restructure the existing
store into the new layout and move forward. The computer-time
already spent (solves, representations, meshes, renders) is
preserved — files MOVE to their content-addressed homes; nothing is
recomputed that already exists.

Migration mechanics:

- **Identities are computed, not invented.** The inputs exist
  (`input/src` → image pool + subset hashes) and the settings exist
  (every run's `specification.json`) — so legacy IDENTITY_HASHes are
  computable with the locked #3 recipe. Legacy executions get a
  retroactive `algo@version` (e.g. `matcha@0` = the welded pre-split
  container; `da3@0`), distinguishing them from future post-split
  versions honestly.
- **Per-identity `metadata.json` written during migration** records
  `migrated: true` + the v2 origin path; genuinely unknowable fields
  stay explicit-unknown (T-002). Migration itself is logged as jobs
  (`mechanism: migrate`) per locked #8.
- **Write new model code as needed during migration** — the
  migration is allowed to drive the new store/model implementation
  into existence.
- Operator rankings (`rankings.jsonl`) translate to `scores.jsonl`
  entries referencing the migrated identities — months of judgment
  carried forward.
- Supersedure: the staged v3 backfill (STO-SCN-077 shape,
  `run_record.json`) is OBSOLETE — close as superseded by this HUG;
  the migration replaces it.

### Locked #10 — represent-via-da3 + license taint as derived ancestry property (2026-06-11)

```
TASK: represent-via-da3
- INPUTS:
  * subset hash (images)
  * orient identity (primary's gauge — the frame to align INTO)
- SETTINGS:
  * tunable: process_res (504 default; measured ceilings: gs<=504, nogs<=756)
  *          mode: gs | nogs
  * frozen: conf thresholds, export set
- IDENTITY: hash(subset + orient_id + settings + da3@version)
- OUTPUT kind: depths + splats + poses   ← typed differently from gaussians+charts
- PLACEMENT: scenes/<scene>/represent/da3/<RID>/
```

- **DA3 solves its own poses** — it consumes primary's cameras only
  as the alignment TARGET: gauge_align runs inside the task, mapping
  its self-solved frame into primary's gauge (residual gate <=10%;
  006 measured 2.9%). Orient identity is a genuine input → in the
  hash. Outputs are born in the scene gauge like everything else.
- **Type-checking routes compatibility** (per #6): meshify-via-tetra
  accepts gaussians+charts only; depths+splats route to TSDF fusion.

**License taint propagates through lineage:**

- License is a fact on `algo@version` (the task def — e.g. DA3
  weights CC-BY-NC-4.0; STO-SCN-078 audit verdicts land there).
- `deliverable-eligible(identity)` = **no NC anywhere in the
  identity's ancestry** — DERIVED at read time by walking inputs;
  never stored, never hand-maintained.
- NC tasks stay in the same graphs (that is their value: same
  runoff, honest comparison); the eligibility query is what keeps
  M11 deliverables clean.

### Locked #11 — DO NOT MANIPULATE DATA OUTSIDE A GRAPH (2026-06-12)

Operator (verbatim): "I want to take a hard stance of: DO NOT
MANIPULATE DATA OUTSIDE A GRAPH."

Context: post-migration repair work (re-orientation, era re-grounding,
render fixes) was performed by one-off scripts manipulating store
artifacts directly. Each fix was honest, but the class is the
disease: artifacts whose history is a chain of out-of-band edits
cannot be trusted or reproduced. Proof-by-construction replaces
repair-by-archaeology:

- Store artifacts are created ONLY by jobs materializing graphs.
- Backfill/migration/repair scripts are DELETED once the native path
  is proven (git history keeps them for forensics).
- Operator data entry (view capture, rankings/scores, refs) is input,
  not manipulation — allowed per locked #7.
- Native proof protocol: wipe a scene to original inputs -> ingest
  -> orient -> reconstruct graphs end-to-end -> compare.

### Flow diagram — worked example of everything locked (refreshed for #5–#7)

Scene `006-kubota`, fake short hashes. One video; primary pool of
200; primary solved once (re-solve Q4K8 shown to motivate identity
levels); curated subset of 17; matcha represent + tetra meshify +
condition + renders; canonical viewset of 3 views.

```
video.mp4
   │  task: video-to-images
   ▼
images/                                   ◄── canonical pool, one dir per image
├── 1A2B/image.jpg ... (5000)
│
│  task: images-subset (mechanism: sample)         [locked #5: identity =
▼                                                   HOH of members ONLY]
subsets/P9F3/  (primary pool, 200)
├── subset.json, metadata.json
│
│  task: solve-cameras (mast3r-sfm@1, settings X)
▼
subsets/P9F3/cameras/S7D2/                ◄── S7D2 = hash(P9F3 + X + mast3r-sfm@1)
├── cameras.json, points.ply                  (re-solve w/ settings Y → cameras/Q4K8/
│                                              coexists — why this level exists)
│  task: orient-cameras (floor method, settings Z)
▼
subsets/P9F3/cameras/S7D2/orient/O3M1/    ◄── THE gauge
├── transform.json, oriented.json
│
├──────────────────────────────────────────────────────────────┐
│  task: generate-views (from gauge data, NEVER from a mesh)   │
▼                                                              │
views/01/view.json  views/02/view.json  views/03/view.json     │ [locked #7]
viewset/canonical/views.json   ◄── MUTABLE members list        │
│                                                              │
subsets/primary ──► P9F3        ◄── refs: set-if-unset by jobs,│
viewset/canonical               moved only by the operator     │
│                                                              │
│  task: images-subset (mechanism: HUMAN viser)                │
▼                                                              │
subsets/C4E8/  (curated 17 ⊆ primary)                          │
├── cameras/S7D2/              ◄── INHERITED row-selection;    │
│                                  same id = same gauge        │
│  task: represent-via-matcha (R = hash(C4E8 + S7D2 + settings + matcha@1))
▼
represent/matcha/R5T9/
├── free_gaussians/..., metadata.json
│
│  task: meshify-via-tetra     ◄── WELDED today: the represent dispatch
▼                                  also writes this → DAG walk finds it → NOOP
represent/matcha/R5T9/meshify/tetra/M2W7/
├── mesh.ply, metadata.json
│
│  task: condition (target_tris 1M)
▼
.../tetra/M2W7/condition/K9J4/
├── mesh.ply
│
│  task: render (per view: hash(mesh + view_content + settings + render@1))
▼
.../condition/K9J4/renders/E1F8/render.png    ◄── one per (mesh × view);
.../condition/K9J4/renders/E2A3/render.png        add view 04 later → only
.../condition/K9J4/renders/E3C5/render.png        (mesh × 04) executes

scores.jsonl   ◄── operator judgments referencing identities; never hashed
```

Reading the why-questions off the diagram:
- **Why `S7D2` under primary?** A re-solve (`Q4K8`) coexists beside
  it. Identity levels exist exactly where settings/code can vary.
- **Why `S7D2` again under C4E8?** Names WHICH solve those 17
  cameras were cut from — gauge lineage readable in every
  downstream path.
- **Mutable entries, total:** `subsets/primary`, `viewset/canonical`
  (the two refs), `viewset/canonical/views.json` (member list), and
  `scores.jsonl` (append-only). Everything else is immutable and
  content-addressed.

## Applied in

_(Reverse pointer — populated by the reconciler from forward `hugs:`
citations on STORY/EPIC frontmatter. Don't hand-author; the
reconciler maintains this field.)_
