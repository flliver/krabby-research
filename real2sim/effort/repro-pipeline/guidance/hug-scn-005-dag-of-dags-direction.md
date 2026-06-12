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
  * (frozen)  - (...?)
  * (pins) - (pinned facts)
- DEDUCED_INPUTs:
  * subset metadata: `scenes/<scene>/images/subsets/<images_hash>/subset.json`
  * images - (via subset metadata)
  * cameras metadata: `scenes/<scene>/images/subsets/primary/cameras/oriented.json`
  * IDENTITY_HASH: Hash of SETTINGS+scene+images_hash
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



## Context

Authored 2026-06-11 during the EPI-SCN-PIPELINE-STUDIO review.
After the Studio MVP shipped its first reproducible-by-record run,
the operator re-thought the pipeline model: a **DAG of DAGs** with
**content-addressed identities** (`IDENTITY_HASH = hash(resolved
inputs + settings)`), replacing run-name paths. The Quote above is
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
scenes/<scene>/images/subsets/primary -> <subset_hash>    # symlink (the ONLY mutable entry)
```

Inherited cameras for a reconstruction subset ARE materialized (downstream
tools need a file), under the same shape as any cameras product, and
**named by the source solve's identity** (locked: identity-propagation
rule — a stage with zero settings and one input does NOT mint a new
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
  are for humans at invocation time. A stage MUST resolve every ref
  to a concrete hash before running, record the resolved hash in
  DEDUCED_INPUTS, and only resolved hashes enter IDENTITY_HASH.
  Re-establishing primary (rare: better sampling, new footage) then
  never changes what past runs meant — only what future runs
  resolve to.

### Locked #3 — the IDENTITY_HASH recipe (2026-06-11)

```
IDENTITY_HASH = hash( resolved input identities     ← per locked #1 (refs resolved first)
                    + tunable settings
                    + frozen settings               ← constant today => same hash;
                                                      future domain-widening re-keys correctly
                    + algo@version )                ← DECLARED behavior version (option B)
```

**Code enters the hash as a declared `algo@version`, NOT the image
digest.** The exact image digest is recorded in the stage's
`metadata.json` for audit, but does not key the identity.

- Rationale: images are rebuilt constantly for non-behavioral reasons
  (tool additions, lib bumps); digest-in-hash would re-key the entire
  store per release and destroy the memoization this design exists to
  provide.
- Identity therefore means **"behaviorally equivalent recipe" — a
  claim, not a bytes guarantee.** Implementers MUST bump a stage's
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

### Flow diagram — worked example of everything locked so far

Scene `006-kubota`, fake short hashes. One video, primary pool of 200,
one curated subset of 17, primary solved once (then once more with
different settings — the reason identity levels exist).

```
video.mp4
   │  video-to-images
   ▼
images/                                  ◄── canonical pool: one dir per image
├── 1A2B/image.jpg  ─┐
├── 7C9D/image.jpg   │  5000 images
├── ...              ┘
│
│  images-subset (mechanism: sample, label: "pool-200")
▼
subsets/P9F3/                            ◄── P9F3 = hash of its 200 image hashes (HOH)
├── subset.json                              (pointers to the 200)
├── metadata.json
│
│  solve-cameras (mast3r-sfm, settings X, image digest D1)
▼
subsets/P9F3/cameras/S7D2/               ◄── S7D2 = hash(P9F3 + X + D1)
├── cameras.json        (arbitrary gauge)    WHY THIS LEVEL EXISTS:
├── points.ply                               re-solve with settings Y
├── metadata.json                            → cameras/Q4K8/ coexists,
├── solve.log                                nothing overwritten
│
│  orient-cameras (method: floor-ransac, settings Z)
▼
subsets/P9F3/cameras/S7D2/orient/O3M1/   ◄── O3M1 = hash(S7D2 + Z)
├── transform.json      (THE gauge: rotation, z_shift)
├── oriented.json
│
subsets/primary ──symlink──► P9F3

   │  images-subset (mechanism: HUMAN viser, label: "curated-17")
   ▼
subsets/C4E8/                            ◄── C4E8 = HOH of the 17 (⊆ primary's 200)
├── subset.json
├── cameras/S7D2/                        ◄── INHERITED: row-selection of primary's
│   ├── cameras.json    (17 rows)            solve. Same string S7D2 = same solve,
│   ├── metadata.json   (mechanism:          same gauge — lineage readable in the
│                        inherit)            path. No new identity minted: inherit
│                                            has zero settings, one input.
└── (represent-via-matcha consumes C4E8 + its cameras/S7D2 from here…)
```

Reading the why-questions off the diagram:
- **Why `S7D2` under primary?** Because `Q4K8` (a re-solve) can exist
  beside it. Identity levels exist exactly where settings/code can
  vary; nowhere else.
- **Why `S7D2` again under C4E8?** It names *which solve* those 17
  cameras were cut from. If primary is re-solved, C4E8 can hold
  `cameras/Q4K8/` too — and every artifact downstream knows which
  gauge it lives in by the identity in its path.
- **Where's the single mutable thing?** `subsets/primary` — the
  symlink. Everything else is immutable, content-addressed.

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

**Open (T-002):** the validated floor-fit runs on a dense mesh,
which doesn't exist at this point in the new order. Candidate
methods — (a) RANSAC on sparse `points.ply` (plausible,
unvalidated), (b) one-time bootstrap from the first reconstruction's
mesh-fit, (c) operator-assisted floor pick. The contract is locked;
the method is the first implementation story's verification job.

## Applied in

_(Reverse pointer — populated by the reconciler from forward `hugs:`
citations on STORY/EPIC frontmatter. Don't hand-author; the
reconciler maintains this field.)_
