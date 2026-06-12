---
xid: EPI-SCN-SCENE-SYNC
parent: ../design.md
kind: epic
effort: scn
status: open
date: 2026-06-04
hugs: []
tenets: [T-013, T-016, T-014]
bd-id: krabby-7c5
assignee: principal
priority: 4
---

# Scene Synchronization

> **Status notes**
> - 2026-06-04: Picked up by principal, beginning review.
> - 2026-06-04: Authored from AID spec (5 functional needs + 4 goals). Grounded in
>   audit of `/var/krabby/workspace/milestones/011-scene-reconstruction/data` (~50 GB),
>   the existing `s3://krabby-real2sim-scenes` bucket, the `krabby-firmware-public`
>   manifest precedent, and a sherpa fleet-topology query.
> - 2026-06-04: Placement RESOLVED (AID) — keep as the existing `EPI-SCN-SCENE-SYNC`;
>   no new foundational design created. Stays under `DES-SCN-TX` for now.
> - 2026-06-04: 7 stories minted (STO-SCN-026..032, all assignee:principal).
> - 2026-06-04: AID directive — model each scene as a **pipeline of
>   transformations** (`input/` + `input/preproc-*/` → `pipeline-<slug>/transform-<id>-<slug>/`
>   {specification.json, results.json, data/} → `pipeline-<slug>/output/` →
>   `output/`). Worked into Design, Goals, Decisions, Success Criteria; STO-SCN-026
>   retitled to own the schema; STO-SCN-031 now also emits `results.json` provenance.
> - 2026-06-04: STO-SCN-026 execution — authored canonical [`SCHEMA.md`](./SCHEMA.md)
>   + JSON schemas (`schemas/`) + worked reference scene (`reference/004-sky-house/`)
>   + [`inventory.md`](./inventory.md) (21 dirs → ~10 scenes). Audit added a **`run-<slug>`
>   level** (param sweeps = parallel runs, AID-confirmed) and found the legacy
>   `manifest.json` already maps near-1:1 onto scene.toml/spec/results → provenance
>   is **mixed, not absent** (Current State corrected).
> - 2026-06-04: Grant cross-check (read `/var/krabby/grants/*`). Findings: (a) M11
>   to date — incl. `environments/reconstructed/`, flat `scenes/<id>/`,
>   `FOLDER_LAYOUT.md` M11 section, the `>100 MB→S3 / ≤100 MB→git` split — is all
>   **prototype** (we created it 2026-05-18; subject to change). Other delivered
>   milestones are **canon**. (b) Established decisions: **data is not code** (no
>   meshes/USD in git); **canon container contract** is code@`/workspace` +
>   `-v <host-data>:/data` (our delivered images already do this); `pipeline-<slug>`
>   ↔ our image names; each transform `data/` holds **third-party tool native
>   output** unchanged ("did not build" — COLMAP/MAtCha/MASt3R/VGGT formats).
>   (c) `krabby scenes` must NOT extend the lean public `krabby-launcher` package —
>   separate dev entry point. (d) The robot-fleet (Jetson Orins, M10/M14) ≠ our
>   compute fleet (gaming PCs) — no fleet-control-plane conflict. All folded into
>   Decisions.
> - 2026-06-04: Fleet-infra facts gathered LIVE from baeprz via ops envoy (not
>   filed). Two design assumptions corrected: (a) disk is abundant — j has a
>   dedicated `/games` 1.8 T disk (624 G free), s/t/b/d have 1.4 T `/home`
>   (~1 T+ free) — the "458 G root" risk is void; (b) LAN (1 GbE, ~110 MB/s) ≈
>   j's WAN-to-S3 (~94 MB/s), so LAN-first wins by **deduplicating S3 egress +
>   offline availability**, NOT raw speed. Gateway = j (confirmed). Only rsync
>   installed fleet-wide; no S3 client or `krabby` profile present anywhere yet.
> - 2026-06-04: **Fleet-sync layer BUILT** (superseded the rclone *fleet-transport*
>   plan; **S3 stays as Layer-2 backing**). Store is a
>   git-LFS repo (`/var/krabby/scenes` Mac → external APFS); **hub on j**
>   (`/games/krabby/scenes`, always-on, source of truth). Transport = git-over-SSH
>   (metadata, 1 `trunk`) + rsync of `.git/lfs/objects` (additive); `GIT_LFS_SKIP_SMUDGE`
>   + `push --no-verify`; no rclone/git-lfs-transfer on the *fleet* leg. **S3 stays
>   as Layer 2** (durable cold store + collab/public, via the j gateway — STO-028).
>   Fleet sync's purpose = **experiment load-distribution** (run anywhere, centralize on j). `krabby-scenes-sync`
>   helper ansible-provisioned fleet-wide (git-lfs 3.6.1 in base role). Migrated 42 GB
>   seeded to j; **Mac + t/s/d verified-identical mirrors** (4702 objs @ `trunk`); b
>   synced (whole) but flaps on a stay-awake hardware fault. Architecture/Approach/
>   Decisions/Risks/Success-Criteria updated to AS-BUILT.

## Problem Statement

M11 (scene reconstruction) produces the project's core asset: reconstructed 3D
scenes. There are currently **~50 GB** of them — `scenes/` (38 G, ~21 scene
directories), `videos/` (11 G raw capture), `sfm-scaling-out/` (956 M) — living
haphazardly under `/var/krabby/workspace/milestones/011-scene-reconstruction/data`.
The per-scene layout is **inconsistent** (`004-sky-house-dining` carries
`mesh/ mast3r_output/ matcha_output/ comparison_renders/…` directly, while
`009-kubota-004` has only `src/`), naming is mixed (numbered captures, `curated`
variants, the `dtu-bicycle` benchmark, the `kubota` series), and the
`s3://krabby-real2sim-scenes` bucket was populated in a rush ("just get the data,
I need to get on a plane") with no schema. The result: data cannot be reliably
**consumed** by Docker jobs on the fleet, **shared** with collaborators in other
physical locations, eventually **published**, or even **inspected** locally
without archaeology — and every fleet host that needs a scene re-pulls it from S3
(slow, costs egress) because there is no LAN-first path. Solving this is
foundational: the rest of the real2sim pipeline (T0–T4) consumes these scenes.

## Goals

- **Canonical scene schema = pipeline of transformations** — every scene is an
  auditable `input/ → pipeline-<slug>/transform-*/ → output/` lineage, where each
  transform records `specification.json` (what) + `results.json` (runtime env).
  One documented contract + a scene-level manifest; migrating the ~21 existing
  scenes to it is in scope.
- **Reproducible provenance** — any artifact traces to the exact params and
  environment (OS/driver/container/software versions) that produced it, so a
  collaborator elsewhere — or the public — can re-run it.
- **Three-tier organization** — the same data is structured for (a) M11
  research use now, (b) expanded use by collaborators in other locations during
  later M-efforts, (c) public release in a later project phase — with an
  explicit gate between tiers.
- **LAN mirror synchronization** *(AS BUILT — was "LAN-first vs S3")* — every node
  holds a **full local mirror**; transfers are additive over the LAN (git refs +
  rsync of `.git/lfs/objects`); **no cloud round-trip on the fleet leg**. *Maximize
  LAN — the fleet sync never round-trips to S3 (S3 is the separate Layer-2 backing).*
- **Docker consumption convention** — containers we stand up on the fleet mount
  scene data through one standard, read-only path/volume convention.
- **Simple, secret-safe sync** — one command (`krabby-scenes-sync`); Layer-1 auth
  is **SSH-key** over the fleet mesh; **no cloud credentials on producer hosts**
  (only j, the S3 gateway, holds the `krabby` profile).
- **Local inspection ergonomics** — pulling a scene (or subset) to a laptop for
  post-experiment inspection is a one-liner that integrates with `camera_viewer`.

## Non-Goals (Out of Scope)

- **Reconstruction algorithms** — how scenes are *produced* is T0–T4's job; this
  epic only organizes, moves, and serves the outputs.
- **A bespoke storage service** — we use a **git-LFS repo + rsync** (fleet hub on
  j) **+ S3** (durable cold store / public tier, via j), not a new daemon. *(AS BUILT.)*
- **Public release itself** — we define the public *tier and gate* but the act of
  publishing M11 data is deferred to the later project phase that owns it.
- **Provisioning new fleet hardware / a dedicated NAS** — flagged as a risk; this
  epic works within the 458 GiB-root hosts that exist today unless ops says
  otherwise.

## Context

**Source:** AID directive, 2026-06-04 — "We have 3D scenes we are trying to
reconstruct… a foundational part of that is being able to work with the scenes we
have." Five functional needs (organize / distribute / consume-in-Docker /
local-inspect / S3-sync) and four goals (min S3, max LAN, secrets-safe, simple).

**Placement (RESOLVED 2026-06-04, AID):** Keep as the existing
`EPI-SCN-SCENE-SYNC`; do **not** spin up a new foundational design. The epic
remains parented to `DES-SCN-TX` for now even though its work is foundational —
the T0–T4 efforts are de-facto *consumers* of it.

**Current state (audited 2026-06-04):**

- Local: `…/011-scene-reconstruction/data/{scenes,videos,sfm-scaling-out}` ≈ 50 GB.
- S3: `s3://krabby-real2sim-scenes` (AWS profile `krabby`); prefixes incl.
  `m11-sfm-scaling/` already exist from yesterday's ad-hoc push. Layout not yet
  schematized.
- Precedent: `krabby-firmware-public.s3` already serves builds via an
  `index.json` manifest + `~/.cache/krabby-firmware` local cache — reuse this shape.
- Secrets: credential is in the `krabby` profile (`~/.aws/`) — already out of code.
- **Maturity:** all historical work to date is **INPUT collection + *prototype*
  transformations + their output**. There are **no promoted/finalized outputs**
  yet — the public-tier scene `output/` is empty until we deliberately promote.
- **Provenance for past runs is MIXED** (corrected by the STO-SCN-026 audit, see
  [`inventory.md`](./inventory.md)): the 7 curated MAtCha runs carry a
  `manifest.json` recording host/GPU/params/duration → **measured** (gaps:
  driver, CUDA, container digest, output hashes). Older runs (001/002/003/
  004-dining) have none → **deduced** from journals. Raw captures (kubota
  006–012, 005-meadow) → input-only, n/a. Never fabricate the gaps (T-002).

**Fleet facts (gathered live from baeprz/ops 2026-06-04 — STO-SCN-030 inputs):**

- **No shared storage today** — no NFS/SMB/ZFS on any host. All ext4.
- **Disk is abundant** — j: `/games` 1.8 T NVMe (624 G free) + `/` 458 G (413 G
  free); s/t/b/d: `/home` 1.4 T each (~1 T+ free). 50 GB→growing fits easily.
- **Tooling** — only `rsync` installed fleet-wide. No rclone/aws-cli/s3cmd/
  syncthing/restic. **No `krabby` AWS profile on any host** — nothing can reach
  S3 yet; an S3 client + the profile must be placed on the gateway (j).
- **Docker** — bind-mounts conventionally rooted at `/games/…` (e.g. game-server
  data). The **canon** image pattern (from our delivered locomotion/isaacsim
  images): code baked at `/workspace`, runtime data bind-mounted `-v <host>:/data`.
  Host-side scene dir follows `/games` bulk storage (e.g. `/games/real2sim/data`).
- **LAN** — flat 1 GbE /24 (~110 MB/s practical), no 10GbE, no WireGuard in path.
- **Gateway** — j (jbeeprz): only always-on host, biggest disk, 125 GiB RAM.

**Dependencies:**

- **`camera_viewer`** (`real2sim/camera_viewer`) — local-inspection integration.
- **T3 Docker design** (`DES-SCN-DOCKER`) — the Docker-consume story must align
  with how locomotion containers are built/run.
- **git-lfs + SSH mesh on the fleet** *(AS BUILT — done)* — sync needs git-lfs on
  each host (ansible base role) + passwordless SSH to the hub; both provisioned.
  Producer hosts need **no S3 client**; j (the gateway) gets the `krabby` profile in STO-028 (Layer 2).

## Stories

| # | XID | Story | Status | Size |
|---|-----|-------|--------|------|
| 0 | `STO-SCN-025` | Scene Synchronization — research spike (what must we build?) — *pre-existing; largely answered by this epic* | open | L |
| 1 | `STO-SCN-026` | Define pipeline-of-transformations scene schema (`input/`→`pipeline-<slug>/transform-*/`→`output/` + per-transform `specification.json`/`results.json`) & inventory existing scenes | open | L |
| 1b | `STO-SCN-033` | Migrate historical scenes (INPUT + *prototype* transforms) into the schema; **reconstruct provenance from M11 journals** (mark deduced vs measured) | open | L |
| 2 | `STO-SCN-027` | Three-tier organization (research/collab/public) + tier gate | open | M |
| 3 | `STO-SCN-028` | **Layer 2: S3 cold store + collab/public distribution** — j↔S3 via the `krabby` profile (gateway): durable backup + public-tier publish + reconcile the existing bucket | open | M |
| 4 | `STO-SCN-029` | Sync CLI — **realized as `krabby-scenes-sync` (git+rsync, ansible-provisioned)**; needs re-scope/close to match as-built | open | L |
| 5 | `STO-SCN-030` | Fleet distribution — **realized: hub-on-j + `krabby-scenes-sync`; all producers mirrored**; close/reconcile to as-built | open | M |
| 6 | `STO-SCN-031` | Docker consume convention — canon `-v <host-data>:/data` (code at `/workspace`) **+ emit `results.json` provenance** per run (contract with STO-SCN-026) | open | M |
| 7 | `STO-SCN-032` | Local inspection ergonomics + `camera_viewer` integration | open | S |

_Stories minted 2026-06-04, all `assignee: principal`, bodies are template stubs awaiting fill. `STO-SCN-025` predates this epic — recommend closing or folding it into STO-SCN-026 now that the epic captures the "what must we build" answer._

## Design

### Approach

> **AS BUILT (2026-06-04) — TWO LAYERS.** **(1) Fleet LAN** (hosts ⇄ j): a
> git-LFS hub-on-j + git+rsync mirror, for **experiment load-distribution** — run
> reconstructions on *any* host, results centralize on j. **(2) Cloud** (j ⇄ S3):
> **S3 remains** the durable/authoritative cold store + collab/public distribution,
> with **j as the gateway**. What changed from the original plan is only the *fleet*
> transport (git+rsync, not rclone/git-lfs-transfer, no creds on producers); **S3
> sits behind j, not in front of it**.

Scene data lives in a **dedicated git-LFS repo** — `/var/krabby/scenes` on the Mac
(→ external APFS drive), separate from the code repo (`krabby-research`). Large
binaries are LFS-tracked (a content-addressed `.git/lfs/objects` store); metadata
(`scene.toml`/`run.json`/`*.json`) is plain git. A scene is still modelled as a
**pipeline of transformations** (canonical spec: [`SCHEMA.md`](./SCHEMA.md)).

Distribution is a **hub-and-spoke mirror on j**. **j (jbeeprz)** — always-on,
`/games/krabby/scenes` — holds the authoritative repo and is the source-of-truth +
merge point. Every other node is a **bidirectional producer**: the **Mac** writes
original *inputs*; the **CUDA hosts t/b/d/s** produce reconstruction *outputs*;
all push to / pull from j.

**Layer 1 — fleet LAN transport = git over SSH (metadata) + rsync (blobs).**
git-lfs's own SSH transport (`git-lfs-transfer`) is **deliberately unused** — it isn't packaged on
Debian 13 and the fleet declined a 3rd-party binary in the base role. So:
- **metadata** moves by `git push/pull` over plain SSH on one shared `trunk`
  (additive merges — different hosts add different scenes/runs, so conflicts are rare);
- **blobs** move by `rsync` of `.git/lfs/objects` — content-addressed, **additive
  only** (`--ignore-existing`, never `--delete`; objects can't conflict);
- every git op runs `GIT_LFS_SKIP_SMUDGE=1` + `push --no-verify` so git never
  touches the absent transport; `git lfs checkout` materializes the working tree
  from the local (rsync'd) objects.

The `krabby-scenes-sync {push|pull|status}` helper wraps this, **ansible-provisioned
fleet-wide** (`krabby-scenes-sync` role; `/etc/krabby-scenes.conf`); the Mac runs
the same portable script. Cadence is **on-demand**. Docker jobs mount the local
repo working tree at `/data`.

**Layer 2 — cloud (j ⇄ S3).** `s3://krabby-real2sim-scenes` (AWS profile `krabby`)
is the **durable/offsite cold store** and the home of the **collab/public tiers**.
**Only j** holds the credential (the gateway) and syncs the store to S3 (backup +
promote-to-public, e.g. mirroring the `krabby-firmware-public` index pattern for
the public tier). **Producer hosts never touch S3** — they only ever talk to j over
the LAN. This layer is **`STO-SCN-028`** (not yet built).

### Architecture

```
   LAYER 2 (cloud)      ┌─────────────────────────────────────────┐
                        │  s3://krabby-real2sim-scenes (profile krabby) │
                        │  durable cold store + collab/public tiers │
                        └────────────────────▲────────────────────┘
                          aws-cli/rclone (krabby profile) — j ONLY   ← STO-SCN-028
   LAYER 1 (fleet LAN)  ┌────────────────────┴─────────────────────┐
                        │  j (jbeeprz) — HUB · source of truth + S3 GATEWAY │
                        │  /games/krabby/scenes  (always-on)        │
                        │  git repo (trunk) + .git/lfs/objects      │
                        │  receive.denyCurrentBranch=updateInstead  │
                        └───────▲────────▲────────▲────────▲────────┘
              push/pull         │        │        │        │   push/pull
        ┌───────────────────────┘        │        │        └───────────────────────┐
   JDP-Mac (inputs)            t         s         d        b   (CUDA: recon outputs)
   /var/krabby/scenes          ~/krabby/scenes  (producers — run experiments anywhere)
        │                                                        each node:
        └─ per op:  git push --no-verify + rsync objects UP      krabby-scenes-sync
                    git pull  + rsync objects DOWN + lfs checkout {push|pull|status}

  Layer 1 (fleet, BUILT): metadata=git/SSH (1 branch, additive) · blobs=rsync .git/lfs/objects · no creds
  Layer 2 (cloud, STO-028): j is the sole S3 gateway (krabby profile); producers never touch S3
```

> **Status (2026-06-04):** live. j seeded (4702 objects @ `trunk`); Mac + t/s/d
> verified-identical mirrors; b synced (its repo is whole) but flaps on a
> stay-awake hardware issue under investigation.

**Per-scene canonical layout = a pipeline of transformations.** The **canonical
spec is [`SCHEMA.md`](./SCHEMA.md)** (authored by STO-SCN-026); the sketch below
is orientation only — SCHEMA.md governs. A scene is an *auditable lineage* from
source to promoted output, where every transform records both **what it did** and
**the exact environment it ran in**.

```
<scene-id>/
  scene.toml                              # scene-level manifest (hashes live in per-transform results.json)
  input/                                  # original source files (raw capture)        [research tier]
    preproc-<NN>-<slug>/{spec,results,data}   # shared preprocessing (transform-shaped)
  pipeline-<slug>/                        # one approach == an image name (matcha, colmap, mast3r, vggt, slam3r)
    run-<slug>/                           # ONE parameterised run (param sweeps = parallel runs)
      run.json                            #   variant identity + promoted flag (← legacy manifest.json)
      transform-<NN>-<slug>/              #   one ordered step
        specification.json                #     WHAT was done (recipe)
        results.json                      #     HOW/WHERE it ran (env + provenance)
        data/                             #     tool-native output, unchanged
      output/                             #   this run's selected output
    output/                               # the pipeline's PROMOTED run                  [collab candidate]
  output/                                 # scene-level PROMOTED outputs (empty today)   [public candidate]
```

**How the data model serves the goals:**
- **Reproducibility / collab / public** — `specification.json` + `results.json`
  per transform mean another engineer (or the public) can see exactly what
  produced an artifact and in what environment, and re-run it. Provenance is
  structural, not a README afterthought.
- **Tiering maps onto pipeline stages** — `input/` (heavy raw) tends to stay
  research-tier; a `pipeline-<slug>/output/` is a collab candidate; the
  scene-level `output/` (promoted, finalized) is the public-tier candidate. Tier
  is still a manifest field, but the stage gives a sensible default.
- **Sync granularity** — the transform directory is the natural sync unit. A host
  running pipeline X pulls only `input/` + `input/preproc-*` + the upstream
  `transform-*/data` it needs — never the whole 50 GB. Manifest/checksum diff at
  this grain is what makes "minimize transfer" real. **Which transforms a pull
  needs is resolved by pipeline code, NOT declared as a DAG in the data** (AID,
  2026-06-04) — `specification.json` records params/inputs, not a dependency graph.
- **`results.json` ⇄ Docker (STO-SCN-031)** — the container image + version that
  the Docker-consume convention runs IS the env recorded in `results.json`. The
  two stories share this contract: the runner emits the provenance the schema
  requires.
- **Maturity is explicit** — every pipeline/transform is `prototype` until
  promoted. **We have no `prototype → promoted` outputs yet** — all historical
  work is INPUT collection + *prototype* transforms. So the scene-level `output/`
  (public tier) is **empty today**; it fills only when a result is deliberately
  promoted. `specification.json` carries the maturity flag.
- **Provenance is dual, going forward** — a transform's data records its env in
  `results.json` AND references the **STO XID it was run under** (AID: "provenance
  captured via Stories and data"). The work-tracking artifact and the data point
  at each other, so the *why* (the story) and the *how* (the env) are both
  recoverable.

### Alternatives Considered

| Alternative | Pros | Cons | Decision |
|-------------|------|------|----------|
| **Split transport: S3 client on j + `rsync` j↔peers** | rsync already on all hosts (zero new dep for LAN leg); S3 client needed on *one* host only | Two tools to reason about | **Selected** — matches what's installed; only j needs an S3 client |
| `rclone` everywhere (S3 + LAN remotes) | One tool spans both legs; checksum dedup | Not installed anywhere; new dep on every host vs rsync already present | Rejected for LAN leg; **viable as j's S3 client** (vs aws-cli) |
| Plain `aws s3 sync` only | Familiar | S3-only — no LAN path; would re-pull per host (no egress dedup) | Rejected: defeats the goal |
| Syncthing (continuous P2P) | Excellent LAN P2P | New daemon; awkward for selective/tiered pulls; sleeping hosts complicate it | Rejected: wrong granularity |
| Custom CLI over urllib + index.json (firmware pattern) | Matches existing precedent | Hand-rolls resumable/checksummed bulk transfer | Partial — reuse the *manifest/index* idea, not the transport |
| Cache host = **j** (always-on, `/games` 1.8 T/624 G free, 125 GiB RAM) | Only 24/7 host; biggest disk; ample page cache | AMD (irrelevant for storage) | **Selected** — confirmed by live probe |
| Cache host = a CUDA box (s/t/b/d) | Closer to compute | Sleep when idle → unreliable as a serve point | Rejected: not always-on |

### Decisions

| XID | Decision | Status | Rationale |
|-----|----------|--------|-----------|
| — | Keep as existing `EPI-SCN-SCENE-SYNC` under `DES-SCN-TX`; no new design | **Resolved** (AID, 2026-06-04) | Operator chose to use the existing artifact, not re-home |
| `HUG-SCN-NNN` | **Two layers.** *Layer 1 (fleet LAN):* git-LFS hub-on-j + git+rsync mirror for experiment load-distribution (built). *Layer 2 (cloud):* **S3 = durable/authoritative cold store + collab/public tiers**, j as gateway (STO-028). S3 sits **behind** j, not in front | **Decided — AS BUILT + AID** | Fleet sync distributes experiments; S3 gives durability + public distribution. Corrects the earlier "no S3" over-rotation |
| `HUG-SCN-NNN` | **Scene = pipeline of transformations** (`input/`→`pipeline-<slug>/run-<slug>/transform-*/`→`output/`); each transform has `specification.json` + `results.json` | **Proposed (AID-directed)** | Makes provenance/reproducibility structural; transform dir = natural sync unit |
| `HUG-SCN-NNN` | **`run-<slug>` level** under each pipeline — param sweeps are *parallel runs*, not sequential steps; `run.json` carries the variant identity (← legacy `manifest.json`) | **Decided (AID)** | The `004-sky-house-curated-*` ×5 sweep proved the gap; canonical in SCHEMA.md |
| `HUG-SCN-NNN` | `results.json` captures full runtime env (OS, NVIDIA driver, software versions) + container `{image, tag, digest}` **reusing M14's tag+digest scheme** (commit-SHA/`<branch>-latest`/semver); **digest is the reproducibility anchor**, captured like `krabby --version` | **Decided** | T-013 — consume M14's contract, don't invent; tags move per-commit, digests don't |
| `HUG-SCN-NNN` | **No runtime scene dependency**: nothing `krabby-launcher` stands up requires scenes (scenes are sim/training assets). A post-launched krabby consumes **public-tier** scenes via a credential-free read client (the `firmware/cli.py` pattern: anonymous S3 + `index.json` + `~/.cache`), distinct from the credentialed dev sync (`krabby-scenes`) | **Decided (AID)** | Keeps launcher lean + no-AWS-creds (M14 posture) while leaving the door open for on-device scene use |
| `HUG-SCN-NNN` | **Pipeline dependency DAG lives in code, not declared in the data** | **Decided (AID)** | Keep `specification.json` to params/inputs; don't bake a graph format into the schema in v1 |
| `HUG-SCN-NNN` | **Dual provenance going forward**: `results.json` (env/data) + the STO XID the transform ran under | **Decided (AID)** | "Provenance captured via Stories and data" — links the *why* to the *how* |
| `HUG-SCN-NNN` | Maturity flag (`prototype` → `promoted`) on each pipeline/transform; `output/` holds only promoted | Proposed | All current work is prototype; public tier stays empty until promotion |
| `HUG-SCN-NNN` | One scene-level `scene.toml` manifest indexing the lineage | Proposed | Inconsistent flat layout is the root problem |
| `HUG-SCN-NNN` | Tier = `scene.toml` field (research/collab/public) | Proposed | Promotion = metadata change; collab/public tiers are also published to their **S3 prefix** (Layer 2, via j) |
| `HUG-SCN-NNN` | **j is the git HUB / source-of-truth + merge point** — `/games/krabby/scenes`, always-on, `receive.denyCurrentBranch=updateInstead`; every node push/pulls j | **Decided — AS BUILT** | Only 24/7 host + biggest disk (probe-chosen); the git hub for the fleet (Layer 1) **and the sole S3 gateway** (Layer 2) |
| `HUG-SCN-NNN` | **Layer 2 transport (j ⇄ S3) = `aws-cli`/`rclone` on j** (the gateway, sole `krabby` profile): backup + collab/public promote | Proposed — STO-028 | Only j needs an S3 client; producers stay creds-free on the LAN leg |
| `HUG-SCN-NNN` | **Scene data lives in a dedicated git-LFS data repo** at `/var/krabby/scenes` (→ external APFS): large binaries via **Git LFS**, metadata (`scene.toml`/`*.json`) in plain git. Separate from the **code** repo (`krabby-research`), which never holds data | **Decided (AID — refines "data is not code")** | Code repo stays binary-free; data gets versioning + LFS-backed large-object storage. Retires the prototype `environments/reconstructed/`-in-git + the `>100 MB→S3 / ≤100 MB→git` split |
| `HUG-SCN-NNN` | **Layer 1 (fleet LAN) transport = git over SSH (metadata; 1 `trunk`, additive merges) + rsync of `.git/lfs/objects` (additive, `--ignore-existing`, never `--delete`)**. git-lfs's SSH transport (`git-lfs-transfer`) deliberately unused (unpackaged on Debian 13; fleet declined a 3rd-party binary in base). Every git op runs `GIT_LFS_SKIP_SMUDGE=1` + `push --no-verify`; `git lfs checkout` materializes the worktree from local objects. Bidirectional — producers push outputs back to j. On the *fleet* leg: **no S3, no rclone, no git-lfs-transfer** (S3 is Layer 2, j-only). | **Decided — AS BUILT** | rsync is fleet-wide + git-over-ssh works; content-addressed objects can't conflict; realizes STO-SCN-029/030 |
| `HUG-SCN-NNN` | `krabby-scenes-sync {push\|pull\|status}` helper wraps the flow; **ansible-provisioned fleet-wide** (`krabby-scenes-sync` role + `/etc/krabby-scenes.conf`; base role ships git-lfs 3.6.1); the Mac runs the same portable script. Cadence **on-demand** | **Decided — AS BUILT** | One uniform tool across producers + hub; the concrete form of STO-SCN-029 |
| `HUG-SCN-NNN` | **Canon container contract**: code baked at `/workspace`; data bind-mounted `-v <host-data>:/data` (RW for transforms, `:ro` for pure consumers) | **Decided** | Matches our *delivered* locomotion/isaacsim images; supersedes the earlier `/games/real2sim/scenes:ro` proposal (that's the host side; `/data` is the in-container mount) |
| `HUG-SCN-NNN` | **`pipeline-<slug>` aligns with our image names** (`colmap`/scene-recon-base, `mast3r`, `matcha`, `vggt`, `slam3r`); transform steps align with `real2sim/run_*.sh` | Proposed | Reuse what exists (T-013); the pipelines already are the images |
| `HUG-SCN-NNN` | **Each transform's `data/` holds the third-party tool's NATIVE output, unchanged** (COLMAP db/sparse/dense, MAtCha tetra/oriented, MASt3R `--save-as`, VGGT COLMAP-format) | **Decided** | "Did not build" — those layouts are fixed by tools we don't own; the schema wraps the *outer* structure, never reorganizes tool internals |
| `HUG-SCN-NNN` | **M11-to-date is prototype**; this schema supersedes the current `environments/reconstructed/`, flat `scenes/<id>/`, and `FOLDER_LAYOUT.md` M11 section. Other delivered milestones are **canon** and are respected/emulated | **Decided (AID)** | Frees the redesign while preserving canon conventions (wheel packages, `/workspace`+`/data`, `krabby-launcher`, ECR) |
| `HUG-SCN-NNN` | **S3 credential (`krabby` profile) lives on j ONLY** (the gateway); producers + Mac have **no AWS creds** — Layer-1 sync is SSH-key over the fleet mesh. T-014 | **Decided — AS BUILT** | Minimizes the credential surface to one host; producers never touch S3 |

### Risks

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| **Layer 2 (j↔S3) not built yet** — no S3 client/profile on j | Medium | Medium | STO-028 puts the `krabby` profile + aws-cli on j (gateway) for backup/public; producers stay creds-free |
| **j (hub) is a single point of failure for pushes/merges** | Medium | Medium | Every node holds a **full mirror** (verified identical), so j is restorable from any peer + all work continues locally while j is down — no central-only data |
| Producers sleep/flap mid-sync (e.g. b's stay-awake fault) | Medium | Low | Additive `--ignore-existing` rsync **converges across windows** — partial pulls accumulate to full parity (proven on b); on-demand wake/kickstart |
| Re-organizing 50 GB risks data loss / broken refs in T0–T4 | Medium | High | Migrate copy-then-verify-then-swap; manifest hashes; never `mv` originals until verified (T-018) |
| Historical provenance unrecoverable from journals (STO-SCN-033) | High | Medium | Reconstruct what the journals support; mark the rest `provenance: deduced`/`unknown` rather than fabricating (T-002); prototypes don't need perfect provenance |
| Legacy ad-hoc objects in `s3://krabby-real2sim-scenes` predate the schema | Low | Low | STO-028 reconciles them into the Layer-2 layout when wiring the j↔S3 backup/publish |

## Success Criteria

- [ ] One documented pipeline-of-transformations schema (`input/`→`pipeline-<slug>/transform-*/`→`output/`) + `scene.toml`; all ~21 existing scenes migrated to conform.
- [ ] Every transform directory carries `specification.json` + `results.json`; a transform can be **re-run on a different host from its records and reproduce its `data/`** (verified, not assumed — T-020).
- [ ] Three tiers defined with an explicit, documented promotion gate (default mapped from pipeline stage).
- [x] `krabby-scenes-sync pull` mirrors a node from j (git refs + additive rsync of `.git/lfs/objects` + `git lfs checkout`); **all nodes verified byte-identical** (4702 objects @ `trunk`, 2026-06-04, T-020).
- [ ] A fleet host + a Docker job consume a scene from its **local mirror** (no network) through the standard `/data` mount (verified, not assumed — T-020).
- [ ] An engineer who has never touched the data gets a full mirror with one command (`krabby-scenes-sync pull`), **no credential handling**.
- [x] **No AWS credential on producer hosts** — Layer-1 sync is SSH-key (fleet mesh); only j (the gateway) holds the `krabby` profile for the Layer-2 S3 link.
- [ ] All stories shipped; M11 data README updated.

## Milestones

| Milestone | Target Date | Actual | Status |
|-----------|-------------|--------|--------|
| Epic re-homed + stories minted | | | open |
| Schema + tiers defined (S001–S003) | | | open |
| Sync CLI + fleet distribution (S004–S005) | | | open |
| Docker + local inspection (S006–S007) | | | open |

## Retrospective

_(Fill in after epic completion.)_

### What Went Well

-

### What Could Be Improved

-

### Lessons Learned

-
