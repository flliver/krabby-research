---
xid: STO-SCN-036
parent: ./epic.md
kind: story
effort: scn
size: M
status: shipped
date: 2026-06-04
depends-on: []
bd-id: krabby-nie
assignee: principal
shipped: 2026-06-05
tasks: 9
complete: 9
---

# Reconstruct legacy-scene provenance from M11 journals (deduced->measured where recoverable)

## Summary

The 12 legacy-transform `results.json` records that STO-SCN-033 migrated with
`provenance: "deduced"` are upgraded to `"measured"` **only where independent
sources agree** (M11 journal threads, the `backfill_manifests.py` pattern, the
on-host `.claude` session histories, the in-repo run-scripts, and tool-native
on-disk evidence) — each field traced to its source in an auditable ledger, with
honestly-unrecoverable records left `deduced`/`unknown` rather than fabricated.

## Context

Split out of **STO-SCN-033** (migration), which deliberately deferred journal
provenance reconstruction here. 033 set every transform's `results.json`
`provenance` to:

- `measured` — the 5 curated `004-sky-house` MAtCha runs + `dtu-bicycle` curated
  (carried a `manifest.json`); **6 records**.
- `deduced` — the legacy multi-tool dirs with no manifest; **12 records** (this story).
- input-only — raw captures (no transform record; out of scope).

Parent epic: `EPI-SCN-SCENE-SYNC`. Sibling follow-on: STO-SCN-037 (`eval/` schema
home). The reconstructed records feed the epic's tiering + sync work with honest
provenance, and exercise the schema's `provenance`/`environment` leniency that
STO-SCN-034 must validate.

## Problem

The 12 `deduced` records assert nothing about *how/where/when* the legacy
reconstructions ran — host, GPU, date, duration, params, software. That is
recoverable for several of them (the scout found the journal's Phase-A recipe was
applied to "001 patio, 003 firepit, 004 sky-house", and `backfill_manifests.py`
already reconstructed the sky-house curated set from journal + on-disk evidence),
but **not all** of them (slam3r has no run-script and is journal-silent). The story
recovers what the evidence supports and stops exactly there.

### The 12 `deduced` records (the work-list)

| Scene | Transforms (pipeline) | Prior recoverability read |
|-------|----------------------|---------------------------|
| `004-sky-house` (legacy/dining) | mast3r, matcha | High — journal + backfill era (2026-05-01, tbeeprz) |
| `001-patio` | colmap, mast3r, matcha, vggt | matcha/mast3r Med (Phase-A recipe); colmap/vggt Low–Med |
| `003-firepit` | mast3r, matcha, slam3r | matcha/mast3r Med; **slam3r Low** (no run-script, journal-silent) |
| `002-patio` | colmap | Low–Med (dewarp variant; params from script, date from db/session) |
| `dtu-bicycle` | colmap, matcha | Med (DTU benchmark; `r=0.1` known per the scene-examples note) |

## Design

### Approach

Source-of-truth precedence, mirroring how `backfill_manifests.py` was built
("journal notes + on-disk evidence"):

1. **M11 journal threads** —
   `…/011-scene-reconstruction/journal/journals/m11-scene-reconstruction/threads/`
   (the `threads/` tree; the top `journal.md` is only a 33-line index). Dated notes
   (`2026-04-30 … 2026-05-06`) give recipe, host pattern (tbeeprz / RTX 5080 16 GB),
   and the explicit "001/003/004" Phase-A attribution.
2. **`backfill_manifests.py` + `manifest_lib.py`** — the canonical reconstructed-record
   *shape* and the already-measured field values to reuse (T-013). Note even the
   measured runs carry `git_sha: None` — so `software.*git_sha` stays `unknown`
   everywhere; do not invent one.
3. **`.claude` session histories** — **scanned 2026-06-05, weak source, do not rely
   on it as primary.** The fleet-host sessions (t/s/d; b down) are tiny (364 KB) and
   **never recorded the recon runs** — M11 reconstruction was script/hand-driven, not
   Claude-driven on the hosts. The Mac history is large (1094 sessions) but the recon
   tokens first appear **2026-05-06**, *after* the original 04-30/05-01 runs — i.e.
   downstream discussion, not the run-commands (the 05-01 orchestration session did not
   survive in greppable form). Use only as soft date-corroboration; the primary basis
   is sources 1, 4, 5.
4. **In-repo run-scripts** — `run_colmap_*.sh`, `run_mast3r.sh`, `run_vggt.sh`,
   `run_mesh_conditioning.sh` give exact per-tool params (the "what"). No
   `run_slam3r.sh` exists → slam3r params unrecoverable from scripts.
5. **Tool-native on-disk evidence** — COLMAP `database.db` internal timestamps,
   `cameras.json`, frame counts (survive the CoW migration) for independent dating.
   The unmounted source volume `/Volumes/Archives-01/krabby/data/scenes` is a
   fallback for original file mtimes if remounted.

**The measured/deduced gate (T-002):** a field is written as fact only when at
least two independent sources agree (e.g. journal date + db timestamp; script
params + cameras.json frame count). A record flips to `provenance: "measured"`
only when host **and** date **and** params are corroborated; otherwise it stays
`deduced`, and any single-source or absent field is `unknown`. No value is ever
invented to fill a slot.

### Changes

| File | Change |
|------|--------|
| `real2sim/scenes/reconstruct_provenance.py` | add — reads the 5 sources, writes per-transform `results.json`, emits the ledger; idempotent, never fabricates |
| `real2sim/scenes/provenance-ledger.md` | add — per-transform source→field table; the auditable record of every deduced-vs-measured call |
| `/var/krabby/scenes/**/run-legacy/transform-01-legacy/results.json` | modify — upgraded records (measured where corroborated; deduced/unknown otherwise) committed to the scene store |

## Definition of Done

- [x] Session histories re-scanned (2026-06-05): fleet-host sessions never recorded
      the runs; Mac sessions post-date them → not a primary source (recorded above).
- [x] All 12 `results.json` updated (+ `specification.json` params): **1 measured**
      (`004-sky-house/matcha` — host tbeeprz, journal+backfill+mtime), **11 deduced**
      but enriched with real on-disk dates + script-derived params; unknown fields `unknown`/null.
- [x] **No fabricated values** — every value traces to a named source in the ledger;
      `slam3r` left params-empty, `002-patio`/`dtu-colmap` dates left null (T-002).
- [x] `provenance-ledger.md` committed: one row per (scene, pipeline) → provenance, date, sources, note.
- [x] All 12 records validate against `results.schema.json` (structural validator,
      incl. measured→host conditional + container `additionalProperties`; formal `jsonschema` CI = STO-SCN-034).
- [x] Changes committed to the `/var/krabby/scenes` store and pushed to the j hub (`f691e4d..dad9f73`).

## Testing

### Unit / fixture tests

- [x] A record with only single-source evidence stays `deduced` (verified against
      real output — `003-firepit/slam3r` is `deduced`).
- [~] A `measured` record fails the ledger check if any field lacks a source _(out of spec — a formal automated guard belongs to the validation harness STO-SCN-034; here the gate is enforced by the reviewed facts table, verified by inspection)_.
- [x] slam3r (no run-script, journal-silent) remains `deduced` with `unknown` env (verified).

### Integration

- [x] Re-running the extractor is idempotent (re-ran 2026-06-05 → 0 store diffs).

## Out of scope

- **`eval/` artifacts + `_unsorted/` re-sorting** → STO-SCN-037.
- **The 6 already-`measured` records** (curated sky-house + dtu) — already carry
  manifest-backed provenance; not re-derived here.
- **Remounting the source volume** — used only if the on-disk/session evidence is
  insufficient; not a required step.
- **Formal `jsonschema` CI gate** → STO-SCN-034 (this story just makes the records conform).

## Implementation Notes

### What Changed

Built `real2sim/scenes/reconstruct_provenance.py` with a reviewed per-(scene,pipeline)
**facts table** (the human judgment, like `migrate.py`'s MAP). It reads live on-disk
mtimes + the facts, writes each transform's `results.json` (the how/where/when) and
enriches `specification.json` `parameters` (the what, from run-scripts), and emits
`provenance-ledger.md`. Outcome: **1 measured, 11 deduced** — but all 11 deduced
records gained real dates and script-derived params, so the practical value is broad
*enrichment*, not just the single provenance upgrade.

The biggest discovery: **the CoW migration (`cp -cR`) preserved original file mtimes**,
making on-disk dates a genuine measured source (001-patio matcha 2026-04-30, sky-house
matcha 2026-04-29, etc.) — this is what made dating possible at all, since the journal
is thin for the legacy scenes and the session histories didn't record the runs.

### Files Modified

- `real2sim/scenes/reconstruct_provenance.py` — add (extractor + facts table).
- `real2sim/scenes/provenance-ledger.md` — add (auditable source→verdict ledger).
- `/var/krabby/scenes/**/run-legacy/transform-01-legacy/{results,specification}.json` —
  24 files updated; pushed to j (`dad9f73`).

### Gotchas

- **mtime traps:** `dtu-bicycle/colmap` mtime is 2022 (upstream DTU dataset, not our
  run) → date left null; `dtu-bicycle/matcha` inputs predate the run → used *max* mtime,
  not min. Per-transform `date_mode` (min/max/none) encodes this.
- **`002-patio/colmap`** has empty `sparse/`+`dense/` → flagged `status: partial`, date null.
- **`measured` is conservative:** only `004-sky-house/matcha` cleared the bar (host known).
  `004-sky-house/mast3r` is *probably* tbeeprz too but isn't separately attested, so it
  stays `deduced` — the honest call (T-002).
- Params live in `specification.json`, not `results.json` (schema is `additionalProperties:false`);
  the extractor writes both.

## Status notes

- 2026-06-05: Picked up by principal, beginning review. (Body is still the
  skeleton — first sub-step is authoring it.)
- 2026-06-05: Scouted all 4 provenance sources (journals, fleet data, `.claude`
  session histories, scripts) + tool-native evidence. Found the gap is exactly 12
  records across 6 scenes; `backfill_manifests.py` is the reconstruction template;
  journal `threads/` is rich for sky-house, thin for patio/firepit; session-history
  token scan needs a quoting fix. Authored the story body from the scout.
- 2026-06-05: Re-scanned session histories (fixed quoting; stdin-heredoc to dodge
  remote-zsh). **Finding: source #3 is weak.** Fleet sessions (364 KB) never recorded
  the recon; Mac sessions (1.7 GB) only mention scenes from 2026-05-06 onward —
  downstream of the 04-30/05-01 runs. Primary basis is journal threads + run-scripts
  + on-disk tool evidence. Story §Approach + DoD updated. Ready to build the extractor
  on operator go-ahead.
- 2026-06-05: Built + ran `reconstruct_provenance.py`. **12/12 records reconstructed
  (1 measured, 11 deduced-but-enriched), all schema-valid, ledger emitted, pushed to j
  (`dad9f73`).** Key enabler: CoW preserved original mtimes → real dates. Work complete;
  closing.
