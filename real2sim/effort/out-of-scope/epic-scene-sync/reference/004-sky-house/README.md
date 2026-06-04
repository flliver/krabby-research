# Reference scene — `004-sky-house` (worked example)

A **structure-only** conforming example for [`../../SCHEMA.md`](../../SCHEMA.md).
No bulk `data/` is copied here — only the metadata/provenance files, so a
validator and a human can both see the shape.

Built by mapping the **real** `manifest.json` of the legacy
`004-sky-house-curated-12-dense-strong-r3` MAtCha run onto the schema — it
doubles as the migration worked-example for `STO-SCN-033`.

```
004-sky-house/
  scene.toml                                    # ← manifest.scene + captured_at
  pipeline-matcha/
    run-12-dense-strong-r3/
      run.json                                  # ← manifest.variant_name + matcha{} knobs
      transform-01-matcha/
        specification.json                      # ← manifest.matcha{} (the recipe)
        results.json                            # ← manifest.execution{} (host/gpu/duration/vram)
        data/                                   # (real tree: mast3r_sfm/ tetra_meshes/ oriented/ — not copied)
      output/                                   # (run's selected mesh — not copied)
    output/                                      # (promoted run — empty: nothing promoted yet)
  output/                                        # (scene-level public tier — empty)
```

**What it demonstrates**
- A **param-sweep run** under one pipeline (`run-12-dense-strong-r3` is one of
  five MAtCha sweeps of this scene).
- **Honest provenance**: `results.json.provenance = "measured"` (the manifest
  recorded host/GPU/duration/VRAM), but `nvidia_driver`, `cuda`, container
  `digest`, `software` versions, and output `sha256` are all `"unknown"` —
  recorded as such, never fabricated (T-002). This is exactly the "moving tag,
  no digest" gap the M14 digest decision guards against.
- **Scale** starts `uncalibrated` (no scene records metric scale — STO-SCN-016).
