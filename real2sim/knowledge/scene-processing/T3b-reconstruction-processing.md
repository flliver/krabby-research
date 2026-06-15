# T3b — Reconstruction Processing

> Phase 5 of [the M11 process](README.md). Run the chosen model on a GPU host: **point
> `primary` at the subset**, **sync data to the host**, **delegate** the GPU run, **monitor**
> it, and **sync outputs back** into the store. `v4exec` orchestrates all of this in one
> command per model — this doc explains what it does so you can drive and debug it.

## Inputs → Outputs

| In | Out |
|---|---|
| a FINAL-N subset (T3a) + model/settings choice | a reconstruction node `represent/<model>/<id>/` + its mesh(es) `meshify/…/mesh.ply`, all synced into the store with a `jobs/` run record |

## 0. Point `primary` at the subset (locked operator act)

`reconstruct-matcha` / `reconstruct-da3` reconstruct **whatever subset is tagged `primary`** —
they do **not** take `--subset`. Re-pointing `primary` is a **deliberate operator act (locked
#1)**:

```bash
# precull can set it once; thereafter re-pointing is deliberate.
v4exec precull <scene> --set-primary        # first time only (opt-in)
# to reconstruct a specific FINAL-N subset, primary must resolve to it.
```

If `primary` is already set, `v4exec` will **not** silently move it — re-point it intentionally
(or the run reconstructs the wrong subset). Always confirm `primary` before a reconstruct.

## 1. Choose a host

GPU-only. The fleet GPU box used for M11 is **`tbeeprz`**. Pass it as `--host U@H` (e.g.
`--host tbeeprz`). The host needs the model's Docker image + a free GPU (DA3 OOMs if another
tenant holds VRAM — check `nvidia-smi` first; STO-SCN-089).

## 2. Run it (one command per model)

```bash
v4exec reconstruct-matcha <scene> --host tbeeprz --sfm posed [--dense-regul default|strong]
v4exec reconstruct-da3    <scene> --host tbeeprz --sfm posed
```

What `v4exec` does under the hood (so you can debug each stage):

| Stage | Mechanism |
|---|---|
| **Sync → host** | `stage_images_on_host()` rsyncs the primary subset's images + posed cameras to `…/<tag>/` on the host |
| **Delegate** | `ssh <host> 'docker run --rm --gpus all -v <workdir>:/work <IMAGE> <tool>'` (then a chown pass so synced files are operator-owned) |
| **Monitor** | progress published to **MQTT** (`publish_progress`); the full container log is captured to `<node>/{infer,solve}.log`. For long runs on a beeprz host, wrap with `nanny-progress` so phase/percent show on `beeprz dash`. |
| **Sync ← outputs** | `rsync -a <host>:<workdir>/out/ <node>/` pulls results into the store; host scratch is removed |
| **Record** | `v4.write_metadata()` (identity, settings, `measured`: host, duration, image digest, tools sha) + a `jobs/` run record (`job_record`) |

For **matcha** the run is welded: `matcha@{0,1}` → orient bootstrap (local) → **tetra + tsdf
meshes** in the canonical gauge (one dispatch, locked #6). For **DA3** it's: infer (host GPU,
posed from the spine) → **fuse** (local CPU) into the orient gauge.

## 3. Monitor progress

- **MQTT** channel — live phase/percent.
- **`<node>/*.log`** — the container stdout/stderr (last 100k) for failures.
- **`nanny-progress set <phase> <pct>`** on the beeprz host → the work shows on `beeprz dash`
  instead of going dark (clear on exit, always — fleet-ops rule).

## 4. Verify outputs landed (T-018)

After the command returns: confirm `represent/<model>/<id>/metadata.json`, the mesh `.ply`,
and the `jobs/` record exist in the store. `v4exec` prints the materialized ids
(`reconstruct-matcha materialized: represent … tetra … tsdf … orient …`).

## Gotchas

- **`primary` is the silent footgun** — a reconstruct against the wrong `primary` wastes a GPU
  run. Confirm it every time.
- **Free VRAM check** before DA3 dispatch (other tenants OOM it).
- **Idempotent:** identical inputs+settings = NOOP (the node already exists) — safe to re-run.
- The **matcha-free DA3 mesh** (spine-gauge `npz` path) sidesteps `reconstruct-da3`'s matcha
  requirement — see [T3a](T3a-reconstruction-preprocessing.md) + STO-SCN-127.

## Automation status

One command per model handles sync→delegate→monitor→sync-back→record. ✅ automated (the
`primary` re-point is the one deliberate manual gate).

## Next

→ [T3c — Reconstruction Post-Processing](T3c-reconstruction-postprocessing.md)
