# krabby — topology consumption map

> krabby is **primarily a consumer** of the fleet topology. This doc
> records what krabby binds to from other projects' `shared` exposures,
> discovered via `ccc-bd topology` rather than hardcoded. It is the
> source of truth for the "bind by discovery, not by hostname" stance.
>
> **Discovery, not duplication:** the live truth is always
> `ccc-bd topology ls` — re-run it; don't trust a stale copy here. This
> map names *which* exposures krabby relies on and *why*, with the
> contact to engage. When the fleet changes, re-discover.

## The binding principle

krabby already runs real work on baeprz hosts (real2sim GPU
reconstruction; parkour IsaacLab training; the envoy fleet). Historically
those hostnames were implicit/hardcoded. The formalization:

1. **Discover** the host/service via `ccc-bd topology ls` (it's `shared`,
   so krabby sees it).
2. **Bind by contact**, not by raw hostname — resolve the `contact`
   (`agent@project`) and engage via `/notify <project>` or the named
   envoy. The sherpa **points**; it never brokers and never speaks a
   secret.
3. **Never hardcode** a fleet hostname in krabby code/config when the
   topology layer can resolve it. If a script must pin a host, it cites
   the exposure `id` + contact in a comment so the binding stays
   traceable.

## What krabby binds to (baeprz `shared` exposures)

Resolve live with: `ccc-bd topology ls --project baeprz`

### Compute / envoy hosts — real2sim + parkour

| Exposure | What krabby uses it for | Contact |
|---|---|---|
| `s` (sbeeprz) | krabby-research envoy host — GPU reconstruction / training runs | `silas@baeprz` |
| `t` (tbeeprz) | krabby-research envoy host — GPU reconstruction / training runs | `theo@baeprz` |
| `b` (bbeeprz) | krabby-research envoy host — GPU reconstruction / training runs | `benny@baeprz` |
| `d` (dbeeprz) | krabby-research envoy host (+ Windows VM) — GPU runs | `dax@baeprz` |
| `j` (jbeeprz) | 24/7 server-workstation; hosts the MQTT broker | `ops@baeprz` |

These back the **real2sim** pipeline (COLMAP / MASt3R / VGGT / SLAM3R
GPU stages — the manifest `execution.host` field already records
s/t/b/d/j runs) and **parkour** IsaacLab training. Krabby reaches them
through the per-host **envoy** agents (`silas`/`theo`/`benny`/`dax` —
the `*beeprz` fleet operators).

### Fleet services

| Exposure | What krabby uses it for | Contact |
|---|---|---|
| `mqtt` (Mosquitto on host j) | Fleet telemetry substrate — available for any krabby fleet-touching work that wants to publish/subscribe (mTLS `:8883`, localhost `:1883`) | `ops@baeprz` |
| `nanny-progress` | **Required** for long-running krabby work on a baeprz host — wrap with `nanny-progress set <phase> <pct>` so the operator sees phase+percent on `beeprz dash` (per the global fleet-ops norm). Publishes to mqtt retained topics. | `ops@baeprz` |

> `nanny-progress` is a standing obligation, not just an option: the
> fleet-ops rule says any turn that lands >~30 s of work on a
> beeprz-managed host wraps it with `/usr/local/bin/nanny-progress`
> (clear on exit, always). That binding is discovered here; the
> mechanics live in the global `~/.claude/rules/fleet-ops.md`.

## What krabby exposes (for symmetry — see `.ccc/topology.json`)

Krabby's own `exposes[]` is modest (it's a consumer): the private
`krabby-robot` host (Jetson hexapod; contact `engineer@krabby`) and two
shared public-artifact services — `firmware-store` (S3) and
`locomotion-image` (ECR), both contact `devex@krabby`. Other projects
discover these the same way krabby discovers baeprz's.

> **Local reachability:** krabby's own locally-hosted services are
> addressed via DNS **`krabby.organl.com`** (not raw IPs / `localhost`).
> Canonical statement: `.ccc/knowledge/architecture.md` §8 (Networking).
> This is the self-hosted counterpart to the bind-by-discovery principle
> above — point here, don't restate (T-023).

## When the map drifts

If `ccc-bd topology ls` shows a baeprz host/service krabby uses but this
map doesn't (or vice-versa), the **live CLI wins** — update this doc,
don't trust the table. New shared exposures krabby starts depending on
get a row here with their contact.
