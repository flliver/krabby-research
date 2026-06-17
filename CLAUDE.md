# krabby — project context

> Auto-loaded into every agent session started in this project.
> Keep this file tiny — a few universal facts plus `@`-pointers to the
> canonical knowledge docs (don't restate them here; T-023 DRY).

## Networking

Local services are reached via DNS **`krabby.organl.com`** — not raw
IPs or `localhost`. Address anything krabby hosts locally (dashboards,
viewers, debug servers, `ccc-combine`, the real2sim verify viewer)
through that name. Topology + reachability detail (and the fleet-side
"bind by discovery" stance) lives in the canonical docs below.

## Canonical knowledge

- System architecture + networking: @.ccc/knowledge/architecture.md
- How we work (the tour): @.ccc/knowledge/orientation.md
